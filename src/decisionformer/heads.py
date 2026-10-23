"""Decision heads. Each returns logits aligned with the question's options, in the order they were shown."""

from __future__ import annotations

import inspect
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from decisionformer.config import HeadConfig
from decisionformer.registry import HEADS

_INTERNAL_ARGUMENTS = {"self", "hidden_size", "max_options", "code_rows"}


def _projection(hidden_size: int, out_features: int, mlp_hidden: int | None, dropout: float) -> nn.Module:
    if mlp_hidden is None:
        return nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, out_features))
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(mlp_hidden, out_features))


class Head(nn.Module):
    pooler_level = "example"
    question_types: tuple[str, ...] = ("choice", "noul", "score")
    needs_code_tokens = False

    def check_rows(self, batch: Any, rows: torch.Tensor) -> None:
        for row in rows.tolist():
            question_type = batch.question_types[row]
            if question_type not in self.question_types:
                raise ValueError(f"{type(self).__name__} does not handle '{question_type}' questions "
                                 f"(record {batch.ids[row]})")


@HEADS.register("fixed_softmax")
class FixedSoftmaxHead(Head):
    """A classifier over a fixed label set; each question's options are looked up by key."""

    question_types = ("choice", "noul")

    def __init__(self, hidden_size: int, labels: list[str], mlp_hidden: int | None = None, dropout: float = 0.0,
                 max_options: int = 26, code_rows: torch.Tensor | None = None) -> None:
        super().__init__()
        if len(set(labels)) != len(labels) or not labels:
            raise ValueError("fixed_softmax needs a nonempty list of unique labels")
        self.labels = list(labels)
        self.index = {label: i for i, label in enumerate(self.labels)}
        self.proj = _projection(hidden_size, len(labels), mlp_hidden, dropout)

    def forward(self, pooled: torch.Tensor, batch: Any, rows: torch.Tensor) -> torch.Tensor:
        logits = self.proj(pooled.float())
        width = batch.option_mask.shape[1]
        gather = torch.zeros((len(rows), width), dtype=torch.long, device=logits.device)
        for i, row in enumerate(rows.tolist()):
            for slot, key in enumerate(batch.option_keys[row]):
                if key not in self.index:
                    raise ValueError(f"record {batch.ids[row]}: option '{key}' is not in fixed_softmax labels")
                gather[i, slot] = self.index[key]
        return logits.gather(1, gather)


@HEADS.register("code_readout")
class CodeReadoutHead(Head):
    """Scores the answer-code tokens (A, B, ...) at the decision position, initialized from the LM head rows."""

    needs_code_tokens = True
    trainable_default = True

    def __init__(self, hidden_size: int, max_options: int = 26, code_rows: torch.Tensor | None = None,
                 trainable: bool | None = None, init_from_lm_head: bool = True) -> None:
        super().__init__()
        self.readout = nn.Linear(hidden_size, max_options, bias=False)
        if init_from_lm_head and code_rows is not None:
            with torch.no_grad():
                self.readout.weight.copy_(code_rows.float())
        trainable = self.trainable_default if trainable is None else trainable
        self.readout.weight.requires_grad_(trainable)

    def forward(self, pooled: torch.Tensor, batch: Any, rows: torch.Tensor) -> torch.Tensor:
        width = batch.option_mask.shape[1]
        return self.readout(pooled.float())[:, :width]


@HEADS.register("lm_verbalizer")
class LMVerbalizerHead(CodeReadoutHead):
    """The frozen LM head restricted to the answer codes: zero new parameters, useful zero-shot."""

    trainable_default = False

    def __init__(self, hidden_size: int, max_options: int = 26, code_rows: torch.Tensor | None = None) -> None:
        if code_rows is None:
            raise ValueError("lm_verbalizer needs the backbone's LM head")
        super().__init__(hidden_size, max_options=max_options, code_rows=code_rows, trainable=False)


@HEADS.register("option_scoring")
class OptionScoringHead(Head):
    """Scores each option from its own pooled representation; any number of options, order-independent."""

    pooler_level = "option"

    def __init__(self, hidden_size: int, mlp_hidden: int | None = None, dropout: float = 0.0,
                 max_options: int = 26, code_rows: torch.Tensor | None = None) -> None:
        super().__init__()
        self.score = _projection(hidden_size, 1, mlp_hidden, dropout)

    def forward(self, pooled: torch.Tensor, batch: Any, rows: torch.Tensor) -> torch.Tensor:
        return self.score(pooled.float()).squeeze(-1)


@HEADS.register("binary")
class BinaryHead(Head):
    """One logit z for 'true'; returned as [0, z] so a softmax over the two options equals sigmoid(z)."""

    question_types = ("noul",)

    def __init__(self, hidden_size: int, mlp_hidden: int | None = None, dropout: float = 0.0,
                 max_options: int = 26, code_rows: torch.Tensor | None = None) -> None:
        super().__init__()
        self.proj = _projection(hidden_size, 1, mlp_hidden, dropout)

    def forward(self, pooled: torch.Tensor, batch: Any, rows: torch.Tensor) -> torch.Tensor:
        z = self.proj(pooled.float())
        return torch.cat([torch.zeros_like(z), z], dim=-1)


@HEADS.register("ordinal")
class OrdinalHead(Head):
    """Cumulative-link model for ordered levels: P(y > k) = sigmoid(s - theta_k) with increasing thresholds."""

    question_types = ("score",)

    def __init__(self, hidden_size: int, num_levels: int, mlp_hidden: int | None = None, dropout: float = 0.0,
                 max_options: int = 26, code_rows: torch.Tensor | None = None) -> None:
        super().__init__()
        if num_levels < 2:
            raise ValueError("ordinal needs num_levels >= 2")
        self.num_levels = num_levels
        self.proj = _projection(hidden_size, 1, mlp_hidden, dropout)
        self.first_threshold = nn.Parameter(torch.tensor(-(num_levels - 2) / 2.0))
        self.gaps = nn.Parameter(torch.zeros(num_levels - 2))

    def thresholds(self) -> torch.Tensor:
        gaps = F.softplus(self.gaps) + 1e-3
        return torch.cat([self.first_threshold.view(1), self.first_threshold + torch.cumsum(gaps, 0)])

    def forward(self, pooled: torch.Tensor, batch: Any, rows: torch.Tensor) -> torch.Tensor:
        for row in rows.tolist():
            if int(batch.num_options[row]) != self.num_levels:
                raise ValueError(f"record {batch.ids[row]}: ordinal head expects {self.num_levels} levels")
        score = self.proj(pooled.float())
        above = torch.sigmoid(score - self.thresholds()[None, :])
        ones, zeros = torch.ones_like(score), torch.zeros_like(score)
        upper = torch.cat([ones, above], dim=-1)
        lower = torch.cat([above, zeros], dim=-1)
        return torch.log((upper - lower).clamp_min(1e-9))


def build_head(config: HeadConfig, hidden_size: int, max_options: int, code_rows: torch.Tensor | None) -> Head:
    cls = HEADS.get(config.type)
    return cls(hidden_size=hidden_size, max_options=max_options, code_rows=code_rows, **config.kwargs())


def validate_head_config(name: str, config: HeadConfig) -> None:
    cls = HEADS.get(config.type)
    signature = inspect.signature(cls.__init__)
    parameters = set(signature.parameters) - _INTERNAL_ARGUMENTS
    unknown = set(config.kwargs()) - parameters
    if unknown:
        raise ValueError(f"head '{name}' ({config.type}) got unknown arguments {sorted(unknown)}; "
                         f"accepted: {sorted(parameters)}")
    missing = [p.name for p in signature.parameters.values()
               if p.name not in _INTERNAL_ARGUMENTS and p.default is inspect.Parameter.empty
               and p.kind is p.POSITIONAL_OR_KEYWORD and p.name not in config.kwargs()]
    if missing:
        raise ValueError(f"head '{name}' ({config.type}) is missing required arguments {missing}")


def head_class(config: HeadConfig) -> type[Head]:
    return HEADS.get(config.type)
