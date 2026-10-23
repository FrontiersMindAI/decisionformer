"""Poolers turn token hidden states into one vector per example or one vector per option."""

from __future__ import annotations

import inspect
from typing import Any

import torch
from torch import nn

from decisionformer.config import PoolerConfig
from decisionformer.registry import POOLERS


class Pooler(nn.Module):
    level = "example"  # "example" -> (batch, hidden); "option" -> (batch, options, hidden)

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size


@POOLERS.register("decision_token")
class DecisionTokenPooler(Pooler):
    """The token the prompt designates: the last token for decoders, CLS for encoders, or the marker token."""

    def forward(self, hidden: torch.Tensor, batch: Any) -> torch.Tensor:
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), batch.decision_index]


@POOLERS.register("last_token")
class LastTokenPooler(Pooler):
    def forward(self, hidden: torch.Tensor, batch: Any) -> torch.Tensor:
        last = batch.attention_mask.sum(-1) - 1
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


@POOLERS.register("cls", "first_token")
class FirstTokenPooler(Pooler):
    def forward(self, hidden: torch.Tensor, batch: Any) -> torch.Tensor:
        return hidden[:, 0]


@POOLERS.register("mean")
class MeanPooler(Pooler):
    def forward(self, hidden: torch.Tensor, batch: Any) -> torch.Tensor:
        weights = batch.attention_mask.to(hidden.dtype).unsqueeze(-1)
        return (hidden * weights).sum(1) / weights.sum(1).clamp_min(1.0)


@POOLERS.register("attention")
class AttentionPooler(Pooler):
    """A learned query attends over all non-padding tokens."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__(hidden_size)
        self.query = nn.Parameter(torch.randn(hidden_size) * 0.02)

    def forward(self, hidden: torch.Tensor, batch: Any) -> torch.Tensor:
        scores = hidden.float() @ self.query.float() / self.hidden_size**0.5
        scores = scores.masked_fill(batch.attention_mask == 0, torch.finfo(scores.dtype).min)
        weights = scores.softmax(-1).to(hidden.dtype)
        return torch.einsum("bt,bth->bh", weights, hidden)


@POOLERS.register("option_spans")
class OptionSpanPooler(Pooler):
    """Mean of the tokens of each option's line in the prompt."""

    level = "option"

    def forward(self, hidden: torch.Tensor, batch: Any) -> torch.Tensor:
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        starts, ends = batch.option_spans[..., 0], batch.option_spans[..., 1]
        inside = (positions[None, None, :] >= starts[..., None]) & (positions[None, None, :] < ends[..., None])
        weights = inside.to(hidden.dtype)
        pooled = torch.einsum("bkt,bth->bkh", weights, hidden)
        return pooled / weights.sum(-1, keepdim=True).clamp_min(1.0)


def default_pooler_type(kind: str, attention: str, head_level: str, reads_codes: bool = False) -> str:
    """Option heads pool each option's tokens; encoders use [CLS]. On decoders, heads that read answer codes use the
    answer position in every attention mode (the Decision 3.0 layout: last token, also when non-causal); other
    heads use the last token under causal attention and the mean otherwise."""
    if head_level == "option":
        return "option_spans"
    if kind == "encoder":
        return "cls"
    return "decision_token" if attention == "causal" or reads_codes else "mean"


def build_pooler(config: PoolerConfig, hidden_size: int) -> Pooler:
    cls = POOLERS.get(config.type)
    return cls(hidden_size=hidden_size, **config.kwargs())


def validate_pooler_config(config: PoolerConfig) -> None:
    if config.type == "auto":
        if config.kwargs():
            raise ValueError("pooler type 'auto' takes no arguments")
        return
    cls = POOLERS.get(config.type)
    parameters = set(inspect.signature(cls.__init__).parameters) - {"self", "hidden_size"}
    unknown = set(config.kwargs()) - parameters
    if unknown:
        raise ValueError(f"pooler '{config.type}' got unknown arguments {sorted(unknown)}")
