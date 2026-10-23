"""Load decision checkpoints and answer System One requests: `AutoDecisionModel.from_pretrained(...).decide(...)`.

Any checkpoint in the shared decision layout loads here: decisionformer exports, AutoJev exports (for example
FrontiersMind/Lumma-fev-27B) and Decision 3.0 checkpoints (`vllm-sr/d3*`). The prompt format is read from
`decision_config.json`; checkpoints trained on other prompts need their format registered first.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from decisionformer.attention import resolve_mode
from decisionformer.backbone import Backbone, context_limit, load_backbone, resolve_kind
from decisionformer.config import HeadConfig, PoolerConfig
from decisionformer.data import make_batch, split_micro_batches
from decisionformer.formats import get_format
from decisionformer.heads import CodeReadoutHead, build_head
from decisionformer.model import DECISION_CONFIG, HEADS, READOUT, DecisionModel, model_card
from decisionformer.poolers import build_pooler
from decisionformer.prompting import ContextLengthError, Encoded, PromptBuilder, prepare_tokenizer
from decisionformer.schema import Decision, DecisionInputError, parse_questions


class UnsupportedInput(ValueError):
    """An input this model cannot read (for example images on a text-only checkpoint)."""


def _local_dir(name_or_path: str | Path, revision: str | None, token: str | None) -> Path:
    path = Path(name_or_path).expanduser()
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(name_or_path), revision=revision, token=token))


def _prompt_name(config: dict[str, Any]) -> str:
    name = config.get("prompt") or config.get("prompt_format")
    if name:
        return str(name)
    return "d3" if str(config.get("format_id", "")).startswith("d3") else "autojev"


def _default_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
    return torch.float32


class Decider:
    """A loaded decision model. `decide(state, questions)` returns `{"model", "answers", "usage"}`."""

    def __init__(self, model: DecisionModel, builder: PromptBuilder, tokenizer: Any, config: dict[str, Any], *,
                 device: torch.device, name: str, max_input_tokens: int | None, batch_size: int = 16,
                 batch_tokens: int = 32768, path: Path | None = None) -> None:
        self.model = model.eval()
        self.builder = builder
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.name = name
        self.max_input_tokens = max_input_tokens
        self.batch_size = batch_size
        self.batch_tokens = batch_tokens
        self.path = path
        self.temperature = float(config.get("temperature", 1.0))
        self.temperature_by_type = {k: float(v) for k, v in (config.get("temperature_by_type") or {}).items()}

    # ------------------------------------------------------------------------------------------- loading
    @classmethod
    def from_pretrained(cls, name_or_path: str | Path, *, revision: str | None = None,
                        device: str | torch.device | None = None, dtype: torch.dtype | None = None,
                        prompt: str | None = None, max_length: int | None = None, token: str | None = None,
                        batch_size: int = 16, trust_remote_code: bool = False, merge: bool = True) -> Decider:
        from transformers import AutoConfig, AutoTokenizer

        root = _local_dir(name_or_path, revision, token)
        config_file = root / DECISION_CONFIG
        if not config_file.is_file():
            raise ValueError(f"{name_or_path} is not a decision checkpoint (no {DECISION_CONFIG}). Train one with "
                             "`decisionformer train`, or evaluate a stock model zero-shot with an lm_verbalizer head.")
        config = json.loads(config_file.read_text())
        ext = config.get("decisionformer") or {}
        style = get_format(prompt or _prompt_name(config))
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = dtype or _default_dtype(device)
        tokenizer_dir = root if (root / "tokenizer_config.json").is_file() else root / "processor"
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=trust_remote_code)
        prepare_tokenizer(tokenizer)
        adapter = ext.get("backbone") == "adapter"
        source = config.get("base_model") if adapter else str(root)
        model_revision = config.get("revision") if adapter else None
        hf_config = AutoConfig.from_pretrained(source, revision=model_revision, trust_remote_code=trust_remote_code)
        kind = ext.get("kind") or resolve_kind(hf_config)
        attention = resolve_mode(kind, {"causal": "causal", "noncausal_full_attention": "bidirectional",
                                        "prefix_bidirectional": "prefix_bidirectional"}.get(
                                            config.get("attention_mode", "causal"), "causal"))
        model_cfg = SimpleNamespace(backbone=source, revision=model_revision, trust_remote_code=trust_remote_code,
                                    attn_implementation="sdpa", init_adapter=None)
        hf_model, _ = load_backbone(model_cfg, None, kind=kind, attention=attention, dtype=dtype, device=device,
                                    code_token_ids=None, source=source if not adapter else None)
        if adapter:
            from peft import PeftModel

            hf_model = PeftModel.from_pretrained(hf_model, str(root / "adapter"))
            if merge:
                hf_model = hf_model.merge_and_unload()
        hf_model.requires_grad_(False)
        backbone = Backbone(hf_model, kind, attention)
        codes = config.get("codes") or []
        if (root / READOUT).is_file() and not (root / HEADS).is_file():
            from safetensors.torch import load_file

            weight = load_file(str(root / READOUT))["weight"]
            head = CodeReadoutHead(backbone.hidden_size, max_options=weight.shape[0], init_from_lm_head=False)
            head.readout.weight.data.copy_(weight.float())
            poolers = {"p0": build_pooler(PoolerConfig(type="decision_token"), backbone.hidden_size)}
            model = DecisionModel(backbone, poolers, {"default": head}, {"default": "p0"})
        else:
            model = _model_from_heads(root, ext, backbone, len(codes))
        model.to(device)
        limit = max_length or config.get("max_length") or context_limit(hf_config, tokenizer)
        builder = PromptBuilder(tokenizer, style, kind, max_length=limit, overflow="error", codes=codes,
                                code_token_ids=config.get("token_ids") or None,
                                chat_template_kwargs=ext.get("chat_template_kwargs"))
        name = ext.get("name") or (str(name_or_path) if not Path(str(name_or_path)).is_dir() else root.name)
        return cls(model, builder, tokenizer, config, device=device, name=name, max_input_tokens=limit,
                   batch_size=batch_size, path=root)

    # ------------------------------------------------------------------------------------------- answering
    def temperature_for(self, question_type: str) -> float:
        return self.temperature_by_type.get(question_type, self.temperature)

    def decide(self, state: Any, questions: dict[str, Any], images: Sequence[Any] | None = None) -> dict[str, Any]:
        """Answer every question about `state` with a probability for each option."""
        return self.decide_many([{"state": state, "questions": questions, "images": images}])[0]

    @torch.inference_mode()
    def decide_many(self, requests: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """Answer several requests; questions from all of them share forward passes."""
        decisions: list[Decision] = []
        owners: list[int] = []
        for number, request in enumerate(requests):
            if request.get("images") or _has_image_parts(request.get("state")):
                raise UnsupportedInput("image inputs are not supported by decisionformer models yet")
            for question in parse_questions(request.get("questions")):
                if question.num_options > len(self.builder.codes):
                    raise ContextLengthError(f"question {question.key!r} has {question.num_options} options; this "
                                             f"model takes at most {len(self.builder.codes)} options per choice")
                decisions.append(Decision(id=question.key, request_id=str(number), state=request.get("state"),
                                          question=question, task=question.task or "default"))
                owners.append(number)
        encoded = [self.builder.encode(d, max_length=self.max_input_tokens) for d in decisions]
        responses = [{"model": self.name, "answers": {}, "usage": {"input_tokens": 0, "output_tokens": 0}}
                     for _ in requests]
        for decision, owner, item, logits in zip(decisions, owners, encoded, self.code_logits(decisions, encoded)):
            scaled = logits[:decision.num_options].float() / self.temperature_for(decision.question_type)
            responses[owner]["answers"][decision.question.key] = self.builder.format.answer(
                decision.question, scaled.softmax(-1).tolist())
            responses[owner]["usage"]["input_tokens"] += len(item.input_ids)
        return responses

    def code_logits(self, decisions: list[Decision], encoded: list[Encoded]) -> list[torch.Tensor]:
        """One row of raw answer-code logits per decision (before temperature)."""
        lengths = [len(e.input_ids) for e in encoded]
        rows: list[torch.Tensor] = [torch.empty(0)] * len(decisions)
        for group in split_micro_batches(range(len(decisions)), lengths, self.batch_size, self.batch_tokens):
            batch = make_batch([decisions[i] for i in group], [list(range(decisions[i].num_options)) for i in group],
                               [encoded[i] for i in group], self.tokenizer.pad_token_id).to(self.device)
            logits = self.model(batch).float()
            for row, index in enumerate(group):
                rows[index] = logits[row]
        return rows

    __call__ = decide

    # ------------------------------------------------------------------------------------------- saving
    def save_pretrained(self, directory: str | Path) -> Path:
        """Copy this checkpoint (merged, standard layout) to `directory`."""
        if self.path is None:
            raise ValueError("This model was not loaded from a checkpoint directory")
        out = Path(directory)
        if out.resolve() != self.path.resolve():
            shutil.copytree(self.path, out, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".cache", ".git*"))
        if not (out / "README.md").is_file():
            (out / "README.md").write_text(model_card(self.config))
        return out

    def push_to_hub(self, repo_id: str, *, private: bool = True, token: str | None = None,
                    commit_message: str = "Upload decision model") -> str:
        from decisionformer.hub import push_folder

        with tempfile.TemporaryDirectory() as tmp:
            folder = self.save_pretrained(Path(tmp) / "checkpoint")
            return push_folder(folder, repo_id, private=private, token=token, commit_message=commit_message)


AutoDecisionModel = Decider


def _has_image_parts(state: Any) -> bool:
    return isinstance(state, list) and any(isinstance(p, dict) and p.get("type") in ("image", "image_url")
                                           for p in state)


def _model_from_heads(root: Path, ext: dict[str, Any], backbone: Backbone, num_codes: int) -> DecisionModel:
    from safetensors.torch import load_file

    if not ext.get("heads"):
        raise ValueError(f"{root}: no {READOUT} and no decisionformer heads; cannot build the decision head")
    poolers = {key: build_pooler(PoolerConfig(**cfg), backbone.hidden_size) for key, cfg in ext["poolers"].items()}
    heads = {name: build_head(HeadConfig(**cfg), backbone.hidden_size, num_codes, None)
             for name, cfg in ext["heads"].items()}
    model = DecisionModel(backbone, poolers, heads, ext["head_poolers"])
    state = load_file(str(root / HEADS))
    missing = model.heads.load_state_dict({k[6:]: v for k, v in state.items() if k.startswith("heads.")})
    model.poolers.load_state_dict({k[8:]: v for k, v in state.items() if k.startswith("poolers.")})
    if missing.missing_keys or missing.unexpected_keys:
        raise ValueError(f"Head weights do not match the checkpoint description: {missing}")
    return model


def validate_request(body: Any) -> tuple[Any, dict[str, Any]]:
    if not isinstance(body, dict) or "questions" not in body:
        raise DecisionInputError("a request needs `state` and `questions`")
    return body.get("state"), body["questions"]
