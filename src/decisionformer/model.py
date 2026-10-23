"""The decision network (backbone -> pooler -> head per task), its construction, and export in the standard layout.

Exports follow the decision-checkpoint convention shared by AutoJev, Decision 3.0 (vLLM), pplx-decider and
others: the backbone's own `config.json` and weights at the repository root, the tokenizer, a
`decision_config.json`, and a 255-row answer-code readout in `readout.safetensors`. Models with other heads
add `heads.safetensors` and describe them in the `decisionformer` section of `decision_config.json`.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import nn

from decisionformer import __version__
from decisionformer.attention import CHECKPOINT_NAMES, resolve_mode
from decisionformer.backbone import Backbone, context_limit, load_backbone, load_tokenizer, resolve_kind
from decisionformer.config import PoolerConfig, RunConfig
from decisionformer.finetune import is_peft, prepare_for_finetuning, resolve_dtype, unwrap_peft
from decisionformer.heads import CodeReadoutHead, Head, build_head, head_class
from decisionformer.losses import MASK_VALUE, masked_logits
from decisionformer.poolers import Pooler, build_pooler, default_pooler_type
from decisionformer.prompting import PromptBuilder, prepare_tokenizer

logger = logging.getLogger(__name__)
DECISION_CONFIG = "decision_config.json"
READOUT = "readout.safetensors"
HEADS = "heads.safetensors"


class DecisionModel(nn.Module):
    def __init__(self, backbone: Backbone, poolers: dict[str, Pooler], heads: dict[str, Head],
                 head_poolers: dict[str, str]) -> None:
        super().__init__()
        self.backbone = backbone
        self.poolers = nn.ModuleDict(poolers)
        self.heads = nn.ModuleDict(heads)
        self.head_poolers = dict(head_poolers)

    def head_for(self, task: str) -> str:
        if task in self.heads:
            return task
        if "default" in self.heads:
            return "default"
        raise ValueError(f"No head for task '{task}'. Heads: {sorted(self.heads)}")

    def forward(self, batch: Any) -> torch.Tensor:
        hidden = self.backbone(batch)
        size, width = batch.option_mask.shape
        logits = torch.full((size, width), MASK_VALUE, dtype=torch.float32, device=hidden.device)
        groups: dict[str, list[int]] = defaultdict(list)
        for row, task in enumerate(batch.tasks):
            groups[self.head_for(task)].append(row)
        pooled_cache: dict[str, torch.Tensor] = {}
        for name, rows in groups.items():
            index = torch.tensor(rows, device=hidden.device)
            head = self.heads[name]
            head.check_rows(batch, index)
            key = self.head_poolers[name]
            if key not in pooled_cache:
                pooled_cache[key] = self.poolers[key](hidden, batch)
            out = head(pooled_cache[key][index], batch, index)
            columns = min(width, out.shape[1])
            logits[index, :columns] = out[:, :columns].float()
        return masked_logits(logits, batch.option_mask)

    def trainable_state(self) -> dict[str, torch.Tensor]:
        return {name: p.detach() for name, p in self.named_parameters() if p.requires_grad}

    def readout_only(self) -> bool:
        """One code-readout head read at the answer position: the standard checkpoint layout."""
        if list(self.heads) != ["default"] or not isinstance(self.heads["default"], CodeReadoutHead):
            return False
        return type(self.poolers[self.head_poolers["default"]]).__name__ == "DecisionTokenPooler"


@dataclass
class BuildResult:
    model: DecisionModel
    builder: PromptBuilder
    tokenizer: Any
    kind: str
    attention: str
    pooler_configs: dict[str, PoolerConfig]


def plan_poolers(heads: dict[str, Any], pooler: PoolerConfig, kind: str,
                 attention: str) -> tuple[dict[str, PoolerConfig], dict[str, str]]:
    poolers: dict[str, PoolerConfig] = {}
    keys: dict[str, str] = {}
    head_poolers: dict[str, str] = {}
    for name, head in heads.items():
        level = head_class(head).pooler_level
        chosen = head.pooler or pooler
        if chosen.type == "auto":
            chosen = PoolerConfig(type=default_pooler_type(kind, attention, level))
        signature = chosen.key()
        if signature not in keys:
            keys[signature] = f"p{len(keys)}"
            poolers[keys[signature]] = chosen
        head_poolers[name] = keys[signature]
    return poolers, head_poolers


def assemble(config: RunConfig, backbone: Backbone, kind: str, attention: str, code_rows: torch.Tensor | None,
             max_options: int) -> tuple[DecisionModel, dict[str, PoolerConfig]]:
    pooler_configs, head_poolers = plan_poolers(config.model.heads, config.model.pooler, kind, attention)
    poolers = {key: build_pooler(cfg, backbone.hidden_size) for key, cfg in pooler_configs.items()}
    heads = {}
    for name, head_cfg in config.model.heads.items():
        head = build_head(head_cfg, backbone.hidden_size, max_options, code_rows)
        if head.pooler_level != poolers[head_poolers[name]].level:
            raise ValueError(f"head '{name}' ({head_cfg.type}) needs a {head.pooler_level}-level pooler, but "
                             f"'{pooler_configs[head_poolers[name]].type}' is {poolers[head_poolers[name]].level}-level")
        heads[name] = head
    return DecisionModel(backbone, poolers, heads, head_poolers), pooler_configs


def build_model(config: RunConfig, device: torch.device, *, training: bool = True,
                backbone_source: str | None = None, tokenizer: Any = None) -> BuildResult:
    from transformers import AutoConfig

    model_cfg, finetune = config.model, config.finetune
    hf_config = AutoConfig.from_pretrained(backbone_source or model_cfg.backbone,
                                           revision=None if backbone_source else model_cfg.revision,
                                           trust_remote_code=model_cfg.trust_remote_code)
    kind = resolve_kind(hf_config, model_cfg.backbone_type)
    attention = resolve_mode(kind, model_cfg.attention)
    if tokenizer is None:
        tokenizer = load_tokenizer(model_cfg.backbone, model_cfg.revision, model_cfg.trust_remote_code)
    prepare_tokenizer(tokenizer)
    need_codes = any(head_class(h).needs_code_tokens for h in model_cfg.heads.values())
    builder = PromptBuilder(tokenizer, config.data.prompt.style, kind, max_length=config.data.max_length,
                            overflow=config.data.overflow, max_options=config.data.max_options,
                            need_code_tokens=need_codes, chat_template_kwargs=config.data.prompt.chat_template_kwargs)
    dtype = resolve_dtype(finetune, config.train.mixed_precision)
    hf_model, code_rows = load_backbone(model_cfg, finetune if training else None, kind=kind, attention=attention,
                                        dtype=dtype, device=device,
                                        code_token_ids=builder.code_token_ids if need_codes else None,
                                        source=backbone_source)
    hf_model = prepare_for_finetuning(hf_model, finetune, added_token_ids=[]) if training else hf_model.requires_grad_(False)
    backbone = Backbone(hf_model, kind, attention)
    model, pooler_configs = assemble(config, backbone, kind, attention, code_rows, len(builder.codes))
    if finetune.mode == "qlora" and training:
        model.poolers.to(device)
        model.heads.to(device)
    else:
        model.to(device)
    return BuildResult(model, builder, tokenizer, kind, attention, pooler_configs)


# --------------------------------------------------------------------------------------------------- export
def decision_config(result: BuildResult, config: RunConfig, *, temperatures: dict[str, float],
                    backbone_kind: str, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    model, builder = result.model, result.builder
    readout = model.readout_only()
    style = builder.format.name
    format_id = ("d3-code-readout-v1" if style == "d3" else "code-readout-v1") if readout else "decisionformer-heads-v1"
    by_type = {k: v for k, v in temperatures.items() if k != "default"}
    payload: dict[str, Any] = {
        "format_version": 1, "format_id": format_id, "prompt": style,
        "base_model": config.model.backbone, "revision": config.model.revision,
        "codes": list(builder.codes), "token_ids": list(builder.code_token_ids or []),
        "temperature": float(temperatures.get("default", 1.0)),
        "attention_mode": CHECKPOINT_NAMES.get(result.attention, result.attention),
        "pooling": "last" if readout else "decisionformer", "max_length": None, "readout_dtype": "float32",
    }
    if by_type:
        payload["temperature_by_type"] = by_type
    payload["decisionformer"] = {
        "version": __version__, "kind": result.kind, "backbone": backbone_kind,
        "chat_template_kwargs": config.data.prompt.chat_template_kwargs,
        "heads": {name: cfg.model_dump(mode="json", exclude_none=True) for name, cfg in config.model.heads.items()},
        "poolers": {key: cfg.model_dump(mode="json") for key, cfg in result.pooler_configs.items()},
        "head_poolers": model.head_poolers, "metrics": metrics or {},
        "training": {"finetune": config.finetune.mode, "max_length": config.data.max_length,
                     "attention": result.attention},
    }
    return payload


def export_checkpoint(result: BuildResult, config: RunConfig, directory: str | Path, *,
                      temperatures: dict[str, float] | None = None, metrics: dict[str, Any] | None = None) -> Path:
    """Write a decision checkpoint (rank 0 only). LoRA models are written as an adapter; `merge_checkpoint`
    turns that into full weights. Everything else is written in its final, standard form."""
    out = Path(directory)
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    model = result.model
    hf_model = model.backbone.model
    if is_peft(hf_model):
        hf_model.save_pretrained(str(tmp / "adapter"))
        backbone_kind = "adapter"
    else:
        unwrap_peft(hf_model).save_pretrained(str(tmp), safe_serialization=True)
        backbone_kind = "full"
    result.tokenizer.save_pretrained(str(tmp))
    if model.readout_only():
        save_file({"weight": model.heads["default"].readout.weight.detach().float().cpu().contiguous()},
                  str(tmp / READOUT))
    else:
        state = {f"heads.{k}": v for k, v in model.heads.state_dict().items()}
        state.update({f"poolers.{k}": v for k, v in model.poolers.state_dict().items()})
        save_file({k: v.detach().cpu().contiguous() for k, v in state.items()}, str(tmp / HEADS))
    payload = decision_config(result, config, temperatures=temperatures or {}, backbone_kind=backbone_kind,
                              metrics=metrics)
    (tmp / DECISION_CONFIG).write_text(json.dumps(payload, indent=2) + "\n")
    (tmp / "README.md").write_text(model_card(payload, config))
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    return out


def merge_checkpoint(directory: str | Path, *, device: str | torch.device = "cpu",
                     dtype: torch.dtype = torch.float32) -> None:
    """Replace an adapter checkpoint with merged full weights at the root (the standard layout)."""
    from peft import PeftModel

    from decisionformer.api import Decider

    out = Path(directory)
    payload = json.loads((out / DECISION_CONFIG).read_text())
    if payload["decisionformer"]["backbone"] != "adapter":
        return
    decider = Decider.from_pretrained(out, device=device, dtype=dtype, merge=False)
    peft_model = decider.model.backbone.model
    if not isinstance(peft_model, PeftModel):
        raise ValueError("Expected a peft adapter checkpoint")
    merged = peft_model.merge_and_unload()
    merged.save_pretrained(str(out), safe_serialization=True)
    shutil.rmtree(out / "adapter")
    payload["decisionformer"]["backbone"] = "full"
    (out / DECISION_CONFIG).write_text(json.dumps(payload, indent=2) + "\n")


def model_card(payload: dict[str, Any], config: RunConfig | None = None) -> str:
    """A Hub model card with metadata, usage for decisionformer, vLLM (when servable) and the Decision Index."""
    ext = payload.get("decisionformer", {})
    metrics = ext.get("metrics") or {}
    servable = payload.get("format_id") == "d3-code-readout-v1"
    lines = ["---", f"base_model: {payload.get('base_model')}", "library_name: decisionformer",
             "pipeline_tag: text-classification", "tags:", "- decisionformer", "- decision-model", "- system-one",
             "---", "", "# Decision model", "",
             f"A one-pass decision model fine-tuned from `{payload.get('base_model')}` with "
             "[decisionformer](https://github.com/FrontiersMindAI/decisionformer). It answers typed questions (choice, yes/no "
             "`noul`, score) about a state with a calibrated probability for every option, in the System One "
             "format used by Jev, vLLM `/v1/systemone` and the Decision Index.", "",
             "| | |", "|---|---|", f"| prompt format | `{payload.get('prompt')}` |",
             f"| attention | `{payload.get('attention_mode')}` |",
             f"| answer codes | {len(payload.get('codes', []))} |",
             f"| temperature | {payload.get('temperature'):.4f} |"]
    if metrics:
        shown = {k: metrics[k] for k in ("accuracy", "ece", "brier", "nll", "count") if k in metrics}
        lines.append("| validation | " + ", ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                                                  for k, v in shown.items()) + " |")
    lines += ["", "## Use", "", "```python", "from decisionformer import AutoDecisionModel", "",
              'model = AutoDecisionModel.from_pretrained("<this repo>")',
              'model.decide("My card was charged twice.", {',
              '    "team": {"type": "choice", "criteria": {"billing": None, "shipping": None}},',
              '    "urgent": {"type": "noul", "instructions": "Does this need a reply within the hour?"},', "})",
              "```", "", "Serve it on `/v1/systemone`: `decisionformer serve <this repo>`."]
    if servable:
        lines += ["", "This checkpoint follows the Decision 3.0 code-readout contract, so vLLM builds with "
                      "structured-decisions support can also serve it (`vllm serve <this repo>`) when the backbone "
                      "is Qwen3.5."]
    return "\n".join(lines) + "\n"


def max_input_tokens(hf_config: Any, tokenizer: Any) -> int | None:
    return context_limit(hf_config, tokenizer)
