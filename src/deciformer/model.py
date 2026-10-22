"""The decision model (backbone -> pooler -> head per task), its construction, export, and reloading."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from deciformer import __version__
from deciformer.attention import resolve_mode
from deciformer.backbone import Backbone, load_backbone, load_tokenizer, resolve_kind
from deciformer.config import PoolerConfig, RunConfig
from deciformer.finetune import is_peft, prepare_for_finetuning, resolve_dtype, unwrap_peft
from deciformer.heads import Head, build_head, head_class
from deciformer.losses import MASK_VALUE, masked_logits
from deciformer.poolers import Pooler, build_pooler, default_pooler_type
from deciformer.prompting import PromptBuilder, prepare_tokenizer

logger = logging.getLogger(__name__)
CONFIG_NAME = "deciformer_config.json"
FORMAT_VERSION = 1


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


@dataclass
class BuildResult:
    model: DecisionModel
    builder: PromptBuilder
    tokenizer: Any
    kind: str
    attention: str
    added_token_ids: list[int]


def _pooler_plan(config: RunConfig, kind: str, attention: str) -> tuple[dict[str, PoolerConfig], dict[str, str]]:
    poolers: dict[str, PoolerConfig] = {}
    keys: dict[str, str] = {}
    head_poolers: dict[str, str] = {}
    for name, head in config.model.heads.items():
        level = head_class(head).pooler_level
        chosen = head.pooler or config.model.pooler
        if chosen.type == "auto":
            chosen = PoolerConfig(type=default_pooler_type(kind, attention, level))
        signature = chosen.key()
        if signature not in keys:
            keys[signature] = f"p{len(keys)}"
            poolers[keys[signature]] = chosen
        head_poolers[name] = keys[signature]
    return poolers, head_poolers


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
    added = prepare_tokenizer(tokenizer, config.data.prompt, kind)
    need_codes = any(head_class(h).needs_code_tokens for h in model_cfg.heads.values())
    builder = PromptBuilder(tokenizer, config.data.prompt, kind, max_length=config.data.max_length,
                            overflow=config.data.overflow, max_options=config.data.max_options,
                            need_code_tokens=need_codes)
    dtype = resolve_dtype(finetune, config.train.mixed_precision)
    hf_model, code_rows = load_backbone(model_cfg, finetune if training else None, kind=kind, attention=attention,
                                        dtype=dtype, device=device,
                                        code_token_ids=builder.code_token_ids if need_codes else None,
                                        vocab_size=len(tokenizer), source=backbone_source)
    if training:
        hf_model = prepare_for_finetuning(hf_model, finetune, added_token_ids=added)
    else:
        hf_model.requires_grad_(False)
    backbone = Backbone(hf_model, kind, attention)
    pooler_configs, head_poolers = _pooler_plan(config, kind, attention)
    poolers = {key: build_pooler(cfg, backbone.hidden_size) for key, cfg in pooler_configs.items()}
    heads = {}
    for name, head_cfg in model_cfg.heads.items():
        head = build_head(head_cfg, backbone.hidden_size, config.data.max_options, code_rows)
        if head.pooler_level != poolers[head_poolers[name]].level:
            raise ValueError(f"head '{name}' ({head_cfg.type}) needs a {head.pooler_level}-level pooler, but "
                             f"'{pooler_configs[head_poolers[name]].type}' is {poolers[head_poolers[name]].level}-level")
        heads[name] = head
    model = DecisionModel(backbone, poolers, heads, head_poolers)
    quantized = finetune.mode == "qlora" and training
    if quantized:
        model.poolers.to(device)
        model.heads.to(device)
    else:
        model.to(device)
    return BuildResult(model, builder, tokenizer, kind, attention, added)


def _head_state(model: DecisionModel) -> dict[str, torch.Tensor]:
    state = {f"heads.{k}": v for k, v in model.heads.state_dict().items()}
    state.update({f"poolers.{k}": v for k, v in model.poolers.state_dict().items()})
    return {k: v.detach().cpu().contiguous() for k, v in state.items()}


def export_model(result: BuildResult, config: RunConfig, directory: str | Path, *,
                 temperatures: dict[str, float] | None = None, metrics: dict[str, Any] | None = None) -> Path:
    """Write a self-contained decision model directory (rank 0 only)."""
    out = Path(directory)
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        import shutil

        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    model = result.model
    hf_model = model.backbone.model
    mode = config.finetune.mode
    if is_peft(hf_model):
        hf_model.save_pretrained(str(tmp / "adapter"))
        backbone_kind = "adapter"
    elif mode == "head_only" and not result.added_token_ids:
        backbone_kind = "reference"
    else:
        unwrap_peft(hf_model).save_pretrained(str(tmp / "backbone"), safe_serialization=True)
        backbone_kind = "full"
    save_file(_head_state(model), str(tmp / "heads.safetensors"))
    result.tokenizer.save_pretrained(str(tmp))
    payload = {
        "format_version": FORMAT_VERSION, "deciformer_version": __version__, "backbone_kind": backbone_kind,
        "kind": result.kind, "attention": result.attention, "codes": result.builder.codes,
        "code_token_ids": result.builder.code_token_ids, "head_poolers": model.head_poolers,
        "temperatures": temperatures or {}, "metrics": metrics or {},
        "config": config.model_dump(mode="json"),
    }
    (tmp / CONFIG_NAME).write_text(json.dumps(payload, indent=2) + "\n")
    if out.exists():
        import shutil

        shutil.rmtree(out)
    tmp.rename(out)
    return out


def merge_adapter_export(directory: str | Path) -> None:
    """Replace an adapter export with merged full backbone weights (for serving without peft)."""
    import shutil

    from peft import PeftModel

    out = Path(directory)
    payload = json.loads((out / CONFIG_NAME).read_text())
    if payload["backbone_kind"] != "adapter":
        return
    config = RunConfig.model_validate(payload["config"])
    loaded = load_decision_model(out, device=torch.device("cpu"), merge=False, dtype=torch.float32)
    hf_model = loaded.model.backbone.model
    if not isinstance(hf_model, PeftModel):
        raise ValueError("Expected a peft adapter export")
    merged = hf_model.merge_and_unload()
    merged.save_pretrained(str(out / "backbone"), safe_serialization=True)
    shutil.rmtree(out / "adapter")
    payload["backbone_kind"] = "full"
    payload["config"] = config.model_dump(mode="json")
    (out / CONFIG_NAME).write_text(json.dumps(payload, indent=2) + "\n")


@dataclass
class LoadedModel:
    model: DecisionModel
    builder: PromptBuilder
    tokenizer: Any
    temperatures: dict[str, float]
    config: RunConfig
    metadata: dict[str, Any]

    def temperature_for(self, task: str) -> float:
        return self.temperatures.get(task, self.temperatures.get("default", 1.0))


def load_decision_model(directory: str | Path, *, device: torch.device | str = "cpu", merge: bool = True,
                        dtype: torch.dtype | None = None) -> LoadedModel:
    out = Path(directory)
    payload = json.loads((out / CONFIG_NAME).read_text())
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported deciformer export format {payload.get('format_version')}")
    config = RunConfig.model_validate(payload["config"])
    device = torch.device(device)
    if dtype is not None:
        names = {torch.float32: "float32", torch.bfloat16: "bfloat16", torch.float16: "float16"}
        config.finetune.model_dtype = names[dtype]
    elif config.finetune.model_dtype == "auto":
        config.finetune.model_dtype = "float32" if device.type == "cpu" else "auto"
    if config.finetune.mode == "qlora":
        config.finetune.mode = "lora"  # inference loads the base in the compute dtype, then applies the adapter
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(out))
    source = str(out / "backbone") if payload["backbone_kind"] == "full" else None
    result = build_model(config, device, training=False, backbone_source=source, tokenizer=tokenizer)
    if result.builder.codes != payload["codes"]:
        raise ValueError("Answer codes differ from the exported model; the tokenizer changed")
    model = result.model
    if payload["backbone_kind"] == "adapter":
        from peft import PeftModel

        hf_model = PeftModel.from_pretrained(model.backbone.model, str(out / "adapter"))
        if merge:
            hf_model = hf_model.merge_and_unload()
        hf_model.requires_grad_(False)
        model.backbone.model = hf_model.to(device)
    head_state = load_file(str(out / "heads.safetensors"))
    missing = model.heads.load_state_dict({k[6:]: v for k, v in head_state.items() if k.startswith("heads.")})
    model.poolers.load_state_dict({k[8:]: v for k, v in head_state.items() if k.startswith("poolers.")})
    if missing.missing_keys or missing.unexpected_keys:
        raise ValueError(f"Head weights do not match the model: {missing}")
    model.to(device).eval()
    return LoadedModel(model, result.builder, tokenizer, payload["temperatures"], config, payload)
