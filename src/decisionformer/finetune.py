"""Fine-tuning modes: full, LoRA, QLoRA, or head-only; plus layer freezing and LoRA configuration."""

from __future__ import annotations

import re
from typing import Any

import torch
from torch import nn

# Module-name suffixes understood by peft; unmatched names are ignored as long as one matches.
ATTENTION_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj", "query", "key", "value", "Wqkv", "Wo",
                     "c_attn", "c_proj", "query_key_value", "dense", "out_proj", "in_proj", "q", "k", "v", "o"]
MLP_TARGETS = ["gate_proj", "up_proj", "down_proj", "gate_up_proj", "fc1", "fc2", "c_fc", "Wi", "dense_h_to_4h",
               "dense_4h_to_h", "wi", "wo", "wi_0", "wi_1", "intermediate.dense", "output.dense", "mlp.Wo"]
TARGET_PRESETS = {"attention": ATTENTION_TARGETS, "mlp": MLP_TARGETS, "attention_mlp": ATTENTION_TARGETS + MLP_TARGETS}
LORA_DEFAULTS: dict[str, Any] = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": "all-linear",
                                 "bias": "none"}
_LAYER_PATTERN = re.compile(r"\.(?:layer|layers|h|block|blocks)\.(\d+)\.")


def router_modules(model: nn.Module) -> list[str]:
    """MoE routers built as Linear subclasses (Phi-MoE, Llama 4) return (logits, weights, experts), which LoRA
    cannot wrap, so "all-linear" leaves them out."""
    return [name for name, module in model.named_modules()
            if isinstance(module, nn.Linear) and type(module).forward is not nn.Linear.forward
            and "router" in type(module).__name__.lower()]


def lora_config_from(lora: dict[str, Any], added_token_ids: list[int], model: nn.Module | None = None) -> Any:
    """Build a `peft.LoraConfig` from the `finetune.lora` block. Field names are peft's own."""
    from peft import LoraConfig

    kwargs = {**LORA_DEFAULTS, **lora}
    if "task_type" in kwargs:
        raise ValueError("finetune.lora.task_type is managed by decisionformer; remove it")
    target = kwargs["target_modules"]
    if target == "all-linear" and model is not None and "exclude_modules" not in kwargs:
        kwargs["exclude_modules"] = router_modules(model) or None
    if isinstance(target, str) and target in TARGET_PRESETS:
        kwargs["target_modules"] = TARGET_PRESETS[target]
    if kwargs.get("trainable_token_indices", "auto") == "auto":
        kwargs["trainable_token_indices"] = list(added_token_ids) or None
    try:
        return LoraConfig(**kwargs)
    except TypeError as error:
        raise ValueError(f"Invalid finetune.lora settings: {error}") from error


def layer_index(name: str) -> int | None:
    match = _LAYER_PATTERN.search(name)
    return int(match.group(1)) if match else None


def num_layers(model: nn.Module) -> int:
    indices = [layer_index(name) for name, _ in model.named_parameters()]
    return 1 + max((i for i in indices if i is not None), default=-1)


def parse_layers(spec: list[int] | str | None) -> set[int]:
    if spec is None:
        return set()
    if isinstance(spec, list):
        return set(spec)
    result: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if "-" in part:
            low, high = part.split("-")
            result.update(range(int(low), int(high) + 1))
        elif part:
            result.add(int(part))
    return result


def is_embedding_parameter(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in ("embed", ".wte.", ".wpe.", "shared.weight"))


def prepare_for_finetuning(model: nn.Module, finetune: Any, *, added_token_ids: list[int]) -> nn.Module:
    """Configure trainable parameters on a loaded transformers backbone and return the (possibly wrapped) model."""
    mode = finetune.mode
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if mode == "qlora":
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=finetune.gradient_checkpointing,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    elif finetune.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if mode in ("lora", "qlora"):
        from peft import get_peft_model

        model = get_peft_model(model, lora_config_from(finetune.lora, added_token_ids, model))
    elif mode == "head_only":
        model.requires_grad_(False)
    else:
        model.requires_grad_(True)
        frozen_layers = parse_layers(finetune.freeze_layers)
        for name, parameter in model.named_parameters():
            if finetune.freeze_embeddings and is_embedding_parameter(name):
                parameter.requires_grad_(False)
            if layer_index(name) in frozen_layers:
                parameter.requires_grad_(False)
    return model


def unwrap_peft(model: nn.Module) -> nn.Module:
    get_base = getattr(model, "get_base_model", None)
    return get_base() if callable(get_base) else model


def is_peft(model: nn.Module) -> bool:
    return callable(getattr(model, "get_base_model", None))


def count_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def resolve_dtype(finetune: Any, mixed_precision: str) -> torch.dtype:
    names = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    if finetune.model_dtype != "auto":
        return names[finetune.model_dtype]
    if finetune.mode == "qlora":
        return names[finetune.quantization.compute_dtype]
    if finetune.mode == "full":
        return torch.float32
    return {"bf16": torch.bfloat16, "fp16": torch.float16}.get(mixed_precision, torch.float32)
