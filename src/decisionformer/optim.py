"""Optimizer parameter groups and learning-rate schedules."""

from __future__ import annotations

import math
from typing import Any

import torch

from decisionformer.finetune import is_embedding_parameter, layer_index, num_layers


def _no_decay(name: str, parameter: torch.nn.Parameter) -> bool:
    lowered = name.lower()
    return parameter.ndim < 2 or lowered.endswith(".bias") or "norm" in lowered


def parameter_groups(model: torch.nn.Module, train: Any, finetune: Any) -> list[dict[str, Any]]:
    """Backbone vs head learning rates, no weight decay on biases/norms, and optional layer-wise LR decay."""
    head_lr = train.head_lr if train.head_lr is not None else train.lr
    decay = finetune.layerwise_lr_decay
    depth = num_layers(model.backbone) if decay else 0
    groups: dict[tuple[float, bool], dict[str, Any]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone."):
            lr = train.lr
            if decay:
                index = layer_index(name)
                level = 0 if index is None and is_embedding_parameter(name) else (depth if index is None else index + 1)
                lr = train.lr * decay ** (depth - min(level, depth))
        else:
            lr = head_lr
        no_decay = _no_decay(name, parameter)
        key = (lr, no_decay)
        group = groups.setdefault(key, {"params": [], "lr": lr, "weight_decay": 0.0 if no_decay else train.weight_decay})
        group["params"].append(parameter)
    if not groups:
        raise ValueError("Nothing to train: no parameter requires gradients")
    return list(groups.values())


def build_optimizer(model: torch.nn.Module, train: Any, finetune: Any, *, distributed: bool) -> torch.optim.Optimizer:
    groups = parameter_groups(model, train, finetune)
    kwargs = {"betas": tuple(train.adam_betas), "eps": train.adam_eps}
    if train.shard_optimizer_states and distributed:
        from torch.distributed.optim import ZeroRedundancyOptimizer

        optimizer = ZeroRedundancyOptimizer(groups[0]["params"], optimizer_class=torch.optim.AdamW,
                                            lr=groups[0]["lr"], weight_decay=groups[0]["weight_decay"], **kwargs)
        for group in groups[1:]:
            optimizer.add_param_group(group)
        return optimizer
    return torch.optim.AdamW(groups, **kwargs)


def lr_lambda(scheduler: Any, total_steps: int):
    warmup = scheduler.warmup_steps if scheduler.warmup_steps is not None else int(scheduler.warmup_ratio * total_steps)

    def factor(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        if scheduler.type == "constant":
            return 1.0
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        floor = scheduler.min_lr_ratio
        if scheduler.type == "linear":
            return floor + (1 - floor) * (1 - progress)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return factor
