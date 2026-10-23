"""Per-example decision losses over masked option logits."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from decisionformer.config import LossConfig

MASK_VALUE = -1e9  # finite, so 0 * log p stays 0 for padded options


def masked_logits(logits: torch.Tensor, option_mask: torch.Tensor) -> torch.Tensor:
    return logits.float().masked_fill(~option_mask, MASK_VALUE)


def decision_loss(logits: torch.Tensor, targets: torch.Tensor, option_mask: torch.Tensor, config: LossConfig,
                  teacher: torch.Tensor | None = None, has_teacher: torch.Tensor | None = None) -> torch.Tensor:
    """Returns one loss per example. `targets` and `teacher` are probability rows over the shown options."""
    logits = masked_logits(logits, option_mask)
    valid = option_mask.float()
    if config.label_smoothing > 0:
        uniform = valid / valid.sum(-1, keepdim=True)
        targets = (1 - config.label_smoothing) * targets + config.label_smoothing * uniform
    log_probs = F.log_softmax(logits, dim=-1)
    if config.type == "focal":
        modulation = (1 - log_probs.exp()).clamp_min(0) ** config.focal_gamma
        loss = -(targets * modulation * log_probs).sum(-1)
    else:
        loss = -(targets * log_probs).sum(-1)
    if teacher is not None and has_teacher is not None and config.distill_alpha > 0 and bool(has_teacher.any()):
        temperature = config.distill_temperature
        student = F.log_softmax(masked_logits(logits, option_mask) / temperature, dim=-1)
        teacher = teacher * valid
        teacher = teacher / teacher.sum(-1, keepdim=True).clamp_min(1e-12)
        kl = (teacher * (teacher.clamp_min(1e-12).log() - student)).sum(-1) * temperature**2
        alpha = config.distill_alpha * has_teacher.float()
        loss = (1 - alpha) * loss + alpha * kl
    return loss
