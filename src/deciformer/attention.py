"""Attention modes for decoder backbones: causal (default), bidirectional, and prefix-bidirectional.

* bidirectional: every token attends to every non-padding token. Implemented through transformers' own
  support: `config.is_causal = False` makes `create_causal_mask` build a bidirectional mask (sliding-window
  layers get a bidirectional window), and every attention module's `is_causal` flag is cleared so SDPA and
  flash-attention kernels do not re-apply causality when no mask is materialized.
* prefix_bidirectional: the prompt (state, question, options) is bidirectional; the answer suffix stays
  causal. Implemented with an explicit 4D mask, which transformers passes through unchanged.

Encoders are natively bidirectional. Recurrent or linear-attention layers (Mamba, DeltaNet, ...) are causal
by construction, so non-causal modes refuse such models instead of silently doing the wrong thing.
"""

from __future__ import annotations

from typing import Any

import torch

MODES = ("causal", "bidirectional", "prefix_bidirectional")
_ATTENTION_LAYER_TYPES = {"full_attention", "sliding_attention", "chunked_attention"}
_RECURRENT_HINTS = ("mamba", "deltanet", "linearattention", "rwkv", "ssm", "recurrent", "retention")


def resolve_mode(kind: str, requested: str) -> str:
    if kind == "encoder":
        if requested not in ("auto", "bidirectional"):
            raise ValueError(f"Encoders are bidirectional; model.attention must be 'auto' or 'bidirectional', not "
                             f"'{requested}'")
        return "bidirectional"
    return "causal" if requested == "auto" else requested


def _text_configs(config: Any) -> list[Any]:
    configs = [config]
    for name in ("text_config", "decoder", "language_config"):
        sub = getattr(config, name, None)
        if sub is not None and hasattr(sub, "to_dict"):
            configs.append(sub)
    return configs


def check_supported(model: torch.nn.Module, mode: str) -> None:
    if mode == "causal":
        return
    for config in _text_configs(model.config):
        layer_types = getattr(config, "layer_types", None) or []
        unsupported = sorted(set(layer_types) - _ATTENTION_LAYER_TYPES)
        if unsupported:
            raise ValueError(f"model.attention='{mode}' needs attention layers everywhere; this model also has "
                             f"{unsupported} layers, which are causal by construction")
        if mode == "prefix_bidirectional" and "sliding_attention" in layer_types and "full_attention" in layer_types:
            raise NotImplementedError("prefix_bidirectional does not yet support models that mix full and "
                                      "sliding-window layers; use 'bidirectional' or 'causal'")
    for module in model.modules():
        name = type(module).__name__.lower()
        if any(hint in name for hint in _RECURRENT_HINTS):
            raise ValueError(f"model.attention='{mode}' cannot be applied: {type(module).__name__} is a "
                             "recurrent/linear-attention layer and is causal by construction")
    implementation = getattr(model.config, "_attn_implementation", "sdpa")
    if mode == "prefix_bidirectional" and implementation not in ("sdpa", "eager"):
        raise ValueError("prefix_bidirectional needs attn_implementation 'sdpa' or 'eager' (custom masks are not "
                         "supported by flash-attention kernels)")


def apply_attention_mode(model: torch.nn.Module, mode: str) -> int:
    """Configure `model` (a transformers backbone) in place. Returns the number of attention modules changed."""
    if mode not in MODES:
        raise ValueError(f"Unknown attention mode '{mode}'")
    check_supported(model, mode)
    bidirectional = mode == "bidirectional"
    configs = _text_configs(model.config)
    if not bidirectional:
        # A checkpoint saved after bidirectional training carries `is_causal: false`; undo it for causal/prefix runs.
        if any(getattr(c, "is_causal", True) is False for c in configs):
            for config in configs:
                config.is_causal = True
            for module in model.modules():
                if isinstance(getattr(module, "is_causal", None), bool) and module is not model:
                    module.is_causal = True
        return 0
    for config in configs:
        config.is_causal = False
        if hasattr(config, "use_bidirectional_attention"):
            config.use_bidirectional_attention = True
    changed = 0
    for module in model.modules():
        if isinstance(getattr(module, "is_causal", None), bool) and module is not model:
            module.is_causal = False
            changed += 1
    if changed == 0:
        raise ValueError(f"Could not find attention modules exposing `is_causal` in {type(model).__name__}; "
                         "bidirectional attention is not supported for this architecture")
    return changed


def sliding_window_of(config: Any) -> int | None:
    window = getattr(config, "sliding_window", None)
    if window is None or getattr(config, "use_sliding_window", True) is False:
        return None
    layer_types = getattr(config, "layer_types", None)
    if layer_types and "sliding_attention" not in layer_types:
        return None
    return int(window)


def prefix_attention_mask(attention_mask: torch.Tensor, prefix_end: torch.Tensor, *, implementation: str,
                          dtype: torch.dtype, sliding_window: int | None = None) -> torch.Tensor:
    """A (batch, 1, query, key) mask: causal everywhere, plus full attention among the first `prefix_end` tokens."""
    length = attention_mask.shape[1]
    positions = torch.arange(length, device=attention_mask.device)
    query, key = positions[None, :, None], positions[None, None, :]
    allowed = (key <= query) | (key < prefix_end[:, None, None])
    if sliding_window is not None:
        allowed &= (query - key).abs() < sliding_window
    allowed &= attention_mask[:, None, :].bool()
    allowed |= torch.eye(length, dtype=torch.bool, device=attention_mask.device)[None]  # never an empty row
    allowed = allowed[:, None]
    if implementation == "sdpa":
        return allowed
    if implementation == "eager":
        return torch.zeros(allowed.shape, dtype=dtype, device=allowed.device).masked_fill(~allowed,
                                                                                            torch.finfo(dtype).min)
    raise ValueError(f"prefix masks are not supported with {implementation}")
