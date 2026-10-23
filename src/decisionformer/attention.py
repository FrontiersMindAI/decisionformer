"""Attention modes for decoder backbones: causal (default), bidirectional, and prefix-bidirectional.

* bidirectional (`noncausal_full_attention` in `decision_config.json`): every full-attention layer reads the
  whole prompt in both directions. Linear-attention layers (Gated DeltaNet in Qwen3.5/3.8) keep their causal
  scan, exactly as Decision 3.0, pplx-decider and GLiDE do. Implemented with transformers' own support:
  `config.is_causal = False` makes `create_causal_mask` build a bidirectional mask (sliding-window layers get a
  bidirectional window), and the text decoder's attention modules drop their `is_causal` flag so SDPA and
  flash kernels do not re-apply causality when no mask is materialized. Vision towers are never touched.
* prefix_bidirectional (experimental): the prompt is bidirectional and the answer suffix causal, through an
  explicit 4D mask. Pure-attention models only.

Models without any full-attention layer (Mamba, RWKV, ...) cannot be made bidirectional and are refused.
"""

from __future__ import annotations

from typing import Any

import torch

MODES = ("causal", "bidirectional", "prefix_bidirectional")
#: How each mode is named in a decision checkpoint's `decision_config.json` (`attention_mode`).
CHECKPOINT_NAMES = {"causal": "causal", "bidirectional": "noncausal_full_attention"}
_ALIASES = {"noncausal_full_attention": "bidirectional", "noncausal": "bidirectional"}
_ATTENTION_LAYERS = {"full_attention", "sliding_attention", "chunked_attention"}
_RECURRENT_LAYERS = {"linear_attention", "mamba", "conv", "recurrent"}


def resolve_mode(kind: str, requested: str) -> str:
    requested = _ALIASES.get(requested, requested)
    if kind == "encoder":
        if requested not in ("auto", "bidirectional"):
            raise ValueError(f"Encoders are bidirectional; model.attention must be 'auto' or 'bidirectional', not "
                             f"'{requested}'")
        return "bidirectional"
    if requested == "auto":
        return "causal"
    if requested not in MODES:
        raise ValueError(f"Unknown attention mode {requested!r}; use one of {MODES}")
    return requested


def text_decoder(model: torch.nn.Module) -> torch.nn.Module:
    """The text decoder inside a (possibly multimodal) backbone."""
    for name in ("language_model", "text_model"):
        inner = getattr(model, name, None)
        if isinstance(inner, torch.nn.Module):
            return inner
    return model


def _text_configs(model: torch.nn.Module) -> list[Any]:
    configs = [model.config]
    for name in ("text_config", "language_config"):
        sub = getattr(model.config, name, None)
        if sub is not None and hasattr(sub, "to_dict"):
            configs.append(sub)
    inner = getattr(text_decoder(model), "config", None)
    if inner is not None and all(inner is not c for c in configs):
        configs.append(inner)
    return configs


def layer_kinds(model: torch.nn.Module) -> set[str]:
    return {kind for config in _text_configs(model) for kind in (getattr(config, "layer_types", None) or [])}


def check_supported(model: torch.nn.Module, mode: str) -> None:
    if mode == "causal":
        return
    kinds = layer_kinds(model)
    if kinds and not kinds & _ATTENTION_LAYERS:
        raise ValueError(f"model.attention='{mode}' needs attention layers; this model only has {sorted(kinds)}")
    hybrid = bool(kinds & _RECURRENT_LAYERS) or bool(kinds - _ATTENTION_LAYERS)
    if mode == "prefix_bidirectional":
        if hybrid:
            raise NotImplementedError("prefix_bidirectional supports pure-attention models only; use 'bidirectional'")
        if "sliding_attention" in kinds and "full_attention" in kinds:
            raise NotImplementedError("prefix_bidirectional does not yet support models that mix full and "
                                      "sliding-window layers; use 'bidirectional' or 'causal'")
        implementation = getattr(model.config, "_attn_implementation", "sdpa")
        if implementation not in ("sdpa", "eager"):
            raise ValueError("prefix_bidirectional needs attn_implementation 'sdpa' or 'eager'")


def _bidirectional_flag(config: Any, on: bool) -> Any:
    """`use_bidirectional_attention` is a bool on Gemma 3 and a 'all'/'vision'/None literal on Gemma 4."""
    annotation = next((str(cls.__annotations__["use_bidirectional_attention"]) for cls in type(config).__mro__
                       if "use_bidirectional_attention" in getattr(cls, "__annotations__", {})), "")
    if "Literal" in annotation or "str" in annotation or isinstance(config.use_bidirectional_attention, str):
        return "all" if on else None
    return on


def apply_attention_mode(model: torch.nn.Module, mode: str) -> int:
    """Configure a transformers backbone in place. Returns the number of attention modules changed."""
    mode = _ALIASES.get(mode, mode)
    if mode not in MODES:
        raise ValueError(f"Unknown attention mode '{mode}'")
    check_supported(model, mode)
    bidirectional = mode == "bidirectional"
    configs = _text_configs(model)
    decoder = text_decoder(model)
    if not bidirectional:
        # A checkpoint saved after bidirectional training carries `is_causal: false`; undo it for causal/prefix runs.
        if any(getattr(c, "is_causal", True) is False for c in configs):
            for config in configs:
                config.is_causal = True
                if getattr(config, "use_bidirectional_attention", None) not in (None, False, "vision"):
                    config.use_bidirectional_attention = _bidirectional_flag(config, False)
            for module in decoder.modules():
                if isinstance(getattr(module, "is_causal", None), bool) and module is not decoder:
                    module.is_causal = True
        return 0
    for config in configs:
        config.is_causal = False
        if hasattr(config, "use_bidirectional_attention"):
            config.use_bidirectional_attention = _bidirectional_flag(config, True)
    changed = 0
    for module in decoder.modules():
        if isinstance(getattr(module, "is_causal", None), bool) and module is not decoder:
            module.is_causal = False
            changed += 1
    if changed == 0:
        raise ValueError(f"Could not find attention modules exposing `is_causal` in {type(model).__name__}; "
                         "bidirectional attention is not supported for this architecture")
    return changed


def sliding_window_of(config: Any) -> int | None:
    config = getattr(config, "text_config", None) or config
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
