"""Load any Hugging Face encoder or decoder as a hidden-state backbone and run it under the chosen attention mode."""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

from decisionformer.attention import apply_attention_mode, prefix_attention_mask, sliding_window_of
from decisionformer.finetune import unwrap_peft

logger = logging.getLogger(__name__)


def text_config(config: Any) -> Any:
    return getattr(config, "text_config", None) or config


def resolve_kind(config: Any, override: str = "auto") -> str:
    if override != "auto":
        return override
    from transformers.models.auto.modeling_auto import MODEL_FOR_MASKED_LM_MAPPING_NAMES

    if getattr(config, "is_encoder_decoder", False):
        return "encoder"
    return "encoder" if config.model_type in MODEL_FOR_MASKED_LM_MAPPING_NAMES else "decoder"


def hidden_size_of(config: Any) -> int:
    inner = text_config(config)
    size = getattr(inner, "hidden_size", None) or getattr(inner, "d_model", None)
    if not size:
        raise ValueError("Could not determine the backbone hidden size")
    return int(size)


def context_limit(config: Any, tokenizer: Any = None) -> int | None:
    limit = getattr(text_config(config), "max_position_embeddings", None)
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if tokenizer_limit and tokenizer_limit < 10**9:
        limit = min(limit, tokenizer_limit) if limit else tokenizer_limit
    return int(limit) if limit else None


def load_tokenizer(name: str, revision: str | None = None, trust_remote_code: bool = False) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=trust_remote_code)
    if not getattr(tokenizer, "is_fast", True):
        raise ValueError("decisionformer needs a fast tokenizer (offset mappings locate option spans)")
    return tokenizer


def _encoder_of_seq2seq(config: Any) -> bool:
    """True for encoder-decoder checkpoints (T5, mT5, UMT5) and their saved encoders (T5EncoderModel)."""
    from transformers.models.auto.modeling_auto import MODEL_FOR_TEXT_ENCODING_MAPPING_NAMES

    saved_encoder = any(str(a).endswith("EncoderModel") for a in getattr(config, "architectures", None) or [])
    if not (getattr(config, "is_encoder_decoder", False) or saved_encoder):
        return False
    if config.model_type not in MODEL_FOR_TEXT_ENCODING_MAPPING_NAMES:
        raise ValueError(f"{config.model_type} is an encoder-decoder model without a standalone text encoder in "
                         "transformers; use an encoder (BERT, ModernBERT, DeBERTa, T5) or a decoder backbone")
    return True


def _drop_unused_pooler(model: nn.Module) -> None:
    # BERT-style models compute a pooled output we never use; its weights would get no gradient under DDP.
    if (getattr(model, "pooler", None) is not None
            and "add_pooling_layer" in inspect.signature(type(model).__init__).parameters):
        model.pooler = None


def _tied(config: Any) -> bool:
    return any(getattr(c, "tie_word_embeddings", False) for c in (config, getattr(config, "text_config", None)) if c)


def output_rows(model: nn.Module, source: str, revision: str | None, token_ids: list[int]) -> torch.Tensor:
    """The LM-head rows of `token_ids`: tied input embeddings, or read straight from the checkpoint files."""
    if _tied(model.config):
        return model.get_input_embeddings().weight[token_ids].detach().float().cpu().clone()
    from safetensors import safe_open
    from transformers.utils import cached_file

    def fetch(filename: str) -> str | None:
        return cached_file(source, filename, revision=revision, _raise_exceptions_for_missing_entries=False)

    def is_head(key: str) -> bool:
        return key == "lm_head.weight" or key.endswith(".lm_head.weight")

    index = fetch("model.safetensors.index.json")
    if index:
        weight_map = json.loads(Path(index).read_text())["weight_map"]
        key = next((k for k in weight_map if is_head(k)), None)
        path = fetch(weight_map[key]) if key else None
    else:
        path = fetch("model.safetensors")
        key = None
        if path:
            with safe_open(path, "pt") as handle:
                key = next((k for k in handle.keys() if is_head(k)), None)  # noqa: SIM118
    if not path or not key:
        raise ValueError(f"{source} has no LM head to initialize answer codes from; use a head that does not need "
                         "it (option_scoring, fixed_softmax, binary, ordinal)")
    with safe_open(path, "pt") as handle:
        rows = handle.get_slice(key)
        return torch.stack([rows[i:i + 1][0] for i in token_ids]).float()


def load_backbone(model_cfg: Any, finetune_cfg: Any, *, kind: str, attention: str, dtype: torch.dtype,
                  device: torch.device, code_token_ids: list[int] | None,
                  source: str | None = None) -> tuple[nn.Module, torch.Tensor | None]:
    """The transformers base model (no LM head) and, if requested, the LM-head rows of the answer codes."""
    from transformers import AutoConfig, AutoModel, AutoModelForTextEncoding, BitsAndBytesConfig

    name = source or model_cfg.backbone
    revision = None if source else model_cfg.revision
    kwargs: dict[str, Any] = {"revision": revision, "trust_remote_code": model_cfg.trust_remote_code, "dtype": dtype,
                              "attn_implementation": model_cfg.attn_implementation}
    if finetune_cfg is not None and finetune_cfg.mode == "qlora":
        quant = finetune_cfg.quantization
        compute = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[quant.compute_dtype]
        kwargs["quantization_config"] = (
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=quant.quant_type,
                               bnb_4bit_use_double_quant=quant.double_quant, bnb_4bit_compute_dtype=compute)
            if quant.bits == 4 else BitsAndBytesConfig(load_in_8bit=True))
        kwargs["device_map"] = {"": device.index if device.type == "cuda" else str(device)}
    config = AutoConfig.from_pretrained(name, revision=revision, trust_remote_code=model_cfg.trust_remote_code)
    loader = AutoModelForTextEncoding if kind == "encoder" and _encoder_of_seq2seq(config) else AutoModel
    try:
        model = loader.from_pretrained(name, **kwargs)
    except (ValueError, ImportError) as error:
        if kwargs["attn_implementation"] == "eager" or "attn" not in str(error).lower():
            raise
        logger.warning("%s does not support %s attention (%s); falling back to eager", name,
                       kwargs["attn_implementation"], error)
        kwargs["attn_implementation"] = "eager"
        model = loader.from_pretrained(name, **kwargs)
    if model_cfg.init_adapter and source is None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, model_cfg.init_adapter).merge_and_unload()
    code_rows = output_rows(model, name, revision, code_token_ids) if code_token_ids is not None else None
    if kind == "encoder":
        _drop_unused_pooler(model)
    else:
        apply_attention_mode(model, attention)
    return model, code_rows


class Backbone(nn.Module):
    """Wraps a transformers base model (optionally peft-wrapped) and returns last hidden states."""

    def __init__(self, model: nn.Module, kind: str, attention: str) -> None:
        super().__init__()
        self.model = model
        self.kind = kind
        self.attention = attention
        base = unwrap_peft(model)
        self.accepts_token_type_ids = "token_type_ids" in inspect.signature(base.forward).parameters
        self.implementation = getattr(base.config, "_attn_implementation", "sdpa")
        self.sliding_window = sliding_window_of(base.config)
        self.hidden_size = hidden_size_of(base.config)

    def forward(self, batch: Any) -> torch.Tensor:
        kwargs: dict[str, Any] = {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask}
        if self.accepts_token_type_ids and batch.token_type_ids is not None:
            kwargs["token_type_ids"] = batch.token_type_ids
        if self.attention == "prefix_bidirectional":
            kwargs["attention_mask"] = prefix_attention_mask(
                batch.attention_mask, batch.prefix_end, implementation=self.implementation, dtype=torch.float32,
                sliding_window=self.sliding_window)
        if self.kind == "decoder":
            kwargs["use_cache"] = False
        return self.model(**kwargs).last_hidden_state
