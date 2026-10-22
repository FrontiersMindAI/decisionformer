"""Load any Hugging Face encoder or decoder as a hidden-state backbone and run it under the chosen attention mode."""

from __future__ import annotations

import inspect
import logging
from typing import Any

import torch
from torch import nn

from deciformer.attention import apply_attention_mode, prefix_attention_mask, sliding_window_of
from deciformer.finetune import unwrap_peft

logger = logging.getLogger(__name__)


def resolve_kind(config: Any, override: str = "auto") -> str:
    if override != "auto":
        return override
    from transformers.models.auto.modeling_auto import MODEL_FOR_MASKED_LM_MAPPING_NAMES

    if getattr(config, "is_encoder_decoder", False):
        return "encoder"
    return "encoder" if config.model_type in MODEL_FOR_MASKED_LM_MAPPING_NAMES else "decoder"


def load_tokenizer(name: str, revision: str | None = None, trust_remote_code: bool = False) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=trust_remote_code)
    if not getattr(tokenizer, "is_fast", True):
        raise ValueError("deciformer needs a fast tokenizer (offset mappings locate option spans)")
    return tokenizer


def _drop_unused_pooler(model: nn.Module) -> None:
    # BERT-style models compute a pooled output we never use; its weights would get no gradient under DDP.
    # Models with an `add_pooling_layer` switch accept `pooler = None`.
    if (getattr(model, "pooler", None) is not None
            and "add_pooling_layer" in inspect.signature(type(model).__init__).parameters):
        model.pooler = None


def load_backbone(model_cfg: Any, finetune_cfg: Any, *, kind: str, attention: str, dtype: torch.dtype,
                  device: torch.device, code_token_ids: list[int] | None, vocab_size: int,
                  source: str | None = None) -> tuple[nn.Module, torch.Tensor | None]:
    """Returns the transformers base model (no LM head) and, if requested, the LM-head rows of the code tokens."""
    from transformers import AutoModel, AutoModelForCausalLM, AutoModelForTextEncoding, BitsAndBytesConfig

    name = source or model_cfg.backbone
    kwargs: dict[str, Any] = {"revision": None if source else model_cfg.revision,
                              "trust_remote_code": model_cfg.trust_remote_code, "dtype": dtype,
                              "attn_implementation": model_cfg.attn_implementation}
    if finetune_cfg is not None and finetune_cfg.mode == "qlora":
        quant = finetune_cfg.quantization
        compute = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[quant.compute_dtype]
        kwargs["quantization_config"] = (
            BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=quant.quant_type,
                               bnb_4bit_use_double_quant=quant.double_quant, bnb_4bit_compute_dtype=compute)
            if quant.bits == 4 else BitsAndBytesConfig(load_in_8bit=True))
        kwargs["device_map"] = {"": device.index if device.type == "cuda" else str(device)}

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(name, revision=kwargs["revision"], trust_remote_code=model_cfg.trust_remote_code)
    if kind == "encoder":
        loader = AutoModelForTextEncoding if getattr(config, "is_encoder_decoder", False) else AutoModel
    else:
        loader = AutoModelForCausalLM if code_token_ids is not None else AutoModel
    try:
        loaded = loader.from_pretrained(name, **kwargs)
    except (ValueError, ImportError) as error:
        if kwargs["attn_implementation"] == "eager" or "attn" not in str(error).lower():
            raise
        logger.warning("%s does not support %s attention (%s); falling back to eager", name,
                       kwargs["attn_implementation"], error)
        kwargs["attn_implementation"] = "eager"
        loaded = loader.from_pretrained(name, **kwargs)

    code_rows = None
    if kind == "decoder" and code_token_ids is not None:
        output = loaded.get_output_embeddings()
        if output is None:
            raise ValueError(f"{name} has no LM head; code_readout/lm_verbalizer need a causal LM checkpoint")
        code_rows = output.weight[code_token_ids].detach().float().cpu().clone()
        model = loaded.base_model
        del loaded
    else:
        model = loaded
    if kind == "encoder":
        _drop_unused_pooler(model)
    if model_cfg.init_adapter and source is None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, model_cfg.init_adapter).merge_and_unload()
    if vocab_size > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(vocab_size)
    if kind == "decoder":
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
        self.hidden_size = int(getattr(base.config, "hidden_size", 0) or getattr(base.config, "d_model", 0))
        if not self.hidden_size:
            raise ValueError("Could not determine the backbone hidden size")

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
