"""A tiny custom-code decoder for offline tests: the same shape of model as FrontiersMind's Nandi and Lumma.

It is loaded only through `trust_remote_code` (an `auto_map` in config.json and tokenizer_config.json), embeds tokens
through a factorized embedding (vocabulary -> rank -> hidden) with a tied LM head behind an `lm_head_proj`
projection, and its tokenizer's `encode` prepends the BOS token and a space, as Nandi's and Lumma's do.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from conftest import make_tokenizer

MODEL_TYPE = "tinycustom"

CONFIGURATION = '''
from transformers import PretrainedConfig


class TinyCustomConfig(PretrainedConfig):
    model_type = "tinycustom"

    def __init__(self, vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                 num_attention_heads=4, embedding_rank=16, max_position_embeddings=512, rms_norm_eps=1e-6,
                 tie_word_embeddings=True, **kwargs):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.embedding_rank = embedding_rank
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
'''

MODELING = '''
import torch
import torch.nn.functional as F
from torch import nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_tinycustom import TinyCustomConfig


class TinyCustomLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.out = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.mlp_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden, mask):
        size, length, width = hidden.shape
        q, k, v = self.qkv(self.attn_norm(hidden)).view(size, length, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        hidden = hidden + self.out(attended.transpose(1, 2).reshape(size, length, width))
        return hidden + self.down(F.silu(self.up(self.mlp_norm(hidden))))


class TinyCustomPreTrainedModel(PreTrainedModel):
    config_class = TinyCustomConfig
    base_model_prefix = "model"
    _supports_sdpa = True


class TinyCustomModel(TinyCustomPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.embedding_rank)
        self.embed_proj = nn.Linear(config.embedding_rank, config.hidden_size, bias=False)
        self.layers = nn.ModuleList(TinyCustomLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, input_ids=None, attention_mask=None, use_cache=None, **kwargs):
        hidden = self.embed_proj(self.embed_tokens(input_ids))
        length = input_ids.shape[1]
        allowed = torch.ones(length, length, dtype=torch.bool, device=hidden.device).tril()[None, None]
        if attention_mask is not None:
            allowed = allowed & attention_mask.bool()[:, None, None, :]
        for layer in self.layers:
            hidden = layer(hidden, allowed)
        return BaseModelOutputWithPast(last_hidden_state=self.norm(hidden))


class TinyCustomForCausalLM(TinyCustomPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = TinyCustomModel(config)
        self.lm_head_proj = nn.Linear(config.hidden_size, config.embedding_rank, bias=False)
        self.lm_head = nn.Linear(config.embedding_rank, config.vocab_size, bias=False)
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return CausalLMOutputWithPast(logits=self.lm_head(self.lm_head_proj(hidden)))
'''

TOKENIZATION = '''
from transformers import PreTrainedTokenizerFast


class TinyCustomTokenizer(PreTrainedTokenizerFast):
    """Like Nandi's and Lumma's tokenizers: `encode` always prepends the BOS token and a space."""

    def encode(self, text, *args, **kwargs):
        if isinstance(text, str):
            text = self.bos_token + " " + text
        return super().encode(text, *args, **kwargs)
'''

BOS = "[CLS]"


def build_custom_decoder(root: Path) -> Path:
    """A tiny random custom-code checkpoint, saved as a causal LM with its code next to the weights."""
    from transformers import AutoConfig, AutoModelForCausalLM

    path = root / MODEL_TYPE
    if path.exists():
        return path
    path.mkdir(parents=True)
    (path / "configuration_tinycustom.py").write_text(CONFIGURATION)
    (path / "modeling_tinycustom.py").write_text(MODELING)
    (path / "tokenization_tinycustom.py").write_text(TOKENIZATION)
    tokenizer = make_tokenizer("decoder")
    tokenizer.chat_template = None  # a plain-text model, like Nandi and Lumma
    tokenizer.bos_token = BOS
    tokenizer.save_pretrained(str(path))
    settings = json.loads((path / "tokenizer_config.json").read_text())
    settings.update(tokenizer_class="TinyCustomTokenizer",
                    auto_map={"AutoTokenizer": ["tokenization_tinycustom.TinyCustomTokenizer", None]})
    (path / "tokenizer_config.json").write_text(json.dumps(settings, indent=2))
    (path / "config.json").write_text(json.dumps({
        "model_type": MODEL_TYPE, "architectures": ["TinyCustomForCausalLM"],
        "auto_map": {"AutoConfig": "configuration_tinycustom.TinyCustomConfig",
                     "AutoModel": "modeling_tinycustom.TinyCustomModel",
                     "AutoModelForCausalLM": "modeling_tinycustom.TinyCustomForCausalLM"},
        "vocab_size": len(tokenizer), "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id, "bos_token_id": tokenizer.convert_tokens_to_ids(BOS)}))
    config = AutoConfig.from_pretrained(str(path), trust_remote_code=True)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model.save_pretrained(str(path))
    return path
