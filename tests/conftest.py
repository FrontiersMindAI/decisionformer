"""Offline fixtures: a word-level tokenizer and tiny random checkpoints for several model families."""

from __future__ import annotations

import json
import random
import string
from pathlib import Path

import pytest
import torch

COLORS = ["red", "blue", "green", "yellow"]
SHAPES = ["circle", "square", "triangle"]
FILLER = ["the", "a", "object", "is", "very", "small", "large", "near", "far", "today", "here", "there", "and"]
SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
            "<|decision|>"]
CHAT_TEMPLATE = ("{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}<|end|>\n{% endfor %}"
                 "{% if add_generation_prompt %}<|assistant|>\n{% endif %}")
DECODERS = ["llama", "mistral", "qwen2", "qwen3", "gemma", "gemma2", "gemma3_text", "phi3", "granite", "olmo2",
            "smollm3"]


def _corpus() -> list[str]:
    from deciformer.config import DEFAULT_SYSTEM, DEFAULT_USER_TEMPLATE, PromptConfig

    prompt = PromptConfig()
    lines = [DEFAULT_SYSTEM, DEFAULT_USER_TEMPLATE, prompt.default_instructions, prompt.answer_instruction,
             prompt.plain_suffix, " ".join(string.ascii_uppercase), " ".join(str(i) for i in range(10)),
             " ".join(COLORS + SHAPES + FILLER), "What is the color ? What is the shape ? Is it red ? true false "
             "Yes No positive negative neutral bad ok good great x _ : , . ? ! - ( ) [ ] { } \" '"]
    return lines * 3


def make_tokenizer(kind: str):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(_corpus(), trainers.WordLevelTrainer(special_tokens=SPECIALS, min_frequency=1))
    if kind == "encoder":
        cls_id, sep_id = tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")
        tok.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                                   sep_token="[SEP]", mask_token="[MASK]",
                                   eos_token="<|end|>" if kind == "decoder" else None)
    if kind == "decoder":
        fast.chat_template = CHAT_TEMPLATE
    return fast


def _decoder_config(model_type: str, vocab: int):
    from transformers import AutoConfig

    return AutoConfig.for_model(
        model_type, vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=512, pad_token_id=0,
        bos_token_id=None, eos_token_id=SPECIALS.index("<|end|>"), tie_word_embeddings=True, attention_dropout=0.0)


def build_decoder(root: Path, model_type: str, plain: bool = False) -> Path:
    from transformers import AutoModelForCausalLM

    path = root / f"{model_type}{'-plain' if plain else ''}"
    if path.exists():
        return path
    tokenizer = make_tokenizer("decoder")
    if plain:
        tokenizer.chat_template = None
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(_decoder_config(model_type, len(tokenizer)))
    model.save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))
    return path


def build_encoder(root: Path, model_type: str) -> Path:
    from transformers import AutoModelForMaskedLM, BertConfig, ModernBertConfig, T5Config, T5ForConditionalGeneration

    path = root / model_type
    if path.exists():
        return path
    tokenizer = make_tokenizer("encoder")
    vocab = len(tokenizer)
    torch.manual_seed(0)
    if model_type == "bert":
        model = AutoModelForMaskedLM.from_config(BertConfig(
            vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
            max_position_embeddings=512, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0, pad_token_id=0))
    elif model_type == "modernbert":
        model = AutoModelForMaskedLM.from_config(ModernBertConfig(
            vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
            max_position_embeddings=512, pad_token_id=0, cls_token_id=2, sep_token_id=3, bos_token_id=2,
            eos_token_id=3, global_attn_every_n_layers=1, attention_dropout=0.0, embedding_dropout=0.0,
            mlp_dropout=0.0))
    else:
        model = T5ForConditionalGeneration(T5Config(vocab_size=vocab, d_model=32, d_kv=8, d_ff=64, num_layers=2,
                                                    num_heads=4, dropout_rate=0.0, decoder_start_token_id=0,
                                                    pad_token_id=0))
    model.save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))
    return path


@pytest.fixture(scope="session")
def model_root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("models")


def synthetic_records(count: int, seed: int, *, kind: str = "choice", long_tail: bool = False) -> list[dict]:
    rng = random.Random(seed)
    records = []
    for i in range(count):
        color, shape = rng.choice(COLORS), rng.choice(SHAPES)
        extra = rng.choice([0, 0, 0, 40, 120]) if long_tail else 0
        words = [rng.choice(FILLER) for _ in range(rng.randint(2, 8) + extra)]
        state = " ".join(words[: len(words) // 2] + ["the", shape, "is", color] + words[len(words) // 2:]) + " ."
        if kind == "binary":
            records.append({"id": f"{seed}-{i}", "state": state, "family": f"f{seed}-{i // 2}", "task": "is_red",
                            "question": {"type": "binary", "instructions": "Is it red ?"}, "label": color == "red"})
        else:
            records.append({"id": f"{seed}-{i}", "state": state, "family": f"f{seed}-{i // 2}", "task": "color",
                            "question": {"type": "choice", "instructions": "What is the color ?",
                                         "options": dict.fromkeys(COLORS)}, "label": color})
    return records


def write_jsonl(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("data")
    write_jsonl(root / "train.jsonl", synthetic_records(96, 1))
    write_jsonl(root / "dev.jsonl", synthetic_records(32, 2))
    write_jsonl(root / "calibration.jsonl", synthetic_records(32, 3))
    write_jsonl(root / "binary.jsonl", synthetic_records(48, 4, kind="binary"))
    return root
