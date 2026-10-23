"""Every model family end to end: fine-tune a tiny random checkpoint, export it in the standard layout, reload it
with AutoDecisionModel and answer a System One request that passes the Decision Index response check."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from conftest import ZOO_DECODERS, ZOO_ENCODERS, build_zoo_model
from test_api import QUESTIONS, index_validate
from test_training import make_config

from decisionformer import AutoDecisionModel
from decisionformer.trainer import train

pytestmark = pytest.mark.slow

LORA = {"finetune": {"mode": "lora", "lora": {"r": 4, "lora_dropout": 0.0}}}
# transformers re-wraps the toy word-level tokenizer as Qwen2Tokenizer for these model types, whose pre-tokenizer
# splits two-letter codes, so only A-Z remain single tokens (real Qwen tokenizers have all 255 codes).
SINGLE_LETTER_CODES = {"qwen2", "qwen3_5"}


def _round_trip(model_root: Path, data_dir: Path, tmp_path: Path, model_type: str, *, head: dict,
                extra: dict | None = None) -> tuple[dict, dict]:
    config = make_config(tmp_path, build_zoo_model(model_root, model_type), data_dir, head=head, steps=4,
                         extra=extra, overrides=[f"data.train={data_dir / 'multi_train.jsonl'}",
                                                 f"data.eval.dev={data_dir / 'multi_dev.jsonl'}",
                                                 f"data.calibration={data_dir / 'multi_calibration.jsonl'}"])
    train(config)
    best = Path(config.train.output_dir) / "best"
    decision_config = json.loads((best / "decision_config.json").read_text())
    model = AutoDecisionModel.from_pretrained(best, device="cpu", dtype=torch.float32)
    response = model.decide("the large circle is red .", QUESTIONS)
    index_validate(QUESTIONS, response)
    return decision_config, response


@pytest.mark.parametrize("model_type", list(ZOO_ENCODERS))
def test_encoders(model_root, data_dir, tmp_path, model_type):
    config, _ = _round_trip(model_root, data_dir, tmp_path, model_type, head={"type": "option_scoring"})
    assert config["decisionformer"]["kind"] == "encoder"


@pytest.mark.parametrize("model_type", [*ZOO_DECODERS, "qwen3_5"])
def test_decoders_with_lora(model_root, data_dir, tmp_path, model_type):
    config, _ = _round_trip(model_root, data_dir, tmp_path, model_type, head={"type": "code_readout"}, extra=LORA)
    assert config["format_id"] == "d3-code-readout-v1"
    assert len(config["token_ids"]) == (26 if model_type in SINGLE_LETTER_CODES else 255)


@pytest.mark.parametrize("model_type, attention", [("qwen3", "bidirectional"), ("qwen3_5", "bidirectional"),
                                                   ("mixtral", "bidirectional"), ("gemma3_text", "bidirectional")])
def test_full_finetune_noncausal(model_root, data_dir, tmp_path, model_type, attention):
    config, _ = _round_trip(model_root, data_dir, tmp_path, model_type, head={"type": "code_readout"},
                            extra={"model": {"attention": attention}, "finetune": {"mode": "full"}})
    assert config["attention_mode"] == "noncausal_full_attention"


def test_seq2seq_without_a_text_encoder_is_refused(model_root, data_dir, tmp_path):
    from conftest import make_tokenizer
    from transformers import AutoConfig, AutoModelForSeq2SeqLM

    path = model_root / "zoo-bart"
    tokenizer = make_tokenizer("encoder")
    AutoModelForSeq2SeqLM.from_config(AutoConfig.for_model(
        "bart", vocab_size=len(tokenizer), d_model=32, encoder_layers=1, decoder_layers=1, encoder_attention_heads=4,
        decoder_attention_heads=4, encoder_ffn_dim=64, decoder_ffn_dim=64)).save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))
    with pytest.raises(ValueError, match="without a standalone text encoder"):
        train(make_config(tmp_path, path, data_dir, head={"type": "option_scoring"}, steps=1))
