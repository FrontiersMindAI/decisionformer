"""Opt-in parity with an actual vLLM engine, using an offline synthetic checkpoint.

The readout is deliberately changed after random initialization. This fixture
tests preservation of that independent head, not training quality or speed.
"""

from __future__ import annotations

import argparse
import json
import os
import runpy
from pathlib import Path

import pytest
import torch
from conftest import make_tokenizer
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

from decisionformer.classifier import export_sequence_classifier

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(os.environ.get("DECISIONFORMER_RUN_VLLM") != "1",
                       reason="Set DECISIONFORMER_RUN_VLLM=1 in a supported vLLM environment"),
]


def _checkpoint(root: Path) -> Path:
    tokenizer = make_tokenizer("decoder")
    torch.manual_seed(19)
    config = Qwen3Config(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                        num_attention_heads=2, num_key_value_heads=2, head_dim=32, max_position_embeddings=256,
                        pad_token_id=tokenizer.pad_token_id, bos_token_id=None,
                        eos_token_id=tokenizer.eos_token_id, tie_word_embeddings=True, attention_dropout=0.0)
    model = Qwen3ForCausalLM(config)
    source = root / "source"
    model.model.save_pretrained(source)
    tokenizer.save_pretrained(source)
    codes = list("ABCDEFGH")
    token_ids = [tokenizer.encode(code, add_special_tokens=False)[0] for code in codes]
    original = model.lm_head.weight[token_ids].detach().float().clone()
    readout = original + torch.randn_like(original) * 0.05
    assert not torch.equal(readout, original), "the independent answer head must differ from the LM rows"
    save_file({"weight": readout.contiguous()}, str(source / "readout.safetensors"))
    decision_config = {
        "format_version": 1, "format_id": "d3-code-readout-v1", "prompt": "d3",
        "codes": codes, "token_ids": token_ids, "attention_mode": "causal", "pooling": "last",
        "max_length": 256, "temperature": 1.2,
        "temperature_by_type": {"choice": 0.7, "noul": 1.4, "score": 1.8},
        "decisionformer": {"backbone": "full", "kind": "decoder", "model_type": "qwen3",
                           "chat_template_kwargs": {"enable_thinking": False},
                           "heads": {"default": {"type": "code_readout"}}},
    }
    (source / "decision_config.json").write_text(json.dumps(decision_config), encoding="utf-8")
    return source


def test_native_classifier_runs_in_actual_vllm(tmp_path, monkeypatch):
    pytest.importorskip("vllm")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    if not torch.cuda.is_available():
        monkeypatch.setenv("VLLM_CPU_KVCACHE_SPACE", os.environ.get("VLLM_CPU_KVCACHE_SPACE", "1"))
        monkeypatch.setenv("VLLM_CPU_OMP_THREADS_BIND", os.environ.get("VLLM_CPU_OMP_THREADS_BIND", "nobind"))
    source = _checkpoint(tmp_path)
    exported = export_sequence_classifier(source, tmp_path / "classifier")
    questions = {
        "color": {"type": "choice", "instructions": "What is the color ?",
                  "criteria": dict.fromkeys(["red", "blue", "green", "yellow"])},
        "red": {"type": "noul", "instructions": "Is it red ?"},
        "size": {"type": "score", "instructions": "How big ?", "criteria": ["small", "medium", "large"]},
        "eight": {"type": "choice", "instructions": "Choose the best matching option.",
                  "criteria": dict.fromkeys(["red", "blue", "green", "yellow", "tiny", "small", "large", "huge"])},
    }
    states = ["the large circle is red .", "the tiny square is blue . " * 5,
              {"object": "triangle", "color": "green"}, ""]
    requests = tmp_path / "requests.jsonl"
    requests.write_text("".join(json.dumps({"state": state, "questions": questions}) + "\n" for state in states),
                        encoding="utf-8")
    example = runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples" / "compare_classifier.py"))
    args = argparse.Namespace(source=str(source), exported=str(exported), requests=requests, backend="vllm",
                              device="cpu", dtype="float32", atol=1e-4,
                              engine_options=json.dumps({"enforce_eager": True, "max_num_seqs": 16,
                                                         "max_num_batched_tokens": 1024,
                                                         "enable_prefix_caching": False,
                                                         "gpu_memory_utilization": 0.1}))
    result = example["compare"](args)
    print(json.dumps(result, indent=2, allow_nan=False))
    assert result["passed"], result
    assert result["requests"] == 4 and result["questions"] == 16
    assert result["choice_mismatches"] == result["prompt_mismatches"] == 0
