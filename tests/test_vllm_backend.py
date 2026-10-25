"""Offline adapter checks using real HF model math behind a small fake engine surface.

These tests do not execute vLLM or establish engine/platform compatibility. D3 fixtures run the bare backbone
and separate readout; native fixtures run an actual Hugging Face sequence classifier.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from test_api import QUESTIONS, _bare_checkpoint, trained  # noqa: F401  (module fixture)

from decisionformer import AutoDecisionModel
from decisionformer.cli import main
from decisionformer.vllm_backend import UnsupportedBackend, VLLMDecider, check_vllm_support

pytestmark = pytest.mark.slow


class FakeLLM:
    """The engine surface used by the adapter, computing raw logits with real Hugging Face models."""

    def __init__(self, model: str, max_model_len: int = 512, **options) -> None:
        from transformers import AutoModel, AutoModelForSequenceClassification

        self.options = options
        config = json.loads((Path(model) / "config.json").read_text())
        self.classifier = None
        if config.get("architectures") == ["Qwen3ForSequenceClassification"]:
            self.classifier = AutoModelForSequenceClassification.from_pretrained(model).eval()
        else:
            self.backbone = AutoModel.from_pretrained(model).eval()
            self.readout = load_file(str(Path(model) / "readout.safetensors"))["weight"]
        self.llm_engine = types.SimpleNamespace(model_config=types.SimpleNamespace(max_model_len=max_model_len))
        self.calls: list[int] = []

    def classify(self, prompts, *, pooling_params=None, use_tqdm=True):
        assert pooling_params.use_activation is False
        self.calls.append(len(prompts))
        outputs = []
        for prompt in prompts:
            ids = torch.tensor([prompt["prompt_token_ids"]])
            with torch.no_grad():
                if self.classifier is not None:
                    logits = self.classifier(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0].tolist()
                else:
                    hidden = self.backbone(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state[0, -1]
                    logits = (self.readout @ hidden).tolist()
            outputs.append(types.SimpleNamespace(outputs=types.SimpleNamespace(probs=logits)))
        return outputs


@pytest.fixture
def fake_vllm(monkeypatch):
    package = types.ModuleType("vllm")
    package.LLM = FakeLLM
    package.PoolingParams = lambda use_activation=None: types.SimpleNamespace(use_activation=use_activation)
    inputs = types.ModuleType("vllm.inputs")
    inputs.TokensPrompt = lambda prompt_token_ids: {"prompt_token_ids": list(prompt_token_ids)}
    modules = {"vllm": package, "vllm.inputs": inputs, "vllm.model_executor": types.ModuleType("x"),
               "vllm.model_executor.models": types.ModuleType("x"),
               "vllm.model_executor.models.d3": types.ModuleType("x")}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return package


def _noncausal(decider) -> None:
    """The fake engine runs the stock backbone; apply the checkpoint's attention mode as vLLM's config does."""
    from decisionformer.attention import apply_attention_mode

    apply_attention_mode(decider.llm.backbone, "bidirectional")


@pytest.mark.parametrize("prompt, attention_mode", [("d3", "noncausal_full_attention"), (None, "causal")])
def test_vllm_backend_answers_like_transformers(model_root, fake_vllm, prompt, attention_mode):
    """Decision 3.0 (d3 prompt, noncausal) and AutoJev-format (no prompt field, causal) Qwen3.5 checkpoints."""
    path, _, _ = _bare_checkpoint(model_root, "qwen3_5", prompt=prompt, attention_mode=attention_mode)
    decider = VLLMDecider.from_pretrained(path)
    assert decider.llm.options["limit_mm_per_prompt"] == {"image": 0, "video": 0}
    if attention_mode == "noncausal_full_attention":
        _noncausal(decider)
    reference = AutoDecisionModel.from_pretrained(path, device="cpu", dtype=torch.float32)
    assert decider.builder.format is reference.builder.format
    state = "the large circle is red ."
    got, expected = decider.decide(state, QUESTIONS), reference.decide(state, QUESTIONS)
    assert decider.llm.calls == [len(QUESTIONS)], "every question of a request goes to vLLM in one call"
    assert got["usage"] == expected["usage"]
    for key, answer in expected["answers"].items():
        for field in ("probabilities", "noul", "score", "confidence"):
            if field in answer:
                assert got["answers"][key][field] == pytest.approx(answer[field], abs=1e-5), (key, field)
        assert got["answers"][key].get("choice") == answer.get("choice")


def test_vllm_backend_refuses_what_vllm_cannot_load(model_root, trained, tmp_path, fake_vllm):  # noqa: F811
    with pytest.raises(UnsupportedBackend, match=r"its backbone is \['LlamaModel'\].*transformers backend"):
        VLLMDecider.from_pretrained(trained / "best")  # a code readout, but on Llama
    path, _, _ = _bare_checkpoint(model_root, "qwen3_5", prompt="d3", attention_mode="noncausal_full_attention")
    import json

    config = json.loads((path / "decision_config.json").read_text())
    with pytest.raises(UnsupportedBackend, match="pooling 'mean'"):
        check_vllm_support(path, {**config, "pooling": "mean"})
    with pytest.raises(UnsupportedBackend, match="attention_mode 'prefix_bidirectional'"):
        check_vllm_support(path, {**config, "attention_mode": "prefix_bidirectional"})
    with pytest.raises(UnsupportedBackend, match="not a decision checkpoint"):
        VLLMDecider.from_pretrained(tmp_path)


def test_vllm_backend_needs_decision_support_in_vllm(model_root, monkeypatch):
    path, _, _ = _bare_checkpoint(model_root, "qwen3_5", prompt="d3", attention_mode="causal")
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.d3", None)  # an older vLLM
    with pytest.raises(UnsupportedBackend, match="no decision-model support"):
        VLLMDecider.from_pretrained(path)


def test_serve_command_reports_unsupported_backend(trained, fake_vllm):  # noqa: F811
    with pytest.raises(SystemExit, match="vLLM cannot serve"):
        main(["serve", str(trained / "best"), "--backend", "vllm"])


@pytest.fixture
def native_classifier(model_root, tmp_path):
    """A native HF artifact built independently of the export implementation under test elsewhere."""
    from transformers import AutoModel, AutoTokenizer, Qwen3ForSequenceClassification

    source, readout, _ = _bare_checkpoint(model_root, "qwen3", prompt="d3", attention_mode="causal")
    backbone = AutoModel.from_pretrained(source)
    backbone.config.num_labels = len(readout)
    backbone.config.head_dtype = "float32"
    classifier = Qwen3ForSequenceClassification(backbone.config)
    classifier.model.load_state_dict(backbone.state_dict(), strict=True)
    with torch.no_grad():
        classifier.score.weight.copy_(readout)
    output = tmp_path / "native-classifier"
    classifier.save_pretrained(output)
    AutoTokenizer.from_pretrained(source).save_pretrained(output)
    config = json.loads((source / "decision_config.json").read_text())
    config.update(max_length=200, temperature_by_type={"noul": 0.7, "score": 1.3})
    config["decisionformer"] = {"export_format": "sequence_classification", "backbone": "full"}
    (output / "decision_config.json").write_text(json.dumps(config))
    # Use precisely the same calibration for the existing canonical reference.
    reference_config = {**config, "decisionformer": {}}
    (source / "decision_config.json").write_text(json.dumps(reference_config))
    return output, source


def test_native_classifier_backend_preserves_decision_contract(native_classifier, fake_vllm, monkeypatch):
    path, source = native_classifier
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.d3", None)
    decider = VLLMDecider.from_pretrained(path)
    assert decider.llm.options["runner"] == "pooling"
    assert decider.llm.options["convert"] == "classify"
    assert decider.max_input_tokens == 200, "native serving keeps the trained decision context limit"
    assert decider.llm.options["pooler_config"] == {
        "task": "classify", "seq_pooling_type": "LAST", "use_activation": False,
    }
    assert decider.llm.options["hf_overrides"]["head_dtype"] == "float32"
    reference = AutoDecisionModel.from_pretrained(source, device="cpu", dtype=torch.float32)
    requests = [{"state": state, "questions": QUESTIONS} for state in (
        "the huge square is blue .", "the circle is red . and the large triangle is near .",
    )]
    actual, expected = decider.decide_many(requests), reference.decide_many(requests)
    assert decider.llm.calls == [len(requests) * len(QUESTIONS)]
    for a, b in zip(actual, expected):
        assert a["usage"] == b["usage"]
        for key, answer in b["answers"].items():
            for field in ("probabilities", "noul", "score", "confidence"):
                if field in answer:
                    assert a["answers"][key][field] == pytest.approx(answer[field], abs=1e-5), (key, field)
            assert a["answers"][key].get("choice") == answer.get("choice")


@pytest.mark.parametrize("options", [
    {"runner": "generate"},
    {"convert": "embed"},
    {"quantization": "awq"},
    {"pooler_config": {"seq_pooling_type": "MEAN"}},
    {"pooler_config": {"use_activation": True}},
    {"pooler_config": {"logit_mean": 1.0}},
    {"pooler_config": {"logit_sigma": 2.0}},
    {"hf_overrides": {"classifier_from_token": ["A", "B"]}},
    {"hf_overrides": {"head_dtype": "model"}},
])
def test_native_classifier_rejects_semantic_engine_overrides(native_classifier, fake_vllm, options):
    path, _ = native_classifier
    with pytest.raises(UnsupportedBackend, match="preserve"):
        VLLMDecider.from_pretrained(path, **options)
