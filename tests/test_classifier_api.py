"""The native classifier retains the decision request/response contract after export."""

from __future__ import annotations

import json

import pytest
import torch
from test_api import QUESTIONS, _bare_checkpoint

from decisionformer import AutoDecisionModel, ContextLengthError
from decisionformer.cli import main


@pytest.fixture
def source(model_root):
    path, _, _ = _bare_checkpoint(model_root, "qwen3", prompt="d3", attention_mode="causal")
    config = json.loads((path / "decision_config.json").read_text())
    config.update(max_length=256, temperature=1.7,
                  temperature_by_type={"choice": 1.3, "noul": 0.6, "score": 2.1})
    (path / "decision_config.json").write_text(json.dumps(config))
    return path


def test_export_command_creates_an_explicit_classifier(source, tmp_path, capsys):
    target = tmp_path / "native"
    main(["export", str(source), str(target), "--format", "sequence-classification"])
    assert json.loads(capsys.readouterr().out)["output"] == str(target)
    config = json.loads((target / "config.json").read_text())
    assert config["architectures"] == ["Qwen3ForSequenceClassification"]
    with pytest.raises(SystemExit, match="exist"):
        main(["export", str(source), str(target), "--format", "sequence-classification"])


def _same_answers(got, expected):
    assert got.keys() == expected.keys()
    for key, answer in expected.items():
        assert got[key].keys() == answer.keys()
        for field, value in answer.items():
            if field in ("probabilities", "noul", "score", "confidence"):
                assert got[key][field] == pytest.approx(value, abs=1e-5), (key, field)
            else:
                assert got[key][field] == value


def test_native_reload_preserves_all_question_types_and_limits(source, tmp_path):
    from decisionformer.classifier import export_sequence_classifier

    target = export_sequence_classifier(source, tmp_path / "native")
    reference = AutoDecisionModel.from_pretrained(source, device="cpu", dtype=torch.float32)
    native = AutoDecisionModel.from_pretrained(target, device="cpu", dtype=torch.float32)
    requests = [{"state": state, "questions": QUESTIONS} for state in (
        "the tiny triangle is yellow .", "the large circle is red . " + "near " * 25,
        {"description": "the huge square is blue ."}, "",
    )]
    for got, expected in zip(native.decide_many(requests), reference.decide_many(requests)):
        _same_answers(got["answers"], expected["answers"])
        assert got["usage"] == expected["usage"]
    assert native.max_input_tokens == reference.max_input_tokens == 256
    for kind in ("choice", "noul", "score"):
        assert native.temperature_for(kind) == reference.temperature_for(kind)
    for model in (reference, native):
        with pytest.raises(ContextLengthError):
            model.decide("red " * 400, {"q": QUESTIONS["red"]})
        with pytest.raises(ContextLengthError):
            model.decide("red", {"q": {"type": "choice", "criteria": dict.fromkeys(map(str, range(256)))}})


def test_native_http_and_predict_preserve_responses(source, tmp_path, capsys):
    from fastapi.testclient import TestClient

    from decisionformer.classifier import export_sequence_classifier
    from decisionformer.serve import create_app

    target = export_sequence_classifier(source, tmp_path / "native")
    request = {"state": "the large circle is green .", "questions": QUESTIONS}
    reference = AutoDecisionModel.from_pretrained(source, device="cpu", dtype=torch.float32)
    native = AutoDecisionModel.from_pretrained(target, device="cpu", dtype=torch.float32)
    with TestClient(create_app(native)) as client:
        response = client.post("/v1/systemone", json=request)
        assert response.status_code == 200
        _same_answers(response.json()["answers"], reference.decide(**request)["answers"])
        response = client.post("/v1/systemone", json={"state": "red " * 400, "questions": QUESTIONS})
        assert response.status_code == 422
    data = tmp_path / "request.jsonl"
    data.write_text(json.dumps(request) + "\n")
    main(["predict", str(target), str(data), "--device", "cpu"])
    got = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    _same_answers(got["answers"], reference.decide(**request)["answers"])


def test_native_save_pretrained_roundtrip(source, tmp_path):
    from decisionformer.classifier import export_sequence_classifier

    native = AutoDecisionModel.from_pretrained(export_sequence_classifier(source, tmp_path / "native"), device="cpu")
    saved = native.save_pretrained(tmp_path / "copied")
    reloaded = AutoDecisionModel.from_pretrained(saved, device="cpu")
    _same_answers(reloaded.decide("red circle", QUESTIONS)["answers"],
                  native.decide("red circle", QUESTIONS)["answers"])


def test_native_reload_refuses_missing_trained_backbone_weights(source, tmp_path):
    from safetensors.torch import load_file, save_file

    from decisionformer.classifier import export_sequence_classifier

    native = export_sequence_classifier(source, tmp_path / "native")
    weights = load_file(str(native / "model.safetensors"))
    weights.pop("model.embed_tokens.weight")
    save_file(weights, str(native / "model.safetensors"))
    with pytest.raises(ValueError, match="missing|Missing|incomplete"):
        AutoDecisionModel.from_pretrained(native, device="cpu", dtype=torch.float32)
