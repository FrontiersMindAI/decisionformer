"""Native HF exports preserve the trained answer head, not the original LM output rows."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from test_api import QUESTIONS, _bare_checkpoint
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from decisionformer import AutoDecisionModel
from decisionformer.schema import Decision, parse_questions


def _module():
    assert importlib.util.find_spec("decisionformer.classifier") is not None, "Native classifier export is not implemented"
    return importlib.import_module("decisionformer.classifier")


def _source(root: Path) -> Path:
    source, _, _ = _bare_checkpoint(root, "qwen3", prompt="d3", attention_mode="causal")
    config = json.loads((source / "decision_config.json").read_text())
    config.update(max_length=256, temperature_by_type={"choice": 0.7, "noul": 1.4, "score": 1.8})
    config["decisionformer"] = {"backbone": "full", "kind": "decoder", "model_type": "qwen3",
                               "chat_template_kwargs": {"enable_thinking": False},
                               "heads": {"default": {"type": "code_readout"}}}
    (source / "decision_config.json").write_text(json.dumps(config))
    return source


def _hashes(root: Path) -> dict[str, str]:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}


def test_export_preserves_trained_tensors_metadata_and_native_logits(tmp_path):
    module = _module()
    source = _source(tmp_path)
    before = _hashes(source)
    original = AutoDecisionModel.from_pretrained(source, device="cpu", dtype=torch.float32)
    weight = load_file(str(source / "readout.safetensors"))["weight"]
    target = module.export_sequence_classifier(source, tmp_path / "classifier")
    assert target == tmp_path / "classifier"
    assert _hashes(source) == before
    assert not (target / "readout.safetensors").exists() and not (target / "heads.safetensors").exists()
    native, loading = AutoModelForSequenceClassification.from_pretrained(
        target, dtype=torch.float32, output_loading_info=True)
    native.eval()
    assert not loading["missing_keys"] and not loading["unexpected_keys"]
    assert native.config.architectures == ["Qwen3ForSequenceClassification"]
    assert native.config.model_type == "qwen3" and native.score.weight.dtype == torch.float32
    assert torch.equal(native.score.weight, weight)
    for key, value in original.model.backbone.model.state_dict().items():
        assert torch.equal(native.model.state_dict()[key], value)
    config = json.loads((target / "decision_config.json").read_text())
    expected = json.loads((source / "decision_config.json").read_text())
    assert config["decisionformer"]["export_format"] == module.CLASSIFIER_FORMAT
    assert config["decisionformer"]["export_dtype"] == "float32"
    for key, value in expected.items():
        if key == "decisionformer":
            assert all(config[key][k] == v for k, v in value.items())
        else:
            assert config[key] == value
    assert not torch.equal(native.score.weight, native.model.get_input_embeddings().weight[config["token_ids"]])
    module.validate_classifier_checkpoint(target, config)
    assert torch.equal(module.classifier_readout(target), weight)
    tokenizer = AutoTokenizer.from_pretrained(target)
    assert tokenizer.chat_template == original.tokenizer.chat_template
    assert tokenizer.pad_token_id == original.tokenizer.pad_token_id

    decisions = [Decision(id=str(i), request_id=str(i), state=state, question=parse_questions({"q": question})[0],
                          task="default") for i, (state, question) in enumerate([
        ("the large circle is red .", QUESTIONS["color"]),
        ("the tiny square is blue . " * 4, QUESTIONS["red"]),
        ("the huge triangle is green .", QUESTIONS["size"]),
    ])]
    encoded = [original.builder.encode(d) for d in decisions]
    length = max(len(e.input_ids) for e in encoded)
    ids = torch.full((len(encoded), length), tokenizer.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, item in enumerate(encoded):
        ids[i, :len(item.input_ids)] = torch.tensor(item.input_ids)
        mask[i, :len(item.input_ids)] = 1
    with torch.inference_mode():
        native_logits = native(input_ids=ids, attention_mask=mask).logits
        hidden = original.model.backbone.model(input_ids=ids, attention_mask=mask).last_hidden_state
        last = hidden[torch.arange(len(encoded)), torch.tensor([len(e.input_ids)-1 for e in encoded])]
        manual = last.float() @ weight.T
    torch.testing.assert_close(native_logits, manual, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(native_logits, hidden[:, -1].float() @ weight.T), "padding must matter in this fixture"


@pytest.mark.parametrize("existing", ["directory", "file"])
def test_export_refuses_existing_destination(tmp_path, existing):
    module = _module()
    source = _source(tmp_path)
    target = tmp_path / "occupied"
    if existing == "directory":
        target.mkdir()
        (target / "keep.txt").write_text("preserve")
    else:
        target.write_text("preserve")
    with pytest.raises(FileExistsError):
        module.export_sequence_classifier(source, target)
    assert (target / "keep.txt").read_text() == "preserve" if existing == "directory" else target.read_text() == "preserve"


@pytest.mark.parametrize("change, match", [
    ("attention", "causal"), ("pooling", "last"), ("adapter", "merged"),
    ("architecture", "Qwen3"), ("heads", "single.*readout"), ("quantization", "quantiz"),
    ("shape", "shape"), ("nonfinite", "finite"),
])
def test_export_refuses_unsupported_sources_before_writing(tmp_path, change, match):
    module = _module()
    source = _source(tmp_path)
    config = json.loads((source / "decision_config.json").read_text())
    hf_config = json.loads((source / "config.json").read_text())
    if change == "attention":
        config["attention_mode"] = "noncausal_full_attention"
    elif change == "pooling":
        config["pooling"] = "mean"
    elif change == "adapter":
        config["decisionformer"] = {"backbone": "adapter"}
    elif change == "architecture":
        hf_config["model_type"] = "llama"
    elif change == "heads":
        save_file({"other": torch.zeros(1)}, str(source / "heads.safetensors"))
    elif change == "quantization":
        hf_config["quantization_config"] = {"quant_method": "bitsandbytes"}
    else:
        weight = load_file(str(source / "readout.safetensors"))["weight"]
        if change == "shape":
            weight = weight[:, :-1].contiguous()
        else:
            weight[0, 0] = torch.nan
        save_file({"weight": weight}, str(source / "readout.safetensors"))
    (source / "decision_config.json").write_text(json.dumps(config))
    (source / "config.json").write_text(json.dumps(hf_config))
    target = tmp_path / "refused"
    with pytest.raises(ValueError, match=match):
        module.export_sequence_classifier(source, target)
    assert not target.exists()


def test_classifier_readout_loads_sharded_head_without_other_tensors(tmp_path):
    module = _module()
    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    save_file({"model.embed_tokens.weight": torch.zeros(7, 4)}, str(tmp_path / "model-00001-of-00002.safetensors"))
    save_file({"score.weight": weight}, str(tmp_path / "model-00002-of-00002.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
        "score.weight": "model-00002-of-00002.safetensors",
    }}))
    assert torch.equal(module.classifier_readout(tmp_path), weight)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_export_explicitly_promotes_stored_backbone_values_to_float32(tmp_path, dtype):
    from transformers import AutoModel

    module = _module()
    source = _source(tmp_path)
    backbone = AutoModel.from_pretrained(source).to(dtype)
    backbone.save_pretrained(source)
    before = _hashes(source)
    target = module.export_sequence_classifier(source, tmp_path / "classifier")
    native = AutoModelForSequenceClassification.from_pretrained(target, dtype=torch.float32)
    for key, value in backbone.state_dict().items():
        assert torch.equal(native.model.state_dict()[key], value.float())
    assert _hashes(source) == before
    config = json.loads((target / "decision_config.json").read_text())
    assert config["decisionformer"]["export_dtype"] == "float32"


def test_export_handles_explicit_null_extension_metadata(tmp_path):
    module = _module()
    source = _source(tmp_path)
    config = json.loads((source / "decision_config.json").read_text())
    config["decisionformer"] = None
    (source / "decision_config.json").write_text(json.dumps(config))
    target = module.export_sequence_classifier(source, tmp_path / "classifier")
    result = json.loads((target / "decision_config.json").read_text())
    assert result["decisionformer"]["export_format"] == module.CLASSIFIER_FORMAT


@pytest.mark.parametrize("field, value", [("classifier_from_token", ["A"]), ("method", "from_2_way_softmax"),
                                         ("logit_mean", 0.0), ("logit_sigma", 1.0)])
def test_native_validation_refuses_logit_or_head_transformations(tmp_path, field, value):
    module = _module()
    source = _source(tmp_path)
    target = module.export_sequence_classifier(source, tmp_path / "classifier")
    config = json.loads((target / "decision_config.json").read_text())
    hf_config = json.loads((target / "config.json").read_text())
    hf_config[field] = value
    (target / "config.json").write_text(json.dumps(hf_config))
    with pytest.raises(ValueError, match=field):
        module.validate_classifier_checkpoint(target, config)


@pytest.mark.parametrize("missing", [True, False], ids=["missing", "unexpected"])
def test_export_refuses_incomplete_or_unexpected_backbone_weights(tmp_path, missing):
    module = _module()
    source = _source(tmp_path)
    tensors = load_file(str(source / "model.safetensors"))
    if missing:
        del tensors["layers.0.self_attn.q_proj.weight"]
    else:
        tensors["unexpected.weight"] = torch.ones(2)
    save_file(tensors, str(source / "model.safetensors"))
    before = _hashes(source)
    target = tmp_path / "classifier"
    with pytest.raises(ValueError, match="[Bb]ackbone.*(missing|unexpected|match)"):
        module.export_sequence_classifier(source, target)
    assert not target.exists()
    assert _hashes(source) == before


@pytest.mark.parametrize("failure", ["weights", "tokenizer", "validation"])
def test_failed_staging_preserves_source_and_allows_retry(tmp_path, monkeypatch, failure):
    from transformers import PreTrainedModel

    module = _module()
    source = _source(tmp_path)
    before = _hashes(source)
    target = tmp_path / "classifier"
    tokenizer = AutoTokenizer.from_pretrained(source)

    def fail(*args, **kwargs):
        raise OSError("injected export write failure")

    with monkeypatch.context() as patch:
        if failure == "weights":
            patch.setattr(PreTrainedModel, "save_pretrained", fail)
        elif failure == "tokenizer":
            patch.setattr(tokenizer, "save_pretrained", fail)
            patch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
        else:
            patch.setattr(module, "validate_classifier_checkpoint", fail)
        with pytest.raises(OSError, match="injected export write failure"):
            module.export_sequence_classifier(source, target)
    assert not target.exists()
    assert not list(tmp_path.glob(".classifier-*"))
    assert _hashes(source) == before
    assert module.export_sequence_classifier(source, target) == target


@pytest.mark.parametrize("occupied", ["empty_directory", "directory", "file"])
def test_export_refuses_a_destination_created_during_staging(tmp_path, monkeypatch, occupied):
    from transformers import PreTrainedModel

    module = _module()
    source = _source(tmp_path)
    before = _hashes(source)
    target = tmp_path / "classifier"
    save = PreTrainedModel.save_pretrained

    def concurrent_destination(self, *args, **kwargs):
        result = save(self, *args, **kwargs)
        if occupied == "file":
            target.write_text("keep")
        else:
            target.mkdir(exist_ok=True)
            if occupied == "directory":
                (target / "keep.txt").write_text("keep")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(PreTrainedModel, "save_pretrained", concurrent_destination)
        with pytest.raises(FileExistsError):
            module.export_sequence_classifier(source, target)
    if occupied == "file":
        assert target.read_text() == "keep"
    elif occupied == "directory":
        assert sorted(p.name for p in target.iterdir()) == ["keep.txt"]
    else:
        assert not list(target.iterdir())
    assert not list(tmp_path.glob(".classifier-*"))
    assert _hashes(source) == before
