"""Custom-code backbones (FrontiersMind's Nandi and Lumma) and training on from decision checkpoints (Lumma-fev v2),
on tiny offline models. tests/test_hub_models.py runs the same paths on the real Hub checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from conftest import build_decoder, make_tokenizer
from custom_model import BOS, build_custom_decoder
from safetensors.torch import load_file, save_file

from decisionformer.api import AutoDecisionModel
from decisionformer.cli import build_parser
from decisionformer.config import load_config
from decisionformer.data import make_batch
from decisionformer.formats import AUTOJEV
from decisionformer.prompting import PromptBuilder, answer_codes, plain_style, prepare_tokenizer
from decisionformer.schema import parse_request

QUESTION = {"type": "choice", "instructions": "What is the color ?",
            "criteria": {"red": None, "blue": None, "green": None, "yellow": None}}
STATE = "the tiny square is green ."


def lumma_reference_prompt(state: str, question: dict, codes: list[str]) -> str:
    """The plain-text prompt of the lumma-fev runtime (lumma_fev.readout.plain, v0.4) for a choice question."""
    lines = "\n".join(f"{code}: {key}" for code, key in zip(codes, question["criteria"]))
    return (AUTOJEV.system + "\n\n" + f"State:\n{state}\n\nQuestion:\n{question['instructions']}\n\nOptions:\n"
            f"{lines}\n\nReturn only the letter code of the best option.\n\nAnswer:")


def _custom_tokenizer(model_root: Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(build_custom_decoder(model_root)), trust_remote_code=True)
    prepare_tokenizer(tokenizer)
    return tokenizer


def _config(tmp_path: Path, backbone: Path, data_dir: Path, *, out: str, steps: int = 6, remote: bool = False,
            prompt: dict | None = None, finetune: dict | None = None, attention: str | None = None):
    model = {"backbone": str(backbone), "trust_remote_code": remote, "head": {"type": "code_readout"}}
    if attention:
        model["attention"] = attention
    config = {
        "model": model,
        "finetune": finetune or {"mode": "full"},
        "data": {"train": str(data_dir / "train.jsonl"), "eval": {"dev": str(data_dir / "dev.jsonl")},
                 "calibration": str(data_dir / "calibration.jsonl"), "max_length": 256, "prompt": prompt or {}},
        "train": {"output_dir": str(tmp_path / out), "max_steps": steps, "global_batch_size": 16,
                  "micro_batch_size": 8, "lr": 3e-3, "seed": 3},
        "eval": {"micro_batch_size": 16},
    }
    path = tmp_path / f"{out}.yaml"
    path.write_text(yaml.safe_dump(config))
    return load_config(path)


# ------------------------------------------------------------------------------------------------ prompt rendering
@pytest.mark.parametrize("requested, model_type, expected", [
    ("auto", "nandi", "lumma"), ("auto", "Lumma", "lumma"), ("auto", "lumma", "lumma"),
    ("auto", "llama", "decisionformer"), ("auto", None, "decisionformer"),
    ("decisionformer", "nandi", "decisionformer"), ("lumma", "llama", "lumma")])
def test_plain_style_follows_the_backbone_family(requested, model_type, expected):
    assert plain_style(requested, model_type) == expected


def test_lumma_plain_prompt_is_token_exact(model_root):
    """BOS + space + the reference prompt, read as one string, even though the tokenizer's own `encode` would
    prepend a second BOS; the answer codes are the space-prefixed letters."""
    tokenizer = _custom_tokenizer(model_root)
    builder = PromptBuilder(tokenizer, "autojev", "decoder", max_length=1024, plain="lumma")
    assert builder.mode == "plain" and builder.lumma_plain
    decision = parse_request({"id": "x", "state": STATE, "questions": {"q": QUESTION}})[0]
    encoded = builder.encode(decision)
    expected = tokenizer.backend_tokenizer.encode(
        f"{BOS} " + lumma_reference_prompt(STATE, QUESTION, builder.codes), add_special_tokens=False).ids
    assert encoded.input_ids == expected
    assert encoded.input_ids[0] == tokenizer.convert_tokens_to_ids(BOS) and encoded.input_ids.count(expected[0]) == 1
    assert tokenizer.convert_ids_to_tokens(encoded.input_ids[-2:]) == ["Answer", ":"]
    assert encoded.decision_index == len(encoded.input_ids) - 1
    for slot, (start, end) in enumerate(encoded.option_spans):
        assert builder.codes[slot] in tokenizer.decode(encoded.input_ids[start:end])
    codes, ids = answer_codes(tokenizer, "plain", {}, plain="lumma")
    assert codes[:3] == ["A", "B", "C"] and ids == builder.code_token_ids


def test_default_plain_prompt_is_unchanged():
    """Plain models of other families keep decisionformer's own convention (existing checkpoints load as before)."""
    tokenizer = make_tokenizer("decoder")
    tokenizer.chat_template = None
    prepare_tokenizer(tokenizer)
    builder = PromptBuilder(tokenizer, "autojev", "decoder", max_length=1024)
    decision = parse_request({"id": "x", "state": STATE, "questions": {"q": QUESTION}})[0]
    text = lumma_reference_prompt(STATE, QUESTION, builder.codes).removesuffix("\n\nAnswer:") + "\nAnswer:"
    assert builder.encode(decision).input_ids == tokenizer(text, add_special_tokens=True)["input_ids"]


# ------------------------------------------------------------------------------------------------ custom-code backbones
def test_code_readout_starts_from_the_lm_head_through_the_factorized_projection(model_root):
    """For a factorized embedding the readout rows must reproduce the causal LM's own code-token logits."""
    from transformers import AutoModel, AutoModelForCausalLM

    from decisionformer.backbone import output_rows

    path = build_custom_decoder(model_root)
    lm = AutoModelForCausalLM.from_pretrained(str(path), trust_remote_code=True).eval()
    base = AutoModel.from_pretrained(str(path), trust_remote_code=True).eval()
    ids = [5, 9, 12]
    rows = output_rows(base, str(path), None, ids)
    assert rows.shape == (3, base.config.hidden_size)
    tokens = torch.tensor([[3, 7, 11, 4]])
    with torch.no_grad():
        hidden = base(input_ids=tokens).last_hidden_state[0, -1]
        logits = lm(input_ids=tokens).logits[0, -1, ids]
    torch.testing.assert_close(rows @ hidden, logits, rtol=1e-4, atol=1e-5)


@pytest.mark.slow
def test_custom_code_backbone_trains_exports_and_reloads(model_root, data_dir, tmp_path):
    from decisionformer.trainer import train

    config = _config(tmp_path, build_custom_decoder(model_root), data_dir, out="custom", remote=True,
                     prompt={"plain": "lumma"})
    train(config)
    best = Path(config.train.output_dir) / "best"
    for name in ("configuration_tinycustom.py", "modeling_tinycustom.py", "tokenization_tinycustom.py"):
        assert (best / name).is_file(), name  # the checkpoint carries its own code
    assert "auto_map" in json.loads((best / "tokenizer_config.json").read_text())
    saved = json.loads((best / "decision_config.json").read_text())
    assert saved["plain_prompt"] == "lumma" and saved["decisionformer"]["trust_remote_code"] is True
    assert "trust_remote_code=True" in (best / "README.md").read_text()

    with pytest.raises(ValueError, match="trust-remote-code"):
        AutoDecisionModel.from_pretrained(best, device="cpu")
    model = AutoDecisionModel.from_pretrained(best, device="cpu", dtype=torch.float32, trust_remote_code=True)
    assert model.builder.lumma_plain
    answer = model.decide(STATE, {"q": QUESTION})["answers"]["q"]
    assert set(answer["probabilities"]) == set(QUESTION["criteria"])
    assert sum(answer["probabilities"].values()) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.slow
def test_custom_code_backbone_with_lora_merges_into_plain_weights(model_root, data_dir, tmp_path):
    from decisionformer.trainer import train

    config = _config(tmp_path, build_custom_decoder(model_root), data_dir, out="custom-lora", remote=True,
                     finetune={"mode": "lora", "lora": {"r": 4, "lora_alpha": 8, "target_modules": "all-linear"}})
    train(config)
    best = Path(config.train.output_dir) / "best"
    assert not (best / "adapter").exists() and (best / "modeling_tinycustom.py").is_file()
    assert json.loads((best / "decision_config.json").read_text())["decisionformer"]["backbone"] == "full"
    model = AutoDecisionModel.from_pretrained(best, device="cpu", dtype=torch.float32, trust_remote_code=True)
    assert model.decide(STATE, {"q": QUESTION})["answers"]["q"]["choice"] in QUESTION["criteria"]


def test_cli_passes_trust_remote_code():
    parser = build_parser()
    for command in (["eval", "m", "d.jsonl"], ["predict", "m", "r.jsonl"], ["serve", "m"]):
        assert parser.parse_args([*command, "--trust-remote-code"]).trust_remote_code is True
        assert parser.parse_args(command).trust_remote_code is False


def test_custom_recipes_validate():
    for name, backbone in (("nandi_mini", "FrontiersMind/Nandi-Mini-150M"),
                           ("lumma_base", "FrontiersMind/Lumma-0.6B-Base"),
                           ("lumma_fev_continue", "FrontiersMind/Lumma-fev-0.1B-v2")):
        config = load_config(name)
        assert config.model.backbone == backbone and config.model.trust_remote_code is True
        assert config.data.prompt.plain == "auto"


# ------------------------------------------------------------------------------------- continuing a decision model
def _lumma_fev_v2_checkpoint(root: Path) -> tuple[Path, torch.Tensor, list[str]]:
    """A checkpoint laid out like FrontiersMind/Lumma-fev-4B-v2 and -27B-v2: a bare Qwen3.5 backbone, format 2,
    no prompt field (AutoJev's prompt), bidirectional full-attention layers and a 255-row readout."""
    from transformers import AutoModel

    path = root / "lumma-fev-v2-like"
    if path.exists():
        saved = json.loads((path / "decision_config.json").read_text())
        return path, load_file(str(path / "readout.safetensors"))["weight"], saved["codes"]
    backbone = AutoModel.from_pretrained(str(build_decoder(root, "qwen3_5")))
    backbone.save_pretrained(str(path))
    tokenizer = make_tokenizer("decoder")
    tokenizer.save_pretrained(str(path))
    prepare_tokenizer(tokenizer)
    codes, ids = answer_codes(tokenizer, "chat", {"enable_thinking": False})
    readout = torch.randn(255, backbone.config.get_text_config().hidden_size,
                          generator=torch.Generator().manual_seed(5))
    save_file({"weight": readout}, str(path / "readout.safetensors"))
    (path / "decision_config.json").write_text(json.dumps({
        "step": 900, "format_version": 2, "base_model": "tiny", "revision": "0" * 40, "codes": codes,
        "token_ids": ids, "temperature": 0.99, "attention_mode": "noncausal_full_attention"}))
    return path, readout, codes


def test_training_on_from_a_decision_checkpoint_keeps_its_prompt_attention_and_readout(model_root):
    from decisionformer.model import build_model

    path, readout, codes = _lumma_fev_v2_checkpoint(model_root)
    config = load_config({"model": {"backbone": str(path), "head": {"type": "code_readout"}},
                          "train": {"output_dir": "unused"}})
    result = build_model(config, torch.device("cpu"), training=True)
    assert result.builder.format is AUTOJEV and result.attention == "bidirectional"
    assert result.builder.codes == codes
    torch.testing.assert_close(result.model.heads["default"].readout.weight.detach(), readout[:len(codes)])
    # Settings given explicitly still win over the checkpoint's own.
    explicit = load_config({"model": {"backbone": str(path), "head": {"type": "code_readout"}, "attention": "causal"},
                            "data": {"prompt": {"style": "d3"}}, "train": {"output_dir": "unused"}})
    result = build_model(explicit, torch.device("cpu"), training=True)
    assert result.builder.format.name == "d3" and result.attention == "causal"


@pytest.mark.slow
def test_continued_model_matches_the_checkpoint_before_training(model_root, data_dir, tmp_path):
    """At step 0 the continued model answers exactly like the checkpoint it starts from; then it trains."""
    from decisionformer.api import Decider
    from decisionformer.model import build_model
    from decisionformer.trainer import train

    path, _, _ = _lumma_fev_v2_checkpoint(model_root)
    config = _config(tmp_path, path, data_dir, out="continued", steps=4)
    original = Decider.from_pretrained(path, device="cpu", dtype=torch.float32)
    rebuilt = build_model(config, torch.device("cpu"), training=False)
    rebuilt.model.float()
    decision = parse_request({"id": "x", "state": STATE, "questions": {"q": QUESTION}})[0]
    encoded = original.builder.encode(decision)
    assert rebuilt.builder.encode(decision).input_ids == encoded.input_ids
    batch = make_batch([decision], [list(range(decision.num_options))], [encoded], original.tokenizer.pad_token_id)
    with torch.no_grad():
        torch.testing.assert_close(rebuilt.model(batch), original.model(batch), rtol=1e-4, atol=1e-4)
    train(config)
    best = Path(config.train.output_dir) / "best"
    saved = json.loads((best / "decision_config.json").read_text())
    assert saved["prompt"] == "autojev" and saved["attention_mode"] == "noncausal_full_attention"
    assert AutoDecisionModel.from_pretrained(best, device="cpu").decide(STATE, {"q": QUESTION})["answers"]["q"]


@pytest.mark.slow
def test_training_on_from_a_custom_code_decision_checkpoint(model_root, data_dir, tmp_path):
    """Like continuing FrontiersMind/Lumma-fev-0.1B-v2: a decision checkpoint on a custom-code backbone."""
    from decisionformer.trainer import train

    first = _config(tmp_path, build_custom_decoder(model_root), data_dir, out="first", remote=True)
    train(first)
    start = Path(first.train.output_dir) / "best"
    second = _config(tmp_path, start, data_dir, out="second", remote=True, steps=4)
    train(second)
    best = Path(second.train.output_dir) / "best"
    saved = json.loads((best / "decision_config.json").read_text())
    assert saved["base_model"] == str(start) and (best / "modeling_tinycustom.py").is_file()
    model = AutoDecisionModel.from_pretrained(best, device="cpu", trust_remote_code=True)
    assert model.decide(STATE, {"q": QUESTION})["answers"]["q"]["choice"] in QUESTION["criteria"]


def test_adapter_checkpoints_cannot_be_continued_directly(tmp_path):
    from decisionformer.model import initial_readout

    config = load_config({"model": {"backbone": str(tmp_path), "head": {"type": "code_readout"}},
                          "train": {"output_dir": "unused"}})
    with pytest.raises(ValueError, match="adapter checkpoint"):
        initial_readout({"decisionformer": {"backbone": "adapter"}, "codes": ["A", "B"]}, config)
