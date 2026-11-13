"""The real FrontiersMind checkpoints from the Hugging Face Hub: inference, fine-tuning and training on.

Opt in, since they download the models:
  DECISIONFORMER_HUB_TESTS=1 pytest tests/test_hub_models.py        Nandi-Mini-150M, Lumma-0.6B-Base, Lumma-fev-0.1B-v2
                                                                     (about 2 GB, runs on a CPU)
  DECISIONFORMER_HUB_LARGE=1 as well                                 Lumma-fev-4B-v2 and -27B-v2 (about 10 GB and 55 GB;
                                                                     a CUDA GPU, 80 GB for the 27B)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
import yaml

from decisionformer.api import AutoDecisionModel
from decisionformer.config import load_config
from decisionformer.formats import AUTOJEV

pytestmark = [pytest.mark.slow, pytest.mark.hub,
              pytest.mark.skipif(os.environ.get("DECISIONFORMER_HUB_TESTS") != "1",
                                 reason="downloads Hub models; set DECISIONFORMER_HUB_TESTS=1")]
large = pytest.mark.skipif(os.environ.get("DECISIONFORMER_HUB_LARGE") != "1" or not torch.cuda.is_available(),
                           reason="multi-GB checkpoints; set DECISIONFORMER_HUB_LARGE=1 on a CUDA machine")

LUMMA_FEV_01B = "FrontiersMind/Lumma-fev-0.1B-v2"
NANDI = "FrontiersMind/Nandi-Mini-150M"
LUMMA_BASE = "FrontiersMind/Lumma-0.6B-Base"
STATE = "My integration keeps failing with a 500 error. Please help, our launch is tomorrow."
QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team should handle this request?",
             "criteria": {"billing": "Charges and refunds", "support": "Technical integration errors",
                          "sales": "Questions about buying a product"}},
    "urgent": {"type": "noul", "instructions": "Does the customer need a reply today?"},
    "mood": {"type": "score", "instructions": "How upset is the customer?", "criteria": ["Calm", "Annoyed", "Angry"]},
}


def _reference_probabilities(repo: str, question: dict) -> list[float]:
    """What the lumma-fev runtime computes for a text-only (Nandi) checkpoint, written out independently: the
    tokenizer's own call adds "<|im_start|> ", the prompt ends in "\\n\\nAnswer:", and the readout reads the last
    token's hidden state, divided by the checkpoint's temperature."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer

    saved = json.loads(Path(hf_hub_download(repo, "decision_config.json")).read_text())
    readout = load_file(hf_hub_download(repo, "readout.safetensors"))["weight"].float()
    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    model = AutoModel.from_pretrained(repo, trust_remote_code=True, dtype=torch.float32).eval()
    if question["type"] == "choice":
        options = [f"{k}: {v}" for k, v in question["criteria"].items()]
    elif question["type"] == "score":
        options = list(question["criteria"])
    else:
        options = ["No / false", "Yes / true"]
    lines = "\n".join(f"{code}: {text}" for code, text in zip(saved["codes"], options))
    prompt = (AUTOJEV.system + "\n\n" + f"State:\n{STATE}\n\nQuestion:\n{question['instructions']}\n\nOptions:\n{lines}"
              "\n\nReturn only the letter code of the best option.\n\nAnswer:")
    ids = tokenizer([prompt], add_special_tokens=False, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        hidden = model(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state[0, -1]
    return (readout[:len(options)] @ hidden / saved["temperature"]).softmax(-1).tolist()


def _probabilities(answer: dict) -> list[float]:
    if answer["type"] == "noul":
        return [1 - answer["noul"], answer["noul"]]
    return list(answer["probabilities"].values())


# ------------------------------------------------------------------------------------------------ inference
def test_lumma_fev_01b_v2_needs_trust_remote_code():
    with pytest.raises(ValueError, match="trust_remote_code=True"):
        AutoDecisionModel.from_pretrained(LUMMA_FEV_01B, device="cpu")


def test_lumma_fev_01b_v2_matches_the_reference_runtime():
    model = AutoDecisionModel.from_pretrained(LUMMA_FEV_01B, device="cpu", dtype=torch.float32,
                                              trust_remote_code=True)
    assert model.builder.format is AUTOJEV and model.builder.lumma_plain and model.model.backbone.attention == "causal"
    answers = model.decide(STATE, QUESTIONS)["answers"]
    for key, question in QUESTIONS.items():
        assert _probabilities(answers[key]) == pytest.approx(_reference_probabilities(LUMMA_FEV_01B, question),
                                                             abs=1e-4), key
    assert answers["team"]["choice"] == "support"


def test_lumma_fev_01b_v2_matches_the_lumma_fev_package():
    lumma_fev = pytest.importorskip("lumma_fev", reason="pip install 'lumma-fev>=0.4' to compare with it")
    ours = AutoDecisionModel.from_pretrained(LUMMA_FEV_01B, device="cpu", dtype=torch.float32,
                                             trust_remote_code=True).decide(STATE, QUESTIONS)["answers"]
    theirs = lumma_fev.load(LUMMA_FEV_01B, device="cpu", dtype="fp32").decide(STATE, QUESTIONS)
    for key in QUESTIONS:
        assert _probabilities(ours[key]) == pytest.approx(_probabilities(theirs[key]), abs=2e-4), key


# ------------------------------------------------------------------------------------------------ fine-tuning
def _config(tmp_path: Path, data_dir: Path, backbone: str, *, out: str, finetune: dict, steps: int = 3,
            remote: bool = True, precision: str = "no"):
    config = {
        "model": {"backbone": backbone, "trust_remote_code": remote, "head": {"type": "code_readout"}},
        "finetune": finetune,
        "data": {"train": str(data_dir / "train.jsonl"), "eval": {"dev": str(data_dir / "dev.jsonl")},
                 "calibration": str(data_dir / "calibration.jsonl"), "max_length": 512},
        "train": {"output_dir": str(tmp_path / out), "max_steps": steps, "global_batch_size": 4,
                  "micro_batch_size": 2, "lr": 1e-5, "mixed_precision": precision, "eval_every": steps},
        "eval": {"micro_batch_size": 8},
    }
    path = tmp_path / f"{out}.yaml"
    path.write_text(yaml.safe_dump(config))
    return load_config(path)


def _check_export(best: Path, code_files: tuple[str, ...]) -> None:
    for name in code_files:
        assert (best / name).is_file(), name
    saved = json.loads((best / "decision_config.json").read_text())
    assert saved["plain_prompt"] == "lumma" and saved["decisionformer"]["trust_remote_code"] is True
    model = AutoDecisionModel.from_pretrained(best, device="cpu", dtype=torch.float32, trust_remote_code=True)
    answers = model.decide(STATE, QUESTIONS)["answers"]
    assert answers["team"]["choice"] in QUESTIONS["team"]["criteria"] and 0 <= answers["urgent"]["noul"] <= 1


def test_nandi_readout_starts_from_its_lm_head():
    """The code readout reproduces Nandi's own next-token logits for the answer codes (factorized embedding)."""
    from transformers import AutoModelForCausalLM

    from decisionformer.model import build_model

    config = load_config({"model": {"backbone": NANDI, "trust_remote_code": True, "head": {"type": "code_readout"}},
                          "train": {"output_dir": "unused", "mixed_precision": "no"}})
    result = build_model(config, torch.device("cpu"), training=False)
    result.model.float()
    lm = AutoModelForCausalLM.from_pretrained(NANDI, trust_remote_code=True, dtype=torch.float32).eval()
    ids = torch.tensor([result.tokenizer.backend_tokenizer.encode("<|im_start|> State:\nhello\n\nAnswer:").ids])
    codes = result.builder.code_token_ids[:5]
    with torch.no_grad():
        hidden = lm.model(input_ids=ids).last_hidden_state[0, -1]
        expected = lm(input_ids=ids).logits[0, -1, codes]
    got = result.model.heads["default"].readout.weight[:5].detach() @ hidden
    torch.testing.assert_close(got, expected, rtol=1e-3, atol=1e-3)


def test_nandi_mini_150m_fine_tunes(data_dir, tmp_path):
    from decisionformer.trainer import train

    config = _config(tmp_path, data_dir, NANDI, out="nandi", finetune={"mode": "full"})
    train(config)
    _check_export(Path(config.train.output_dir) / "best",
                  ("configuration_nandi.py", "modeling_nandi.py", "tokenization_nandi.py"))


def test_lumma_06b_base_fine_tunes_with_lora(data_dir, tmp_path):
    from decisionformer.trainer import train

    config = _config(tmp_path, data_dir, LUMMA_BASE, out="lumma", steps=2,
                     finetune={"mode": "lora", "lora": {"r": 8, "lora_alpha": 16, "target_modules": "all-linear"}})
    train(config)
    _check_export(Path(config.train.output_dir) / "best",
                  ("configuration_lumma.py", "modeling_lumma.py", "tokenization_lumma.py"))


def test_lumma_fev_01b_v2_trains_on(data_dir, tmp_path):
    """Training on from the finished decision model: its readout, prompt and codes carry over."""
    from safetensors.torch import load_file

    from decisionformer.model import build_model, read_decision_config
    from decisionformer.trainer import train

    config = _config(tmp_path, data_dir, LUMMA_FEV_01B, out="fev", finetune={"mode": "full"})
    saved = read_decision_config(LUMMA_FEV_01B)
    result = build_model(config, torch.device("cpu"), training=False)
    assert result.builder.format is AUTOJEV and result.builder.codes == saved["codes"]
    from huggingface_hub import hf_hub_download

    original = load_file(hf_hub_download(LUMMA_FEV_01B, "readout.safetensors"))["weight"].float()
    torch.testing.assert_close(result.model.heads["default"].readout.weight.detach().float(), original)
    train(config)
    _check_export(Path(config.train.output_dir) / "best", ("configuration_nandi.py", "modeling_nandi.py"))


# ------------------------------------------------------------------------------------------------ large checkpoints
@large
@pytest.mark.gpu
@pytest.mark.parametrize("repo", ["FrontiersMind/Lumma-fev-4B-v2", "FrontiersMind/Lumma-fev-27B-v2"])
def test_lumma_fev_v2_qwen_checkpoints_load_and_decide(repo):
    model = AutoDecisionModel.from_pretrained(repo)
    assert model.builder.format is AUTOJEV and model.model.backbone.attention == "bidirectional"
    answers = model.decide(STATE, QUESTIONS)["answers"]
    assert answers["team"]["choice"] == "support"


@large
@pytest.mark.gpu
def test_lumma_fev_4b_v2_trains_on_with_lora(data_dir, tmp_path):
    from decisionformer.trainer import train

    config = _config(tmp_path, data_dir, "FrontiersMind/Lumma-fev-4B-v2", out="fev4b", steps=2, remote=False,
                     precision="bf16",
                     finetune={"mode": "lora", "lora": {"r": 8, "lora_alpha": 16, "target_modules": "all-linear"}})
    train(config)
    best = Path(config.train.output_dir) / "best"
    saved = json.loads((best / "decision_config.json").read_text())
    assert saved["prompt"] == "autojev" and saved["attention_mode"] == "noncausal_full_attention"
    assert AutoDecisionModel.from_pretrained(best).decide(STATE, QUESTIONS)["answers"]["team"]["choice"]
