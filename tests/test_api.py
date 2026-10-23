"""The transformers-like surface: load any decision checkpoint, decide, serve, and run under the Decision Index."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from conftest import COLORS, SIZES, build_decoder, make_tokenizer
from safetensors.torch import save_file
from test_training import make_config

from decisionformer import AutoDecisionModel, ContextLengthError, UnsupportedInput
from decisionformer.formats import AUTOJEV, D3
from decisionformer.prompting import answer_codes, prepare_tokenizer
from decisionformer.trainer import train

pytestmark = pytest.mark.slow

QUESTIONS = {
    "color": {"type": "choice", "instructions": "What is the color ?", "criteria": dict.fromkeys(COLORS)},
    "red": {"type": "noul", "instructions": "Is it red ?"},
    "size": {"type": "score", "instructions": "How big ?", "criteria": SIZES},
}


def index_validate(questions: dict, response: dict) -> None:
    """The Decision Index kit's response check (decision_index/engines/base.py, MIT); score answers are checked
    elsewhere because the Index text suite has none."""
    assert set(response["answers"]) == set(questions)
    for key, question in questions.items():
        answer = response["answers"][key]
        assert answer["type"] == question["type"]
        if question["type"] == "choice":
            assert answer["choice"] in question["criteria"]
            probabilities = answer["probabilities"]
            assert set(probabilities) == set(question["criteria"])
            assert all(math.isfinite(v) and 0 <= v <= 1 for v in probabilities.values())
            assert abs(sum(probabilities.values()) - 1) <= 0.01
        elif question["type"] == "noul":
            assert math.isfinite(answer["noul"]) and 0 <= answer["noul"] <= 1


@pytest.fixture(scope="module")
def trained(model_root, data_dir, tmp_path_factory) -> Path:
    """A multi-question model (choice, noul, score) trained with LoRA and exported in the standard layout."""
    tmp = tmp_path_factory.mktemp("trained")
    config = make_config(tmp, build_decoder(model_root, "llama"), data_dir, head={"type": "code_readout"}, steps=6,
                         extra={"finetune": {"mode": "lora", "lora": {"r": 4, "lora_dropout": 0.0}}},
                         overrides=[f"data.train={data_dir / 'multi_train.jsonl'}",
                                    f"data.eval.dev={data_dir / 'multi_dev.jsonl'}",
                                    f"data.calibration={data_dir / 'multi_calibration.jsonl'}"])
    train(config)
    return Path(config.train.output_dir)


def test_standard_export_loads_and_decides(trained):
    best = trained / "best"
    config = json.loads((best / "decision_config.json").read_text())
    assert config["format_id"] == "d3-code-readout-v1" and config["prompt"] == "d3"
    assert config["attention_mode"] == "causal" and config["pooling"] == "last" and len(config["token_ids"]) == 255
    assert set(config["temperature_by_type"]) == {"choice", "noul", "score"}
    assert (best / "readout.safetensors").exists() and (best / "model.safetensors").exists()
    card = (best / "README.md").read_text()
    assert card.startswith("---\nlibrary_name: decisionformer"), "a local base model is not a Hub id: no base_model"
    model = AutoDecisionModel.from_pretrained(best, device="cpu", dtype=torch.float32)
    response = model.decide("the large circle is red .", QUESTIONS)
    index_validate(QUESTIONS, response)
    size = response["answers"]["size"]
    assert size["type"] == "score" and 0 <= size["score"] <= 4 and set(size["legend"]) == {"0", "1", "2", "3", "4"}
    assert response["usage"]["input_tokens"] > 0 and response["usage"]["output_tokens"] == 0


def test_loaded_model_reproduces_training_predictions(trained):
    """The exported model, reloaded, gives the calibrated probabilities the trainer recorded at the best step."""
    from conftest import multi_records

    step = json.loads((trained / "best" / "decision_config.json").read_text())["decisionformer"]["metrics"]["step"]
    recorded: dict[str, dict] = {}
    for line in (trained / "predictions" / f"dev-step{step:06d}.jsonl").read_text().splitlines():
        row = json.loads(line)
        recorded.setdefault(row["request_id"], {})[row["question"]] = row["probabilities"]
    model = AutoDecisionModel.from_pretrained(trained / "best", device="cpu", dtype=torch.float32)
    checked = 0
    for record in multi_records(16, 6)[:4]:  # the dev file the trainer scored (see conftest.data_dir)
        for key, answer in model.decide(record["state"], record["questions"])["answers"].items():
            got = list(answer["probabilities"].values()) if "probabilities" in answer else [1 - answer["noul"],
                                                                                            answer["noul"]]
            assert got == pytest.approx(recorded[record["id"]][key], abs=1e-5)
            checked += 1
    assert checked == 12


def _bare_checkpoint(root: Path, model_type: str, *, prompt: str | None, attention_mode: str | None,
                     temperature: float = 0.9) -> tuple[Path, torch.Tensor, list[int]]:
    """A checkpoint as AutoJev or Decision 3.0 write it: bare backbone, decision_config.json, readout.safetensors."""
    from transformers import AutoModel

    path = root / f"foreign-{model_type}-{prompt}"
    backbone = AutoModel.from_pretrained(str(build_decoder(root, model_type)))
    backbone.save_pretrained(str(path))
    tokenizer = make_tokenizer("decoder")
    tokenizer.save_pretrained(str(path))
    prepare_tokenizer(tokenizer)
    codes, ids = answer_codes(tokenizer, "chat", {"enable_thinking": False})
    hidden = backbone.config.get_text_config().hidden_size
    readout = torch.randn(255, hidden, generator=torch.Generator().manual_seed(3))
    save_file({"weight": readout}, str(path / "readout.safetensors"))
    config = {"format_version": 1, "base_model": "tiny", "revision": "0" * 40, "codes": codes, "token_ids": ids,
              "temperature": temperature}
    if prompt:
        config["prompt"] = prompt
    if attention_mode:
        config.update(attention_mode=attention_mode, pooling="last")
    (path / "decision_config.json").write_text(json.dumps(config))
    return path, readout, codes


def _manual_probabilities(path: Path, fmt, state: str, question: dict, readout: torch.Tensor, codes: list[str],
                          temperature: float, noncausal: bool) -> list[float]:
    from transformers import AutoModel, AutoTokenizer

    from decisionformer.schema import parse_question

    tokenizer = AutoTokenizer.from_pretrained(str(path))  # the checkpoint's tokenizer, as transformers loads it
    parsed = parse_question("q", question)
    user, _ = fmt.user_text(state, parsed, codes, list(range(parsed.num_options)))
    text = tokenizer.apply_chat_template([{"role": "system", "content": fmt.system}, {"role": "user", "content": user}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = torch.tensor([tokenizer(text, add_special_tokens=False)["input_ids"]])
    model = AutoModel.from_pretrained(str(path)).eval()
    if noncausal:  # what Decision 3.0 does: full-attention layers see the whole prompt
        model.config.get_text_config().is_causal = False
        for module in model.language_model.modules():
            if isinstance(getattr(module, "is_causal", None), bool):
                module.is_causal = False
    with torch.no_grad():
        hidden = model(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state[0, -1]
    logits = readout[:parsed.num_options] @ hidden / temperature
    return logits.softmax(-1).tolist()


@pytest.mark.parametrize("model_type, prompt, attention_mode, fmt", [
    ("llama", None, None, AUTOJEV),  # an AutoJev export (Lumma-style): no prompt field, causal
    ("qwen3_5", "d3", "noncausal_full_attention", D3),  # a Decision 3.0 export on a hybrid Qwen3.5 backbone
])
def test_foreign_checkpoints_match_a_manual_forward_pass(model_root, model_type, prompt, attention_mode, fmt):
    path, readout, codes = _bare_checkpoint(model_root, model_type, prompt=prompt, attention_mode=attention_mode)
    model = AutoDecisionModel.from_pretrained(path, device="cpu", dtype=torch.float32)
    assert model.builder.format is fmt
    state, question = "the tiny square is green .", QUESTIONS["color"]
    got = model.decide(state, {"q": question})["answers"]["q"]["probabilities"]
    expected = _manual_probabilities(path, fmt, state, question, readout, codes, 0.9,
                                     noncausal=attention_mode == "noncausal_full_attention")
    assert list(got.values()) == pytest.approx(expected, abs=1e-5)


def test_refusals_never_truncate(trained):
    model = AutoDecisionModel.from_pretrained(trained / "best", device="cpu", dtype=torch.float32, max_length=96)
    with pytest.raises(ContextLengthError, match="maximum context length"):
        model.decide(" ".join(["red"] * 200), {"q": {"type": "noul"}})
    with pytest.raises(ContextLengthError, match="options per choice"):
        model.decide("x", {"q": {"type": "choice", "criteria": {f"o{i}": None for i in range(256)}}})
    with pytest.raises(UnsupportedInput):
        model.decide("x", {"q": {"type": "noul"}}, images=["data:image/png;base64,AAAA"])


def test_systemone_server(trained):
    from fastapi.testclient import TestClient

    from decisionformer.serve import create_app

    model = AutoDecisionModel.from_pretrained(trained / "best", device="cpu", dtype=torch.float32, max_length=200)
    client = TestClient(create_app(model))
    assert client.get("/health").json()["prompt_format"] == "d3"
    body = {"model": "x", "state": "the huge triangle is yellow .", "questions": QUESTIONS}
    response = client.post("/v1/systemone", json=body)
    assert response.status_code == 200 and response.json()["object"] == "structured_decision"
    index_validate(QUESTIONS, response.json())
    too_long = client.post("/v1/systemone", json={"state": " ".join(["red"] * 300), "questions": {"q": {"type": "noul"}}})
    assert too_long.status_code == 422 and "maximum context length" in too_long.text
    bad = client.post("/v1/systemone", json={"state": "x", "questions": {"q": {"type": "score", "criteria": ["a"]}}})
    assert bad.status_code == 400 and "a score takes 2 to 10 levels" in bad.text
    locked = TestClient(create_app(model, api_key="secret"))
    assert locked.post("/v1/systemone", json=body).status_code == 401
    assert locked.post("/v1/systemone", json=body, headers={"Authorization": "Bearer secret"}).status_code == 200


def test_decision_index_engine(trained):
    from decisionformer.integrations.decision_index import DecisionFormerEngine, Unsupported

    engine = DecisionFormerEngine(model=str(trained / "best"), device="cpu", dtype="float32", max_length="96")
    choice_and_noul = {k: v for k, v in QUESTIONS.items() if v["type"] != "score"}
    response, raw = engine("the medium circle is blue .", choice_and_noul)
    index_validate(choice_and_noul, response)
    assert raw is None and engine.provenance["prompt_format"] == "d3"
    with pytest.raises(Unsupported, match="maximum context length"):
        engine(" ".join(["red"] * 200), {"q": {"type": "noul"}})


def test_predict_and_eval_commands(trained, data_dir, tmp_path, capsys):
    from decisionformer.cli import main

    requests = tmp_path / "requests.jsonl"
    requests.write_text(json.dumps({"id": "r1", "state": "the small circle is red .", "questions": QUESTIONS}) + "\n")
    main(["predict", str(trained / "best"), str(requests), "--output", str(tmp_path / "out.jsonl"), "--device", "cpu"])
    out = json.loads((tmp_path / "out.jsonl").read_text())
    assert out["id"] == "r1" and set(out["answers"]) == set(QUESTIONS)
    main(["eval", str(trained / "best"), str(data_dir / "multi_dev.jsonl"), "--device", "cpu"])
    report = json.loads(capsys.readouterr().out)
    assert report["overall"]["count"] == 48 and set(report["types"]) == {"choice", "noul", "score"}


def test_push_fills_the_repo_id_into_the_model_card(trained, monkeypatch):
    import huggingface_hub

    from decisionformer.hub import push_folder

    calls: dict = {}

    class FakeApi:
        def __init__(self, token=None):
            pass

        def create_repo(self, repo_id, private, exist_ok):
            calls["private"] = private

        def upload_folder(self, repo_id, folder_path, commit_message, ignore_patterns):
            calls["ignored"] = ignore_patterns

        def create_commit(self, repo_id, commit_message, operations):
            calls["card"] = operations[0].path_or_fileobj.decode()

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    assert push_folder(trained / "best", "org/decider") == "https://huggingface.co/org/decider"
    assert calls["private"] is True and "README.md" in calls["ignored"]
    assert 'AutoDecisionModel.from_pretrained("org/decider")' in calls["card"] and "<this repo>" not in calls["card"]
