"""DDP correctness: gradients match a single process exactly (up to float summation order) for any world size."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from conftest import build_decoder, synthetic_records, write_jsonl
from safetensors.torch import load_file
from test_training import make_config

pytestmark = pytest.mark.slow
WORKER = Path(__file__).with_name("ddp_worker.py")


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def torchrun(processes: int, *arguments: str) -> None:
    env = {**os.environ, "DECIFORMER_DEVICE": os.environ.get("DECIFORMER_TEST_DEVICE", "cpu"),
           "OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}
    command = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(processes),
               "--master_addr", "127.0.0.1", "--master_port", str(_port()), *arguments]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise AssertionError(f"torchrun failed:\n{result.stdout[-4000:]}\n{result.stderr[-6000:]}")


def _yaml(config, path: Path) -> Path:
    from deciformer.config import dump_config

    path.write_text(dump_config(config))
    return path


@pytest.mark.parametrize("variant", ["full", "lora", "multitask"])
def test_gradients_match_across_world_sizes(model_root, data_dir, tmp_path, variant):
    head = {"type": "code_readout"}
    extra: dict = {"train": {"micro_batch_size": 5}, "data": {"eval": {}}}  # 3 ranks: shares 6/5/5 -> 2/1/1 micro-batches
    write_jsonl(tmp_path / "varied.jsonl", synthetic_records(64, 5, long_tail=True))
    overrides = ["data.calibration=null", f"data.train={tmp_path / 'varied.jsonl'}"]
    if variant == "lora":
        extra["finetune"] = {"mode": "lora", "lora": {"r": 4, "lora_dropout": 0.0}}
    if variant == "multitask":
        write_jsonl(tmp_path / "mixed.jsonl", synthetic_records(40, 7, long_tail=True)
                    + synthetic_records(40, 8, kind="binary", long_tail=True))
        overrides.append(f"data.train={tmp_path / 'mixed.jsonl'}")
    config = make_config(tmp_path, build_decoder(model_root, "llama"), data_dir, head=head, extra=extra,
                         overrides=overrides)
    if variant == "multitask":
        config.model.heads = {"color": config.model.heads["default"],
                              "is_red": config.model.heads["default"].model_copy(update={"type": "binary"})}
    path = _yaml(config, tmp_path / "config.yaml")
    results = {}
    for processes in (1, 2, 3):
        output = tmp_path / f"grads-{processes}.pt"
        torchrun(processes, str(WORKER), str(path), str(output))
        results[processes] = torch.load(output)
    reference = results[1]
    assert reference["count"] == 16
    assert results[3]["padding"] > 0, "test data must force zero-weight padding micro-batches"
    assert len({r["step"] for r in results.values()}) == 1
    for processes in (2, 3):
        other = results[processes]
        assert other["count"] == 16 and other["loss"] == pytest.approx(reference["loss"], rel=1e-5)
        assert reference["gradients"].keys() == other["gradients"].keys()
        for name, gradient in reference["gradients"].items():
            torch.testing.assert_close(other["gradients"][name], gradient, rtol=1e-4, atol=1e-7, msg=name)


def test_ddp_training_with_sharded_optimizer_and_exact_resume(model_root, data_dir, tmp_path):
    def run(name: str, stops: list[int | None], shard: bool) -> dict[str, torch.Tensor]:
        for stop in stops:
            config = make_config(tmp_path, build_decoder(model_root, "llama"), data_dir, head={"type": "code_readout"},
                                 out=name, steps=6,
                                 extra={"train": {"save_every": 3, "stop_after": stop, "resume": "auto",
                                                  "shard_optimizer_states": shard, "eval_every": 3}})
            torchrun(2, "-m", "deciformer", "train", str(_yaml(config, tmp_path / f"{name}.yaml")))
        output = tmp_path / name
        assert (output / "best").exists() and (output / "summary.json").exists()
        state = load_file(str(output / "last" / "model.safetensors"))
        state.update(load_file(str(output / "last" / "readout.safetensors")))
        return state

    plain = run("plain", [None], shard=False)
    sharded = run("sharded", [None], shard=True)
    resumed = run("resumed", [3, None], shard=True)
    for key, value in plain.items():
        torch.testing.assert_close(sharded[key], value, rtol=1e-5, atol=1e-6, msg=key)
        assert torch.equal(resumed[key], sharded[key]), f"resume changed {key}"
