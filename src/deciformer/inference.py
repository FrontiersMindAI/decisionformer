"""Batched inference, prediction records, and grouped metrics shared by training, `eval`, and `predict`."""

from __future__ import annotations

import contextlib
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
import torch

from deciformer.data import Collator, DecisionDataset, eval_batches
from deciformer.metrics import compute_metrics, fit_temperature, softmax


def autocast(device: torch.device, mixed_precision: str) -> contextlib.AbstractContextManager[Any]:
    if mixed_precision == "no":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


@torch.inference_mode()
def collect_logits(model: torch.nn.Module, dataset: DecisionDataset, collator: Collator, indices: Sequence[int], *,
                   device: torch.device, micro_batch_size: int, token_budget: int | None,
                   mixed_precision: str = "no") -> list[dict[str, Any]]:
    was_training = model.training
    model.eval()
    rows: list[dict[str, Any]] = []
    try:
        for batch_indices in eval_batches(indices, dataset.lengths, micro_batch_size, token_budget):
            batch = collator(dataset, batch_indices).to(device)
            with autocast(device, mixed_precision):
                logits = model(batch)
            for row in range(len(batch)):
                count = int(batch.num_options[row])
                rows.append({"index": batch.indices[row], "logits": logits[row, :count].float().cpu().tolist()})
    finally:
        model.train(was_training)
    return rows


def fit_task_temperatures(dataset: DecisionDataset, rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Fit one temperature per task, plus 'default' over all calibration examples."""
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    labeled = [r for r in rows if dataset.decisions[r["index"]].label_index is not None]
    for row in labeled:
        by_task[dataset.decisions[row["index"]].task].append(row)
    if not labeled:
        return {}
    temperatures = {"default": fit_temperature([r["logits"] for r in labeled],
                                               [dataset.decisions[r["index"]].label_index for r in labeled])}
    for task, task_rows in by_task.items():
        temperatures[task] = fit_temperature([r["logits"] for r in task_rows],
                                             [dataset.decisions[r["index"]].label_index for r in task_rows])
    return temperatures


def prediction_records(dataset: DecisionDataset, rows: Sequence[dict[str, Any]],
                       temperature_for: Callable[[str], float]) -> list[dict[str, Any]]:
    records = []
    for row in sorted(rows, key=lambda r: r["index"]):
        decision = dataset.decisions[row["index"]]
        temperature = temperature_for(decision.task)
        probabilities = softmax(row["logits"], temperature)
        best = int(np.argmax(probabilities))
        record = {"id": decision.id, "task": decision.task, "family": decision.family,
                  "question_type": decision.question_type, "options": decision.option_keys,
                  "probabilities": probabilities.tolist(), "prediction": decision.option_keys[best],
                  "confidence": float(probabilities[best]), "temperature": temperature, "logits": row["logits"]}
        if decision.label_index is not None:
            record["label"] = decision.option_keys[decision.label_index]
            record["label_index"] = decision.label_index
            record["correct"] = best == decision.label_index
        if decision.target is not None:
            record["soft_target"] = decision.target
        records.append(record)
    return records


def grouped_metrics(records: Sequence[dict[str, Any]], *, bins: int) -> dict[str, Any]:
    labeled = [r for r in records if "label_index" in r]
    if not labeled:
        return {}

    def score(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
        result = compute_metrics([r["probabilities"] for r in items], [r["label_index"] for r in items], bins=bins,
                                 soft_targets=[r.get("soft_target") for r in items])
        result.pop("reliability")
        return result

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in labeled:
        by_task[record["task"]].append(record)
    overall = score(labeled)
    tasks = {task: score(items) for task, items in sorted(by_task.items())}
    if len(tasks) > 1:
        for name in ("accuracy", "ece", "brier", "nll"):
            overall[f"macro_{name}"] = float(np.mean([t[name] for t in tasks.values()]))
    return {"overall": overall, "tasks": tasks}


def shard(indices: Sequence[int], rank: int, world_size: int) -> list[int]:
    return list(indices[rank::world_size])


def iterate_jsonl(records: Sequence[dict[str, Any]]) -> Iterator[str]:
    import json

    for record in records:
        yield json.dumps(record, ensure_ascii=False) + "\n"
