"""Batched logits, per-type temperature fitting, prediction records and grouped metrics (training and `eval`)."""

from __future__ import annotations

import contextlib
import json
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
import torch

from deciformer.data import Collator, DecisionDataset, eval_batches
from deciformer.metrics import compute_metrics, fit_temperature, softmax
from deciformer.schema import QUESTION_TYPES, Decision


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


def temperature_for(temperatures: dict[str, float], question_type: str) -> float:
    return temperatures.get(question_type, temperatures.get("default", 1.0))


def fit_type_temperatures(dataset: DecisionDataset, rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    """One temperature per question type (choice, noul, score) plus 'default' over all calibration questions."""
    labeled = [r for r in rows if dataset.decisions[r["index"]].label_index is not None]
    if not labeled:
        return {}
    labels = [dataset.decisions[r["index"]].label_index for r in labeled]
    temperatures = {"default": fit_temperature([r["logits"] for r in labeled], labels)}
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in labeled:
        by_type[dataset.decisions[row["index"]].question_type].append(row)
    for kind, group in by_type.items():
        temperatures[kind] = fit_temperature([r["logits"] for r in group],
                                             [dataset.decisions[r["index"]].label_index for r in group])
    return temperatures


def prediction_records(dataset: DecisionDataset, rows: Sequence[dict[str, Any]],
                       temperature: Callable[[Decision], float]) -> list[dict[str, Any]]:
    records = []
    for row in sorted(rows, key=lambda r: r["index"]):
        decision = dataset.decisions[row["index"]]
        t = temperature(decision)
        probabilities = softmax(row["logits"], t)
        best = int(np.argmax(probabilities))
        record = {"id": decision.id, "request_id": decision.request_id, "question": decision.question.key,
                  "task": decision.task, "family": decision.family, "question_type": decision.question_type,
                  "options": decision.option_keys, "probabilities": probabilities.tolist(),
                  "prediction": decision.option_keys[best], "confidence": float(probabilities[best]),
                  "temperature": t, "logits": row["logits"]}
        if decision.label_index is not None:
            record.update(label=decision.option_keys[decision.label_index], label_index=decision.label_index,
                          correct=best == decision.label_index)
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

    groups: dict[str, dict[str, list[dict[str, Any]]]] = {"tasks": defaultdict(list), "types": defaultdict(list)}
    for record in labeled:
        groups["tasks"][record["task"]].append(record)
        groups["types"][record["question_type"]].append(record)
    overall = score(labeled)
    tasks = {name: score(items) for name, items in sorted(groups["tasks"].items())}
    types = {name: score(items) for name, items in sorted(groups["types"].items(), key=lambda kv: (
        QUESTION_TYPES.index(kv[0]) if kv[0] in QUESTION_TYPES else 99))}
    if len(tasks) > 1:
        for name in ("accuracy", "ece", "brier", "nll"):
            overall[f"macro_{name}"] = float(np.mean([t[name] for t in tasks.values()]))
    return {"overall": overall, "tasks": tasks, "types": types}


def shard(indices: Sequence[int], rank: int, world_size: int) -> list[int]:
    return list(indices[rank::world_size])


def iterate_jsonl(records: Sequence[dict[str, Any]]) -> Iterator[str]:
    for record in records:
        yield json.dumps(record, ensure_ascii=False) + "\n"
