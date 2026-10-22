"""Accuracy and calibration metrics, temperature scaling, checkpoint selection, and paired bootstrap comparison."""

from __future__ import annotations

import math
import operator
import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np


def softmax(logits: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64) / temperature
    values = np.exp(values - values.max())
    return values / values.sum()


def compute_metrics(probabilities: Sequence[Sequence[float]], labels: Sequence[int], *, bins: int = 15,
                    soft_targets: Sequence[Sequence[float] | None] | None = None) -> dict[str, Any]:
    """Accuracy, top-label ECE, multiclass Brier (summed over classes), and NLL, averaged over examples."""
    if not probabilities:
        raise ValueError("Cannot score an empty evaluation")
    correct, brier, nll, confidences = [], [], [], []
    soft_nll: list[float] = []
    for index, (row, label) in enumerate(zip(probabilities, labels)):
        p = np.asarray(row, dtype=np.float64)
        predicted = int(p.argmax())
        correct.append(float(predicted == label))
        confidences.append(float(p[predicted]))
        onehot = np.zeros_like(p)
        onehot[label] = 1.0
        brier.append(float(((p - onehot) ** 2).sum()))
        nll.append(-math.log(max(float(p[label]), 1e-12)))
        if soft_targets is not None and soft_targets[index] is not None:
            q = np.asarray(soft_targets[index], dtype=np.float64)
            soft_nll.append(float(-(q * np.log(np.clip(p, 1e-12, None))).sum()))
    confidence_array, correct_array = np.asarray(confidences), np.asarray(correct)
    bucket = np.minimum((confidence_array * bins).astype(int), bins - 1)
    ece = 0.0
    reliability = []
    for b in range(bins):
        members = bucket == b
        count = int(members.sum())
        if count:
            gap = abs(confidence_array[members].mean() - correct_array[members].mean())
            ece += count / len(confidences) * gap
        reliability.append({"lower": b / bins, "upper": (b + 1) / bins, "count": count,
                            "confidence": float(confidence_array[members].mean()) if count else None,
                            "accuracy": float(correct_array[members].mean()) if count else None})
    result: dict[str, Any] = {"count": len(correct), "accuracy": float(correct_array.mean()), "ece": float(ece),
                              "brier": float(np.mean(brier)), "nll": float(np.mean(nll)), "reliability": reliability}
    if soft_nll:
        result["soft_nll"] = float(np.mean(soft_nll))
    return result


def fit_temperature(logits: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    """One scalar temperature minimizing hard-label NLL, found by golden-section search over log T in [0.05, 20]."""
    if not logits:
        raise ValueError("Temperature fitting needs at least one example")
    rows = [np.asarray(row, dtype=np.float64) - max(row) for row in logits]
    targets = [row[label] for row, label in zip(rows, labels)]

    def loss(log_t: float) -> float:
        inverse = math.exp(-log_t)
        return float(np.mean([math.log(np.exp(row * inverse).sum()) - t * inverse for row, t in zip(rows, targets)]))

    low, high = math.log(0.05), math.log(20.0)
    ratio = (math.sqrt(5) - 1) / 2
    left, right = high - ratio * (high - low), low + ratio * (high - low)
    left_loss, right_loss = loss(left), loss(right)
    for _ in range(60):
        if left_loss <= right_loss:
            high, right, right_loss = right, left, left_loss
            left = high - ratio * (high - low)
            left_loss = loss(left)
        else:
            low, left, left_loss = left, right, right_loss
            right = low + ratio * (high - low)
            right_loss = loss(right)
    return math.exp(min([math.log(0.05), (low + high) / 2, math.log(20.0), 0.0], key=loss))


_CONSTRAINT = re.compile(r"^\s*([a-z_]+)\s*(<=|>=|<|>|==)\s*(-?[0-9.eE+-]+)\s*$")
_OPERATORS = {"<=": operator.le, ">=": operator.ge, "<": operator.lt, ">": operator.gt, "==": operator.eq}


def parse_constraint(text: str) -> tuple[str, Any, float]:
    match = _CONSTRAINT.match(text)
    if not match:
        raise ValueError(f"Constraint '{text}' must look like 'ece <= 0.05'")
    return match.group(1), _OPERATORS[match.group(2)], float(match.group(3))


def satisfies(metrics: dict[str, Any], constraints: Sequence[str]) -> bool:
    for text in constraints:
        name, compare, value = parse_constraint(text)
        if name not in metrics:
            raise ValueError(f"Constraint metric '{name}' is not available; have {sorted(metrics)}")
        if not compare(metrics[name], value):
            return False
    return True


def is_improvement(candidate: dict[str, Any], best: dict[str, Any] | None, metric: str, mode: str) -> bool:
    if best is None:
        return True
    return candidate[metric] > best[metric] if mode == "max" else candidate[metric] < best[metric]


def paired_bootstrap(first: Sequence[dict[str, Any]], second: Sequence[dict[str, Any]], *, replicates: int = 2000,
                     seed: int = 0, bins: int = 15) -> dict[str, Any]:
    """Compare two prediction lists on the same ids. Resamples whole families so related examples move together."""
    a = {row["id"]: row for row in first}
    b = {row["id"]: row for row in second}
    if a.keys() != b.keys():
        raise ValueError("Both prediction files must contain exactly the same ids")
    groups: dict[str, list[str]] = defaultdict(list)
    for identifier, row in a.items():
        groups[str(row.get("family") or identifier)].append(identifier)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)

    def score(rows: dict[str, dict[str, Any]], ids: list[str]) -> dict[str, float]:
        m = compute_metrics([rows[i]["probabilities"] for i in ids], [rows[i]["label_index"] for i in ids], bins=bins)
        return {name: m[name] for name in ("accuracy", "ece", "brier", "nll")}

    all_ids = [i for key in keys for i in groups[key]]
    observed_a, observed_b = score(a, all_ids), score(b, all_ids)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(replicates):
        sample = [i for key in rng.choice(keys, size=len(keys), replace=True) for i in groups[key]]
        sa, sb = score(a, sample), score(b, sample)
        for name in sa:
            draws[name].append(sa[name] - sb[name])
    return {"count": len(all_ids), "families": len(keys), "replicates": replicates,
            "metrics": {name: {"first": observed_a[name], "second": observed_b[name],
                               "difference": observed_a[name] - observed_b[name],
                               "ci95": [float(np.quantile(draws[name], 0.025)), float(np.quantile(draws[name], 0.975))]}
                        for name in observed_a}}
