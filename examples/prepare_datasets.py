"""Convert public datasets into deciformer JSONL: train.jsonl, calibration.jsonl and dev.jsonl.

    python examples/prepare_datasets.py mnli --output data/mnli --train-size 20000
    python examples/prepare_datasets.py boolq --output data/boolq

The labeled validation split is divided into calibration and dev by whole families (same premise / passage
stay together), so the temperature is never fitted on the examples it is evaluated on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from datasets import load_dataset


def _family(text: str) -> str:
    return hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()[:16]


NLI_OPTIONS = {"entailment": "The premise shows the hypothesis is true.",
               "neutral": "The premise neither proves nor disproves the hypothesis.",
               "contradiction": "The premise shows the hypothesis is false."}


def mnli(row: dict[str, Any], split: str, index: int) -> dict[str, Any] | None:
    if row["label"] < 0:
        return None
    return {"id": f"mnli-{split}-{index}", "task": "nli", "family": _family(row["premise"]),
            "state": f"Premise: {row['premise']}\nHypothesis: {row['hypothesis']}",
            "question": {"type": "choice", "instructions": "How does the premise relate to the hypothesis?",
                         "options": NLI_OPTIONS},
            "label": ["entailment", "neutral", "contradiction"][row["label"]]}


def sst2(row: dict[str, Any], split: str, index: int) -> dict[str, Any] | None:
    if row["label"] < 0:
        return None
    return {"id": f"sst2-{split}-{index}", "task": "sentiment", "family": _family(row["sentence"]),
            "state": row["sentence"],
            "question": {"type": "choice", "instructions": "What is the sentiment of this movie review?",
                         "options": {"negative": "The reviewer dislikes the film.",
                                     "positive": "The reviewer likes the film."}},
            "label": ["negative", "positive"][row["label"]]}


def boolq(row: dict[str, Any], split: str, index: int) -> dict[str, Any] | None:
    return {"id": f"boolq-{split}-{index}", "task": "boolq", "family": _family(row["passage"]),
            "state": f"Passage: {row['passage']}\nQuestion: {row['question']}?",
            "question": {"type": "binary", "instructions": "Based on the passage, is the answer to the question yes?",
                         "options": {"true": "Yes", "false": "No"}},
            "label": bool(row["answer"])}


DATASETS: dict[str, tuple[Callable[[], Any], str, str, Callable]] = {
    "mnli": (lambda: load_dataset("nyu-mll/glue", "mnli"), "train", "validation_matched", mnli),
    "sst2": (lambda: load_dataset("nyu-mll/glue", "sst2"), "train", "validation", sst2),
    "boolq": (lambda: load_dataset("google/boolq"), "train", "validation", boolq),
}


def convert(rows: Iterable[dict[str, Any]], split: str, fn: Callable) -> list[dict[str, Any]]:
    return [record for index, row in enumerate(rows) if (record := fn(row, split, index)) is not None]


def write(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    print(f"{path}: {len(records)} records")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=sorted(DATASETS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=None, help="random subset of the training split")
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    loader, train_split, validation_split, fn = DATASETS[args.dataset]
    dataset = loader()
    rng = random.Random(args.seed)
    validation = convert(dataset[validation_split], "validation", fn)
    held_out = {r["family"] for r in validation}
    converted = convert(dataset[train_split], "train", fn)
    # Official splits can share passages/premises; drop those training families so evaluation stays unseen.
    train = [r for r in converted if r["family"] not in held_out]
    if len(train) < len(converted):
        print(f"Dropped {len(converted) - len(train)} training records whose family appears in {validation_split}")
    if args.train_size is not None and args.train_size < len(train):
        train = rng.sample(train, args.train_size)
    families = sorted({r["family"] for r in validation})
    rng.shuffle(families)
    calibration_families = set(families[: int(len(families) * args.calibration_fraction)])
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / "train.jsonl", train)
    write(args.output / "calibration.jsonl", [r for r in validation if r["family"] in calibration_families])
    write(args.output / "dev.jsonl", [r for r in validation if r["family"] not in calibration_families])


if __name__ == "__main__":
    main()
