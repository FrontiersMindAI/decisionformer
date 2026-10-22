"""Split hygiene checks: duplicate ids, cross-split family/state overlap, near-duplicates, and label balance."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from typing import Any

from deciformer.schema import Decision, load_decisions


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class NearDuplicates:
    """Five-word shingles, candidates found through eight minimum-hash anchors, then exact Jaccard >= threshold."""

    def __init__(self, threshold: float = 0.85) -> None:
        self.threshold = threshold
        self.parts: list[tuple[set[str], str]] = []
        self.index: dict[str, list[int]] = defaultdict(list)

    @staticmethod
    def shingles(text: str) -> set[str]:
        words = re.findall(r"\w+", _normalized(text))
        return {" ".join(words[i:i + 5]) for i in range(len(words) - 4)} if len(words) >= 20 else set()

    def add(self, text: str, label: str) -> None:
        values = self.shingles(text)
        if values:
            position = len(self.parts)
            self.parts.append((values, label))
            for anchor in sorted(values, key=_digest)[:8]:
                self.index[anchor].append(position)

    def match(self, text: str) -> str | None:
        values = self.shingles(text)
        if not values:
            return None
        candidates = {i for anchor in sorted(values, key=_digest)[:8] for i in self.index.get(anchor, [])}
        for i in candidates:
            other, label = self.parts[i]
            if len(values & other) / len(values | other) >= self.threshold:
                return label
        return None


def summarize(decisions: list[Decision]) -> dict[str, Any]:
    return {"rows": len(decisions),
            "tasks": dict(Counter(d.task for d in decisions)),
            "question_types": dict(Counter(d.question_type for d in decisions)),
            "options": dict(sorted(Counter(d.num_options for d in decisions).items())),
            "labels": dict(Counter(d.option_keys[d.label_index] for d in decisions if d.label_index is not None)
                           .most_common(20)),
            "unlabeled": sum(d.label_index is None for d in decisions),
            "with_family": sum(d.family is not None for d in decisions)}


def check_splits(splits: dict[str, str], *, near_duplicates: bool = True) -> dict[str, Any]:
    loaded = {name: load_decisions(path) for name, path in splits.items()}
    report: dict[str, Any] = {"splits": {name: summarize(items) for name, items in loaded.items()}, "problems": []}
    seen_ids: dict[str, str] = {}
    seen_families: dict[str, str] = {}
    seen_states: dict[str, str] = {}
    index = NearDuplicates()
    for name, items in loaded.items():
        families = {d.family for d in items if d.family is not None}
        states = {_digest(_normalized(d.state)): d.id for d in items}
        id_clash = sorted({d.id for d in items if d.id in seen_ids})
        family_clash = sorted(f for f in families if f in seen_families)
        state_clash = sorted(states[s] for s in states if s in seen_states)
        near = sorted(d.id for d in items if near_duplicates and index.match(d.state) is not None) if seen_ids else []
        for kind, values in (("id", id_clash), ("family", family_clash), ("identical_state", state_clash),
                             ("near_duplicate_state", near)):
            if values:
                report["problems"].append({"split": name, "overlap": kind, "count": len(values),
                                           "examples": values[:5]})
        for d in items:
            seen_ids.setdefault(d.id, name)
            if near_duplicates:
                index.add(d.state, name)
        for f in families:
            seen_families.setdefault(f, name)
        for s in states:
            seen_states.setdefault(s, name)
    report["ok"] = not report["problems"]
    return report
