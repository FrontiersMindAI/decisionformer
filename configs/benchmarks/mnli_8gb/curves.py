import json
import sys
from pathlib import Path

for run in sys.argv[1:]:
    path = Path(f"runs/{run}/metrics.jsonl")
    if not path.exists():
        print(run, "no metrics")
        continue
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    evals = []
    for e in lines:
        if e["event"] == "eval":
            dev = e["sets"]["dev"]
            evals.append((e["step"], round(dev["raw"]["overall"]["accuracy"], 3),
                          round(dev.get("calibrated", dev["raw"])["overall"]["ece"], 4)))
    losses = [(e["step"], round(e["loss"], 3)) for e in lines if e["event"] == "train"]
    print(f"{run:28} evals (step, acc, ece_cal): {evals}")
    print(f"{'':28} loss: {losses}")
