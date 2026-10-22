"""Command line: train, eval, predict, adapt, compare, data check, config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _train(args: argparse.Namespace) -> None:
    from deciformer.config import load_config
    from deciformer.trainer import train

    summary = train(load_config(args.config, args.overrides))
    if summary:
        print(json.dumps(summary, indent=2))


def _adapt(args: argparse.Namespace) -> None:
    from deciformer.adapt import adapt
    from deciformer.config import load_config

    config = load_config(args.config, args.overrides, validate_components=False)
    print(adapt(config))


def _config(args: argparse.Namespace) -> None:
    from deciformer.config import dump_config, load_config

    print(dump_config(load_config(args.config, args.overrides)))


def _load(args: argparse.Namespace):
    import torch

    from deciformer.model import load_decision_model

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return load_decision_model(args.model, device=device), torch.device(device)


def _infer(args: argparse.Namespace, paths: list[str], calibration: str | None = None):
    from deciformer.data import Collator, DecisionDataset
    from deciformer.inference import collect_logits, fit_task_temperatures

    loaded, device = _load(args)
    collator = Collator(loaded.builder, loaded.tokenizer.pad_token_id)
    temperatures = dict(loaded.temperatures)
    if calibration:
        calib = DecisionDataset.from_paths(calibration, loaded.builder)
        temperatures = fit_task_temperatures(calib, collect_logits(
            loaded.model, calib, collator, range(len(calib)), device=device,
            micro_batch_size=args.batch_size, token_budget=None))
    dataset = DecisionDataset.from_paths(paths, loaded.builder)
    rows = collect_logits(loaded.model, dataset, collator, range(len(dataset)), device=device,
                          micro_batch_size=args.batch_size, token_budget=None)

    def temperature_for(task: str) -> float:
        if args.raw:
            return 1.0
        return temperatures.get(task, temperatures.get("default", 1.0))

    return dataset, rows, temperature_for


def _write(records, path: str | None) -> None:
    from deciformer.inference import iterate_jsonl

    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("".join(iterate_jsonl(records)))


def _eval(args: argparse.Namespace) -> None:
    from deciformer.inference import grouped_metrics, prediction_records

    dataset, rows, temperature_for = _infer(args, args.data, args.calibration)
    records = prediction_records(dataset, rows, temperature_for)
    _write(records, args.predictions)
    print(json.dumps(grouped_metrics(records, bins=args.ece_bins), indent=2))


def _predict(args: argparse.Namespace) -> None:
    from deciformer.inference import prediction_records

    dataset, rows, temperature_for = _infer(args, args.data)
    records = prediction_records(dataset, rows, temperature_for)
    for record in records:
        record.pop("logits")
    _write(records, args.output)
    if not args.output:
        for record in records:
            print(json.dumps(record, ensure_ascii=False))


def _compare(args: argparse.Namespace) -> None:
    from deciformer.metrics import paired_bootstrap

    def read(path: str) -> list[dict]:
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

    print(json.dumps(paired_bootstrap(read(args.first), read(args.second), replicates=args.replicates,
                                      seed=args.seed), indent=2))


def _data_check(args: argparse.Namespace) -> None:
    from deciformer.datacheck import check_splits

    splits = {}
    for item in args.splits:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"Use NAME=PATH, got {item!r}")
        splits[name] = path
    report = check_splits(splits, near_duplicates=not args.no_near_duplicates)
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deciformer", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    for name, handler, help_text in (("train", _train, "Fine-tune a decision model (use torchrun for multi-GPU)"),
                                     ("adapt", _adapt, "MNTP adaptation of a decoder to bidirectional attention"),
                                     ("config", _config, "Print the fully resolved configuration")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("config")
        command.add_argument("overrides", nargs="*", help="dotted overrides, e.g. train.lr=1e-5")
        command.set_defaults(handler=handler)

    for name, handler, help_text in (("eval", _eval, "Score an exported model on labeled data"),
                                     ("predict", _predict, "Predict option probabilities for new records")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("model", help="exported model directory (e.g. runs/x/best)")
        command.add_argument("data", nargs="+", help="JSONL files")
        command.add_argument("--device")
        command.add_argument("--batch-size", type=int, default=16)
        command.add_argument("--raw", action="store_true", help="ignore the fitted temperature")
        if name == "eval":
            command.add_argument("--calibration", help="refit the temperature on this labeled file")
            command.add_argument("--predictions", help="write per-example predictions here")
            command.add_argument("--ece-bins", type=int, default=15)
        else:
            command.add_argument("--output", help="write predictions here instead of stdout")
        command.set_defaults(handler=handler)

    compare = commands.add_parser("compare", help="Paired family bootstrap between two prediction files")
    compare.add_argument("first")
    compare.add_argument("second")
    compare.add_argument("--replicates", type=int, default=2000)
    compare.add_argument("--seed", type=int, default=0)
    compare.set_defaults(handler=_compare)

    data = commands.add_parser("data", help="Data utilities")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    check = data_commands.add_parser("check", help="Check splits for leakage and summarize them")
    check.add_argument("splits", nargs="+", help="NAME=PATH, in priority order (e.g. test=... dev=... train=...)")
    check.add_argument("--no-near-duplicates", action="store_true")
    check.set_defaults(handler=_data_check)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
