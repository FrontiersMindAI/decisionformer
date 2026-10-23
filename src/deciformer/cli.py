"""deciformer command line: train, eval, predict, serve, push, compare, data check, config (adapt: experimental)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _train(args: argparse.Namespace) -> None:
    from deciformer.config import load_config
    from deciformer.trainer import train

    summary = train(load_config(args.config, args.overrides))
    if summary:
        print(json.dumps(summary, indent=2))


def _adapt(args: argparse.Namespace) -> None:
    from deciformer.adapt import adapt
    from deciformer.config import load_config

    print(adapt(load_config(args.config, args.overrides, validate_components=False)))


def _config(args: argparse.Namespace) -> None:
    from deciformer.config import dump_config, load_config

    print(dump_config(load_config(args.config, args.overrides)))


def _load(args: argparse.Namespace) -> Any:
    from deciformer.api import Decider

    return Decider.from_pretrained(args.model, device=args.device, batch_size=args.batch_size,
                                   max_length=args.max_length, revision=args.revision)


def _write(lines: list[str], path: str | None) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("".join(lines))
    else:
        sys.stdout.write("".join(lines))


def _eval(args: argparse.Namespace) -> None:
    from deciformer.data import Collator, DecisionDataset
    from deciformer.inference import (
        collect_logits,
        fit_type_temperatures,
        grouped_metrics,
        iterate_jsonl,
        prediction_records,
        temperature_for,
    )

    decider = _load(args)
    collator = Collator(decider.builder, decider.tokenizer.pad_token_id)
    temperatures = {"default": decider.temperature, **decider.temperature_by_type}
    if args.calibration:
        calibration = DecisionDataset.from_paths(args.calibration, decider.builder, labeled_only=True)
        temperatures = fit_type_temperatures(calibration, collect_logits(
            decider.model, calibration, collator, range(len(calibration)), device=decider.device,
            micro_batch_size=args.batch_size, token_budget=None))
    dataset = DecisionDataset.from_paths(args.data, decider.builder)
    rows = collect_logits(decider.model, dataset, collator, range(len(dataset)), device=decider.device,
                          micro_batch_size=args.batch_size, token_budget=None)
    records = prediction_records(dataset, rows, lambda d: 1.0 if args.raw else
                                 temperature_for(temperatures, d.question_type))
    if args.predictions:
        _write(list(iterate_jsonl(records)), args.predictions)
    print(json.dumps({"temperatures": temperatures, **grouped_metrics(records, bins=args.ece_bins)}, indent=2))


def _predict(args: argparse.Namespace) -> None:
    from deciformer.schema import read_records

    decider = _load(args)
    lines = []
    for number, record in enumerate(r for path in args.data for r in read_records(path)):
        response = decider.decide(record.get("state"), record["questions"], images=record.get("images"))
        lines.append(json.dumps({"id": record.get("id", number), **response}, ensure_ascii=False) + "\n")
    _write(lines, args.output)


def _serve(args: argparse.Namespace) -> None:
    from deciformer.serve import serve

    serve(args.model, host=args.host, port=args.port, device=args.device, api_key_env=args.api_key_env,
          batch_size=args.batch_size, max_length=args.max_length, revision=args.revision)


def _push(args: argparse.Namespace) -> None:
    from deciformer.hub import push_folder

    print(push_folder(args.checkpoint, args.repo_id, private=not args.public))


def _compare(args: argparse.Namespace) -> None:
    from deciformer.metrics import paired_bootstrap

    def read(path: str) -> list[dict[str, Any]]:
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


def _model_options(command: argparse.ArgumentParser) -> None:
    command.add_argument("model", help="decision checkpoint: a local directory or a Hub repo id")
    command.add_argument("--revision")
    command.add_argument("--device")
    command.add_argument("--batch-size", type=int, default=16)
    command.add_argument("--max-length", type=int, help="override the input limit (default: the model's context)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deciformer", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    for name, handler, help_text in (("train", _train, "Fine-tune a decision model (torchrun for multi-GPU)"),
                                     ("config", _config, "Print the fully resolved configuration"),
                                     ("adapt", _adapt, "Experimental: MNTP adaptation of a decoder")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("config")
        command.add_argument("overrides", nargs="*", help="dotted overrides, e.g. train.lr=1e-5")
        command.set_defaults(handler=handler)

    evaluate = commands.add_parser("eval", help="Score a decision model on labeled requests (JSONL)")
    _model_options(evaluate)
    evaluate.add_argument("data", nargs="+")
    evaluate.add_argument("--calibration", help="refit per-type temperatures on this labeled file first")
    evaluate.add_argument("--predictions", help="write per-question predictions here")
    evaluate.add_argument("--raw", action="store_true", help="ignore temperatures")
    evaluate.add_argument("--ece-bins", type=int, default=15)
    evaluate.set_defaults(handler=_eval)

    predict = commands.add_parser("predict", help="Answer System One requests (JSONL of {state, questions})")
    _model_options(predict)
    predict.add_argument("data", nargs="+")
    predict.add_argument("--output", help="write responses here instead of stdout")
    predict.set_defaults(handler=_predict)

    serve = commands.add_parser("serve", help="Serve a decision model on /v1/systemone")
    _model_options(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--api-key-env", default="DECIFORMER_API_KEY",
                       help="require `Authorization: Bearer <key>` when this variable is set")
    serve.set_defaults(handler=_serve)

    push = commands.add_parser("push", help="Upload a checkpoint directory to the Hugging Face Hub")
    push.add_argument("checkpoint")
    push.add_argument("repo_id")
    push.add_argument("--public", action="store_true", help="create the repo as public (default: private)")
    push.set_defaults(handler=_push)

    compare = commands.add_parser("compare", help="Paired family bootstrap between two prediction files")
    compare.add_argument("first")
    compare.add_argument("second")
    compare.add_argument("--replicates", type=int, default=2000)
    compare.add_argument("--seed", type=int, default=0)
    compare.set_defaults(handler=_compare)

    data = commands.add_parser("data", help="Data utilities")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    check = data_commands.add_parser("check", help="Check splits (and evaluation suites) for leakage")
    check.add_argument("splits", nargs="+", help="NAME=PATH (.jsonl or .jsonl.gz), protected sets first, "
                                                 "e.g. index=suite-rows.jsonl.gz dev=dev.jsonl train=train.jsonl")
    check.add_argument("--no-near-duplicates", action="store_true")
    check.set_defaults(handler=_data_check)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
