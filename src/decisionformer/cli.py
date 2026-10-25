"""decisionformer: fine-tune, evaluate, serve and publish one-pass decision models on Hugging Face transformers.

Start with a bundled recipe:
  decisionformer recipes                                   list them
  decisionformer data split hf:FrontiersMind/deciformer-15k:train --out data/deciformer-15k
  decisionformer train deciformer15k_qwen3                 fine-tune (a recipe name or your own YAML file)
  decisionformer serve runs/deciformer15k-qwen3-0.6b/best  answer requests on /v1/systemone"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _run_config(args: argparse.Namespace, **options: Any) -> Any:
    from decisionformer.config import explain, load_config

    try:
        return load_config(args.config, args.overrides, **options)
    except (FileNotFoundError, ValueError) as error:  # pydantic's ValidationError is a ValueError
        raise SystemExit(f"error: {explain(error)}") from None


def _train(args: argparse.Namespace) -> None:
    from decisionformer.trainer import train

    config = _run_config(args)
    try:
        summary = train(config)
    except FileNotFoundError as error:
        raise SystemExit(f"error: {error}") from None
    if summary:
        print(json.dumps(summary, indent=2))


def _adapt(args: argparse.Namespace) -> None:
    from decisionformer.adapt import adapt

    print(adapt(_run_config(args, validate_components=False)))


def _config(args: argparse.Namespace) -> None:
    from decisionformer.config import dump_config

    print(dump_config(_run_config(args)))


def _recipes(args: argparse.Namespace) -> None:
    from decisionformer.config import list_recipes

    recipes = list_recipes()
    width = max(map(len, recipes))
    for name, description in recipes.items():
        print(f"{name:<{width}}  {description}")
    print("\nRun one with `decisionformer train NAME`, or copy it to edit: `decisionformer init NAME`.")


def _init(args: argparse.Namespace) -> None:
    from decisionformer.config import config_path

    try:
        source = config_path(args.recipe)
    except FileNotFoundError as error:
        raise SystemExit(f"error: {error}") from None
    target = Path(args.path or f"{source.stem}.yaml")
    if target.exists() and not args.force:
        raise SystemExit(f"error: {target} exists (use --force to overwrite)")
    target.write_text(source.read_text())
    print(f"Wrote {target}. Edit it, then: decisionformer train {target}")


def _load(args: argparse.Namespace) -> Any:
    from decisionformer.api import Decider

    return Decider.from_pretrained(args.model, device=args.device, batch_size=args.batch_size,
                                   max_length=args.max_length, revision=args.revision)


def _write(lines: list[str], path: str | None) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("".join(lines))
    else:
        sys.stdout.write("".join(lines))


def _eval(args: argparse.Namespace) -> None:
    from decisionformer.data import Collator, DecisionDataset
    from decisionformer.inference import (
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
    from decisionformer.schema import as_request, read_records

    decider = _load(args)
    records = [r for path in args.data for r in read_records(path)]
    lines: list[str] = []
    for start in range(0, len(records), 64):  # requests in a chunk share forward passes
        chunk = list(enumerate(records[start:start + 64], start))
        requests: list[dict[str, Any] | Exception] = []
        for number, record in chunk:
            try:
                state, questions = as_request(record, number)
                requests.append({"state": state, "questions": questions, "images": record.get("images")})
            except ValueError as error:
                requests.append(error)
        valid = [request for request in requests if not isinstance(request, Exception)]
        try:
            answers = iter(decider.decide_many(valid))
            results: list[dict[str, Any] | Exception] = [r if isinstance(r, Exception) else next(answers)
                                                         for r in requests]
        except ValueError:  # one request does not fit: answer the others and report it alone
            results = [r if isinstance(r, Exception) else _decide_one(decider, r) for r in requests]
        for (number, record), result in zip(chunk, results):
            body = {"error": {"message": str(result)}} if isinstance(result, Exception) else result
            lines.append(json.dumps({"id": record.get("id", number), **body}, ensure_ascii=False) + "\n")
    _write(lines, args.output)
    failed = sum('"error": {"message"' in line for line in lines)
    if failed:
        print(f"{failed} of {len(lines)} requests could not be answered; see their `error` lines", file=sys.stderr)


def _decide_one(decider: Any, request: dict[str, Any]) -> dict[str, Any] | Exception:
    try:
        return decider.decide_many([request])[0]
    except ValueError as error:
        return error


def _serve(args: argparse.Namespace) -> None:
    from decisionformer.serve import serve
    from decisionformer.vllm_backend import UnsupportedBackend

    try:
        serve(args.model, host=args.host, port=args.port, device=args.device, backend=args.backend,
              api_key_env=args.api_key_env, max_batch=args.max_batch, batch_size=args.batch_size,
              max_length=args.max_length, revision=args.revision)
    except UnsupportedBackend as error:
        raise SystemExit(f"error: {error}") from None


def _push(args: argparse.Namespace) -> None:
    from decisionformer.hub import push_folder

    print(push_folder(args.checkpoint, args.repo_id, private=not args.public))


def _export(args: argparse.Namespace) -> None:
    from decisionformer.classifier import export_sequence_classifier

    try:
        output = export_sequence_classifier(args.model, args.output, revision=args.revision)
    except (ValueError, OSError) as error:
        raise SystemExit(f"error: {error}") from None
    print(json.dumps({"output": str(output), "format": args.format, "dtype": "float32"}, indent=2))


def _compare(args: argparse.Namespace) -> None:
    from decisionformer.metrics import paired_bootstrap

    def read(path: str) -> list[dict[str, Any]]:
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

    print(json.dumps(paired_bootstrap(read(args.first), read(args.second), replicates=args.replicates,
                                      seed=args.seed), indent=2))


def _data_check(args: argparse.Namespace) -> None:
    from decisionformer.datacheck import check_splits

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


def _data_split(args: argparse.Namespace) -> None:
    from collections import Counter

    from decisionformer.datacheck import split_records
    from decisionformer.schema import read_records

    holdout = {}
    for item in args.holdout:
        name, _, share = item.partition("=")
        if not share:
            raise SystemExit(f"Use NAME=FRACTION, got {item!r}")
        holdout[name] = float(share)
    records = [record for source in args.sources for record in read_records(source)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, rows in split_records(records, holdout, seed=args.seed).items():
        _write([json.dumps(row, ensure_ascii=False) + "\n" for row in rows], str(out / f"{name}.jsonl"))
        report[name] = {"records": len(rows), "families": len({str(r.get("family", r.get("id"))) for r in rows}),
                        "tasks": dict(sorted(Counter(str(r.get("task", "default")) for r in rows).items()))}
    print(json.dumps(report, indent=2))


def _model_options(command: argparse.ArgumentParser) -> None:
    command.add_argument("model", help="decision checkpoint: a local directory or a Hub repo id")
    command.add_argument("--revision")
    command.add_argument("--device")
    command.add_argument("--batch-size", type=int, default=16)
    command.add_argument("--max-length", type=int, help="override the input limit (default: the model's context)")


def build_parser() -> argparse.ArgumentParser:
    from decisionformer import __version__

    raw = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(prog="decisionformer", description=__doc__, formatter_class=raw)
    parser.add_argument("--version", action="version", version=f"decisionformer {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    examples = """examples:
  decisionformer {name} deciformer15k_qwen3
  decisionformer {name} my_config.yaml train.lr=1e-4 train.max_steps=200
  torchrun --nproc-per-node 8 -m decisionformer {name} my_config.yaml        (multi-GPU training)

`decisionformer recipes` lists the bundled recipes; `decisionformer init NAME` copies one to edit."""
    for name, handler, help_text in (("train", _train, "Fine-tune a decision model (torchrun for multi-GPU)"),
                                     ("config", _config, "Print the fully resolved configuration"),
                                     ("adapt", _adapt, "Experimental: MNTP adaptation of a decoder")):
        command = commands.add_parser(name, help=help_text, description=help_text, formatter_class=raw,
                                      epilog=examples.format(name=name))
        command.add_argument("config", help="a YAML config file or a bundled recipe name")
        command.add_argument("overrides", nargs="*", help="dotted overrides, e.g. train.lr=1e-5")
        command.set_defaults(handler=handler)

    recipes = commands.add_parser("recipes", help="List the bundled training recipes")
    recipes.set_defaults(handler=_recipes)
    init = commands.add_parser("init", help="Copy a bundled recipe to a YAML file you can edit")
    init.add_argument("recipe", help="recipe name (see `decisionformer recipes`)")
    init.add_argument("path", nargs="?", help="where to write it (default: RECIPE.yaml)")
    init.add_argument("--force", action="store_true", help="overwrite an existing file")
    init.set_defaults(handler=_init)

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
    serve.add_argument("--backend", choices=("transformers", "vllm"), default="transformers",
                       help="vllm: native Qwen3 classifier exports or Qwen3.5/3.8 code-readout checkpoints")
    serve.add_argument("--max-batch", type=int, default=16,
                       help="requests arriving together are answered in one batch of up to this many")
    serve.add_argument("--api-key-env", default="DECISIONFORMER_API_KEY",
                       help="require `Authorization: Bearer <key>` when this variable is set")
    serve.set_defaults(handler=_serve)

    push = commands.add_parser("push", help="Upload a checkpoint directory to the Hugging Face Hub")
    push.add_argument("checkpoint")
    push.add_argument("repo_id")
    push.add_argument("--public", action="store_true", help="create the repo as public (default: private)")
    push.set_defaults(handler=_push)

    export = commands.add_parser("export", help="Export a merged Qwen3 code-readout model as a native HF classifier")
    export.add_argument("model", help="decision checkpoint: local directory or Hub repo id")
    export.add_argument("output", help="new export directory (existing paths are refused)")
    export.add_argument("--revision")
    export.add_argument("--format", choices=("sequence-classification",), required=True,
                        help="explicit output layout; sequence-classification exports CPU float32 weights")
    export.set_defaults(handler=_export)

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
    split = data_commands.add_parser("split", help="Split records into train and held-out sets by whole families")
    split.add_argument("sources", nargs="+", help=".jsonl, .jsonl.gz or hf:<name>[:<config>]:<split>")
    split.add_argument("--out", required=True, help="directory for train.jsonl and one file per held-out set")
    split.add_argument("--holdout", nargs="+", default=["calibration=0.1", "dev=0.1"], metavar="NAME=FRACTION")
    split.add_argument("--seed", type=int, default=0)
    split.set_defaults(handler=_data_split)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
