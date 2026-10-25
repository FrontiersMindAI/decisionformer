"""Compare a source decision checkpoint with its native classifier export.

The reference runs on CPU in FP32. The candidate uses DecisionFormer's
Transformers or actual vLLM adapter. This checks behavior, not model accuracy
or serving performance. Run from an environment with decisionformer installed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any

import torch

from decisionformer import AutoDecisionModel
from decisionformer.schema import Decision, parse_questions

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def read_requests(path: Path) -> list[dict[str, Any]]:
    requests = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or "state" not in row or "questions" not in row:
            raise ValueError(f"{path}:{line_number}: each request needs state and questions")
        parse_questions(row["questions"])
        requests.append(row)
    if not requests:
        raise ValueError(f"{path}: no requests")
    return requests


def flatten(requests: list[dict[str, Any]]) -> list[Decision]:
    return [Decision(id=question.key, request_id=str(i), state=request["state"], question=question,
                     task=question.task or "default")
            for i, request in enumerate(requests) for question in parse_questions(request["questions"])]


def absolute_error(a: float, b: float) -> float:
    return abs(a-b) if math.isfinite(a) and math.isfinite(b) else math.inf


def answer_errors(expected: Any, actual: Any) -> tuple[float, int]:
    """Compare answer schema/values recursively, including legends and confidence."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            return 0.0, 1
        comparisons = [answer_errors(value, actual[key]) for key, value in expected.items()]
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return 0.0, 1
        comparisons = [answer_errors(a, b) for a, b in zip(expected, actual)]
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            return 0.0, 1
        return absolute_error(float(expected), float(actual)), 0
    else:
        return 0.0, int(expected != actual)
    return max((error for error, _ in comparisons), default=0.0), sum(n for _, n in comparisons)


def probability_error(expected: dict[str, Any], actual: dict[str, Any]) -> float:
    if "probabilities" in expected:
        a, b = expected["probabilities"], actual.get("probabilities")
        if not isinstance(b, dict) or a.keys() != b.keys():
            return math.inf
        return max((absolute_error(float(value), float(b[key])) for key, value in a.items()), default=0.0)
    if "noul" in expected and "noul" in actual:
        return absolute_error(float(expected["noul"]), float(actual["noul"]))
    return math.inf


def package_versions(backend: str) -> dict[str, str]:
    names = ["decisionformer", "torch", "transformers"] + (["vllm"] if backend == "vllm" else [])
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not recorded"
    return result


@torch.inference_mode()
def compare(args: argparse.Namespace) -> dict[str, Any]:
    requests = read_requests(args.requests)
    reference = AutoDecisionModel.from_pretrained(args.source, device="cpu", dtype=torch.float32)
    if args.backend == "transformers":
        if args.engine_options:
            raise ValueError("--engine-options only applies to --backend vllm")
        candidate = AutoDecisionModel.from_pretrained(args.exported, device=args.device, dtype=DTYPES[args.dtype])
    else:
        from decisionformer.vllm_backend import VLLMDecider

        options = json.loads(args.engine_options or "{}")
        if not isinstance(options, dict):
            raise ValueError("--engine-options must be a JSON object")
        reserved = {"model", "name_or_path", "dtype"} & options.keys()
        if reserved:
            raise ValueError(f"--engine-options cannot override {sorted(reserved)}")
        candidate = VLLMDecider.from_pretrained(args.exported, dtype=args.dtype, **options)

    decisions = flatten(requests)
    reference_encoded = [reference.builder.encode(d, max_length=reference.max_input_tokens) for d in decisions]
    candidate_encoded = [candidate.builder.encode(d, max_length=candidate.max_input_tokens) for d in decisions]
    prompt_mismatches = sum(a.input_ids != b.input_ids for a, b in zip(reference_encoded, candidate_encoded))
    reference_logits = reference.code_logits(decisions, reference_encoded)
    candidate_logits = candidate.code_logits(decisions, candidate_encoded)
    if len(reference_logits) != len(decisions) or len(candidate_logits) != len(decisions):
        raise ValueError("A backend returned the wrong number of raw-logit rows")
    max_logit_error = 0.0
    for decision, a, b in zip(decisions, reference_logits, candidate_logits):
        if len(a) < decision.num_options or len(b) < decision.num_options:
            raise ValueError(f"question {decision.question.key!r}: a backend omitted active option logits")
        values_a = a[:decision.num_options].float().cpu().tolist()
        values_b = b[:decision.num_options].float().cpu().tolist()
        max_logit_error = max(max_logit_error, max(absolute_error(x, y) for x, y in zip(values_a, values_b)))

    reference_responses = reference.decide_many(requests)
    candidate_responses = candidate.decide_many(requests)
    if len(reference_responses) != len(requests) or len(candidate_responses) != len(requests):
        raise ValueError("A backend returned the wrong number of responses")
    max_probability_error, max_answer_error = 0.0, 0.0
    answer_mismatches, choice_mismatches = 0, 0
    responses = []
    for i, (request, expected, actual) in enumerate(zip(requests, reference_responses, candidate_responses)):
        expected_answers, actual_answers = expected["answers"], actual["answers"]
        error, mismatches = answer_errors(expected_answers, actual_answers)
        max_answer_error = max(max_answer_error, error)
        answer_mismatches += mismatches
        for key, answer in expected_answers.items():
            got = actual_answers.get(key, {})
            max_probability_error = max(max_probability_error, probability_error(answer, got))
            if "choice" in answer:
                choice_mismatches += int(answer["choice"] != got.get("choice"))
        responses.append({"id": request.get("id", i), "reference": expected_answers, "exported": actual_answers})
    code_mapping_matches = (reference.builder.codes == candidate.builder.codes
                            and reference.builder.code_token_ids == candidate.builder.code_token_ids)
    temperatures_match = all(reference.temperature_for(d.question_type) == candidate.temperature_for(d.question_type)
                             for d in decisions)
    input_limit_matches = reference.max_input_tokens == candidate.max_input_tokens
    finite = all(math.isfinite(value) for value in (max_logit_error, max_probability_error, max_answer_error))
    passed = (finite and max_logit_error <= args.atol and max_probability_error <= args.atol
              and max_answer_error <= args.atol and not answer_mismatches and not choice_mismatches
              and not prompt_mismatches and code_mapping_matches and temperatures_match and input_limit_matches)
    return {
        "passed": passed, "backend": args.backend, "reference_device": "cpu", "reference_dtype": "float32",
        "candidate_dtype": args.dtype, "atol": args.atol, "versions": package_versions(args.backend),
        "source": args.source, "exported": args.exported,
        "requests_sha256": hashlib.sha256(args.requests.read_bytes()).hexdigest(),
        "requests": len(requests), "questions": len(decisions), "prompt_mismatches": prompt_mismatches,
        "code_mapping_matches": code_mapping_matches, "temperatures_match": temperatures_match,
        "input_limit_matches": input_limit_matches,
        "max_active_logit_abs_error": max_logit_error if math.isfinite(max_logit_error) else None,
        "max_probability_abs_error": max_probability_error if math.isfinite(max_probability_error) else None,
        "max_answer_abs_error": max_answer_error if math.isfinite(max_answer_error) else None,
        "nonfinite_or_missing_outputs": not finite, "answer_mismatches": answer_mismatches,
        "choice_mismatches": choice_mismatches,
        "scope": "Behavior comparison on supplied requests. No accuracy or performance claim.",
        "responses": responses,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="original decision checkpoint")
    parser.add_argument("--exported", required=True, help="native classifier export")
    parser.add_argument("--requests", type=Path, required=True, help="JSONL of {state, questions} requests")
    parser.add_argument("--backend", choices=("transformers", "vllm"), default="transformers")
    parser.add_argument("--device", default="cpu", help="candidate Transformers device (reference is always CPU)")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="float32", help="candidate inference dtype")
    parser.add_argument("--atol", type=float, default=1e-5, help="absolute tolerance for logits and numeric answers")
    parser.add_argument("--engine-options", help="JSON object of additional vLLM constructor options")
    args = parser.parse_args()
    if not math.isfinite(args.atol) or args.atol < 0:
        parser.error("--atol must be finite and nonnegative")
    try:
        result = compare(args)
    except (OSError, ValueError, ImportError, RuntimeError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
