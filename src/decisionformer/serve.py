"""`/v1/systemone` server for any decision checkpoint (the wire format of Jev, vLLM and the Decision Index).

    decisionformer serve org/my-decider --port 8000
    decisionformer serve org/my-qwen3.5-decider --backend vllm   # vLLM computes the readout (Qwen3.5 code readouts)
    curl localhost:8000/v1/systemone -d '{"state": "...", "questions": {"q": {"type": "noul"}}}'

Capacity refusals use HTTP 422 with the phrases the Decision Index `http` engine recognizes ("maximum context
length", "options per choice"), so over-long requests count as unsupported instead of failing the run.
A request that arrives while the model is idle runs at once, alone, so single-request latency is unchanged; requests
that arrive while it is busy are answered together in one batch. The request and response schemas are in /docs.
"""


import asyncio
import hmac
import os
import queue
import threading
import time
import uuid
from concurrent.futures import Future
from typing import Any, Literal

from pydantic import BaseModel, Field

from decisionformer import __version__
from decisionformer.api import Decider, UnsupportedInput
from decisionformer.prompting import ContextLengthError
from decisionformer.schema import DecisionInputError


# ----------------------------------------------------------------------------------------------- API schema (docs)
class QuestionSpec(BaseModel):
    type: Literal["choice", "noul", "score"] = Field(description="choice: pick one option; noul: yes/no; "
                                                                  "score: pick one of 2-10 ordered levels")
    instructions: Any = Field(None, description="the question, in words")
    criteria: dict[str, Any] | list[Any] | None = Field(
        None, description="choice: option -> description (or null), or a list of options; score: the ordered levels; "
                          "noul: optional descriptions of 'false' and 'true'")


class SystemOneRequest(BaseModel):
    state: Any = Field(description="what to decide about: text or any JSON value")
    questions: dict[str, QuestionSpec] = Field(description="question id -> question")

    model_config = {"json_schema_extra": {"examples": [{
        "state": "I was charged twice for my subscription this month.",
        "questions": {"angry": {"type": "noul", "instructions": "Is the customer angry?"},
                      "team": {"type": "choice", "criteria": {"billing": "Payments and refunds", "technical": None}}},
    }]}}


class Answer(BaseModel):
    type: Literal["choice", "noul", "score"]
    choice: str | None = Field(None, description="choice: the most probable option")
    noul: float | None = Field(None, description="noul: the probability of yes")
    score: float | None = Field(None, description="score: the expected level, 0-based")
    probabilities: dict[str, float] | None = Field(None, description="choice and score: per option or level")
    confidence: float | None = None
    legend: dict[str, Any] | None = Field(None, description="score: level index -> level")


class SystemOneResponse(BaseModel):
    id: str
    object: Literal["structured_decision"]
    created: int
    model: str
    answers: dict[str, Answer]
    usage: dict[str, int]


class ErrorResponse(BaseModel):
    error: dict[str, str]


# ----------------------------------------------------------------------------------------------- batching
class Batcher:
    """Answers requests on one worker thread. The worker takes the next request and every request already queued
    behind it (up to `max_batch`) and answers them in one `decide_many` call; if that fails, it answers them one by
    one so each request gets its own result or error."""

    def __init__(self, decider: Any, max_batch: int = 16) -> None:
        self.decider = decider
        self.max_batch = max(1, max_batch)
        self.queue: queue.Queue[tuple[dict[str, Any], Future]] = queue.Queue()
        threading.Thread(target=self._run, name="decisionformer-batcher", daemon=True).start()

    def submit(self, request: dict[str, Any]) -> Future:
        future: Future = Future()
        self.queue.put((request, future))
        return future

    def _answer(self, requests: list[dict[str, Any]]) -> list[Any]:
        try:
            return list(self.decider.decide_many(requests))
        except Exception as error:  # noqa: BLE001 - delivered to the request(s) it belongs to
            if len(requests) == 1:
                return [error]
            return [self._answer([request])[0] for request in requests]

    def _run(self) -> None:
        while True:
            batch = [self.queue.get()]
            while len(batch) < self.max_batch:
                try:
                    batch.append(self.queue.get_nowait())
                except queue.Empty:
                    break
            for (_, future), result in zip(batch, self._answer([request for request, _ in batch])):
                if isinstance(result, BaseException):
                    future.set_exception(result)
                else:
                    future.set_result(result)


# ----------------------------------------------------------------------------------------------- app
def create_app(decider: Any, *, api_key: str | None = None, max_batch: int = 16) -> Any:
    try:
        from fastapi import FastAPI, Request
        from fastapi.openapi.utils import get_openapi
        from fastapi.responses import JSONResponse
    except ImportError as error:
        raise ImportError("`decisionformer serve` needs `pip install decisionformer[serve]`") from error

    app = FastAPI(title="decisionformer", version=__version__,
                  description="Answer System One requests: typed questions about a state, one forward pass each.")
    batcher = Batcher(decider, max_batch)

    def error(status: int, message: str, kind: str) -> Any:
        return JSONResponse(status_code=status, content={"error": {"message": message, "type": kind}})

    def authorized(request: Request) -> bool:
        if not api_key:
            return True
        return hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {api_key}")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ready", "model": decider.name, "max_input_tokens": decider.max_input_tokens,
                "max_options": len(decider.builder.codes), "prompt_format": decider.builder.format.name,
                "max_batch": batcher.max_batch}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": decider.name, "object": "model", "owned_by": "decisionformer"}]}

    @app.post("/v1/systemone", summary="Answer every question about a state")
    async def systemone(request: Request) -> Any:
        if not authorized(request):
            return error(401, "missing or invalid API key", "authentication_error")
        try:
            body = await request.json()
        except ValueError:
            return error(400, "the body must be JSON", "invalid_request")
        if not isinstance(body, dict) or "questions" not in body or "state" not in body:
            return error(400, "a request needs `state` and `questions`", "invalid_request")
        started = time.perf_counter()
        try:
            response = await asyncio.wrap_future(batcher.submit(
                {"state": body.get("state"), "questions": body["questions"], "images": body.get("images")}))
        except DecisionInputError as exc:
            return error(400, str(exc), "invalid_request")
        except (ContextLengthError, UnsupportedInput) as exc:
            return error(422, str(exc), "unsupported_request")
        response = {**response, "id": f"decision-{uuid.uuid4().hex}", "object": "structured_decision",
                    "created": int(time.time())}
        return JSONResponse(content=response, headers={
            "server-timing": f"total;dur={(time.perf_counter() - started) * 1000:.1f}"})

    def openapi() -> dict[str, Any]:
        # The endpoint parses requests itself (its error codes and messages are part of the protocol), so the
        # schemas are added to the generated document for /docs instead of being enforced by FastAPI.
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes)
        components = schema.setdefault("components", {}).setdefault("schemas", {})
        for model in (SystemOneRequest, SystemOneResponse, ErrorResponse):
            body = model.model_json_schema(ref_template="#/components/schemas/{model}")
            components.update(body.pop("$defs", {}))
            components[model.__name__] = body

        def content(name: str) -> dict[str, Any]:
            return {"application/json": {"schema": {"$ref": f"#/components/schemas/{name}"}}}

        operation = schema["paths"]["/v1/systemone"]["post"]
        operation["requestBody"] = {"required": True, "content": content("SystemOneRequest")}
        operation["responses"] = {
            "200": {"description": "One answer per question", "content": content("SystemOneResponse")},
            "400": {"description": "Malformed request", "content": content("ErrorResponse")},
            "401": {"description": "Missing or invalid API key", "content": content("ErrorResponse")},
            "422": {"description": "Too long or too many options for this model (never truncated)",
                    "content": content("ErrorResponse")},
        }
        app.openapi_schema = schema
        return schema

    app.openapi = openapi
    return app


def load_decider(model: str, *, backend: str = "transformers", device: str | None = None,
                 **load_options: Any) -> Any:
    """A `Decider` (transformers) or a `VLLMDecider`; unsupported vLLM checkpoints raise UnsupportedBackend."""
    if backend == "transformers":
        return Decider.from_pretrained(model, device=device, **load_options)
    if backend == "vllm":
        from decisionformer.vllm_backend import VLLMDecider

        load_options.pop("batch_size", None)  # vLLM schedules its own batches
        return VLLMDecider.from_pretrained(model, **load_options)
    raise ValueError(f"unknown backend {backend!r}; use 'transformers' or 'vllm'")


def serve(model: str, *, host: str = "127.0.0.1", port: int = 8000, device: str | None = None,
          backend: str = "transformers", api_key_env: str = "DECISIONFORMER_API_KEY", max_batch: int = 16,
          **load_options: Any) -> None:
    decider = load_decider(model, backend=backend, device=device, **load_options)  # model errors first
    import uvicorn

    uvicorn.run(create_app(decider, api_key=os.environ.get(api_key_env), max_batch=max_batch), host=host, port=port)
