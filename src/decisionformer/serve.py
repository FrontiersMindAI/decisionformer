"""`/v1/systemone` server for any decision checkpoint (the wire format of Jev, vLLM and the Decision Index).

    decisionformer serve org/my-decider --port 8000
    decisionformer serve org/my-qwen3.5-decider --backend vllm   # vLLM computes the readout (Qwen3.5 code readouts)
    curl localhost:8000/v1/systemone -d '{"state": "...", "questions": {"q": {"type": "noul"}}}'

Capacity refusals use HTTP 422 with the phrases the Decision Index `http` engine recognizes ("maximum context
length", "options per choice"), so over-long requests count as unsupported instead of failing the run.
Requests run one at a time: the latency the Index measures is single-request latency.
"""


import hmac
import os
import threading
import time
import uuid
from typing import Any

from decisionformer import __version__
from decisionformer.api import Decider, UnsupportedInput
from decisionformer.prompting import ContextLengthError
from decisionformer.schema import DecisionInputError


def create_app(decider: Any, *, api_key: str | None = None) -> Any:
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
    except ImportError as error:
        raise ImportError("`decisionformer serve` needs `pip install decisionformer[serve]`") from error

    app = FastAPI(title="decisionformer", version=__version__)
    lock = threading.Lock()

    def error(status: int, message: str, kind: str) -> Any:
        return JSONResponse(status_code=status, content={"error": {"message": message, "type": kind}})

    def authorized(request: Request) -> bool:
        if not api_key:
            return True
        return hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {api_key}")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ready", "model": decider.name, "max_input_tokens": decider.max_input_tokens,
                "max_options": len(decider.builder.codes), "prompt_format": decider.builder.format.name}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": decider.name, "object": "model", "owned_by": "decisionformer"}]}

    @app.post("/v1/systemone")
    async def systemone(request: Request) -> Any:
        if not authorized(request):
            return error(401, "missing or invalid API key", "authentication_error")
        try:
            body = await request.json()
        except ValueError:
            return error(400, "the body must be JSON", "invalid_request")
        if not isinstance(body, dict) or "questions" not in body:
            return error(400, "a request needs `state` and `questions`", "invalid_request")
        started = time.perf_counter()
        try:
            with lock:
                response = decider.decide(body.get("state"), body["questions"], images=body.get("images"))
        except DecisionInputError as exc:
            return error(400, str(exc), "invalid_request")
        except (ContextLengthError, UnsupportedInput) as exc:
            return error(422, str(exc), "unsupported_request")
        response.update(id=f"decision-{uuid.uuid4().hex}", object="structured_decision", created=int(time.time()))
        return JSONResponse(content=response, headers={
            "server-timing": f"total;dur={(time.perf_counter() - started) * 1000:.1f}"})

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
          backend: str = "transformers", api_key_env: str = "DECISIONFORMER_API_KEY", **load_options: Any) -> None:
    import uvicorn

    decider = load_decider(model, backend=backend, device=device, **load_options)
    uvicorn.run(create_app(decider, api_key=os.environ.get(api_key_env)), host=host, port=port)
