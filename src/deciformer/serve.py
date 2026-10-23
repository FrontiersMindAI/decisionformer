"""`/v1/systemone` server for any decision checkpoint (the wire format of Jev, vLLM and the Decision Index).

    deciformer serve org/my-decider --port 8000
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

from deciformer import __version__
from deciformer.api import Decider, UnsupportedInput
from deciformer.prompting import ContextLengthError
from deciformer.schema import DecisionInputError


def create_app(decider: Decider, *, api_key: str | None = None) -> Any:
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
    except ImportError as error:
        raise ImportError("`deciformer serve` needs `pip install deciformer[serve]`") from error

    app = FastAPI(title="deciformer", version=__version__)
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
        return {"object": "list", "data": [{"id": decider.name, "object": "model", "owned_by": "deciformer"}]}

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


def serve(model: str, *, host: str = "127.0.0.1", port: int = 8000, device: str | None = None,
          api_key_env: str = "DECIFORMER_API_KEY", **load_options: Any) -> None:
    import uvicorn

    decider = Decider.from_pretrained(model, device=device, **load_options)
    uvicorn.run(create_app(decider, api_key=os.environ.get(api_key_env)), host=host, port=port)
