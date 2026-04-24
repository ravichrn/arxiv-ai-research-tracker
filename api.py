import logging
import logging.config
import os
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from prometheus_client import Info
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, ConfigDict, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from agents.supervisor import run_supervisor_once, stream_supervisor_once
from guardrails.sanitizer import InputRejected, validate_user_input

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": "pythonjsonlogger.json.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "json",
            }
        },
        "root": {"level": "INFO", "handlers": ["console"]},
    }
)

_log = logging.getLogger(__name__)

try:
    _MODEL_INFO = Info("agent_llm", "Active agent LLM configuration")
except ValueError:
    from prometheus_client import REGISTRY

    _MODEL_INFO = REGISTRY._names_to_collectors.get("agent_llm_info")  # type: ignore[assignment]

_limiter = Limiter(key_func=get_remote_address)


def _ollama_reachable() -> bool:
    """Match stores.py summarizer routing — cheap TCP check, no model load."""
    try:
        import urllib.request

        urllib.request.urlopen("http://localhost:11434", timeout=1)
        return True
    except Exception:
        return False


@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
    backend = os.getenv("AGENT_LLM", "openai")
    model = (
        os.getenv("VLLM_MODEL")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("ANTHROPIC_MODEL")
        or "default"
    )
    _MODEL_INFO.info({"backend": backend, "model": model})
    _log.info("Starting arXiv AI Research Tracker API", extra={"backend": backend, "model": model})
    yield


app = FastAPI(title="arXiv AI Research Tracker API", version="0.1.0", lifespan=_lifespan)
app.state.limiter = _limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"query": "find papers on diffusion models", "thread_id": "demo"},
                {"query": "fetch NLP papers then summarize", "thread_id": "session-1"},
            ]
        }
    )

    query: str = Field(..., min_length=1, description="Natural language query for the supervisor")
    thread_id: str = Field(
        default="default",
        min_length=1,
        description="Conversation thread identifier for memory",
    )


class ChatResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "response": "Top papers on diffusion models include...",
                    "thread_id": "demo",
                    "error": None,
                }
            ]
        }
    )

    response: str
    thread_id: str
    error: str | None = None


@app.get("/health", summary="Liveness check")
def health() -> dict[str, str]:
    backend = os.getenv("AGENT_LLM", "openai")
    model = (
        os.getenv("VLLM_MODEL")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("ANTHROPIC_MODEL")
        or "default"
    )
    return {"status": "ok", "backend": backend, "model": model}


@app.get("/models", summary="Active model configuration")
def models() -> dict[str, str]:
    """Returns the active LLM backend and model name."""
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")
    summarizer = f"ollama/{ollama_model}" if _ollama_reachable() else "openai/gpt-4o-mini"
    return {
        "backend": os.getenv("AGENT_LLM", "openai"),
        "agent_model": os.getenv("VLLM_MODEL")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("ANTHROPIC_MODEL")
        or "default",
        "summarizer_model": summarizer,
        "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
    }


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Run one chat turn",
    description="Returns a single non-streaming supervisor response for the provided query.",
    responses={
        400: {"description": "Invalid user query"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
)
@_limiter.limit("20/minute")
def chat(request: Request, req: ChatRequest) -> ChatResponse:
    try:
        validated_query = validate_user_input(req.query)
    except (InputRejected, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        response = run_supervisor_once(validated_query, thread_id=req.thread_id)
        return ChatResponse(response=response, thread_id=req.thread_id, error=None)
    except InputRejected as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        _log.exception("Unhandled error in /chat")
        raise HTTPException(status_code=500, detail="Internal server error") from exc


@app.post(
    "/chat/stream",
    summary="Stream one chat turn as SSE",
    description=(
        "Streams supervisor output chunks as Server-Sent Events (text/event-stream). "
        "Each chunk is sent as a `data:` frame and terminates with `event: done`."
    ),
    responses={
        200: {
            "description": "SSE stream",
            "content": {
                "text/event-stream": {
                    "example": "data: partial text\\n\\nevent: done\\ndata: [DONE]\\n\\n"
                }
            },
        },
        400: {"description": "Invalid user query (emitted as SSE error event)"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal error (emitted as SSE error event)"},
    },
)
@_limiter.limit("20/minute")
def chat_stream(request: Request, req: ChatRequest) -> StreamingResponse:
    # Align with /chat by rejecting invalid input before opening the SSE stream.
    cleaned_query = req.query.strip()
    if not cleaned_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        validated_query = validate_user_input(cleaned_query)
    except InputRejected as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    def event_stream() -> Iterator[str]:
        try:
            for chunk in stream_supervisor_once(validated_query, thread_id=req.thread_id):
                # SSE format: one event chunk per generated token/message piece.
                safe_chunk = chunk.replace("\r", "")
                for line in safe_chunk.split("\n"):
                    yield f"data: {line}\n"
                yield "\n"
            yield "event: done\ndata: [DONE]\n\n"
        except InputRejected as exc:
            yield f"event: error\ndata: {exc}\n\n"
        except ValueError as exc:
            yield f"event: error\ndata: {exc}\n\n"
        except Exception:
            _log.exception("Unhandled error in /chat/stream")
            yield "event: error\ndata: Internal server error\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
