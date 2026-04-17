from collections.abc import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from agents.supervisor import run_supervisor_once, stream_supervisor_once
from guardrails.sanitizer import InputRejected, validate_user_input

app = FastAPI(title="arXiv AI Research Tracker API", version="0.1.0")


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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Run one chat turn",
    description="Returns a single non-streaming supervisor response for the provided query.",
    responses={
        400: {"description": "Invalid user query"},
        500: {"description": "Internal server error"},
    },
)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        response = run_supervisor_once(req.query, thread_id=req.thread_id)
        return ChatResponse(response=response, thread_id=req.thread_id, error=None)
    except InputRejected as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


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
        500: {"description": "Internal error (emitted as SSE error event)"},
    },
)
def chat_stream(req: ChatRequest) -> StreamingResponse:
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
        except Exception as exc:
            yield f"event: error\ndata: {type(exc).__name__}: {exc}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
