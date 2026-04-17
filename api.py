from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agents.supervisor import run_supervisor_once, stream_supervisor_once
from guardrails.sanitizer import InputRejected

app = FastAPI(title="arXiv AI Research Tracker API", version="0.1.0")


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language query for the supervisor")
    thread_id: str = Field(
        default="default",
        min_length=1,
        description="Conversation thread identifier for memory",
    )


class ChatResponse(BaseModel):
    response: str
    thread_id: str
    error: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
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


@app.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    def event_stream():
        try:
            for chunk in stream_supervisor_once(req.query, thread_id=req.thread_id):
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
