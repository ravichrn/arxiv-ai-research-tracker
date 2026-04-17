from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agents.supervisor import run_supervisor_once
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
