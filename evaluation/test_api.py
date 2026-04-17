import importlib
import sys
import types

from fastapi.testclient import TestClient


def _load_api_module():
    # Build minimal stubs so api.py can import without the full runtime stack.
    agents_pkg = types.ModuleType("agents")
    supervisor_mod = types.ModuleType("agents.supervisor")

    def _run_supervisor_once(query: str, thread_id: str = "default") -> str:
        return f"stub:{thread_id}:{query}"

    def _stream_supervisor_once(query: str, thread_id: str = "default"):
        yield f"stub:{thread_id}:{query}"

    supervisor_mod.run_supervisor_once = _run_supervisor_once
    supervisor_mod.stream_supervisor_once = _stream_supervisor_once
    agents_pkg.supervisor = supervisor_mod

    guardrails_pkg = types.ModuleType("guardrails")
    sanitizer_mod = types.ModuleType("guardrails.sanitizer")

    class _InputRejected(Exception):
        pass

    sanitizer_mod.InputRejected = _InputRejected
    guardrails_pkg.sanitizer = sanitizer_mod

    sys.modules["agents"] = agents_pkg
    sys.modules["agents.supervisor"] = supervisor_mod
    sys.modules["guardrails"] = guardrails_pkg
    sys.modules["guardrails.sanitizer"] = sanitizer_mod
    sys.modules.pop("api", None)
    return importlib.import_module("api")


api = _load_api_module()
client = TestClient(api.app)


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_success(monkeypatch):
    def _fake_run(query: str, thread_id: str = "default") -> str:
        return f"ok:{thread_id}:{query}"

    monkeypatch.setattr(api, "run_supervisor_once", _fake_run)
    resp = client.post(
        "/chat",
        json={"query": "find papers on diffusion models", "thread_id": "demo"},
    )
    assert resp.status_code == 200
    assert resp.json() == {
        "response": "ok:demo:find papers on diffusion models",
        "thread_id": "demo",
        "error": None,
    }


def test_chat_rejected(monkeypatch):
    def _fake_run(_query: str, thread_id: str = "default") -> str:
        raise ValueError("Query cannot be empty.")

    monkeypatch.setattr(api, "run_supervisor_once", _fake_run)
    resp = client.post("/chat", json={"query": "x", "thread_id": "demo"})
    assert resp.status_code == 400
    assert "Query cannot be empty." in resp.json()["detail"]


def test_chat_stream_done(monkeypatch):
    def _fake_stream(_query: str, thread_id: str = "default"):
        yield "hello"
        yield "world"

    monkeypatch.setattr(api, "stream_supervisor_once", _fake_stream)
    with client.stream(
        "POST", "/chat/stream", json={"query": "hello", "thread_id": "demo"}
    ) as resp:
        assert resp.status_code == 200
        body = "".join(part for part in resp.iter_text())
    assert "data: hello" in body
    assert "data: world" in body
    assert "event: done" in body
