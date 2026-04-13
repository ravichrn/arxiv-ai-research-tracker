"""
Multi-agent supervisor for the arXiv AI research tracker.

The supervisor routes natural language requests to specialized sub-agents and
supports multi-step chaining (e.g. "fetch NLP papers then find the best on
transformers").

Graph structure:
    [START] → route_node → ingestion_node  ─┐
                        → rag_node         ─┤→ _route_after_capability → finalize_node → [END]
                        → summarize_node   ─┘        │ (chain pending)
                        → clarify_node → [END]  ← ───┘ (loops back to route_node)

Sub-agents:
- route_node:      LLM parses intent + topics; sets pending_chain for multi-step requests.
- ingestion_node:  Calls fetch_and_summarize_papers() for the resolved topics.
- rag_node:        Delegates to the compiled Self-RAG graph from agents/runner.py.
- summarize_node:  Hybrid-searches papers and returns an LLM-generated batch summary.
- clarify_node:    Asks the user to rephrase when intent is unclear.
- finalize_node:   Assembles and returns the final answer.
"""

import json
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agents.runner import rag_graph
from agents.tools import _format_docs
from databases.stores import hybrid_search, llm_agent, papers_store
from guardrails.sanitizer import InputRejected, validate_user_input
from ingestion.arxiv_fetcher import TOPICS, fetch_and_summarize_papers

_MAX_CHAIN_STEPS = 3
_MAX_HISTORY_TURNS = 10

_TOPIC_ALIASES: dict[str, str] = {
    "ai": "cs.AI",
    "artificial intelligence": "cs.AI",
    "ml": "cs.LG",
    "machine learning": "cs.LG",
    "nlp": "cs.CL",
    "llm": "cs.CL",
    "language": "cs.CL",
    "robotics": "cs.RO",
    "robot": "cs.RO",
}

_SYSTEM_MESSAGE = SystemMessage(
    content=(
        "You are a supervisor for an AI research assistant. You help users fetch "
        "recent arXiv papers, search the paper database, and get summaries.\n"
        "Available capabilities: fetch (ingest new papers), rag (Q&A search), "
        "summarize (batch summary of papers)."
    )
)

_ROUTE_PROMPT = """\
You are a routing assistant for an AI research paper tool.

Given the user request, respond with a JSON object only (no markdown, no explanation):
{{
  "steps": [...],      // ordered list from: "fetch", "rag", "summarize"
  "topics": [...],     // arXiv topic codes needed for fetch: cs.AI, cs.LG, cs.CL, cs.RO
  "rag_query": "..."   // refined search query for rag/summarize steps (empty string if not needed)
}}

Rules:
- Use "fetch" when the user wants to ingest/download new papers.
- Use "rag" when the user wants to search or ask questions about papers.
- Use "summarize" when the user wants a summary or overview of papers.
- Multi-step requests get multiple entries in "steps" (max {max_steps}).
- If topics are not mentioned for a fetch, return an empty list (means fetch all).
- If the request is unclear or unrelated, return {{"steps": [], "topics": [], "rag_query": ""}}.

User request: {query}
"""


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str  # current step being executed: "fetch" | "rag" | "summarize" | "unknown"
    resolved_topics: list[str]  # parsed arXiv topic codes for fetch
    pending_chain: list[str]  # remaining steps in a multi-step request
    rag_query: str  # refined query for rag/summarize nodes
    last_result: str  # output from the last completed sub-agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _last_human_content(state: SupervisorState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def _parse_route(raw: str) -> dict:
    """Extract JSON from the LLM routing response, tolerating minor formatting issues."""
    raw = raw.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"steps": [], "topics": [], "rag_query": ""}


def _resolve_topics(raw_topics: list[str]) -> list[str]:
    """Map LLM-returned topic strings to valid arXiv category codes."""
    resolved = []
    for t in raw_topics:
        t_lower = t.lower().strip()
        if t in set(TOPICS):
            resolved.append(t)
        elif t_lower in _TOPIC_ALIASES:
            resolved.append(_TOPIC_ALIASES[t_lower])
    # Deduplicate while preserving order
    seen: set[str] = set()
    out = []
    for t in resolved:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def route_node(state: SupervisorState) -> dict:
    """Parse user intent and build the execution chain.

    On subsequent calls (mid-chain), pops the next intent from pending_chain
    instead of re-parsing the original query — this is what enables chaining.
    """
    pending = list(state.get("pending_chain", []))
    if pending:
        # Already have a chain in progress — advance to the next step.
        next_intent = pending.pop(0)
        print(f"  [Supervisor] chain advancing → intent={next_intent}, remaining={pending}")
        return {"intent": next_intent, "pending_chain": pending}

    # First call — parse intent from the user query.
    query = _last_human_content(state)
    prompt = _ROUTE_PROMPT.format(query=query, max_steps=_MAX_CHAIN_STEPS)
    raw = str(llm_agent.invoke(prompt).content)
    parsed = _parse_route(raw)

    steps: list[str] = parsed.get("steps", [])[:_MAX_CHAIN_STEPS]
    raw_topics: list[str] = parsed.get("topics", [])
    rag_query: str = parsed.get("rag_query", "") or query

    resolved = _resolve_topics(raw_topics)
    # Fall back to all topics if fetch is requested but none resolved
    if "fetch" in steps and not resolved:
        resolved = list(TOPICS)

    if not steps:
        intent = "unknown"
        remaining: list[str] = []
    else:
        intent = steps[0]
        remaining = steps[1:]

    print(f"  [Supervisor] intent={intent}, chain={remaining}, topics={resolved}")
    return {
        "intent": intent,
        "resolved_topics": resolved,
        "pending_chain": remaining,
        "rag_query": rag_query,
    }


def ingestion_node(state: SupervisorState) -> dict:
    """Fetch and ingest new arXiv papers for the resolved topics."""
    topics = state.get("resolved_topics") or list(TOPICS)
    print(f"  [Supervisor] Fetching topics: {', '.join(topics)}")
    fetch_and_summarize_papers(topics=topics)
    result = f"Fetched and indexed new papers for: {', '.join(topics)}."
    return {"last_result": result, "messages": [AIMessage(content=result)]}


def rag_node(state: SupervisorState) -> dict:
    """Delegate to the Self-RAG sub-agent for paper Q&A."""
    query = state.get("rag_query") or _last_human_content(state)
    print(f"  [Supervisor] RAG query: {query}")
    result = rag_graph.invoke(
        {
            "messages": [_SYSTEM_MESSAGE, HumanMessage(content=query)],
            "retrieval_context": [],
            "rewrite_count": 0,
            "hallucination_verdict": "",
        }
    )
    answer = str(result["messages"][-1].content)
    return {"last_result": answer, "messages": [AIMessage(content=answer)]}


def summarize_node(state: SupervisorState) -> dict:
    """Retrieve a batch of papers and return an LLM-generated summary."""
    query = state.get("rag_query") or _last_human_content(state)
    print(f"  [Supervisor] Summarizing papers for: {query}")
    docs = hybrid_search(papers_store, query, k=5)
    formatted = _format_docs(docs)
    prompt = (
        "You are a research assistant. Summarize the following papers for a researcher "
        "in 3-5 bullet points per paper, highlighting key contributions and methods.\n\n"
        f"{formatted}"
    )
    answer = str(llm_agent.invoke(prompt).content)
    return {"last_result": answer, "messages": [AIMessage(content=answer)]}


def clarify_node(state: SupervisorState) -> dict:
    """Ask the user to rephrase when intent is unclear."""
    clarification = (
        "I'm not sure what you'd like to do. You can ask me to:\n"
        "- **Fetch** new papers (e.g. 'fetch recent NLP papers')\n"
        "- **Search** for papers (e.g. 'find papers on diffusion models')\n"
        "- **Summarize** papers (e.g. 'summarize recent robotics papers')\n"
        "- **Chain** steps (e.g. 'fetch ML papers then find the best on LLMs')"
    )
    return {"last_result": clarification, "messages": [AIMessage(content=clarification)]}


def finalize_node(state: SupervisorState) -> dict:
    """Pass-through; last_result is already set by the capability node."""
    return {}


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------
def _dispatch_intent(state: SupervisorState) -> str:
    intent = state.get("intent", "unknown")
    if intent == "fetch":
        return "ingestion"
    if intent == "rag":
        return "rag"
    if intent == "summarize":
        return "summarize"
    return "clarify"


def _route_after_capability(state: SupervisorState) -> str:
    """After a capability node: continue chain or finalize."""
    pending = state.get("pending_chain", [])
    if pending:
        return "route"  # loop back to handle next step
    return "finalize"


# ---------------------------------------------------------------------------
# Graph definition
# ---------------------------------------------------------------------------
def _build_supervisor_graph():
    graph = StateGraph(SupervisorState)

    graph.add_node("route", route_node)
    graph.add_node("ingestion", ingestion_node)
    graph.add_node("rag", rag_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", _dispatch_intent)

    # After each capability, either chain to next step or finalize
    for cap in ("ingestion", "rag", "summarize", "clarify"):
        graph.add_conditional_edges(
            cap, _route_after_capability, {"route": "route", "finalize": "finalize"}
        )

    graph.add_edge("finalize", END)

    return graph.compile()


_supervisor_graph = _build_supervisor_graph()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def launch_supervisor() -> None:
    chat_history: list[HumanMessage | AIMessage] = []

    print("\n[Supervisor Ready] Ask me to fetch papers, search, summarize, or chain tasks.")
    print("Examples:")
    print("  - 'fetch recent robotics papers'")
    print("  - 'find papers on diffusion models'")
    print("  - 'fetch NLP papers then summarize the best on transformers'")
    print("  - 'summarize recent cs.AI papers'\n")

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit"):
            print("Exiting. Goodbye!")
            break
        try:
            query = validate_user_input(query)

            trimmed = chat_history[-(_MAX_HISTORY_TURNS * 2) :]
            messages = [_SYSTEM_MESSAGE, *trimmed, HumanMessage(content=query)]

            result = _supervisor_graph.invoke(
                {
                    "messages": messages,
                    "intent": "",
                    "resolved_topics": [],
                    "pending_chain": [],
                    "rag_query": "",
                    "last_result": "",
                }
            )
            answer = result.get("last_result") or str(result["messages"][-1].content)

            chat_history.extend([HumanMessage(content=query), AIMessage(content=answer)])
            if len(chat_history) > _MAX_HISTORY_TURNS * 2:
                chat_history = chat_history[-(_MAX_HISTORY_TURNS * 2) :]
            print(f"\nAgent: {answer}\n")

        except InputRejected as e:
            print(f"Blocked: {e}\n")
        except KeyboardInterrupt:
            print("\nExiting. Goodbye!")
            break
        except Exception as e:
            print(f"Error ({type(e).__name__}): {e}\n")
