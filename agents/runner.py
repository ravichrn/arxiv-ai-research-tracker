"""
LangGraph-based agent runner with Self-RAG.

Graph structure:
    [START] → agent → tools_condition → tool_node → grade_docs
                                                          │ (docs pass)
                                                        agent → hallucination_check → [END]
                                                          ↑            │ (not grounded)
                                                    rewrite_query ←───┘
                                                               ↑
                                                    grade_docs (all filtered)

Self-RAG nodes:
- grade_docs_node:         filters retrieved docs by relevance before generation.
- rewrite_query_node:      reformulates the user query when retrieval fails or answer
                           is not grounded; capped at _MAX_REWRITES to prevent loops.
- hallucination_check_node: verifies the generated answer is grounded in retrieved
                           context; routes to rewrite or END accordingly.
"""

from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from agents.tools import (
    add_paper_to_saved,
    delete_paper_from_saved,
    search_papers,
    search_saved_papers,
)
from databases.stores import llm_agent

_MAX_REWRITES = 2

# Sentinel strings used in comparisons — defined once to avoid raw string duplication.
_NO_RESULTS_MSG = "No relevant papers found."
_DISCLAIMER_PREFIX = "⚠️ Note:"

_TOOLS = [search_papers, search_saved_papers, add_paper_to_saved, delete_paper_from_saved]

_SYSTEM_MESSAGE = SystemMessage(
    content=(
        "You are an expert AI research assistant with access to a database of recent papers "
        "from arXiv covering AI, ML, NLP, and Robotics (cs.AI, cs.LG, cs.CL, cs.RO).\n"
        "When answering:\n"
        "- Always use the search tools to ground your answer in retrieved papers.\n"
        "- Cite paper titles and URLs when referencing specific work.\n"
        "- If the database has no relevant results, say so clearly — do not speculate.\n"
        "- For questions about recency, note the published date from metadata.\n"
        "- Never fabricate paper titles, authors, or results."
    )
)

# ---------------------------------------------------------------------------
# Grader prompts — inline, no file I/O needed.
# ---------------------------------------------------------------------------
_DOC_GRADE_PROMPT = (
    "You are grading document relevance.\n"
    "Question: {question}\n"
    "Document: {doc}\n"
    "Is this document relevant to the question? Reply with a single word: YES or NO."
)

_HALLUCINATION_PROMPT = (
    "You are checking answer faithfulness.\n"
    "Context:\n{context}\n\n"
    "Answer: {answer}\n"
    "Is the answer fully grounded in the context above with no fabricated claims? "
    "Reply with YES or NO only."
)

_REWRITE_PROMPT = (
    "You are improving a search query to retrieve better research paper results.\n"
    "Original query: {query}\n"
    "Rewrite it to be more specific and focused on research concepts. "
    "Reply with only the rewritten query, no explanation."
)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    retrieval_context: list[str]  # filtered doc texts from grade_docs_node
    rewrite_count: int  # guards against infinite rewrite loops
    hallucination_verdict: str  # "YES" | "NO" | "" — set by hallucination_check_node


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_last_question(state: AgentState) -> str:
    """Return the most recent HumanMessage content."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def _get_last_answer(state: AgentState) -> str:
    """Return the most recent non-tool AIMessage content."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            return str(msg.content)
    return ""


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
_llm_with_tools = llm_agent.bind_tools(_TOOLS)


def agent_node(state: AgentState) -> dict:
    """Call the LLM; it either returns an answer or requests tool calls."""
    response = _llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


_tool_node = ToolNode(_TOOLS)


def grade_docs_node(state: AgentState) -> dict:
    """Grade each retrieved document for relevance; keep only passing ones.

    Parses the last ToolMessage (search result string) into individual doc
    blocks, grades each with the LLM, and stores survivors in retrieval_context.
    """
    question = _get_last_question(state)

    # Find the most recent ToolMessage (search results).
    tool_content = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            tool_content = str(msg.content)
            break

    if not tool_content or tool_content.strip() == _NO_RESULTS_MSG:
        return {"retrieval_context": [], "rewrite_count": state.get("rewrite_count", 0)}

    # Each doc block is separated by a blank line in _format_docs output.
    doc_blocks = [b.strip() for b in tool_content.split("\n\n") if b.strip()]

    passing: list[str] = []
    for doc in doc_blocks:
        prompt = _DOC_GRADE_PROMPT.format(question=question, doc=doc[:800])
        verdict = str(llm_agent.invoke(prompt).content).strip().upper()
        if verdict.startswith("YES"):
            passing.append(doc)

    return {
        "retrieval_context": passing,
        "rewrite_count": state.get("rewrite_count", 0),
    }


def rewrite_query_node(state: AgentState) -> dict:
    """Rewrite the user query to improve retrieval on the next attempt."""
    question = _get_last_question(state)
    prompt = _REWRITE_PROMPT.format(query=question)
    rewritten = str(llm_agent.invoke(prompt).content).strip()
    print(f"  [Self-RAG] Rewriting query → {rewritten}")
    return {
        "messages": [HumanMessage(content=rewritten)],
        "rewrite_count": state.get("rewrite_count", 0) + 1,
    }


def hallucination_check_node(state: AgentState) -> dict:
    """Verify the generated answer is grounded in the retrieved context."""
    context = state.get("retrieval_context", [])
    if not context:
        # No retrieved context to check against — accept the answer as-is.
        return {"hallucination_verdict": "YES"}

    answer = _get_last_answer(state)

    prompt = _HALLUCINATION_PROMPT.format(
        context="\n\n".join(context)[:2000],
        answer=answer[:1000],
    )
    verdict = str(llm_agent.invoke(prompt).content).strip().upper()
    verdict = "YES" if verdict.startswith("YES") else "NO"

    if verdict == "NO":
        print("  [Self-RAG] Hallucination detected.")
        if state.get("rewrite_count", 0) >= _MAX_REWRITES:
            # Max retries reached — append disclaimer and end.
            disclaimer = (
                f"\n\n{_DISCLAIMER_PREFIX} This answer could not be fully verified against "
                "the retrieved papers. Please treat with caution."
            )
            return {
                "hallucination_verdict": verdict,
                "messages": [AIMessage(content=answer + disclaimer)],
            }

    return {"hallucination_verdict": verdict}


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------
def _route_after_grade_docs(state: AgentState) -> str:
    if state.get("retrieval_context"):
        return "agent"
    # No relevant docs — rewrite if retries remain, else generate anyway.
    if state.get("rewrite_count", 0) < _MAX_REWRITES:
        return "rewrite_query"
    return "agent"


def _route_after_hallucination_check(state: AgentState) -> str:
    if state.get("rewrite_count", 0) >= _MAX_REWRITES:
        return END
    verdict = state.get("hallucination_verdict", "YES")
    return "rewrite_query" if verdict == "NO" else END


# ---------------------------------------------------------------------------
# Graph definition
# ---------------------------------------------------------------------------
def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", _tool_node)
    graph.add_node("grade_docs", grade_docs_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("hallucination_check", hallucination_check_node)

    graph.add_edge(START, "agent")
    # After agent: use tools_condition to go to tools or END,
    # but intercept the END case to run hallucination_check first.
    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", END: "hallucination_check"},
    )
    graph.add_edge("tools", "grade_docs")
    graph.add_conditional_edges("grade_docs", _route_after_grade_docs)
    graph.add_edge("rewrite_query", "agent")
    graph.add_conditional_edges("hallucination_check", _route_after_hallucination_check)

    return graph.compile()


_graph = _build_graph()
rag_graph = _graph  # expose for supervisor
