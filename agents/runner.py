"""
LangGraph-based agent runner.

Graph structure:
    [START] → agent_node → tool_node → agent_node → ... → [END]

- agent_node: calls the LLM with bound tools; decides whether to invoke a tool
  or return a final answer.
- tool_node: executes whichever tool(s) the LLM requested, then loops back to
  agent_node so the LLM can see the result and continue reasoning.

This explicit graph replaces create_agent(), making control flow inspectable
and traceable (each node appears as a separate span in Arize Phoenix / LangSmith).
"""

from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing_extensions import TypedDict

from databases.stores import llm_agent
from agents.tools import search_papers, search_saved_papers, add_paper_to_saved, delete_paper_from_saved
from guardrails.sanitizer import validate_user_input, InputRejected

_MAX_HISTORY_TURNS = 10

_TOOLS = [search_papers, search_saved_papers, add_paper_to_saved, delete_paper_from_saved]

_SYSTEM_MESSAGE = SystemMessage(content=(
    "You are an expert AI research assistant with access to a database of recent papers "
    "from arXiv covering AI, ML, NLP, and Robotics (cs.AI, cs.LG, cs.CL, cs.RO).\n"
    "When answering:\n"
    "- Always use the search tools to ground your answer in retrieved papers.\n"
    "- Cite paper titles and URLs when referencing specific work.\n"
    "- If the database has no relevant results, say so clearly — do not speculate.\n"
    "- For questions about recency, note the published date from metadata.\n"
    "- Never fabricate paper titles, authors, or results."
))


# ---------------------------------------------------------------------------
# Graph state — messages list is the only shared state.
# add_messages reducer appends new messages rather than replacing the list.
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ---------------------------------------------------------------------------
# Nodes
# bind_tools at module level — tools don't change, no need to rebind per session.
# ---------------------------------------------------------------------------
_llm_with_tools = llm_agent.bind_tools(_TOOLS)


def agent_node(state: AgentState) -> dict:
    """Call the LLM; it either returns an answer or requests tool calls."""
    response = _llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


_tool_node = ToolNode(_TOOLS)


# ---------------------------------------------------------------------------
# Graph definition
# ---------------------------------------------------------------------------
def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", _tool_node)

    graph.add_edge(START, "agent")
    # tools_condition routes to "tools" if the LLM made tool calls, else END
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile()


_graph = _build_graph()


# ---------------------------------------------------------------------------
# CLI loop
# ---------------------------------------------------------------------------
def launch_agent() -> None:
    chat_history: list[HumanMessage | AIMessage] = []

    print("\n[Agent Ready] Ask about AI research papers or type 'exit' to quit.\n")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("Exiting. Goodbye!")
            break
        try:
            query = validate_user_input(query)

            # Prepend system message + trimmed history on every turn
            trimmed = chat_history[-(_MAX_HISTORY_TURNS * 2):]
            messages = [_SYSTEM_MESSAGE] + trimmed + [HumanMessage(content=query)]

            result = _graph.invoke({"messages": messages})
            answer = str(result["messages"][-1].content)

            chat_history.extend([HumanMessage(content=query), AIMessage(content=answer)])
            # Cap the stored history — trim oldest turns when limit exceeded
            if len(chat_history) > _MAX_HISTORY_TURNS * 2:
                chat_history = chat_history[-(_MAX_HISTORY_TURNS * 2):]
            print(f"Agent: {answer}\n")

        except InputRejected as e:
            print(f"Blocked: {e}\n")
        except KeyboardInterrupt:
            print("\nExiting. Goodbye!")
            break
        except Exception as e:
            print(f"Error ({type(e).__name__}): {e}\n")
