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
import re as _re
from pathlib import Path
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agents.runner import rag_graph
from agents.tools import _format_docs
from databases.stores import hybrid_search, llm_agent, papers_store
from guardrails.sanitizer import InputRejected, validate_user_input
from ingestion.arxiv_fetcher import (
    TOPICS,
    fetch_paper_content,
    fetch_papers,
    get_paper_by_arxiv_id,
    get_paper_by_title,
    get_recent_papers,
    list_papers,
)

_MAX_CHAIN_STEPS = 3

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

_TOPIC_SET = set(TOPICS)

_SYSTEM_MESSAGE = SystemMessage(
    content=(
        "You are a supervisor for an AI research assistant. You help users fetch "
        "recent arXiv papers, search the paper database, get summaries, compare papers, "
        "auto-tag/cluster papers, generate digests, create Mermaid diagrams, and extract "
        "figures from papers.\n"
        "Available capabilities: fetch (ingest new papers), rag (Q&A search), "
        "summarize (batch summary), compare (side-by-side comparison), "
        "tag (cluster papers by research theme), digest (recent papers digest), "
        "diagram (Mermaid flowchart of a paper's methodology), "
        "figures (extract real images/figures from a paper)."
    )
)

_ROUTE_PROMPT = """\
You are a routing assistant for an AI research paper tool.

Given the user request, respond with a JSON object only (no markdown, no explanation):
{{
  "steps": [...],      // ordered list: fetch|list|rag|summarize|compare|tag|digest|diagram|figures
  "topics": [...],     // arXiv topic codes needed for fetch: cs.AI, cs.LG, cs.CL, cs.RO
  "rag_query": "..."   // refined query for rag/summarize/tag/diagram (empty string if not needed)
}}

Rules:
- Use "fetch" when the user wants to download/retrieve new papers from arXiv.
- Use "list" when the user wants to see saved/fetched papers with their IDs and titles.
- Use "rag" when the user wants to search or ask questions about papers.
- Use "summarize" when the user wants a summary or overview of papers.
- Use "compare" when the user wants to compare papers side by side.
- Use "tag" when the user wants to cluster, tag, or group papers by research theme.
- Use "digest" when the user wants a digest or overview of recent papers (e.g. "daily digest").
- Use "diagram" when the user wants a visual diagram or flowchart of a paper's methodology.
- Use "figures" when the user wants actual images/figures from the paper (not a generated diagram).
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
    intent: str  # "fetch" | "rag" | "summarize" | "compare" | "list" | "unknown"
    resolved_topics: list[str]  # parsed arXiv topic codes for fetch
    pending_chain: list[str]  # remaining steps in a multi-step request
    rag_query: str  # refined query for rag/summarize nodes
    last_result: str  # output from the last completed sub-agent
    pinned_paper: str  # title of a specific paper pinned via #arxiv_id (empty = no pin)
    pinned_papers: list[str]  # all titles pinned via #arxiv_id refs (used by compare_node)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_paper_ref(query: str) -> tuple[str, str, list[str]]:
    """Replace #<arxiv_id> references with paper titles.

    Returns (resolved_query, pinned_title, all_pinned_titles).
    - resolved_query: query with #id tokens replaced by quoted paper titles
    - pinned_title: non-empty only when exactly one paper is referenced (used by summarize_node)
    - all_pinned_titles: all resolved titles in order (used by compare_node)

    Matches patterns like #2301.12345v2 or #2301.12345.
    """
    pinned: list[str] = []

    def _replace(m: _re.Match) -> str:  # type: ignore[type-arg]
        paper = get_paper_by_arxiv_id(m.group(1))
        if paper:
            pinned.append(paper["title"])
            return f'"{paper["title"]}"'
        return m.group(0)

    # Match #2301.12345v2 or #2301.12345 (digits, dot, digits, optional version)
    resolved = _re.sub(r"#(\d{4}\.\d{4,5}(?:v\d+)?)", _replace, query)
    pinned_title = pinned[0] if len(pinned) == 1 else ""
    return resolved, pinned_title, pinned


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
        print(f"  [Supervisor] WARNING: LLM returned malformed routing JSON — {raw[:120]!r}")
        return {"steps": [], "topics": [], "rag_query": ""}


def _resolve_topics(raw_topics: list[str]) -> list[str]:
    """Map LLM-returned topic strings to valid arXiv category codes."""
    resolved = []
    for t in raw_topics:
        t_clean = t.strip()
        t_lower = t_clean.lower()
        if t_clean in _TOPIC_SET:
            resolved.append(t_clean)
        elif t_lower in _TOPIC_ALIASES:
            resolved.append(_TOPIC_ALIASES[t_lower])
    return list(dict.fromkeys(resolved))


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
    query, pinned_title, all_pinned = _resolve_paper_ref(_last_human_content(state))
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
        "pinned_paper": pinned_title,
        "pinned_papers": all_pinned,
    }


def ingestion_node(state: SupervisorState) -> dict:
    """Fetch paper metadata from arXiv, display results, and index into vector store."""
    topics = state.get("resolved_topics") or list(TOPICS)
    print(f"  [Supervisor] Fetching topics: {', '.join(topics)}")
    n = fetch_papers(topics=topics)
    result = (
        f"Fetched and indexed {n} new paper(s) for: {', '.join(topics)}. You can now search them."
        if n > 0
        else f"No new papers found for: {', '.join(topics)}."
    )
    return {"last_result": result, "messages": [AIMessage(content=result)]}


def list_node(state: SupervisorState) -> dict:
    """List all saved papers with their arXiv IDs and titles."""
    print("  [Supervisor] Listing saved papers...")
    result = list_papers()
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
    citations = _format_citations(result.get("retrieval_context", []))
    if citations:
        answer = f"{answer}\n\nSources:\n{citations}"
    return {"last_result": answer, "messages": [AIMessage(content=answer)]}


def summarize_node(state: SupervisorState) -> dict:
    """Retrieve papers and return an LLM-generated summary.

    If a specific paper was pinned via #N, retrieve it directly from the fetch
    session and summarize only that paper — skipping the broad similarity search.
    """
    query = state.get("rag_query") or _last_human_content(state)
    pinned = state.get("pinned_paper", "")

    if pinned:
        paper = get_paper_by_title(pinned)
        if paper:
            print(f"  [Supervisor] Summarizing pinned paper: {pinned}")
            prompt = (
                "You are a research assistant. Summarize the following paper in 5-7 bullet points, "
                "highlighting the problem it solves, key methods, and main contributions.\n\n"
                f"Title: {paper['title']}\n"
                f"Authors: {paper['authors']}\n"
                f"Abstract: {paper['abstract']}"
            )
            answer = str(llm_agent.invoke(prompt).content)
            # Prepend S2 TLDR if available
            s2_tldr = paper.get("s2_tldr")
            if s2_tldr:
                answer = f"TL;DR (Semantic Scholar): {s2_tldr}\n\n---\n\n{answer}"
            return {"last_result": answer, "messages": [AIMessage(content=answer)]}

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
        "- **Compare** papers (e.g. 'compare #2301.12345 and #2504.08123')\n"
        "- **Tag** papers by research theme (e.g. 'tag papers')\n"
        "- **Digest** recent papers (e.g. 'daily digest' or 'digest last 14 days')\n"
        "- **Diagram** a paper's methodology (e.g. 'diagram #2504.08123')\n"
        "- **Figures** extract real images from a paper (e.g. 'get figures from #2504.08123')\n"
        "- **Chain** steps (e.g. 'fetch ML papers then find the best on LLMs')"
    )
    return {"last_result": clarification, "messages": [AIMessage(content=clarification)]}


_COMPARE_PROMPT = """\
You are a research analyst comparing AI papers side by side.

{papers_section}

Produce a structured comparison with these sections:

## Comparison: {titles}

### Problem & Motivation
For each paper: what problem does it address and why does it matter?

### Approach & Methodology
For each paper: what is the core technical approach?

### Key Results & Contributions
For each paper: what are the headline results or contributions?

### Limitations & Future Work
For each paper: what are the stated or apparent limitations?

### When to Prefer Each
Give a clear, opinionated verdict: in what scenario would a researcher choose one over the other?
"""


def compare_node(state: SupervisorState) -> dict:
    """Compare two or more papers side by side with a structured LLM analysis.

    If papers are pinned via #arxiv_id, looks them up directly from JSONL.
    Otherwise falls back to hybrid search using the rag_query and compares the top results.
    """
    pinned = state.get("pinned_papers", [])
    query = state.get("rag_query") or _last_human_content(state)

    papers: list[dict] = []

    if len(pinned) >= 2:
        # Direct lookup for each pinned title
        for title in pinned:
            paper = get_paper_by_title(title)
            if paper:
                papers.append(paper)
        if len(papers) < 2:
            missing = len(pinned) - len(papers)
            return {
                "last_result": (
                    f"Could not find {missing} of the referenced paper(s). "
                    "Check the arXiv IDs and ensure the papers have been fetched."
                ),
                "messages": [
                    AIMessage(
                        content=f"Could not find {missing} of the referenced paper(s). "
                        "Check the arXiv IDs and ensure papers have been fetched."
                    )
                ],
            }
    else:
        # No pins or only one — search and take top results
        print(f"  [Supervisor] Compare: no papers pinned, searching for: {query}")
        docs = hybrid_search(papers_store, query, k=4, rerank=True)
        seen_titles: set[str] = set()
        for doc in docs:
            title = doc.metadata.get("title", "")
            if title and title not in seen_titles:
                paper = get_paper_by_title(title)
                if paper:
                    papers.append(paper)
                    seen_titles.add(title)
            if len(papers) >= 2:
                break

        if len(papers) < 2:
            result = (
                "Not enough papers found to compare. Try referencing papers directly "
                "with their arXiv IDs (e.g. 'compare #2301.12345 and #2504.08123'), "
                "or fetch more papers first."
            )
            return {"last_result": result, "messages": [AIMessage(content=result)]}

    print(f"  [Supervisor] Comparing {len(papers)} paper(s): {[p['title'] for p in papers]}")

    papers_section_parts = []
    for i, paper in enumerate(papers, 1):
        papers_section_parts.append(
            f"**Paper {i}: {paper['title']}**\n"
            f"Authors: {paper['authors']}\n"
            f"Published: {paper['published'][:10]}\n"
            f"Abstract: {paper['abstract']}"
        )
    papers_section = "\n\n---\n\n".join(papers_section_parts)

    short_titles = " vs ".join(
        p["title"][:50] + ("…" if len(p["title"]) > 50 else "") for p in papers
    )
    prompt = _COMPARE_PROMPT.format(papers_section=papers_section, titles=short_titles)
    answer = str(llm_agent.invoke(prompt).content)
    return {"last_result": answer, "messages": [AIMessage(content=answer)]}


_TAG_PROMPT = """\
You are a research analyst. Below is a list of AI paper titles and their arXiv categories.

{papers_list}

Group these papers into 4-7 concise research theme clusters (e.g. "Diffusion Models", \
"LLM Alignment", "Embodied Agents"). For each cluster:
- Give it a short, descriptive name (2-4 words)
- List the paper numbers that belong to it

Respond in this exact format (no extra text):

### <Theme Name>
- [<num>] <title>
...

### <Theme Name>
- [<num>] <title>
...
"""

_DIGEST_PROMPT = """\
You are a research newsletter editor. Below are AI papers published in the last {days} days, \
grouped by arXiv category.

{papers_section}

Write a concise research digest (newsletter style):
1. A one-paragraph executive summary of the most notable trends across all papers.
2. For each category, 2-3 bullet highlights (what problem, what approach, why it matters).

Keep the tone professional but accessible to a senior ML engineer. \
Total length: 300-500 words.
"""

_DIAGRAM_PROMPT = """\
You are a technical writer producing Mermaid diagrams for AI papers.

Title: {title}
Abstract: {abstract}

Generate a Mermaid flowchart (flowchart TD) that illustrates the paper's methodology: \
key inputs, processing steps, model components, and outputs. Use concise node labels \
(≤6 words each). Include at least 6 nodes.

Output ONLY the raw Mermaid code block — no explanation, no markdown prose:

```mermaid
flowchart TD
    ...
```
"""


def tag_node(state: SupervisorState) -> dict:
    """Cluster all fetched papers into research themes via a single LLM call.

    Loads all papers from JSONL, sends titles + categories to the LLM for batch
    clustering into 4-7 named themes, and prints the grouped result.
    """
    papers = get_recent_papers(days=365)  # tag across all fetched papers
    if not papers:
        result = "No papers found. Run a fetch first, then try tagging."
        return {"last_result": result, "messages": [AIMessage(content=result)]}

    print(f"  [Supervisor] Clustering {len(papers)} paper(s) into research themes...")

    # Prefer Semantic Scholar fields of study if available for >=80% of papers
    s2_covered = sum(1 for p in papers if p.get("s2_fields"))
    if s2_covered / len(papers) >= 0.8:
        print(f"  [Supervisor] Using S2 fields for {s2_covered} papers (no LLM call).")
        groups: dict[str, list[str]] = {}
        for p in papers:
            fields = p.get("s2_fields") or []
            primary = fields[0] if fields else "Other"
            groups.setdefault(primary, []).append(p["title"])
        lines = []
        for field, titles in sorted(groups.items()):
            lines.append(f"\n### {field}")
            for title in titles:
                lines.append(f"- {title}")
        answer = "\n".join(lines)
        return {"last_result": answer, "messages": [AIMessage(content=answer)]}

    lines = []
    for i, p in enumerate(papers, 1):
        cats = p.get("categories", "")
        lines.append(f"{i}. [{cats}] {p['title']}")
    papers_list = "\n".join(lines)

    prompt = _TAG_PROMPT.format(papers_list=papers_list)
    answer = str(llm_agent.invoke(prompt).content)
    return {"last_result": answer, "messages": [AIMessage(content=answer)]}


def digest_node(state: SupervisorState) -> dict:
    """Generate a newsletter-style digest of papers from the last N days.

    Defaults to 7 days. Optionally sends via email if SMTP env vars are set.
    """
    import os

    days = 7
    # Allow "last 14 days digest" etc. via rag_query hint
    query = state.get("rag_query") or _last_human_content(state)
    for token in query.split():
        if token.isdigit():
            days = max(1, min(int(token), 90))
            break

    papers = get_recent_papers(days=days)
    if not papers:
        result = (
            f"No papers found in the last {days} days. "
            "Try fetching first or increase the day range."
        )
        return {"last_result": result, "messages": [AIMessage(content=result)]}

    print(f"  [Supervisor] Generating digest for {len(papers)} paper(s) from last {days} day(s)...")

    # Group by primary arXiv category
    groups: dict[str, list[dict]] = {}
    for p in papers:
        primary_cat = p.get("categories", "unknown").split(",")[0].strip()
        groups.setdefault(primary_cat, []).append(p)

    section_parts = []
    for cat, cat_papers in sorted(groups.items()):
        lines = [f"**{cat}** ({len(cat_papers)} paper(s)):"]
        for p in cat_papers[:10]:  # cap per-category to keep prompt manageable
            lines.append(f"  - {p['title']} ({p.get('published', '')[:10]})")
        section_parts.append("\n".join(lines))
    papers_section = "\n\n".join(section_parts)

    prompt = _DIGEST_PROMPT.format(days=days, papers_section=papers_section)
    answer = str(llm_agent.invoke(prompt).content)

    # Optionally send via email
    email_to = os.getenv("DIGEST_EMAIL_TO", "")
    if email_to:
        _send_digest_email(answer, days, email_to)

    return {"last_result": answer, "messages": [AIMessage(content=answer)]}


def _send_digest_email(body: str, days: int, to_addr: str) -> None:
    """Send the digest via SMTP. Only called when DIGEST_EMAIL_TO is set."""
    import os
    import smtplib
    from email.mime.text import MIMEText

    host = os.getenv("DIGEST_SMTP_HOST", "")
    port = int(os.getenv("DIGEST_SMTP_PORT", "587"))
    user = os.getenv("DIGEST_SMTP_USER", "")
    password = os.getenv("DIGEST_SMTP_PASS", "")
    from_addr = os.getenv("DIGEST_EMAIL_FROM", user)

    if not host or not user or not password:
        print("  [Digest] Email skipped — DIGEST_SMTP_HOST/USER/PASS not configured.")
        return

    from datetime import date

    subject = f"arXiv AI Research Digest — last {days} days ({date.today()})"
    msg = MIMEText(body, "plain")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr

    try:
        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(from_addr, [to_addr], msg.as_string())
        print(f"  [Digest] Email sent to {to_addr}.")
    except Exception as e:
        print(f"  [Digest] Email failed: {e}")


def diagram_node(state: SupervisorState) -> dict:
    """Generate a Mermaid flowchart diagram of a paper's methodology.

    Requires a paper pinned via #arxiv_id. Falls back to the top search result
    if no paper is pinned.
    """
    pinned = state.get("pinned_paper", "")
    query = state.get("rag_query") or _last_human_content(state)

    paper: dict | None = None
    if pinned:
        paper = get_paper_by_title(pinned)
    if paper is None:
        docs = hybrid_search(papers_store, query, k=3, rerank=True)
        for doc in docs:
            title = doc.metadata.get("title", "")
            if title:
                paper = get_paper_by_title(title)
                if paper:
                    break

    if paper is None:
        result = (
            "No paper found to diagram. Reference one with its arXiv ID "
            "(e.g. 'diagram #2301.12345') or fetch papers first."
        )
        return {"last_result": result, "messages": [AIMessage(content=result)]}

    print(f"  [Supervisor] Generating Mermaid diagram for: {paper['title']}")
    prompt = _DIAGRAM_PROMPT.format(title=paper["title"], abstract=paper["abstract"])
    answer = str(llm_agent.invoke(prompt).content)
    return {"last_result": answer, "messages": [AIMessage(content=answer)]}


def figures_node(state: SupervisorState) -> dict:
    """Extract real figures from a paper (arXiv HTML first, PDF fallback).

    Requires a paper pinned via #arxiv_id. Falls back to a Mermaid diagram
    (using _DIAGRAM_PROMPT) if no figures can be found in HTML or PDF.
    """
    pinned = state.get("pinned_paper", "")

    if not pinned:
        result = (
            "Please specify a paper by its arXiv ID to extract figures "
            "(e.g. 'get figures from #2504.08123')."
        )
        return {"last_result": result, "messages": [AIMessage(content=result)]}

    paper = get_paper_by_title(pinned)
    if paper is None:
        result = (
            f"Could not find paper '{pinned}'. "
            "Check the arXiv ID and ensure the paper has been fetched."
        )
        return {"last_result": result, "messages": [AIMessage(content=result)]}

    arxiv_id = paper.get("arxiv_id", "")
    pdf_url = paper.get("pdf_url", "")
    print(f"  [Supervisor] Extracting figures for: {paper['title']}")

    content = fetch_paper_content(arxiv_id, pdf_url)

    if content and content.get("figures"):
        figures = content["figures"]
        source = content["source"]

        if source == "html":
            lines = [f"Found {len(figures)} figure(s) in paper (source: arXiv HTML):\n"]
            for i, fig in enumerate(figures, 1):
                caption = fig.get("caption") or "(no caption)"
                lines.append(f"[Figure {i}] {caption}")
                lines.append(f"  -> {fig['url']}\n")
            sections = content.get("sections", [])
            if sections:
                lines.append(f"Sections available: {', '.join(sections[:10])}")
        else:  # pdf
            lines = [f"Found {len(figures)} figure(s) extracted from PDF:\n"]
            for fig in figures:
                lines.append(f"[Page {fig['page']}] -> {fig['path']}")

        answer = "\n".join(lines)
        return {"last_result": answer, "messages": [AIMessage(content=answer)]}

    # Fallback to Mermaid diagram when no figures found
    print(f"  [Supervisor] No figures found for '{paper['title']}' — falling back to Mermaid.")
    fallback_note = (
        "No figures could be extracted from this paper (HTML unavailable, PDF has no images). "
        "Generating a Mermaid methodology diagram instead:\n\n"
    )
    prompt = _DIAGRAM_PROMPT.format(title=paper["title"], abstract=paper["abstract"])
    mermaid = str(llm_agent.invoke(prompt).content)
    answer = fallback_note + mermaid
    return {"last_result": answer, "messages": [AIMessage(content=answer)]}


_MAX_SUPERVISOR_MESSAGES = 20  # keep last N messages in the supervisor's conversation history


def finalize_node(state: SupervisorState) -> dict:
    """Trim conversation history to bound context growth across turns.

    Keeps the SystemMessage(s) and the most recent _MAX_SUPERVISOR_MESSAGES
    non-system messages. Older messages are deleted via RemoveMessage so the
    checkpointer doesn't restore them on the next turn.
    """
    msgs = state["messages"]
    other_msgs = [m for m in msgs if not isinstance(m, SystemMessage)]

    if len(other_msgs) <= _MAX_SUPERVISOR_MESSAGES:
        return {}

    to_remove = other_msgs[: len(other_msgs) - _MAX_SUPERVISOR_MESSAGES]
    removals = [RemoveMessage(id=m.id) for m in to_remove if getattr(m, "id", None)]
    if not removals:
        return {}

    kept = len(other_msgs) - len(to_remove)
    print(f"  [Supervisor] Trimmed {len(to_remove)} old message(s); keeping {kept}.")
    return {"messages": removals}


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------
def _dispatch_intent(state: SupervisorState) -> str:
    intent = state.get("intent", "unknown")
    if intent == "fetch":
        return "ingestion"
    if intent == "list":
        return "list"
    if intent == "rag":
        return "rag"
    if intent == "summarize":
        return "summarize"
    if intent == "compare":
        return "compare"
    if intent == "tag":
        return "tag"
    if intent == "digest":
        return "digest"
    if intent == "diagram":
        return "diagram"
    if intent == "figures":
        return "figures"
    return "clarify"


def _format_citations(context: list[str]) -> str:
    """Parse title + arxiv ID from retrieval_context blocks and format as a Sources list.

    Each block in retrieval_context is a formatted string from _format_docs:
        Title    : <title>
        ...
        URL      : https://arxiv.org/abs/<arxiv_id>
    """
    citations: list[str] = []
    for block in context:
        title_match = _re.search(r"Title\s*:\s*(.+)", block)
        url_match = _re.search(r"URL\s*:\s*(https?://\S+)", block)
        if not title_match:
            continue
        title = title_match.group(1).strip()
        arxiv_id = ""
        if url_match:
            id_match = _re.search(r"/abs/(\S+)", url_match.group(1))
            if id_match:
                arxiv_id = id_match.group(1)
        line = f"  [{arxiv_id}] {title}" if arxiv_id else f"  {title}"
        citations.append(line)
    return "\n".join(dict.fromkeys(citations))


def _route_after_capability(state: SupervisorState) -> str:
    """After a capability node: continue chain or finalize."""
    pending = state.get("pending_chain", [])
    if pending:
        return "route"  # loop back to handle next step
    return "finalize"


# ---------------------------------------------------------------------------
# Graph definition
# ---------------------------------------------------------------------------
def _build_supervisor_graph(checkpointer=None):
    graph = StateGraph(SupervisorState)

    graph.add_node("route", route_node)
    graph.add_node("ingestion", ingestion_node)
    graph.add_node("list", list_node)
    graph.add_node("rag", rag_node)
    graph.add_node("summarize", summarize_node)
    graph.add_node("compare", compare_node)
    graph.add_node("tag", tag_node)
    graph.add_node("digest", digest_node)
    graph.add_node("diagram", diagram_node)
    graph.add_node("figures", figures_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", _dispatch_intent)

    # After each capability, either chain to next step or finalize
    _CAPS = (
        "ingestion",
        "list",
        "rag",
        "summarize",
        "compare",
        "tag",
        "digest",
        "diagram",
        "figures",
        "clarify",
    )
    for cap in _CAPS:
        graph.add_conditional_edges(
            cap, _route_after_capability, {"route": "route", "finalize": "finalize"}
        )

    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

# Nodes whose LLM calls produce token-level output worth streaming to the user.
# (rag_node invokes a sub-graph — its tokens don't surface here; ingestion/list
# return brief status strings, not long LLM-generated text.)
_STREAMING_NODES = {"summarize", "clarify", "compare", "tag", "digest", "diagram", "figures"}

_DB_DIR = Path(__file__).parent.parent / "databases"


def launch_supervisor() -> None:
    print("\n[Supervisor Ready] Fetch, search, summarize, compare, tag, digest, or diagram papers.")
    print("Examples:")
    print("  - 'fetch recent robotics papers'")
    print("  - 'find papers on diffusion models'")
    print("  - 'summarize recent cs.AI papers'")
    print("  - 'compare #2301.12345 and #2504.08123'")
    print("  - 'tag papers'                        (cluster all papers by research theme)")
    print("  - 'daily digest'                      (recent papers digest, last 7 days)")
    print("  - 'digest last 14 days'               (custom day range)")
    print("  - 'diagram #2504.08123'               (Mermaid flowchart of methodology)")
    print("  - 'get figures from #2504.08123'      (extract real images from paper)")
    print("  - 'fetch NLP papers then find the best on transformers'")
    print("  - Type 'new session' to start a fresh conversation\n")

    with SqliteSaver.from_conn_string(str(_DB_DIR / "agent_memory.db")) as checkpointer:
        graph = _build_supervisor_graph(checkpointer)
        config: dict = {"configurable": {"thread_id": "default"}}

        while True:
            query = input("You: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                print("Exiting. Goodbye!")
                break
            if query.lower() in ("new session", "reset"):
                config = {"configurable": {"thread_id": _new_thread_id()}}
                print("  [Supervisor] Started a new session.\n")
                continue

            try:
                query = validate_user_input(query)

                # Include the system message only on the first turn for a thread.
                # The checkpointer accumulates messages across turns; subsequent
                # turns just append the new HumanMessage.
                checkpoint = checkpointer.get(config)
                has_history = checkpoint is not None
                initial_msgs = (
                    [HumanMessage(content=query)]
                    if has_history
                    else [_SYSTEM_MESSAGE, HumanMessage(content=query)]
                )

                initial_state = {
                    "messages": initial_msgs,
                    "intent": "",
                    "resolved_topics": [],
                    "pending_chain": [],
                    "rag_query": "",
                    "last_result": "",
                    "pinned_paper": "",
                    "pinned_papers": [],
                }

                # Stream both state snapshots (values) and LLM token chunks (messages).
                # tokens from summarize/clarify nodes are printed inline as they arrive;
                # results from rag/ingestion/list are printed after the stream ends.
                streamed_nodes: set[str] = set()
                final_result = ""

                for event_type, event_data in graph.stream(
                    initial_state, config=config, stream_mode=["values", "messages"]
                ):
                    if event_type == "messages":
                        chunk, meta = event_data
                        node = meta.get("langgraph_node", "")
                        if node in _STREAMING_NODES and hasattr(chunk, "content") and chunk.content:
                            if node not in streamed_nodes:
                                print("\nAgent: ", end="", flush=True)
                                streamed_nodes.add(node)
                            print(chunk.content, end="", flush=True)
                    elif event_type == "values":
                        r = event_data.get("last_result", "")
                        if r:
                            final_result = r

                if streamed_nodes:
                    print("\n")  # close inline token output
                elif final_result:
                    print(f"\nAgent: {final_result}\n")

            except InputRejected as e:
                print(f"Blocked: {e}\n")
            except KeyboardInterrupt:
                print("\nExiting. Goodbye!")
                break
            except Exception as e:
                print(f"Error ({type(e).__name__}): {e}\n")


def _new_thread_id() -> str:
    """Generate a unique thread ID for a fresh session."""
    import time

    return f"session-{int(time.time())}"
