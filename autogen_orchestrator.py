"""
Multi-agent orchestrator using LangGraph.

Workflow:
  user query
      │
      ▼
  [Planner]  — decomposes query into 3-4 sub-questions
      │
      ▼
  [Researcher] — web + paper search for each sub-question  (loops up to 2x)
      │
      ▼
  [Critic]   — scores evidence quality, flags gaps
      │ (if gaps flagged) ──► back to Researcher (once)
      │ (otherwise)
      ▼
  [Writer]   — synthesizes cited final answer
      │
      ▼
  final answer + citations + trace
"""

import os
import json
import time
from typing import TypedDict, Annotated, Any
import operator

# ── LangGraph ──────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ── Local tools ────────────────────────────────────────────────────────────
from src.tools.web_search import web_search, format_search_results
from src.tools.paper_search import paper_search, format_paper_results
from src.tools.citation_tool import CitationTracker, inject_inline_citations, build_citation_summary


# =========================================================================== #
# State schema
# =========================================================================== #

class ResearchState(TypedDict):
    query: str                          # original user query
    sub_questions: list[str]            # from Planner
    web_results: list[dict]             # raw web results
    paper_results: list[dict]           # raw paper results
    evidence_text: str                  # formatted evidence for Writer
    critique: str                       # Critic's notes
    needs_more_research: bool           # Critic flag
    research_iterations: int            # loop counter (max 2)
    final_answer: str                   # Writer output
    citations: list[dict]               # structured citation list
    trace: Annotated[list[str], operator.add]  # agent trace log
    error: str                          # error message if any


# =========================================================================== #
# LLM client builder
# =========================================================================== #

def _strip_think(text: str) -> str:
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


def _call_llm(llm, system: str, user: str) -> str:
    """Call LLM via OpenAI client directly with thinking disabled."""
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY", ""),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    try:
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "Qwen/Qwen3-8B"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=1500,
            temperature=0.3,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = resp.choices[0].message.content or ""
        return _strip_think(raw)
    except Exception as e:
        return f"[LLM ERROR: {e}]"


def _build_llm(temperature: float = 0.3):
    return None  # calls go through _call_llm directly


# =========================================================================== #
# Agent node functions
# =========================================================================== #

def planner_node(state: ResearchState) -> dict:
    """Decompose the user query into 4 focused sub-questions."""
    llm = _build_llm(temperature=0.2)
    system = (
        "/no_think\n"
        "You are a research planning assistant for HCI and AI. "
        "Output ONLY a numbered list of exactly 4 sub-questions. Example format:\n"
        "1. First sub-question here\n"
        "2. Second sub-question here\n"
        "3. Third sub-question here\n"
        "4. Fourth sub-question here\n"
        "No preamble, no explanation, nothing else."
    )
    user = "/no_think\nDecompose into 4 numbered sub-questions:\n" + state["query"]
    raw = _call_llm(llm, system, user)

    # Parse numbered lines robustly — skip any think tag leakage
    import re as _re
    sub_questions = []
    for line in raw.splitlines():
        line = line.strip()
        m = _re.match(r"^\d+[\.\'\)]\s*(.+)", line)
        if m:
            q = m.group(1).strip()
            if not q.startswith("<") and len(q) > 10:
                sub_questions.append(q)

    # Guaranteed fallback
    if len(sub_questions) < 2:
        orig = state["query"]
        sub_questions = [
            f"What are the main user-facing challenges of: {orig}",
            f"What technical barriers exist for: {orig}",
            f"What design patterns or frameworks address: {orig}",
            f"What empirical research evaluates approaches to: {orig}",
        ]

    trace_msg = (
        f"**[Planner]** Decomposed query into {len(sub_questions)} sub-questions:\n"
        + "\n".join(f"  {i+1}. {q}" for i, q in enumerate(sub_questions))
    )
    return {"sub_questions": sub_questions, "trace": [trace_msg]}


def researcher_node(state: ResearchState) -> dict:
    """Search web + Semantic Scholar for each sub-question."""
    all_web: list[dict] = []
    all_papers: list[dict] = []
    trace_lines = [f"**[Researcher]** Iteration {state['research_iterations'] + 1}"]

    for q in state["sub_questions"]:
        # Web search
        web = web_search(q, max_results=3)
        all_web.extend(web)
        trace_lines.append(f"  🌐 Web search: '{q[:60]}' → {len(web)} results")

        # Paper search
        papers = paper_search(q, max_results=3)
        all_papers.extend(papers)
        trace_lines.append(f"  📄 Paper search: '{q[:60]}' → {len(papers)} results")

        time.sleep(0.3)  # gentle rate limiting

    # Deduplicate by URL
    seen_urls = set()
    deduped_web = []
    for r in all_web:
        url = r.get("url", "")
        if url not in seen_urls:
            deduped_web.append(r)
            seen_urls.add(url)

    seen_ids = set()
    deduped_papers = []
    for p in all_papers:
        pid = p.get("paper_id") or p.get("title", "")
        if pid not in seen_ids:
            deduped_papers.append(p)
            seen_ids.add(pid)

    # Format evidence text
    evidence = "## Web Sources\n\n"
    evidence += format_search_results(deduped_web[:8])
    evidence += "\n\n## Academic Sources\n\n"
    evidence += format_paper_results(deduped_papers[:6])

    trace_lines.append(
        f"  ✅ Total evidence: {len(deduped_web)} web + {len(deduped_papers)} papers"
    )

    return {
        "web_results": deduped_web,
        "paper_results": deduped_papers,
        "evidence_text": evidence,
        "research_iterations": state["research_iterations"] + 1,
        "trace": ["\n".join(trace_lines)],
    }


def critic_node(state: ResearchState) -> dict:
    """Evaluate evidence quality and decide if more research is needed."""
    llm = _build_llm(temperature=0.1)
    system = (
        "/no_think\nYou are a critical research evaluator. "
        "Review the evidence collected and assess whether it adequately answers the query. "
        "Return a JSON object with keys: "
        "\"quality_score\" (1-5), \"gaps\" (list of missing aspects), "
        "\"needs_more_research\" (boolean), \"summary\" (one sentence)."
    )
    user = (
        f"/no_think\nOriginal query: {state['query']}\n\n"
        f"Evidence collected:\n{state['evidence_text'][:3000]}\n\n"
        "Evaluate the evidence quality."
    )
    raw = _call_llm(llm, system, user)

    try:
        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        result = json.loads(clean)
        quality = result.get("quality_score", 3)
        gaps = result.get("gaps", [])
        needs_more = result.get("needs_more_research", False) and state["research_iterations"] < 2
        summary = result.get("summary", "Evidence evaluated.")
    except Exception:
        quality = 3
        gaps = []
        needs_more = False
        summary = raw[:200]

    critique = f"Quality score: {quality}/5. {summary}"
    if gaps:
        critique += f"\nGaps identified: {', '.join(gaps[:3])}"

    trace_msg = (
        f"**[Critic]** Evidence quality: {quality}/5. "
        f"Needs more research: {needs_more}. {summary}"
    )
    return {
        "critique": critique,
        "needs_more_research": needs_more,
        "trace": [trace_msg],
    }


def writer_node(state: ResearchState) -> dict:
    """Synthesize all evidence into a well-cited final answer."""
    llm = _build_llm(temperature=0.4)

    # Build citation tracker
    tracker = CitationTracker()
    for r in state["web_results"]:
        tracker.add(r)
    for p in state["paper_results"]:
        tracker.add(p)

    system = (
        "/no_think\nYou are a research writer specializing in HCI and AI. "
        "Write a comprehensive, well-structured answer to the research query. "
        "Use the evidence provided. Where you cite a source, note it as [Web N] or [Paper N] "
        "referencing the numbered sources in the evidence. "
        "Organize with clear headings. Aim for 400-700 words."
    )
    user = (
        f"/no_think\nResearch query: {state['query']}\n\n"
        f"Critic notes: {state['critique']}\n\n"
        f"Evidence:\n{state['evidence_text'][:4000]}\n\n"
        "Write a comprehensive research synthesis."
    )
    answer = _call_llm(llm, system, user)

    # Append citation list
    citation_data = build_citation_summary(tracker)
    final = inject_inline_citations(answer, citation_data["sources"])

    trace_msg = (
        f"**[Writer]** Synthesized final answer ({len(answer.split())} words). "
        f"Cited {citation_data['total']} sources "
        f"({citation_data['web_count']} web, {citation_data['academic_count']} academic)."
    )
    return {
        "final_answer": final,
        "citations": citation_data["sources"],
        "trace": [trace_msg],
    }


# =========================================================================== #
# Routing
# =========================================================================== #

def _route_after_critic(state: ResearchState) -> str:
    if state.get("needs_more_research") and state["research_iterations"] < 2:
        return "researcher"
    return "writer"


# =========================================================================== #
# Graph construction
# =========================================================================== #

def build_graph() -> StateGraph:
    g = StateGraph(ResearchState)

    g.add_node("planner", planner_node)
    g.add_node("researcher", researcher_node)
    g.add_node("critic", critic_node)
    g.add_node("writer", writer_node)

    g.set_entry_point("planner")
    g.add_edge("planner", "researcher")
    g.add_edge("researcher", "critic")
    g.add_conditional_edges("critic", _route_after_critic, {
        "researcher": "researcher",
        "writer": "writer",
    })
    g.add_edge("writer", END)

    return g.compile()


# =========================================================================== #
# Main entry point
# =========================================================================== #

def run_research(query: str) -> dict:
    """
    Run the full multi-agent research pipeline on a query.

    Returns:
      {
        "query": str,
        "final_answer": str,
        "citations": list[dict],
        "trace": list[str],
        "sub_questions": list[str],
        "critique": str,
      }
    """
    graph = build_graph()

    initial_state: ResearchState = {
        "query": query,
        "sub_questions": [],
        "web_results": [],
        "paper_results": [],
        "evidence_text": "",
        "critique": "",
        "needs_more_research": False,
        "research_iterations": 0,
        "final_answer": "",
        "citations": [],
        "trace": [],
        "error": "",
    }

    try:
        final_state = graph.invoke(initial_state)
    except Exception as e:
        return {
            "query": query,
            "final_answer": f"Research pipeline error: {e}",
            "citations": [],
            "trace": [f"[ERROR] {e}"],
            "sub_questions": [],
            "critique": "",
            "error": str(e),
        }

    return {
        "query": query,
        "final_answer": final_state.get("final_answer", ""),
        "citations": final_state.get("citations", []),
        "trace": final_state.get("trace", []),
        "sub_questions": final_state.get("sub_questions", []),
        "critique": final_state.get("critique", ""),
        "error": final_state.get("error", ""),
    }
