"""
Streamlit web UI for the Multi-Agent HCI Research System.

Run with:  streamlit run src/ui/streamlit_app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import time
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from src.autogen_orchestrator import run_research
from src.guardrails.safety_manager import safety_manager
from src.evaluation.judge import run_both_judges, format_judge_results

# =========================================================================== #
# Page config
# =========================================================================== #
st.set_page_config(
    page_title="HCI Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================================== #
# Sidebar
# =========================================================================== #
with st.sidebar:
    st.title("🔬 HCI Research Agent")
    st.caption("Multi-Agent Deep Research System")
    st.divider()

    st.subheader("⚙️ Settings")
    run_judge = st.toggle("Run LLM-as-a-Judge", value=True)
    show_raw_json = st.toggle("Show raw JSON export", value=False)

    st.divider()
    st.subheader("💡 Example Queries")
    examples = [
        "What are the main challenges of designing explainable AI for non-expert users?",
        "How does AR improve usability in industrial training?",
        "What evaluation methods work best for conversational AI UX?",
        "How does cognitive load theory apply to dashboard design?",
        "What are ethical considerations in AI-driven UI personalization?",
    ]
    for ex in examples:
        if st.button(ex[:60] + "...", key=ex, use_container_width=True):
            st.session_state["prefill_query"] = ex

    st.divider()
    st.subheader("🛡️ Safety Log")
    log_summary = safety_manager.get_log_summary()
    col1, col2 = st.columns(2)
    col1.metric("Blocked", log_summary["blocked_count"])
    col2.metric("Sanitized", log_summary["sanitized_count"])

    if st.button("Clear safety log"):
        safety_manager.clear_log()
        st.rerun()


# =========================================================================== #
# Main area
# =========================================================================== #
st.title("🧠 Multi-Agent HCI Research System")
st.caption("Powered by LangGraph · Planner → Researcher → Critic → Writer")

# Query input
prefill = st.session_state.pop("prefill_query", "")
query = st.text_area(
    "Enter your research question:",
    value=prefill,
    height=100,
    placeholder="e.g. What are the main challenges of designing explainable AI interfaces for non-expert users?",
)

col_run, col_clear = st.columns([1, 4])
run_btn = col_run.button("🚀 Research", type="primary", use_container_width=True)
if col_clear.button("🗑️ Clear", use_container_width=True):
    st.session_state.pop("last_result", None)
    st.rerun()

# =========================================================================== #
# Run pipeline
# =========================================================================== #
if run_btn and query.strip():

    # ── Safety check ──────────────────────────────────────────────────────
    safety_result = safety_manager.validate_input(query)

    if not safety_result["allowed"]:
        st.error(f"🚫 **Query blocked by safety policy**")
        st.warning(
            f"**Policy category:** `{safety_result['category']}`\n\n"
            f"**Reason:** {safety_result['reason']}"
        )
        st.info("Please revise your query to focus on HCI/AI research topics.")
        st.stop()

    if safety_result["category"] != "SAFE":
        st.warning(f"⚠️ Query was sanitized (`{safety_result['category']}`). Proceeding with cleaned input.")

    safe_query = safety_result["sanitized_query"]

    # ── Agent pipeline ─────────────────────────────────────────────────────
    with st.status("🤖 Agents working...", expanded=True) as status:
        st.write("**[Planner]** Decomposing your query...")
        t0 = time.time()
        result = run_research(safe_query)
        elapsed = round(time.time() - t0, 1)

        if result.get("error"):
            status.update(label="❌ Pipeline error", state="error")
            st.error(f"Error: {result['error']}")
            st.stop()

        # Output safety check
        out_safety = safety_manager.validate_output(
            result.get("final_answer", ""), query_context=safe_query
        )
        if not out_safety["allowed"]:
            status.update(label="🚫 Output blocked", state="error")
            st.error("**Output blocked by safety policy.**")
            st.warning(f"Category: `{out_safety['category']}`\n\n{out_safety['reason']}")
            st.stop()

        result["final_answer"] = out_safety["sanitized_text"]
        if out_safety["category"] not in ("SAFE",):
            st.warning(f"⚠️ Output sanitized: `{out_safety['category']}`")

        status.update(label=f"✅ Research complete in {elapsed}s", state="complete")

    st.session_state["last_result"] = result
    st.session_state["run_judge"] = run_judge


# =========================================================================== #
# Display results
# =========================================================================== #
if "last_result" in st.session_state:
    result = st.session_state["last_result"]

    tab_answer, tab_trace, tab_citations, tab_judge = st.tabs([
        "📝 Answer", "🔍 Agent Trace", "📚 Citations", "⚖️ Evaluation"
    ])

    # ── Answer tab ─────────────────────────────────────────────────────────
    with tab_answer:
        st.subheader("Research Answer")

        if result.get("sub_questions"):
            with st.expander("🗂️ Sub-questions generated by Planner"):
                for i, q in enumerate(result["sub_questions"], 1):
                    st.markdown(f"{i}. {q}")

        st.markdown(result.get("final_answer", "_No answer generated._"))

        # Download button
        st.download_button(
            "⬇️ Download answer (.md)",
            data=result.get("final_answer", ""),
            file_name="research_answer.md",
            mime="text/markdown",
        )

    # ── Trace tab ──────────────────────────────────────────────────────────
    with tab_trace:
        st.subheader("Agent Execution Trace")
        trace = result.get("trace", [])
        if trace:
            for i, step in enumerate(trace):
                agent_name = "Agent"
                if "[Planner]" in step:
                    agent_name = "🗺️ Planner"
                elif "[Researcher]" in step:
                    agent_name = "🔍 Researcher"
                elif "[Critic]" in step:
                    agent_name = "🧐 Critic"
                elif "[Writer]" in step:
                    agent_name = "✍️ Writer"
                with st.expander(f"Step {i+1}: {agent_name}", expanded=(i == 0)):
                    st.markdown(step)
        else:
            st.info("No trace available.")

        if result.get("critique"):
            st.subheader("Critic Notes")
            st.info(result["critique"])

    # ── Citations tab ──────────────────────────────────────────────────────
    with tab_citations:
        st.subheader("Sources & Citations")
        citations = result.get("citations", [])
        if citations:
            web = [c for c in citations if c.get("source_type") in ("web", "web_mock")]
            acad = [c for c in citations if c.get("source_type") in ("academic", "academic_mock")]

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Sources", len(citations))
            col2.metric("Web", len(web))
            col3.metric("Academic", len(acad))

            if web:
                st.subheader("🌐 Web Sources")
                for c in web:
                    with st.expander(f"[{c.get('citation_number', '?')}] {c.get('title', 'Untitled')}"):
                        st.markdown(f"**URL:** {c.get('url', 'N/A')}")
                        st.markdown(f"**Content:** {c.get('content', '')[:300]}...")

            if acad:
                st.subheader("📄 Academic Papers")
                for c in acad:
                    with st.expander(f"[P{c.get('citation_number', '?')}] {c.get('title', 'Untitled')} ({c.get('year', '')})"):
                        st.markdown(f"**Authors:** {c.get('authors', 'N/A')}")
                        st.markdown(f"**Venue:** {c.get('venue', 'N/A')}")
                        st.markdown(f"**Citations:** {c.get('citations', 0)}")
                        if c.get("abstract"):
                            st.markdown(f"**Abstract:** {c['abstract'][:300]}...")
                        st.markdown(f"**URL:** {c.get('url', 'N/A')}")
        else:
            st.info("No citations collected.")

    # ── Evaluation tab ─────────────────────────────────────────────────────
    with tab_judge:
        st.subheader("LLM-as-a-Judge Evaluation")

        if st.session_state.get("run_judge"):
            if "judge_results" not in st.session_state:
                with st.spinner("Running judges..."):
                    judge_results = run_both_judges(
                        result["query"],
                        result.get("final_answer", "")
                    )
                    st.session_state["judge_results"] = judge_results
            else:
                judge_results = st.session_state["judge_results"]

            # Display scores
            sb = judge_results.get("scores_breakdown", {})
            composite = judge_results.get("composite_score", 0)

            st.metric("Composite Score", f"{composite:.1f} / 5.0")

            cols = st.columns(5)
            labels = ["Relevance", "Evidence", "Accuracy", "Clarity", "Safety"]
            keys = ["relevance", "evidence", "accuracy", "clarity", "safety"]
            for col, label, key in zip(cols, labels, keys):
                col.metric(label, f"{sb.get(key, 0)}/5")

            st.markdown(format_judge_results(judge_results))

            if show_raw_json:
                st.json(judge_results)
        else:
            if st.button("▶️ Run judges now"):
                with st.spinner("Evaluating..."):
                    judge_results = run_both_judges(
                        result["query"],
                        result.get("final_answer", "")
                    )
                    st.session_state["judge_results"] = judge_results
                    st.rerun()

    # ── Raw JSON export ────────────────────────────────────────────────────
    if show_raw_json:
        with st.expander("📦 Raw session JSON"):
            export = {k: v for k, v in result.items() if k != "final_answer"}
            export["final_answer_preview"] = result.get("final_answer", "")[:500]
            st.json(export)
            st.download_button(
                "⬇️ Download full session (.json)",
                data=json.dumps(result, indent=2),
                file_name="session_export.json",
                mime="application/json",
            )

elif not run_btn:
    st.info("👆 Enter a research question above and click **Research** to get started.")

    st.subheader("🏗️ System Architecture")
    st.markdown("""
    | Agent | Role |
    |-------|------|
    | 🗺️ **Planner** | Decomposes query into 3–4 focused sub-questions |
    | 🔍 **Researcher** | Searches web (Tavily) + academic papers (Semantic Scholar) |
    | 🧐 **Critic** | Scores evidence quality; requests re-search if needed |
    | ✍️ **Writer** | Synthesizes cited, structured answer |
    
    **Safety:** Input/output guardrails check for harmful content, prompt injection, and PII.
    """)
