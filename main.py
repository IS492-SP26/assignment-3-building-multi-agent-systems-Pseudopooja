"""
main.py — entry points for the Multi-Agent HCI Research System.

Usage:
  python main.py                    # AutoGen example (default demo)
  python main.py --mode autogen     # Same as above
  python main.py --mode cli         # Interactive CLI
  python main.py --mode web         # Launch Streamlit UI
  python main.py --mode evaluate    # Run batch evaluation
  python main.py --mode demo        # Single end-to-end demo with judge output
"""

import argparse
import os
import sys
import json
import subprocess
from dotenv import load_dotenv

load_dotenv()


def mode_autogen():
    """Quick demo: run one query end-to-end and print results."""
    from src.autogen_orchestrator import run_research
    from src.guardrails.safety_manager import safety_manager
    from src.evaluation.judge import run_both_judges, format_judge_results

    query = (
        "What are the main challenges of designing explainable AI interfaces "
        "for non-expert users, and what design patterns have been proposed?"
    )
    print(f"\n{'='*60}")
    print("Multi-Agent HCI Research System — Demo")
    print(f"{'='*60}")
    print(f"Query: {query}\n")

    # Safety check
    safety = safety_manager.validate_input(query)
    print(f"[Safety] Input check: {safety['category']}")

    # Research
    print("\n[Pipeline] Running agents...\n")
    result = run_research(query)

    print("── Agent Trace ──────────────────────────────────────────")
    for step in result.get("trace", []):
        print(step)
        print()

    print("── Final Answer ─────────────────────────────────────────")
    print(result.get("final_answer", "No answer generated."))

    print(f"\n── Citations ({len(result.get('citations', []))}) ──────────────────────────────")
    for c in result.get("citations", [])[:5]:
        print(f"  [{c.get('citation_number')}] {c.get('title')} — {c.get('url', '')[:60]}")

    # Output safety
    out_safety = safety_manager.validate_output(result.get("final_answer", ""))
    print(f"\n[Safety] Output check: {out_safety['category']}")

    # Judge
    print("\n── LLM-as-a-Judge ───────────────────────────────────────")
    judge = run_both_judges(query, result.get("final_answer", ""))
    print(format_judge_results(judge))

    # Export
    os.makedirs("outputs", exist_ok=True)
    export_path = "outputs/demo_session.json"
    with open(export_path, "w") as f:
        json.dump({**result, "judge_results": judge}, f, indent=2)
    print(f"\n📁 Session exported to {export_path}")


def mode_cli():
    """Interactive CLI research loop."""
    from src.autogen_orchestrator import run_research
    from src.guardrails.safety_manager import safety_manager
    from src.evaluation.judge import run_both_judges, format_judge_results

    print("\n🔬 HCI Research Agent — Interactive CLI")
    print("Type 'quit' to exit, 'log' to see safety log, 'eval' to run batch evaluation.\n")

    while True:
        try:
            query = input("🔍 Enter research question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if query.lower() == "log":
            import json
            print(json.dumps(safety_manager.get_log(), indent=2))
            continue
        if query.lower() == "eval":
            mode_evaluate()
            continue

        # Safety check
        safety = safety_manager.validate_input(query)
        if not safety["allowed"]:
            print(f"\n🚫 Blocked [{safety['category']}]: {safety['reason']}\n")
            continue

        print("\n⏳ Running research pipeline...\n")
        result = run_research(safety["sanitized_query"])

        print("── Trace ────────────────────────────────────────────────")
        for step in result.get("trace", []):
            print(step)

        print("\n── Answer ───────────────────────────────────────────────")
        print(result.get("final_answer", "No answer."))

        print(f"\n── Sources ({len(result.get('citations', []))}) ──")
        for c in result.get("citations", [])[:5]:
            print(f"  [{c.get('citation_number')}] {c.get('title', '')[:80]}")

        # Judge
        do_judge = input("\n⚖️  Run LLM judge? (y/N): ").strip().lower()
        if do_judge == "y":
            print("Running judges...")
            judge = run_both_judges(query, result.get("final_answer", ""))
            print(format_judge_results(judge))

        print()


def mode_web():
    """Launch Streamlit web UI."""
    script = os.path.join("src", "ui", "streamlit_app.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", script], check=True)


def mode_evaluate():
    """Run batch evaluation on all queries in data/example_queries.json."""
    from src.evaluation.evaluator import run_evaluation, load_eval_queries
    queries = load_eval_queries()
    print(f"\n📋 Loaded {len(queries)} evaluation queries.")
    run_evaluation(queries)


def mode_demo():
    """Full end-to-end demo with rich output — same as autogen mode."""
    mode_autogen()


# =========================================================================== #
# Entry point
# =========================================================================== #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent HCI Research System")
    parser.add_argument(
        "--mode",
        choices=["autogen", "cli", "web", "evaluate", "demo"],
        default="autogen",
        help="Run mode (default: autogen)",
    )
    args = parser.parse_args()

    modes = {
        "autogen": mode_autogen,
        "cli": mode_cli,
        "web": mode_web,
        "evaluate": mode_evaluate,
        "demo": mode_demo,
    }
    modes[args.mode]()
