"""
Batch evaluator: runs the research system on multiple queries and aggregates judge scores.
"""

import os
import json
import time
import datetime
from pathlib import Path


def load_eval_queries(path: str = "data/example_queries.json") -> list[dict]:
    """Load evaluation queries from a JSON file."""
    p = Path(path)
    if not p.exists():
        # Fallback inline queries
        return _default_queries()
    with open(p) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("queries", _default_queries())


def run_evaluation(queries = None, output_dir: str = "outputs") -> dict:
    """
    Run full evaluation pipeline: research + judge for each query.
    Saves results to outputs/eval_results_<timestamp>.json.
    Returns aggregate report dict.
    """
    # Import here to avoid circular deps at module level
    from src.autogen_orchestrator import run_research
    from src.evaluation.judge import run_both_judges
    from src.guardrails.safety_manager import safety_manager

    if queries is None:
        queries = load_eval_queries()

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    results = []
    all_scores = {
        "relevance": [], "evidence": [], "accuracy": [],
        "clarity": [], "safety": [], "composite": []
    }

    print(f"\n{'='*60}")
    print(f"Running evaluation on {len(queries)} queries")
    print(f"{'='*60}\n")

    for i, q_item in enumerate(queries, 1):
        query = q_item.get("query", q_item) if isinstance(q_item, dict) else q_item
        print(f"[{i}/{len(queries)}] Query: {query[:80]}...")

        # Safety check
        safety_result = safety_manager.validate_input(query)
        if not safety_result["allowed"]:
            print(f"  ⚠️  Blocked by safety: {safety_result['category']}")
            results.append({
                "query": query,
                "blocked": True,
                "safety_category": safety_result["category"],
                "judge_results": None,
            })
            continue

        # Run research
        t0 = time.time()
        research_result = run_research(query)
        elapsed = round(time.time() - t0, 1)
        print(f"  ✅ Research done in {elapsed}s")

        # Run judges
        judge_results = run_both_judges(query, research_result.get("final_answer", ""))
        sb = judge_results.get("scores_breakdown", {})
        print(f"  📊 Scores: {sb}")

        # Accumulate
        for k in ["relevance", "evidence", "accuracy", "clarity", "safety"]:
            v = sb.get(k, 0)
            if v > 0:
                all_scores[k].append(v)
        c = judge_results.get("composite_score", 0)
        if c > 0:
            all_scores["composite"].append(c)

        results.append({
            "query": query,
            "blocked": False,
            "research": {
                "final_answer": research_result.get("final_answer", ""),
                "sub_questions": research_result.get("sub_questions", []),
                "citation_count": len(research_result.get("citations", [])),
                "elapsed_seconds": elapsed,
            },
            "judge_results": judge_results,
        })

        time.sleep(1)  # rate limit

    # Aggregate stats
    def avg(lst):
        return round(sum(lst) / len(lst), 2) if lst else 0.0

    aggregate = {
        "total_queries": len(queries),
        "completed": len([r for r in results if not r.get("blocked")]),
        "blocked": len([r for r in results if r.get("blocked")]),
        "mean_scores": {k: avg(v) for k, v in all_scores.items()},
        "timestamp": timestamp,
    }

    report = {"aggregate": aggregate, "results": results}

    # Save
    out_path = os.path.join(output_dir, f"eval_results_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n📁 Results saved to {out_path}")

    _print_summary(aggregate)
    return report


def _print_summary(agg: dict):
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total queries: {agg['total_queries']}")
    print(f"Completed: {agg['completed']} | Blocked: {agg['blocked']}")
    print("\nMean scores (1–5):")
    for k, v in agg["mean_scores"].items():
        bar = "█" * int(v) + "░" * (5 - int(v))
        print(f"  {k:<12} {bar} {v}")
    print(f"{'='*60}\n")


def _default_queries() -> list[dict]:
    return [
        {"query": "What are the main challenges of designing explainable AI interfaces for non-expert users?"},
        {"query": "How does augmented reality improve usability in industrial training applications?"},
        {"query": "What evaluation methods are most effective for measuring UX in conversational AI systems?"},
        {"query": "What are the ethical considerations in AI-powered personalization of user interfaces?"},
        {"query": "How do cognitive load theories apply to dashboard design in data visualization?"},
        {"query": "What research exists on accessibility features in large language model interfaces?"},
        {"query": "How does agentic AI change the role of the user in human-computer interaction?"},
    ]
