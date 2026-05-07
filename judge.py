"""
LLM-as-a-Judge: scores system outputs using two independent rubrics.
Uses OpenAI client directly to support extra_body for disabling Qwen3 thinking.
"""

import os
import json
import re


def _call_judge_llm(system: str, user: str) -> str:
    """Call the judge LLM with thinking disabled."""
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
            max_tokens=800,
            temperature=0.1,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = resp.choices[0].message.content or ""
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        raw = re.sub(r"<think>.*", "", raw, flags=re.DOTALL)
        return raw.strip()
    except Exception as e:
        return '{}'


JUDGE_1_SYSTEM = """You are a research quality evaluator. Score the response on THREE criteria (1-5 each).
Return ONLY a JSON object like this example - no preamble, no explanation:
{"relevance_score": 4, "evidence_score": 3, "accuracy_score": 4, "relevance_feedback": "Good coverage", "evidence_feedback": "Some citations", "accuracy_feedback": "Mostly accurate", "overall_research_score": 3.67}"""

JUDGE_2_SYSTEM = """You are a communication quality evaluator. Score the response on TWO criteria (1-5 each).
Return ONLY a JSON object like this example - no preamble, no explanation:
{"clarity_score": 4, "safety_score": 5, "clarity_feedback": "Well structured", "safety_feedback": "No unsafe content", "overall_communication_score": 4.5}"""


def run_judge_1(query: str, response: str) -> dict:
    user = f"Query: {query}\n\nResponse:\n{response[:3000]}\n\nReturn JSON scores only."
    try:
        raw = _call_judge_llm(JUDGE_1_SYSTEM, user)
        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        result = json.loads(clean)
        result["judge"] = "research_quality"
        result["status"] = "success"
        return result
    except Exception as e:
        return {"judge": "research_quality", "status": "error", "error": str(e),
                "relevance_score": 0, "evidence_score": 0, "accuracy_score": 0, "overall_research_score": 0.0}


def run_judge_2(query: str, response: str) -> dict:
    user = f"Query: {query}\n\nResponse:\n{response[:3000]}\n\nReturn JSON scores only."
    try:
        raw = _call_judge_llm(JUDGE_2_SYSTEM, user)
        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        result = json.loads(clean)
        result["judge"] = "communication_safety"
        result["status"] = "success"
        return result
    except Exception as e:
        return {"judge": "communication_safety", "status": "error", "error": str(e),
                "clarity_score": 0, "safety_score": 0, "overall_communication_score": 0.0}


def run_both_judges(query: str, response: str) -> dict:
    j1 = run_judge_1(query, response)
    j2 = run_judge_2(query, response)
    scores = [j1.get("relevance_score", 0), j1.get("evidence_score", 0),
              j1.get("accuracy_score", 0), j2.get("clarity_score", 0), j2.get("safety_score", 0)]
    valid = [s for s in scores if s > 0]
    composite = round(sum(valid) / len(valid), 2) if valid else 0.0
    return {
        "query": query,
        "judge_1_research": j1,
        "judge_2_communication": j2,
        "composite_score": composite,
        "scores_breakdown": {
            "relevance": j1.get("relevance_score", 0),
            "evidence": j1.get("evidence_score", 0),
            "accuracy": j1.get("accuracy_score", 0),
            "clarity": j2.get("clarity_score", 0),
            "safety": j2.get("safety_score", 0),
        }
    }


def format_judge_results(results: dict) -> str:
    j1 = results.get("judge_1_research", {})
    j2 = results.get("judge_2_communication", {})
    sb = results.get("scores_breakdown", {})
    lines = [
        "### LLM-as-a-Judge Evaluation Results", "",
        f"**Composite Score: {results.get('composite_score', 0):.1f} / 5.0**", "",
        "| Criterion | Score | Feedback |",
        "|-----------|-------|----------|",
        f"| Relevance & Coverage | {sb.get('relevance', 0)}/5 | {j1.get('relevance_feedback', '')} |",
        f"| Evidence & Citations | {sb.get('evidence', 0)}/5 | {j1.get('evidence_feedback', '')} |",
        f"| Factual Accuracy | {sb.get('accuracy', 0)}/5 | {j1.get('accuracy_feedback', '')} |",
        f"| Clarity & Organization | {sb.get('clarity', 0)}/5 | {j2.get('clarity_feedback', '')} |",
        f"| Safety Compliance | {sb.get('safety', 0)}/5 | {j2.get('safety_feedback', '')} |",
    ]
    return "\n".join(lines)
