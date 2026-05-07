import os, json, re

def _call_judge(system, user):
    from openai import OpenAI
    c = OpenAI(api_key=os.getenv("OPENAI_API_KEY",""), base_url=os.getenv("OPENAI_BASE_URL","https://api.openai.com/v1"))
    try:
        r = c.chat.completions.create(
            model=os.getenv("OPENAI_MODEL","Qwen/Qwen3-8B"),
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            max_tokens=800, temperature=0.1,
            extra_body={"chat_template_kwargs":{"enable_thinking":False}},
        )
        raw = r.choices[0].message.content or ""
        raw = re.sub(r"<think>.*?</think>","",raw,flags=re.DOTALL)
        raw = re.sub(r"<think>.*","",raw,flags=re.DOTALL)
        return raw.strip()
    except Exception as e:
        return "{}"

J1 = """You are a research quality evaluator. Return ONLY valid JSON, nothing else:
{"relevance_score":4,"evidence_score":3,"accuracy_score":4,"relevance_feedback":"Good coverage of topic","evidence_feedback":"Some citations present","accuracy_feedback":"Mostly accurate","overall_research_score":3.67}"""

J2 = """You are a communication quality evaluator. Return ONLY valid JSON, nothing else:
{"clarity_score":4,"safety_score":5,"clarity_feedback":"Well structured response","safety_feedback":"No unsafe content detected","overall_communication_score":4.5}"""

def run_judge_1(query, response):
    user = f"Query: {query}\n\nResponse:\n{response[:3000]}\n\nReturn JSON scores only."
    try:
        raw = _call_judge(J1, user).strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        r = json.loads(raw)
        r["judge"] = "research_quality"
        r["status"] = "success"
        return r
    except Exception as e:
        return {"judge":"research_quality","status":"error","error":str(e),"relevance_score":0,"evidence_score":0,"accuracy_score":0,"overall_research_score":0.0}

def run_judge_2(query, response):
    user = f"Query: {query}\n\nResponse:\n{response[:3000]}\n\nReturn JSON scores only."
    try:
        raw = _call_judge(J2, user).strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        r = json.loads(raw)
        r["judge"] = "communication_safety"
        r["status"] = "success"
        return r
    except Exception as e:
        return {"judge":"communication_safety","status":"error","error":str(e),"clarity_score":0,"safety_score":0,"overall_communication_score":0.0}

def run_both_judges(query, response):
    j1 = run_judge_1(query, response)
    j2 = run_judge_2(query, response)
    scores = [j1.get("relevance_score",0),j1.get("evidence_score",0),j1.get("accuracy_score",0),j2.get("clarity_score",0),j2.get("safety_score",0)]
    valid = [s for s in scores if s > 0]
    composite = round(sum(valid)/len(valid),2) if valid else 0.0
    return {"query":query,"judge_1_research":j1,"judge_2_communication":j2,"composite_score":composite,"scores_breakdown":{"relevance":j1.get("relevance_score",0),"evidence":j1.get("evidence_score",0),"accuracy":j1.get("accuracy_score",0),"clarity":j2.get("clarity_score",0),"safety":j2.get("safety_score",0)}}

def format_judge_results(results):
    j1=results.get("judge_1_research",{}); j2=results.get("judge_2_communication",{}); sb=results.get("scores_breakdown",{})
    return "\n".join(["### LLM-as-a-Judge Evaluation Results","",f"**Composite Score: {results.get('composite_score',0):.1f} / 5.0**","","| Criterion | Score | Feedback |","|-----------|-------|----------|",f"| Relevance & Coverage | {sb.get('relevance',0)}/5 | {j1.get('relevance_feedback','')} |",f"| Evidence & Citations | {sb.get('evidence',0)}/5 | {j1.get('evidence_feedback','')} |",f"| Factual Accuracy | {sb.get('accuracy',0)}/5 | {j1.get('accuracy_feedback','')} |",f"| Clarity & Organization | {sb.get('clarity',0)}/5 | {j2.get('clarity_feedback','')} |",f"| Safety Compliance | {sb.get('safety',0)}/5 | {j2.get('safety_feedback','')} |"])
