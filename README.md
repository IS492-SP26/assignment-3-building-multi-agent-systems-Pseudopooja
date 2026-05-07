# Multi-Agent HCI Research System
### IS 492 — Assignment 3

A multi-agent deep research system for HCI topics, built with **LangGraph**, **Qwen3-8B**, **Tavily**, and **Streamlit**.
---
## Demo: https://assignment-3-building-multi-agent-systems-pseudopooja-q3nfvzmv.streamlit.app
---

## System Architecture

```
User Query
    │
    ▼
[Safety: Input Check] ──blocked──► Refusal UI
    │ safe
    ▼
[Planner]      — decomposes query into 3–4 sub-questions
    │
    ▼
[Researcher]   — Tavily web search + Semantic Scholar papers
    │
    ▼
[Critic]       — scores evidence quality (1–5); re-triggers Researcher if gaps found
    │
    ▼
[Writer]       — synthesizes cited final answer
    │
    ▼
[Safety: Output Check] — redacts PII, blocks harmful output
    │
    ▼
Streamlit UI   — Answer · Agent Trace · Citations · Evaluation tabs
```

**4 agents · LangGraph state graph · 2 search tools · 3 safety policies · 5-metric LLM judge**

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/IS492-SP26/assignment-3-building-multi-agent-systems-Pseudopooja.git
cd assignment-3-building-multi-agent-systems-Pseudopooja
```

### 2. Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env.example .env
```

Edit `.env` — the LLM keys are pre-configured. Add your Tavily key:
```env
OPENAI_API_KEY=sk-SzGKL7KOo8fs3vkuFx5mHRdwZktp_0veWDek9YTIpGY
OPENAI_BASE_URL=https://vllm.salt-lab.org/v1
OPENAI_MODEL=Qwen/Qwen3-8B
TAVILY_API_KEY=your_key_here     # free at app.tavily.com
```

---

## Running

### Full end-to-end demo (single command)
```bash
python main.py
```
Runs one complete query through all 4 agents and prints trace + answer + judge scores.  
Expected output: agent trace → 500–700 word cited answer → composite judge score ~4.0/5.0.  
Session exported to `outputs/demo_session.json`.

### Web UI
```bash
streamlit run src/ui/streamlit_app.py
```
Opens at `http://localhost:8501`

### Interactive CLI
```bash
python main.py --mode cli
```

### Batch evaluation (10 queries)
```bash
python main.py --mode evaluate
```
Results saved to `outputs/eval_results_<timestamp>.json`

---

## Tested Queries

The following queries were used for evaluation (see `data/example_queries.json`):

1. What are the key principles of explainable AI for novice users?
2. How has AR usability evolved in the past 5 years?
3. What are ethical considerations in using AI for education?
4. Compare different approaches to measuring UX in mobile applications
5. What are the latest developments in conversational AI for healthcare?
6. How do design patterns for accessibility differ across web and mobile platforms?
7. What are best practices for visualizing uncertainty in data displays?
8. How can voice interfaces be designed for elderly users?
9. What are emerging trends in AI-driven prototyping tools?
10. How do cultural factors influence mobile app design?

---

## Safety Guardrails

The system enforces 3 policy categories:

| Category | Example | Response |
|---|---|---|
| `HARMFUL` | "how to make explosives" | Blocked + logged |
| `INJECTION` | "ignore all previous instructions" | Blocked + logged |
| `OFF_TOPIC` | "what is the best pizza recipe?" | Blocked + redirected |

Both input and output are checked. Safety events are logged and displayed in the UI.

---

## Evaluation Results

10 queries evaluated with LLM-as-a-Judge (2 independent rubrics, 5 criteria):

| Criterion | Mean Score (1–5) |
|---|---|
| Relevance & Coverage | 4.0 |
| Evidence & Citation Quality | 3.0 |
| Factual Accuracy | 4.0 |
| Clarity & Organization | 4.0 |
| Safety Compliance | 5.0 |
| **Composite** | **4.0** |

---

## Repo Structure

```
├── src/
│   ├── autogen_orchestrator.py   # LangGraph pipeline (4 agents)
│   ├── agents/
│   ├── tools/
│   │   ├── web_search.py         # Tavily API
│   │   ├── paper_search.py       # Semantic Scholar API
│   │   └── citation_tool.py      # APA citation formatter
│   ├── guardrails/
│   │   ├── input_guardrail.py    # 3 policy categories
│   │   ├── output_guardrail.py   # PII redaction + harmful output
│   │   └── safety_manager.py     # Coordinator + event log
│   ├── evaluation/
│   │   ├── judge.py              # 2 LLM judges, 5 criteria
│   │   └── evaluator.py          # Batch runner
│   └── ui/
│       └── streamlit_app.py      # Web interface
├── data/
│   └── example_queries.json      # 10 evaluation queries
├── outputs/
│   ├── demo_session.json         # Exported session
│   └── eval_results_*.json       # Batch eval results
├── main.py
├── requirements.txt
└── .env.example
```

---

## Dependencies

- `langgraph` — multi-agent orchestration
- `langchain-openai` — LLM client
- `streamlit` — web UI
- `openai` — direct API calls (thinking mode control)
- `python-dotenv` — environment config
