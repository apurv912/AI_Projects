
# Build 2.4 — RAG Evaluation + Scorecard (A/B: Rerank OFF vs ON)

## Problem
RAG quality can regress silently as models/prompt/chunking change. We need a repeatable way to measure:
- trust (citations grounded in retrieved chunks)
- reliability (quota/429 behavior)
- latency (rerank cost)

## Goal
Ship a low-friction local evaluation harness that:
- runs a fixed “golden set” of questions
- compares rerank OFF vs ON (A/B)
- outputs CSV + markdown scorecard to prevent regressions

## Non-goals
- No hosted eval service / dashboards
- No LLM-judge grading (we use simple proxies)
- Do not modify Build 2.3

## Metrics (proxies)
- Citation validity rate: cited chunk IDs must be within retrieved IDs
- Citation coverage rate: % answered rows with ≥1 citation
- No-answer rate: guardrail refusals
- Rerank fallback rate (ON): rerank failed → baseline order used
- Avg latency per stage: retrieve / rerank / generate / total
- Gen error rate: % rows with generation_error (run continues)

## Acceptance Criteria
- `python3 eval_runner.py --doc sample_doc.txt --max-q 3 --retrieve-only` produces:
  - `outputs/eval_results.csv`
  - `outputs/eval_report.md`
- A/B is present: every question has OFF + ON rows
- 429/quota never crashes: rerank falls back; generation errors recorded

## How to Run

### 1) Setup venv
~~~bash
cd ~/Projects/microbuild-template/build2_4_rag_eval_scorecard
python3.11 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install openai numpy
~~~

### 2) Env vars (Gemini OpenAI-compatible)
~~~bash
export GEMINI_API_KEY="YOUR_KEY"
export GEMINI_CHAT_MODEL="gemini-2.0-flash"
export GEMINI_RERANK_MODEL="gemini-2.0-flash"
export GEMINI_EMBED_MODEL="gemini-embedding-001"

export RAG_TOP_N="8"
export RAG_TOP_K="4"
export RAG_SIM_THRESHOLD="0.20"
~~~

### 3) Run eval
~~~bash
# Recommended smoke test (quota-safe)
python3 eval_runner.py --doc sample_doc.txt --max-q 3 --retrieve-only

# Full run (may hit quota)
python3 eval_runner.py --doc sample_doc.txt
~~~

## Outputs
- `outputs/eval_results.csv` = per-question rows (audit trail)
- `outputs/eval_report.md` = aggregated scorecard (decision view)

## Demo Script (60s)
1) Run smoke test command above
2) Open `outputs/eval_report.md`
3) Explain: OFF baseline vs ON (fallback + latency under quota)
4) Show `outputs/eval_results.csv` for traceability

## What This Proves (PM)
- Regression prevention via reproducible golden set
- Trust proxy via citation validity (when generation available)
- Rerank ROI tradeoff visible (quality vs latency/reliability)
- Graceful degraded mode under quota (retrieve-only)

