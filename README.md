# AI PM Portfolio — RAG Microbuild Ladder (PM-first)

This repo is my **AI PM portfolio** built as a **RAG microbuild ladder**: each build is a small, shippable artifact with clear acceptance criteria, eval notes, and proof-ready docs.

## Why this exists (PM framing)
- **Quality:** improve retrieval relevance and answer correctness over iterations
- **Trust:** enforce citations + validate them
- **Reliability:** handle quota/rate limits gracefully (no crashes)
- **Cost/latency:** add switches like retrieve-only + caching/guardrails
- **Proof:** each build ships with README + eval set + results template

---

## Builds

### ✅ Build 2 — Basic RAG
**Folder:** `build2_rag_qa/`  
**What:** baseline RAG using Gemini via OpenAI-compatible `base_url`.

### ✅ Build 2.1 — Caching + Similarity Guardrail
**Folder:** `build2_1_rag_cached_guardrails/`  
**What:** reuse answers for very similar queries to reduce cost/latency.

### ✅ Build 2.2 — Citations + Citation Validator
**Folder:** `build2_2_rag_citations/`  
**What:** answers include chunk citations like `[C001]` + validator flags invalid citations.

### ✅ Build 2.3 — 2-Stage Retrieval + Reranking + Quota-Safe UX
**Folder:** `build2_3_rag_reranking/`  
**What:**
- 2-stage retrieval: **embedding Top-N → LLM rerank → Top-K**
- Preserves: caching/guardrails + citations + validator
- Adds: **retrieve-only mode** (skip generation to save quota)
- Adds: **graceful degradation on 429** (fallback paths; no crash)
- Adds: rerank debug expander (stage1 vs final + fallback flags)

### ✅ Build 2.4 — Evaluation + Scorecard (A/B: Rerank OFF vs ON)
**Folder:** `build2_4_rag_eval_scorecard/`  
**What:** adds a production-like evaluation layer to prevent regressions and prove quality changes.

**Adds:**
- Golden set questions (reproducible): `eval_questions.md`
- Automated A/B runner (CLI): rerank **OFF vs ON**
- Scorecard report: `outputs/eval_report.md` (reliability + latency proxies; trust proxy when generation available)
- Audit trail: `outputs/eval_results.csv` (retrieved chunk IDs, fallback flags, timings, errors)
- Quota-safe switches: `--retrieve-only` and `--max-q`

**Current signal:** under quota constraints, rerank can fall back (100%) and add latency—this scorecard makes the tradeoff visible and supports a default decision (**OFF for now**).

### ✅ Build 2.5 — Telemetry + Ops Dashboard (Run-ready Observability)
**Folder:** `build2_5_rag_telemetry_ops/`  
**What:** adds production-like observability + a lightweight ops cockpit so the engine looks “run-ready”.

**Adds:**
- Telemetry schema (1 record per run): timings by stage, flags, errors, citation validity, cost proxies
- Persistent logs:
  - `outputs/telemetry.jsonl` (append-only)
  - `outputs/telemetry.csv` (flat/analysis-friendly)
- Streamlit app (2 tabs):
  1) **Run Q&A** (runs pipeline + logs telemetry; supports **retrieve-only** mode)
  2) **Ops Dashboard** (reads telemetry and shows):
     - p50/p95 total latency
     - p50/p95 per-stage latency
     - error rate overall + by stage
     - rerank fallback rate under quota (429)
     - guardrail trigger rate
     - citation validity rate
     - last 20 runs table
- Reliability under quota:
  - If **429** occurs during rerank → fallback to embedding order + flag `rerank_fallback_used`
  - If generation fails (incl 429) → readable error + telemetry still logs + dashboard still works

**PM takeaway:** this build turns “a working demo” into an **operable system** with SLO-style visibility and graceful degradation.
### ✅ Build 2.6 — Safety + Robustness Guardrails (Trustworthy RAG)
**Folder:** `build2_6_rag_safety_guardrails/`  
**What:** adds production-grade safety + robustness so the RAG engine can be trusted under adversarial prompts and high-stakes queries.

**Adds:**
- **Prompt injection detection** (lightweight patterns) + strict-mode behavior  
  - Logs: `injection_detected`
- **Strict grounding (fail-closed)**  
  - Answer only from retrieved chunks; if evidence weak / citation invalid → **refuse / not enough info**  
  - Logs: `grounding_failed`, `refusal_triggered`
- **Risky-domain routing** (medical/legal/financial)  
  - No definitive advice; safe next steps + escalation UX  
  - Logs: `risk_domain`
- **Optional PII redaction toggle** (emails/phones masked before model call)  
  - Logs: `pii_redacted`
- **Safety proof pack**
  - `safety_tests.md` (10 adversarial prompts)
  - `safety_report.md` (template with run_id evidence)
  - `safety_runner.py` (automated retrieve-only safety runner → PASS/FAIL/SKIP report)
- **Telemetry + Ops dashboard extensions**
  - New safety rates: injection detected rate, refusal rate, grounding failure rate, PII redaction rate

**Current signal:** generation may be quota-limited (429), but the build remains **demoable** via `retrieve_only=true` + `python3 safety_runner.py` with report outputs in `outputs/`.

