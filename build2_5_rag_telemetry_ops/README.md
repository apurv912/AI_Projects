# Build 2.5 — RAG Telemetry + Ops Dashboard (PM-first)

## Problem

A RAG engine that “answers questions” isn’t production-ready unless we can **observe** it:

* Where is latency coming from? (p95 by stage)
* How often do we degrade? (rerank fallback under quota)
* How often do we fail, and where? (error rate by stage)
* Are we maintaining quality signals? (citation validity rate)

Without this, we can’t run SLOs / error budgets, and we can’t troubleshoot incidents.

## Goal

Add **production-like observability**:

* Structured telemetry (1 record per run) with stage timings, reliability counters, fallbacks, and errors
* Persist to `outputs/telemetry.jsonl` and `outputs/telemetry.csv`
* Streamlit ops dashboard with SLO-style summaries:

  * p50/p95 total latency
  * p50/p95 per-stage latency
  * error rate overall + by stage
  * rerank fallback rate (429)
  * guardrail trigger rate
  * citation validity rate
  * last 20 runs table
* Quota resilience:

  * If rerank hits 429 → fallback to embedding order and log fallback
  * If generation fails (incl 429) → user sees readable error, telemetry still logs, dashboard still works
  * Retrieval-only mode to conserve quota

## Non-goals

* No production deployment, auth, multi-user tenancy, or tracing backend (Prometheus/OTel)
* No advanced analytics or anomaly detection
* No new retrieval/reranking algorithms (we reuse prior engine logic)

---

## Metrics / SLO thinking (starter)

These are “starter SLOs” for a small RAG service:

**Latency SLO (interactive)**

* p95 total latency ≤ 6,000 ms (tune once real keys + load exist)

**Reliability SLO**

* Error rate (generate stage) ≤ 2%
* Rerank fallback rate (429) ≤ 10% (signals quota pressure)

**Quality SLO**

* Citation validity rate ≥ 95%

When SLOs are breached, treat it as consuming **error budget**, and prioritize fixes.

---

## Telemetry schema (1 record per run)

Identity:

* `timestamp`, `run_id`

Doc shape:

* `doc_length_chars`, `num_chunks`

Params:

* `params.top_n`, `params.top_k`, `params.min_score`
* `params.rerank_enabled`, `params.retrieve_only`

Flags:

* `flags.guardrail_triggered`
* `flags.rerank_fallback_used` (only True when rerank hit 429)

Stage timings (ms):

* `timings_ms.chunking_ms`
* `timings_ms.embed_doc_ms`
* `timings_ms.embed_query_ms`
* `timings_ms.retrieve_ms`
* `timings_ms.rerank_ms`
* `timings_ms.generate_ms`
* `timings_ms.validate_ms`
* `timings_ms.total_ms`

Errors:

* `errors.error_stage` (e.g., `client_init`, `rerank`, `generate`)
* `errors.status_code` (e.g., `429`)
* `errors.error_message_short`

Quality outputs:

* `citation_valid` (bool)
* `cited_ids_count`

Cost proxy:

* `cost.prompt_chars`, `cost.context_chars`, `cost.answer_chars`
* token fields reserved if available later

Storage:

* JSONL: `outputs/telemetry.jsonl`
* CSV: `outputs/telemetry.csv` (flattened keys)

---

## How to run

### 1) Setup venv

Run this sequence:
cd ~/Projects/microbuild-template/build2_5_rag_telemetry_ops
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

### 2) Set env vars (Gemini via OpenAI SDK)

Run this sequence:
export GEMINI_API_KEY="YOUR_KEY"
export GEMINI_CHAT_MODEL="gemini-1.5-flash"
export GEMINI_RERANK_MODEL="gemini-1.5-flash"
export GEMINI_EMBED_MODEL="text-embedding-004"

### 3) Run Streamlit

Run:
streamlit run app.py

---

## 60-second demo script (what to show)

1. Open app → show two tabs: **Run Q&A** and **Ops Dashboard**
2. Run Q&A with defaults → show `run_id` + Answer + Citations
3. Flip to Ops Dashboard → show:

   * p50/p95 total latency
   * per-stage p95 (where time is spent)
   * fallback rate + citation validity
   * last 20 runs table
4. Toggle **retrieve_only** → run again → show it logs but skips generation (quota saver)
5. (Optional) simulate quota pressure by repeated runs; if 429 happens, point to:

   * `flags.rerank_fallback_used = True`
   * rerank errors recorded, but answer still served (degraded mode)

---

## What this proves (PM lens)

* You can design **SLO-based observability** for an AI workflow (not just “it works”)
* You can manage **reliability under quota** using graceful degradation + fallback
* You understand **cost/latency tradeoffs** (retrieve-only mode + stage breakdown)
* You can build a lightweight “ops cockpit” that supports incident response and prioritization

---

## Definition of Done (checklist)

* [ ] `outputs/telemetry.jsonl` appends 1 record per run
* [ ] `outputs/telemetry.csv` appends the same record in flat format
* [ ] Run tab logs telemetry even when `GEMINI_API_KEY` is missing
* [ ] Ops dashboard shows p50/p95 total + p50/p95 per-stage
* [ ] Ops dashboard shows rates: error overall/by stage, fallback (429), guardrail, citation validity
* [ ] Last 20 runs table works
* [ ] Rerank 429 triggers fallback + sets `flags.rerank_fallback_used=True`
* [ ] Generation failure shows readable error and still logs
* [ ] Retrieval-only mode works end-to-end
* [ ] No modifications to prior builds
