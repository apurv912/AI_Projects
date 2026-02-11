# Build 2.3 — RAG Reranking (2-Stage Retrieval)

## Problem
Embedding-only retrieval often surfaces “keyword traps” in the top results. In real usage, APIs also hit **quota/rate limits (429)** and products should **not crash** or go blank.

## Goal
Ship a PM-grade RAG demo that improves retrieval precision via **2-stage retrieval** while preserving:
- **Trust:** citations + citation validator
- **Reliability:** graceful degradation on 429 (no crashes)
- **Cost control:** retrieve-only mode + caching/guardrails

## Non-goals
- Production auth/tenancy, vector DB infra, advanced chunking strategies
- Full automated eval harness (this build uses a lightweight golden set)

---

## What’s in this build
### 2-stage retrieval flow
1) **Stage-1 (Embedding Top-N):** fast candidate shortlist  
2) **Stage-2 (LLM Rerank):** re-orders Top-N → selects Top-K  
3) **Fallback:** if rerank hits 429 / bad output, use embedding order (no crash)

### Trust layer (from Build 2.2)
- Answers cite chunks like `[C001]`
- Validator flags invalid citations

### Reliability under quota (429)
- If answer generation hits 429, UI shows a **retrieval-based fallback** that still includes citations so the validator can remain ✅

### Cost controls
- **Retrieve-only mode:** skip answer generation to save quota/latency
- **Similarity guardrail cache:** reuse answers for very similar queries

### Debuggability
- Optional expander shows:
  - Stage-1 candidates
  - Final Top-K
  - Fallback used + reason

---

## Metrics (what this build demonstrates)
- **Quality:** rerank changes ordering when embedding is confused by keyword traps
- **Trust:** citation validity rate (validator ✅/❌)
- **Reliability:** app remains usable under 429 (fallbacks, no crash)
- **Cost/latency:** retrieve-only + cache reduce calls

---

## Acceptance Criteria
1) 2-stage retrieval: embedding Top-N → LLM rerank → Top-K  
2) Preserve caching + similarity guardrail  
3) Preserve citations + citation validator  
4) Graceful degradation for rerank 429 (fallback to embedding order)  
5) Models configurable via env vars:
   - `GEMINI_CHAT_MODEL`, `GEMINI_RERANK_MODEL`, `GEMINI_EMBED_MODEL`  
6) Retrieve-only mode to skip answer generation  
7) Debug expander shows stage1, final top-K, fallback flag/reason

---

## Run it (Build 2.3)
```bash
cd ~/Projects/microbuild-template/build2_3_rag_reranking

export GEMINI_API_KEY="YOUR_REAL_KEY"
export GEMINI_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
export GEMINI_EMBED_MODEL="gemini-embedding-001"
export GEMINI_RERANK_MODEL="gemini-2.5-flash"
export GEMINI_CHAT_MODEL="gemini-2.5-flash"

./.venv/bin/python -m streamlit run app.py
