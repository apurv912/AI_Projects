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

---


