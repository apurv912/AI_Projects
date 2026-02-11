# Build 2.3 Sample Doc

## Goal
Build 2.3 adds 2-stage retrieval: embedding top-N → LLM rerank → top-K.

## Reliability
If rerank hits 429/rate limit, the system must fall back to embedding order and never crash.

## Trust
Answers must cite sources using chunk ids like [C001]. A citation validator must flag invalid citations.

## Cost Controls
Retrieve-only mode skips answer generation to save quota and reduce latency.

## Config
Models are set via env vars:
- GEMINI_CHAT_MODEL
- GEMINI_RERANK_MODEL
- GEMINI_EMBED_MODEL
