# Build 2.3 Results — Rerank A/B

## Setup
- Docs: sample.md + extra.md + noisy.md
- Chunking: MAX_CHUNK_CHARS=200, CHUNK_OVERLAP=50
- Top-N: 20
- Top-K: 5
- Cache: OFF for A/B
- Retrieve-only: ON for A/B (quota-safe)

## Query
"What are the correct model config environment variables?"

## A) Rerank OFF (embedding-only)
Final Top-K:
1.[C006] sim=0.782 — data/noisy.md
2.[C005] sim=0.780 — data/noisy.md
3.[C010] sim=0.739 — data/sample.md
4.[C002] sim=0.681 — data/extra.md
5.[C009] sim=0.677 — data/sample.md

Notes:
- Noisy chunks present? (Y/N)
- Any obviously wrong chunk higher ranked? (Y/N)

## B) Rerank ON (2-stage)
Final Top-K:
1.[C010] sim=0.739 — data/sample.md
2.[C002] sim=0.681 — data/extra.md
3.[C009] sim=0.677 — data/sample.md
4.[C005] sim=0.780 — data/noisy.md
5.[C004] sim=0.664 — data/noisy.md


Notes:
- Order changed vs A? (Y/N)Y
- Noisy chunks demoted? (Y/N)Y

## Reliability proof (429)
- Cache OFF + repeated runs triggered 429: (Y/N)-Y
- App did not crash: (Y/N)-did not crash
- Fallback answer shown with citations: (Y/N)-Y
- Citation validation ✅ under 429: (Y/N)-Y







