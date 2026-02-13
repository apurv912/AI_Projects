Definition of Done (Build 2.5)

 outputs/telemetry.jsonl appends 1 record per run

 outputs/telemetry.csv appends the same record in flat format

 Run tab logs telemetry even when GEMINI_API_KEY is missing

 Ops dashboard shows p50/p95 total + p50/p95 per-stage

 Ops dashboard shows rates: error overall/by stage, fallback (429), guardrail, citation validity

 Last 20 runs table works

 Rerank 429 triggers fallback + sets flags.rerank_fallback_used=True

 Generation failure shows readable error and still logs

 Retrieval-only mode works end-to-end

 No modifications to prior builds