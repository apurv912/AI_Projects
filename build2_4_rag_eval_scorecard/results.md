# Build 2.4 — Results Template (A/B: Rerank OFF vs ON)

> Fill this after running eval. Source of truth is:
- `outputs/eval_report.md` (aggregates)
- `outputs/eval_results.csv` (per-question detail)

## Summary (paste from eval_report)
- Citation validity rate (OFF vs ON):
- No-answer rate (OFF vs ON):
- Rerank fallback rate (ON only):
- Avg total latency ms (OFF vs ON):
- Notes:

## A/B table (per question)
| QID | OFF: Answer ok? | OFF: Cit valid | OFF: No-answer | ON: Answer ok? | ON: Cit valid | ON: No-answer | Winner (OFF/ON/Tie) | Notes |
|---|---|---|---|---|---|---|---|---|
| Q01 |  |  |  |  |  |  |  |  |
| Q02 |  |  |  |  |  |  |  |  |
| Q03 |  |  |  |  |  |  |  |  |

## Decision
- Keep rerank ON by default? (Y/N)
- Why (quality lift vs latency/cost vs fallback rate):
- Next experiment (if needed):
## Decision
- Keep rerank ON by default? **N (for now)**

### Why
- **Rerank fallback rate = 100%** → rerank never succeeds under current quota limits.
- **Latency tax is high (~+1.7–1.9s per question)** even when it fails, so ON currently adds cost/UX delay without quality benefit.
- In quota-constrained conditions, **retrieve-only mode** still provides reliable pipeline measurements (retrieval latency, rerank reliability) without crashing.

### Next experiment
- Re-run with live generation once quota is available (start with `--max-q 3`).
- If rerank succeeds reliably (<10% fallback), compare OFF vs ON on:
  - citation validity + coverage
  - no-answer rate
  - total latency delta

