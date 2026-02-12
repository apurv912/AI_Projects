# Golden Set Questions (v1)

## How this file is used
- The eval runner reads this file and extracts questions in the format: `Q##: ...`
- Each question is run twice: **rerank OFF** vs **rerank ON**
- This is a *reproducible* set you re-run after changes to detect regressions.

## Labeling rubric (simple + production-like)
For each run (OFF/ON), label using:
- **Relevance (0/1)**:
  - 1 = Answer addresses the question and uses correct facts from the doc
  - 0 = Off-topic, hallucinated, or contradicts doc
- **Citation valid (Y/N)** (automatic validator proxy):
  - Y = Every cited chunk ID is within the retrieved chunk IDs
  - N = Any citation points to a chunk not retrieved
- **No-answer (Y/N)**:
  - Y = Model refused appropriately because retrieval similarity was too low
  - N = Model answered when it should have refused (or refused when it shouldn’t)

> Note: This build auto-calculates citation-valid + no-answer. Relevance is intentionally manual.

---

## Questions (10–20)
Q01: What is the stated objective of the “Phoenix” migration project?
Q02: Which three KPIs are used to judge success, and what are their target directions?
Q03: What is the rollback policy if error rate increases after release?
Q04: Name two risks called out in the doc and the planned mitigations.
Q05: What does the document say about data retention, and what is the retention period?
Q06: Who are the key stakeholders (roles), and what is the escalation path?
Q07: What is the planned timeline (phases) and the criteria to exit each phase?
Q08: What is explicitly out of scope for this project?
Q09: What is the customer impact if the project succeeds (as described)?
Q10: What is the monitoring/alerting plan and which metrics are monitored?
Q11: What is the definition of “SEV-1” in this doc, and what is the response SLA?
Q12: If asked “Should we ship by default?”, what does the doc recommend and why?
Q13: What is the dependency that could block the launch, and how do we de-risk it?
Q14: What is the budget/cost constraint mentioned (if any), and its implication?
Q15: Summarize the go/no-go checklist in 4–6 bullet points.

---

## Optional: How to adapt to your own doc
- Replace `sample_doc.txt` with your doc
- Update questions to match your product domain
- Keep IDs stable so you can compare runs over time
