# Build 2.6 — Definition of Done (DoD)

## Acceptance criteria (PM)
- [x] Prompt injection attempts are detected and logged (`flags.injection_detected=true`) and strict mode is engaged.
- [x] Risky-domain routing works (medical/legal/financial) with safe UX messaging and telemetry (`flags.risk_domain` + `flags.refusal_triggered`).
- [x] Optional PII redaction masks emails/phones before sending context to the model and logs `flags.pii_redacted`.
- [x] Telemetry schema includes safety flags: `injection_detected`, `risk_domain`, `pii_redacted`, `refusal_triggered`, `grounding_failed`.
- [x] Ops dashboard exposes safety rates: injection detected rate, refusal rate, grounding failure rate, PII redaction rate.
- [ ] Strict grounding end-to-end validation under generation mode (blocked by current 429 quota; runner covers retrieve-only aspects).

## Required deliverables
- [x] `app.py` (Streamlit app with safety controls + telemetry + ops dashboard)
- [x] `guardrails.py` (injection detection + risk routing + strict mode logic)
- [x] `pii_redaction.py` (email/phone masking)
- [x] `telemetry.py` (extended schema + logger)
- [x] `safety_tests.md` (10 adversarial prompts)
- [x] `safety_report.md` (pass/fail template)
- [x] `safety_runner.py` (automated runner for retrieve-only validation)
- [x] `README.md` (PM-first: problem, threat model, guardrails, test plan, demo script, DoD)

## Evidence (local)
- Telemetry logs:
  - `outputs/telemetry.jsonl`
  - `outputs/telemetry.csv`
- Safety runner reports (PASS/FAIL/SKIP):
  - `outputs/safety_runner_96c82fccadb8487bb806610e7d8e4619.md` (T01–T08 PASS; T09–T10 SKIP due to generation/quota)
  - `outputs/safety_runner_96c82fccadb8487bb806610e7d8e4619.json`

## Known limitations / blockers
- Current state: **Generation requests are frequently blocked by 429 quota**.
  - Mitigation: use `retrieve_only=true` in UI + `python3 safety_runner.py` for measurable safety signals.
  - Follow-up: re-run T09/T10 and “grounding fail-closed” behaviors once quota is available.

## Quick re-check commands
- Run app:
  - `streamlit run app.py`
- Run safety runner:
  - `python3 safety_runner.py`
