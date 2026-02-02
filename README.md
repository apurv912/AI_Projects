# Build 1 — JD Analyzer MicroBuild
## What it does
Paste a Job Description + my resume highlights → generates a structured analysis prompt for:
- ATS keyword pack
- match score + gaps
- tailored resume bullets (impact-first)
- interview questions + story outline

## How to run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m streamlit run app/app.py
## PM Story (5 bullets)
Problem: Job applications need fast JD-to-resume alignment and ATS keyword coverage.

User: PM/TPM job seeker applying to multiple roles weekly.

Approach: Lightweight Streamlit app that generates a consistent, structured prompt using JD + candidate highlights.

Risks: Output quality depends on the model used; needs guardrails for hallucinations and keyword stuffing.

Next step: Add optional LLM API integration + “keyword diff” view + export to PDF.