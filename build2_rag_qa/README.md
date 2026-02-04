# Build 2 — Doc Q&A (RAG-lite) — Gemini API

## What it does
Paste any document + ask a question → the app:
1) chunks the document
2) embeds chunks + question
3) retrieves Top-K relevant chunks (with similarity scores)
4) generates a grounded answer using ONLY retrieved chunks (refuses if insufficient info)

## Why this matters (AI PM angle)
- Demonstrates the core RAG pattern: chunking → retrieval → grounded generation
- Adds transparency: shows retrieved evidence + similarity scores
- Adds a basic hallucination guardrail: “I don’t have enough information…”

## How to run (Mac)
```bash
cd ~/Projects/microbuild-template
source .venv/bin/activate
export GEMINI_API_KEY="YOUR_KEY"
python -m streamlit run build2_rag_qa/app.py


##PM Story

Problem: Users need fast, trustworthy answers from long docs without hallucinations.

User: PM/teams scanning JDs, PRDs, meeting notes, policies.

Approach: RAG-lite app using embeddings retrieval + grounded generation.

Risks: Retrieval misses key chunks; model may still over-generalize if context is thin.

Next step: Add caching, better chunking, and citations per chunk in the final answer.