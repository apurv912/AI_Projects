# Build 2.1 — RAG Doc Q&A — Cached + Guardrails

# Build 2.1 — RAG Doc Q&A — Cached + Guardrails (Gemini API)

## What it does
Paste a document + ask a question → the app:
1) chunks the document
2) embeds chunks + question (Gemini embeddings)
3) retrieves Top-K chunks with similarity scores
4) generates a grounded answer using ONLY retrieved chunks
5) blocks unsafe answers using a similarity threshold guardrail

## What’s new vs Build 2
### 1) Embedding Cache (cost + latency)
- Doc chunk embeddings are cached so the same document doesn’t get re-embedded on every rerun.
- Result: faster UI + fewer paid API calls.

### 2) Similarity Threshold Guardrail (trust + safety)
- If best retrieval score < threshold, the app refuses to answer.
- It suggests how to fix it: paste more relevant text / rephrase with doc keywords.

## Why this matters (AI PM angle)
- Shows productization thinking: performance + unit economics (caching)
- Shows responsible AI behavior: “when to refuse” (guardrails)
- Adds transparency: retrieved evidence + similarity scores

## Tech stack
- Streamlit UI
- Gemini API via OpenAI-compatible base URL
- Embeddings: `text-embedding-004`
- Generation: `gemini-2.5-flash`
- Retrieval: cosine similarity over embeddings
- Caching: `st.cache_data` for doc chunk embeddings

## How to run (Mac)
```bash
cd ~/Projects/microbuild-template
source .venv/bin/activate
export GEMINI_API_KEY="YOUR_KEY"
python -m streamlit run build2_1_rag_cached_guardrails/app.py
