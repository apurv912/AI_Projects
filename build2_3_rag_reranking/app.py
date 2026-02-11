"""
Build 2.3 — RAG with 2-stage retrieval (embedding → LLM rerank)

PM-level intent:
- Quality: reranking improves relevance among top-N embedding candidates.
- Trust: citations + validator prevent “confident nonsense”.
- Reliability: quota/rate-limit never crashes the app; it degrades gracefully.
- Cost/latency: retrieve-only mode + knobs (topN/topK) reduce spend.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from openai import OpenAI

from reranker import LLMReranker


DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL") or os.getenv("GEMINI_BASE_URL") or "https://generativelanguage.googleapis.com/v1beta/openai/"
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or ""

GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
GEMINI_RERANK_MODEL = os.getenv("GEMINI_RERANK_MODEL", "gemini-2.5-flash")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", ".cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = CACHE_DIR / "index.pkl"
QA_CACHE_PATH = CACHE_DIR / "qa_cache.pkl"

SIMILARITY_GUARDRAIL = float(os.getenv("SIMILARITY_GUARDRAIL", "0.92"))
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))


def _client() -> OpenAI:
    return OpenAI(api_key=API_KEY, base_url=DEFAULT_BASE_URL)


def _looks_like_rate_limit(err: Exception) -> bool:
    msg = str(err).lower()
    return ("rate limit" in msg) or ("ratelimit" in msg) or ("429" in msg) or ("resource exhausted" in msg) or ("quota" in msg)


def _cosine(a: List[float], b: List[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


def embed_texts(texts: List[str], model: str) -> List[List[float]]:
    c = _client()
    out: List[List[float]] = []
    B = 8
    for i in range(0, len(texts), B):
        batch = texts[i : i + B]
        resp = c.embeddings.create(model=model, input=batch)
        out.extend([d.embedding for d in resp.data])
        time.sleep(0.05)
    return out


@dataclass
class Chunk:
    id: str
    source: str
    text: str


@dataclass
class Index:
    doc_fingerprint: str
    chunks: List[Chunk]
    embeddings: List[List[float]]


def _fingerprint_docs(files: List[Path]) -> str:
    h = hashlib.sha256()
    for fp in sorted(files, key=lambda p: str(p)):
        stat = fp.stat()
        h.update(str(fp).encode("utf-8"))
        h.update(str(stat.st_mtime_ns).encode("utf-8"))
        h.update(str(stat.st_size).encode("utf-8"))
    return h.hexdigest()


def _load_files() -> List[Path]:
    if not DATA_DIR.exists():
        return []
    files = []
    for ext in ("*.md", "*.txt"):
        files.extend(DATA_DIR.rglob(ext))
    return [f for f in files if f.is_file()]


def _read_text(fp: Path) -> str:
    try:
        return fp.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _chunk_text(text: str, max_chars: int, overlap: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def build_or_load_index(embed_model: str) -> Tuple[Optional[Index], str]:
    files = _load_files()
    if not files:
        return None, "No docs found. Create ./data and add .md/.txt files."

    fp = _fingerprint_docs(files)

    if INDEX_PATH.exists():
        try:
            idx: Index = pickle.loads(INDEX_PATH.read_bytes())
            if idx.doc_fingerprint == fp:
                return idx, "Loaded cached index."
        except Exception:
            pass

    all_chunks: List[Chunk] = []
    for f in sorted(files, key=lambda p: str(p)):
        raw = _read_text(f)
        for ch in _chunk_text(raw, MAX_CHUNK_CHARS, CHUNK_OVERLAP):
            cid = f"C{len(all_chunks)+1:03d}"
            all_chunks.append(Chunk(id=cid, source=str(f), text=ch))

    embeddings = embed_texts([c.text for c in all_chunks], model=embed_model)
    idx = Index(doc_fingerprint=fp, chunks=all_chunks, embeddings=embeddings)
    INDEX_PATH.write_bytes(pickle.dumps(idx))
    return idx, f"Built index: {len(all_chunks)} chunks from {len(files)} files."


@dataclass
class CachedQA:
    query: str
    query_emb: List[float]
    topk_ids: List[str]
    answer: str
    created_at: float


def _load_qa_cache() -> List[CachedQA]:
    if not QA_CACHE_PATH.exists():
        return []
    try:
        return pickle.loads(QA_CACHE_PATH.read_bytes())
    except Exception:
        return []


def _save_qa_cache(items: List[CachedQA]) -> None:
    QA_CACHE_PATH.write_bytes(pickle.dumps(items))


def _find_similar_cached_answer(cache: List[CachedQA], q_emb: List[float], threshold: float) -> Optional[CachedQA]:
    best = None
    best_sim = 0.0
    for item in cache:
        sim = _cosine(q_emb, item.query_emb)
        if sim > best_sim:
            best_sim = sim
            best = item
    if best and best_sim >= threshold:
        return best
    return None


CITATION_RE = re.compile(r"\[(C\d{3})\]")


def extract_citations(text: str) -> List[str]:
    return sorted(set(CITATION_RE.findall(text or "")))


def validate_citations(answer: str, allowed_ids: List[str]) -> Tuple[bool, List[str]]:
    cited = extract_citations(answer)
    allowed = set(allowed_ids)
    invalid = [c for c in cited if c not in allowed]
    return (len(invalid) == 0), invalid


def format_context(chunks: List[Chunk]) -> str:
    blocks = []
    for c in chunks:
        blocks.append(f"[{c.id}] SOURCE: {c.source}\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def generate_answer(query: str, context_chunks: List[Chunk], chat_model: str) -> str:
    c = _client()
    context = format_context(context_chunks)
    prompt = (
        "You are a helpful RAG assistant.\n"
        "Use ONLY the provided context.\n"
        "Cite every important claim using chunk ids in square brackets like [C001].\n"
        "If the answer is not in the context, say you don't know.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{query}\n"
    )
    resp = c.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": "Answer with high precision and citations."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


def fallback_answer_from_retrieval(final_chunks: List[Chunk]) -> str:
    # Quota-safe fallback that still includes citations so validator can pass
    if not final_chunks:
        return "Quota/rate limit hit and no sources were retrieved. Please try again later."
    sources = "\n".join([f"- [{c.id}] {c.source}" for c in final_chunks])
    cites = " ".join([f"[{c.id}]" for c in final_chunks])
    return (
        "Quota/rate limit hit for answer generation. Showing retrieval-based fallback.\n\n"
        f"Most relevant sources:\n{sources}\n\n"
        f"Citations: {cites}"
    )


def retrieve_stage1(idx: Index, query: str, embed_model: str, top_n: int) -> Tuple[List[Dict[str, Any]], List[float]]:
    q_emb = embed_texts([query], model=embed_model)[0]
    scored = []
    for chunk, emb in zip(idx.chunks, idx.embeddings):
        scored.append((_cosine(q_emb, emb), chunk))
    scored.sort(key=lambda x: x[0], reverse=True)

    out = []
    for score, ch in scored[:top_n]:
        out.append({"id": ch.id, "source": ch.source, "text": ch.text, "score": float(score)})
    return out, q_emb


def retrieve_stage2_rerank(
    query: str,
    stage1: List[Dict[str, Any]],
    top_k: int,
    rerank_enabled: bool,
    reranker: LLMReranker,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    meta = {"fallback_used": False, "fallback_reason": None}
    if not rerank_enabled:
        meta["fallback_used"] = True
        meta["fallback_reason"] = "rerank_disabled"
        return stage1[:top_k], meta

    rr = reranker.rerank(query=query, candidates=stage1, top_k=top_k)
    meta["fallback_used"] = rr.fallback_used
    meta["fallback_reason"] = rr.fallback_reason
    meta["rerank_usage"] = rr.usage
    meta["raw_rerank_output"] = rr.raw_model_output
    return rr.ranked, meta


st.set_page_config(page_title="Build 2.3 — RAG Reranking", layout="wide")
st.title("Build 2.3 — RAG with 2-Stage Retrieval (Embedding → LLM Rerank)")

with st.sidebar:
    st.subheader("Models (env-configurable)")
    st.code(
        f"BASE_URL={DEFAULT_BASE_URL}\n"
        f"GEMINI_EMBED_MODEL={GEMINI_EMBED_MODEL}\n"
        f"GEMINI_RERANK_MODEL={GEMINI_RERANK_MODEL}\n"
        f"GEMINI_CHAT_MODEL={GEMINI_CHAT_MODEL}\n",
        language="bash",
    )

    st.subheader("Controls")
    top_n = st.slider("Stage-1 Top-N (embedding)", 5, 50, 20, 5)
    top_k = st.slider("Final Top-K", 2, 10, 5, 1)
    rerank_enabled = st.checkbox("Enable LLM rerank", value=True)
    retrieve_only = st.checkbox("Retrieve-only (skip answer generation)", value=False)
    show_debug = st.checkbox("Show rerank debug expander", value=False)

    st.subheader("Guardrails & cache")
    enable_cache = st.checkbox("Enable QA cache + similarity guardrail", value=True)
    st.caption(f"Similarity reuse threshold: {SIMILARITY_GUARDRAIL}")

    if st.button("Rebuild index"):
        if INDEX_PATH.exists():
            INDEX_PATH.unlink(missing_ok=True)
        st.success("Index cache cleared. Reload / run a query to rebuild.")


if "index" not in st.session_state:
    idx, msg = build_or_load_index(GEMINI_EMBED_MODEL)
    st.session_state.index = idx
    st.session_state.index_msg = msg

idx: Optional[Index] = st.session_state.index
st.info(st.session_state.index_msg)

if idx is None:
    st.stop()

query = st.text_input("Ask a question", placeholder="e.g., What is the goal of Build 2.3?")
run = st.button("Run")

if run and query.strip():
    query = query.strip()

    qa_cache = _load_qa_cache() if enable_cache else []

    try:
        stage1, q_emb = retrieve_stage1(idx, query, GEMINI_EMBED_MODEL, top_n=top_n)
    except Exception as e:
        st.error(f"Embedding retrieval failed: {e}")
        st.stop()

    cached_hit = _find_similar_cached_answer(qa_cache, q_emb, SIMILARITY_GUARDRAIL) if enable_cache else None
    if cached_hit and not retrieve_only:
        st.success("Cache hit (similarity guardrail): reused prior answer for speed/cost.")
        allowed_ids = cached_hit.topk_ids
        cached_chunks = [c for c in idx.chunks if c.id in set(allowed_ids)]
        st.subheader("Answer")
        st.write(cached_hit.answer)

        ok, invalid = validate_citations(cached_hit.answer, allowed_ids)
        st.subheader("Citation validation")
        st.write("✅ Valid" if ok else f"❌ Invalid citations: {invalid}")

        with st.expander("Retrieved chunks"):
            for ch in cached_chunks:
                st.markdown(f"**[{ch.id}]** — `{ch.source}`")
                st.write(ch.text)
        st.stop()

    reranker = LLMReranker(client=_client(), model=GEMINI_RERANK_MODEL, max_candidates_in_prompt=min(20, top_n))
    final_ranked, meta = retrieve_stage2_rerank(
        query=query,
        stage1=stage1,
        top_k=top_k,
        rerank_enabled=rerank_enabled,
        reranker=reranker,
    )

    final_ids = [c["id"] for c in final_ranked]
    id_to_chunk = {c.id: c for c in idx.chunks}
    final_chunks = [id_to_chunk[cid] for cid in final_ids if cid in id_to_chunk]

    colA, colB = st.columns([1, 1])

    with colA:
        st.subheader("Retrieval result")
        st.write(
            f"Rerank fallback: **{meta.get('fallback_used')}**"
            + (f" — reason: `{meta.get('fallback_reason')}`" if meta.get("fallback_used") else "")
        )
        for rank_i, c in enumerate(final_ranked, start=1):
            st.markdown(f"**{rank_i}. [{c['id']}]** — sim={c['score']:.3f} — `{c['source']}`")

        if show_debug:
            with st.expander("Rerank debug (stage1 vs final)"):
                st.markdown("**Stage-1 (embedding order)**")
                for i, c in enumerate(stage1, start=1):
                    st.markdown(f"{i}. [{c['id']}] sim={c['score']:.3f} — `{c['source']}`")

                st.markdown("---")
                st.markdown("**Final Top-K**")
                for i, c in enumerate(final_ranked, start=1):
                    st.markdown(f"{i}. [{c['id']}] sim={c['score']:.3f} — `{c['source']}`")

                st.markdown("---")
                st.markdown(f"**Fallback used:** `{meta.get('fallback_used')}`")
                if meta.get("fallback_reason"):
                    st.markdown(f"**Fallback reason:** `{meta.get('fallback_reason')}`")

    with colB:
        st.subheader("Answer")
        if retrieve_only:
            st.warning("Retrieve-only mode is ON — skipping answer generation to save quota.")
        else:
            try:
                answer = generate_answer(query, final_chunks, GEMINI_CHAT_MODEL)
            except Exception as e:
                st.error(f"Answer generation failed (degrading gracefully): {e}")
                answer = fallback_answer_from_retrieval(final_chunks) if _looks_like_rate_limit(e) else ""

            if answer:
                st.write(answer)
                ok, invalid = validate_citations(answer, final_ids)
                st.subheader("Citation validation")
                st.write("✅ Valid" if ok else f"❌ Invalid citations: {invalid}")

                if enable_cache and not _looks_like_rate_limit(Exception(answer)):
                    qa_cache.append(CachedQA(query=query, query_emb=q_emb, topk_ids=final_ids, answer=answer, created_at=time.time()))
                    qa_cache = qa_cache[-50:]
                    _save_qa_cache(qa_cache)

    with st.expander("Retrieved chunks (top-K)"):
        for ch in final_chunks:
            st.markdown(f"**[{ch.id}]** — `{ch.source}`")
            st.write(ch.text)
