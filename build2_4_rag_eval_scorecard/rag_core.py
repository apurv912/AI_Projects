import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from openai import OpenAI


GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


@dataclass
class RagConfig:
    chat_model: str
    rerank_model: str
    embed_model: str
    top_n: int
    top_k: int
    sim_threshold: float
    cache_dir: Path


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def build_client() -> OpenAI:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY env var.")
    return OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL, timeout=60)


def chunk_document(text: str, max_chars: int = 900, overlap_chars: int = 150) -> List[Tuple[str, str]]:
    """
    Returns list of (chunk_id, chunk_text).
    Pragmatic chunking: paragraph-aware + sliding window fallback.
    """
    text = text.strip().replace("\r\n", "\n")
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []

    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 <= max_chars:
            buf = (buf + "\n\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            if len(p) <= max_chars:
                buf = p
            else:
                start = 0
                while start < len(p):
                    end = min(len(p), start + max_chars)
                    chunks.append(p[start:end].strip())
                    if end >= len(p):
                        break
                    start = max(0, end - overlap_chars)
                buf = ""
    if buf:
        chunks.append(buf)

    out: List[Tuple[str, str]] = []
    for i, c in enumerate(chunks, start=1):
        out.append((f"C{i:02d}", c))
    return out


def _doc_fingerprint(text: str, embed_model: str) -> str:
    h = hashlib.sha256()
    h.update(embed_model.encode("utf-8"))
    h.update(b"\n---\n")
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:16]


def embed_texts(client: OpenAI, model: str, texts: List[str]) -> np.ndarray:
    resp = client.embeddings.create(model=model, input=texts)
    vecs = [np.array(item.embedding, dtype=np.float32) for item in resp.data]
    mat = np.vstack(vecs)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


def get_chunk_embeddings_cached(
    client: OpenAI,
    cfg: RagConfig,
    doc_text: str,
    chunks: List[Tuple[str, str]],
) -> Tuple[List[str], List[str], np.ndarray]:
    """
    File cache: embeddings + chunk ids/texts keyed by (doc hash, embed_model).
    """
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    fp = _doc_fingerprint(doc_text, cfg.embed_model)

    emb_path = cfg.cache_dir / f"{fp}_embeddings.npy"
    meta_path = cfg.cache_dir / f"{fp}_chunks.json"

    if emb_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        chunk_ids = meta["chunk_ids"]
        chunk_texts = meta["chunk_texts"]
        mat = np.load(emb_path)
        return chunk_ids, chunk_texts, mat

    chunk_ids = [cid for cid, _ in chunks]
    chunk_texts = [ct for _, ct in chunks]
    mat = embed_texts(client, cfg.embed_model, chunk_texts)

    meta_path.write_text(
        json.dumps({"chunk_ids": chunk_ids, "chunk_texts": chunk_texts}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.save(emb_path, mat)
    return chunk_ids, chunk_texts, mat


def retrieve_top_n(
    client: OpenAI,
    cfg: RagConfig,
    chunk_ids: List[str],
    chunk_texts: List[str],
    chunk_embs: np.ndarray,
    question: str,
) -> Tuple[List[Tuple[str, float, str]], float, float]:
    """
    Returns list of (chunk_id, score, chunk_text) sorted desc, plus:
    - q_embed_ms
    - retrieve_ms (scoring + selection only)
    """
    t0 = _now_ms()
    q_emb = embed_texts(client, cfg.embed_model, [question])[0:1, :]
    q_embed_ms = _now_ms() - t0

    t1 = _now_ms()
    scores = (chunk_embs @ q_emb.T).reshape(-1)
    order = np.argsort(-scores)

    top = []
    for idx in order[: max(cfg.top_n, cfg.top_k)]:
        top.append((chunk_ids[idx], float(scores[idx]), chunk_texts[idx]))
    retrieve_ms = _now_ms() - t1
    return top, q_embed_ms, retrieve_ms


def llm_rerank(
    client: OpenAI,
    cfg: RagConfig,
    question: str,
    candidates: List[Tuple[str, float, str]],
) -> Tuple[List[str], bool, float, Optional[str]]:
    """
    LLM rerank: returns ordered chunk_ids.
    Fallback on quota / parse issues (records fallback_used=True).
    """
    t0 = _now_ms()
    err_msg: Optional[str] = None

    packed = []
    for cid, score, text in candidates[: cfg.top_n]:
        excerpt = text.replace("\n", " ")[:350]
        packed.append({"id": cid, "baseline_score": round(score, 4), "excerpt": excerpt})

    sys = "You are a ranking function for RAG. Return ONLY valid JSON."
    user = {
        "question": question,
        "candidates": packed,
        "task": "Return JSON: {\"ranked_ids\": [\"C01\",\"C02\",...]} using only provided ids.",
    }

    try:
        resp = client.chat.completions.create(
            model=cfg.rerank_model,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
            ],
            temperature=0,
        )
        content = (resp.choices[0].message.content or "").strip()
        data = json.loads(content)
        ranked = data.get("ranked_ids", [])
        ranked = [r for r in ranked if isinstance(r, str)]
        valid_ids = {cid for cid, _, _ in candidates[: cfg.top_n]}
        ranked = [r for r in ranked if r in valid_ids]
        if not ranked:
            raise ValueError("Rerank returned empty/invalid ranked_ids.")
        for cid, _, _ in candidates[: cfg.top_n]:
            if cid not in ranked:
                ranked.append(cid)
        return ranked, False, (_now_ms() - t0), None
    except Exception as e:
        err_msg = str(e)
        ranked = [cid for cid, _, _ in candidates[: cfg.top_n]]
        return ranked, True, (_now_ms() - t0), err_msg


def build_context_block(selected: List[Tuple[str, float, str]]) -> str:
    parts = []
    for cid, score, text in selected:
        parts.append(f"[{cid}] (score={score:.3f})\n{text}")
    return "\n\n---\n\n".join(parts)


def generate_answer(
    client: OpenAI,
    cfg: RagConfig,
    question: str,
    selected: List[Tuple[str, float, str]],
) -> Tuple[str, float, Optional[str]]:
    """
    Answer ONLY from chunks. Cite as [C##]. If insufficient, output exactly NO_ANSWER.
    """
    t0 = _now_ms()
    ctx = build_context_block(selected)

    sys = (
        "You are a precise assistant for retrieval-augmented QA. "
        "Use ONLY the provided context chunks. "
        "Cite sources using chunk IDs like [C01]. "
        "If the context does not contain the answer, output exactly: NO_ANSWER"
    )
    user = f"Question:\n{question}\n\nContext chunks:\n{ctx}\n\nAnswer concisely with citations."

    try:
        resp = client.chat.completions.create(
            model=cfg.chat_model,
            messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
            temperature=0,
        )
        ans = (resp.choices[0].message.content or "").strip()
        return ans, (_now_ms() - t0), None
    except Exception as e:
        return "", (_now_ms() - t0), str(e)


_CIT_RE = re.compile(r"\[(C\d{2,3})\]")


def extract_citations(answer: str) -> List[str]:
    return list(dict.fromkeys(_CIT_RE.findall(answer or "")))


def citation_valid(cited_ids: List[str], retrieved_ids: List[str]) -> bool:
    if not cited_ids:
        return True
    s = set(retrieved_ids)
    return all(cid in s for cid in cited_ids)


def should_no_answer(best_score: float, threshold: float) -> bool:
    return best_score < threshold
