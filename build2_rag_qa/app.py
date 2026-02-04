import os
import re
import numpy as np
import streamlit as st
from openai import OpenAI

# Gemini (OpenAI-compatible)
client = OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY", ""),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

EMBED_MODEL = "text-embedding-004"
CHAT_MODEL = "gemini-2.5-flash"
def chunk_text(text: str, max_chars: int = 900, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks for retrieval."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    # Split into sentences (simple heuristic, good enough for microbuild)
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    buf = ""
    for s in sentences:
        if len(buf) + len(s) + 1 <= max_chars:
            buf = (buf + " " + s).strip()
        else:
            chunks.append(buf)
            # keep a tail overlap to preserve context across chunk boundaries
            buf = ((buf[-overlap:] + " " + s).strip()) if buf else s

    if buf:
        chunks.append(buf)

    return [c for c in chunks if c]
def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts into a 2D numpy array."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vectors = [d.embedding for d in resp.data]
    return np.array(vectors, dtype=np.float32)
def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between matrix a (n,d) and vector b (d,)."""
    if a.size == 0 or b.size == 0:
        return np.array([])
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return a_norm @ b_norm

def retrieve_top_k(chunks: list[str], q: str, k: int = 4) -> list[tuple[int, float, str]]:
    """Return (index, score, chunk) for top-k chunks relevant to question."""
    if not chunks or not q.strip():
        return []

    chunk_vecs = embed_texts(chunks)
    q_vec = embed_texts([q])[0]
    sims = cosine_sim(chunk_vecs, q_vec)

    idxs = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i]), chunks[int(i)]) for i in idxs]
def generate_answer(question: str, retrieved: list[tuple[int, float, str]]) -> str:
    context = "\n\n".join([f"[Chunk {i} | score {s:.3f}]\n{c}" for i, s, c in retrieved])

    prompt = f"""You are a helpful assistant.
Answer the user's question using ONLY the information in the provided chunks.
If the chunks do not contain enough information, say: "I don't have enough information in the document to answer that."

Question:
{question}

Chunks:
{context}
"""
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()
def generate_answer(question: str, retrieved: list[tuple[int, float, str]]) -> str:
    context = "\n\n".join([f"[Chunk {i} | score {s:.3f}]\n{c}" for i, s, c in retrieved])

    prompt = f"""You are a helpful assistant.
Answer the user's question using ONLY the information in the provided chunks.
If the chunks do not contain enough information, say: "I don't have enough information in the document to answer that."

Question:
{question}

Chunks:
{context}
"""
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()

st.set_page_config(page_title="Build 2 — RAG Q&A", layout="wide")
st.title("Build 2 — Doc Q&A (RAG-lite)")
st.caption("Paste a doc → ask a question → retrieves relevant chunks → answers grounded in retrieved text.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Inputs")
    doc = st.text_area("Document", height=260, placeholder="Paste a doc here (JD, notes, PRD, etc.)")
    question = st.text_input("Question", placeholder="e.g., What are the top requirements and risks?")
    k = st.slider("Top K chunks to retrieve", 1, 8, 4)

with col2:
    st.subheader("Output")
    st.write("Will show retrieved chunks + grounded answer here.")

st.divider()
st.subheader("Run")

run = st.button("Retrieve + Answer")
if run:
    if not os.environ.get("GEMINI_API_KEY"):
        st.error("GEMINI_API_KEY not set. Set it in Terminal with: export GEMINI_API_KEY='...'")
        st.stop()

    if not doc.strip():
        st.warning("Paste a document first.")
        st.stop()

    if not question.strip():
        st.warning("Type a question first.")
        st.stop()

    chunks = chunk_text(doc)
    st.write(f"Chunked into **{len(chunks)}** chunks.")

    results = retrieve_top_k(chunks, question, k=k)

    st.subheader("Retrieved chunks")
    for idx, score, chunk in results:
        st.markdown(f"**Chunk {idx} — score {score:.3f}**")
        st.write(chunk)
        st.divider()
    st.subheader("Grounded answer")
    answer = generate_answer(question, results)
    st.write(answer)


