from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import rag_core
import guardrails
import citations
from telemetry import (
    Stopwatch,
    TelemetryLogger,
    TelemetryRun,
    build_run_record,
    detect_status_code,
)
from utils import env_str, load_env, read_text_file


OUTPUTS_DIR = "outputs"
TELEMETRY_JSONL = Path(OUTPUTS_DIR) / "telemetry.jsonl"
TELEMETRY_CSV = Path(OUTPUTS_DIR) / "telemetry.csv"


def _percentile_ms(series: pd.Series, p: float) -> Optional[float]:
    if series is None or series.empty:
        return None
    return float(np.percentile(series.dropna().astype(float), p))


def load_telemetry_df() -> pd.DataFrame:
    if TELEMETRY_CSV.exists() and TELEMETRY_CSV.stat().st_size > 0:
        return pd.read_csv(TELEMETRY_CSV)

    if TELEMETRY_JSONL.exists() and TELEMETRY_JSONL.stat().st_size > 0:
        rows = []
        for line in TELEMETRY_JSONL.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        if not rows:
            return pd.DataFrame()

        flat_rows = []
        for r in rows:
            flat = {}
            for k, v in r.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        flat[f"{k}.{k2}"] = v2
                else:
                    flat[k] = v
            flat_rows.append(flat)
        return pd.DataFrame(flat_rows)

    return pd.DataFrame()


def compute_rates(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {}
    n = len(df)

    def rate(col: str) -> float:
        if col not in df.columns:
            return 0.0
        s = df[col]
        return float((s == True).sum()) / n  # noqa: E712

    err_rate = 0.0
    if "errors.error_stage" in df.columns:
        err_rate = float((df["errors.error_stage"].fillna("") != "").sum()) / n

    by_stage = {}
    if "errors.error_stage" in df.columns:
        vc = df["errors.error_stage"].fillna("").value_counts()
        for stage, cnt in vc.items():
            if stage == "":
                continue
            by_stage[stage] = float(cnt) / n

    return {
        "error_rate_overall": err_rate,
        "rerank_fallback_rate": rate("flags.rerank_fallback_used"),
        "guardrail_trigger_rate": rate("flags.guardrail_triggered"),
        "citation_validity_rate": rate("citation_valid"),
        **{f"error_rate_{k}": v for k, v in by_stage.items()},
    }


def run_pipeline_with_telemetry(
    *,
    doc_text: str,
    question: str,
    top_n: int,
    top_k: int,
    min_score: float,
    rerank_enabled: bool,
    retrieve_only: bool,
) -> Tuple[TelemetryRun, Dict[str, Any]]:
    total_sw = Stopwatch()

    # 1) chunking
    sw = Stopwatch()
    chunks = rag_core.chunk_document(doc_text)
    run = build_run_record(
        doc_text=doc_text,
        num_chunks=len(chunks),
        top_n=top_n,
        top_k=top_k,
        min_score=min_score,
        rerank_enabled=rerank_enabled,
        retrieve_only=retrieve_only,
    )
    run.timings_ms.chunking_ms = sw.ms()
    run.cost.prompt_chars = len(question or "")

    try:
        client = rag_core.build_client()
        cfg = rag_core.RagConfig(
            chat_model=env_str("GEMINI_CHAT_MODEL", "gemini-1.5-flash"),
            rerank_model=env_str("GEMINI_RERANK_MODEL", "gemini-1.5-flash"),
            embed_model=env_str("GEMINI_EMBED_MODEL", "text-embedding-004"),
            top_n=top_n,
            top_k=top_k,
            min_score=min_score,
            rerank_enabled=rerank_enabled,
        )

        # 2) embed doc (cached)
        sw = Stopwatch()
        chunk_ids, chunk_texts, chunk_mat = rag_core.get_chunk_embeddings_cached(
            client, cfg, doc_text, chunks
        )
        run.timings_ms.embed_doc_ms = sw.ms()
        run.cost.context_chars = int(sum(len(t) for t in chunk_texts))

        # 3) embed query
        sw = Stopwatch()
        q_emb = rag_core.embed_texts(client, cfg.embed_model, [question])[0:1, :]
        run.timings_ms.embed_query_ms = sw.ms()

        # 4) retrieve
        sw = Stopwatch()
        retrieved = rag_core.retrieve_top_n(
            question_emb=q_emb,
            chunk_ids=chunk_ids,
            chunk_texts=chunk_texts,
            chunk_mat=chunk_mat,
            top_n=cfg.top_n,
            top_k=cfg.top_k,
            min_score=cfg.min_score,
        )
        run.timings_ms.retrieve_ms = sw.ms()

        # Guardrail stage
        guardrail = guardrails.guardrail_triggered(retrieved)
        run.flags.guardrail_triggered = bool(guardrail)

        if retrieve_only:
            run.timings_ms.total_ms = total_sw.ms()
            return run, {
                "warning": "Retrieval-only mode enabled (quota saver).",
                "retrieved": retrieved,
            }

        # 5) rerank (optional, 429 fallback flag required)
        reranked = retrieved
        status_code = ""
        rerank_fallback_429 = False

        if cfg.rerank_enabled:
            sw = Stopwatch()
            try:
                reranked = rag_core.llm_rerank(client, cfg, question, retrieved)
            except Exception as e:
                status_code = detect_status_code(e)
                # record as a degraded stage error (but we can still proceed)
                run.errors.error_stage = "rerank"
                run.errors.status_code = status_code
                run.errors.error_message_short = str(e)[:180]

                # fallback always to embedding order
                reranked = retrieved

                # requirement: if 429 occurs, flag fallback
                if status_code == "429":
                    rerank_fallback_429 = True
            run.timings_ms.rerank_ms = sw.ms()

        run.flags.rerank_fallback_used = bool(rerank_fallback_429)

        # 6) generate
        sw = Stopwatch()
        try:
            context_block = rag_core.build_context_block(reranked)
            answer = rag_core.generate_answer(client, cfg, question, context_block)
            run.timings_ms.generate_ms = sw.ms()
            run.cost.answer_chars = len(answer or "")

            # 7) validate
            sw2 = Stopwatch()
            is_valid, cited_count = citations.validate_citations(answer, reranked)
            cited_ids = citations.extract_cited_ids(answer)
            run.timings_ms.validate_ms = sw2.ms()

            run.citation_valid = bool(is_valid)
            run.cited_ids_count = int(cited_count)

            run.timings_ms.total_ms = total_sw.ms()
            return run, {
                "answer": answer,
                "cited_ids": cited_ids,
                "retrieved": retrieved,
                "reranked": reranked,
                "rerank_fallback": bool(rerank_fallback_429),
                "guardrail": guardrail,
            }

        except Exception as e:
            run.timings_ms.generate_ms = sw.ms()
            run.errors.error_stage = "generate"
            run.errors.status_code = detect_status_code(e)
            run.errors.error_message_short = str(e)[:180]
            run.timings_ms.total_ms = total_sw.ms()

            msg = "Generation failed. "
            if run.errors.status_code == "429":
                msg += "Quota/Rate limit (429). Try retrieval-only, or wait and retry."
            else:
                msg += "See telemetry for details."
            return run, {"error": msg, "retrieved": retrieved}

    except Exception as e:
        run.errors.error_stage = run.errors.error_stage or "client_init"
        run.errors.status_code = detect_status_code(e)
        run.errors.error_message_short = str(e)[:180]
        run.timings_ms.total_ms = total_sw.ms()
        return run, {"error": "Client/init failed (check GEMINI_API_KEY, model env vars).", "details": str(e)[:300]}


def ui_tab_run() -> None:
    st.subheader("Run Q&A (logs telemetry)")

    colA, colB = st.columns(2)
    with colA:
        doc_path = st.text_input("Document path", value="data/sample_doc.txt")
    with colB:
        question = st.text_input("Question", value="What is this document about?")

    with st.expander("Run parameters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            top_n = st.number_input("top_n", min_value=1, max_value=50, value=8, step=1)
        with c2:
            top_k = st.number_input("top_k", min_value=1, max_value=50, value=5, step=1)
        with c3:
            min_score = st.number_input("min_score", min_value=0.0, max_value=1.0, value=0.15, step=0.01)
        with c4:
            rerank_enabled = st.checkbox("rerank_enabled", value=True)

        retrieve_only = st.checkbox("retrieve_only (quota saver)", value=False)

    if st.button("Run", type="primary"):
        try:
            doc_text = read_text_file(doc_path)
        except Exception as e:
            st.error(f"Could not read doc: {e}")
            return

        logger = TelemetryLogger(outputs_dir=OUTPUTS_DIR)

        run, result = run_pipeline_with_telemetry(
            doc_text=doc_text,
            question=question,
            top_n=int(top_n),
            top_k=int(top_k),
            min_score=float(min_score),
            rerank_enabled=bool(rerank_enabled),
            retrieve_only=bool(retrieve_only),
        )

        logger.append(run)
        st.success(f"Logged run_id: {run.run_id}")

        if "details" in result:
            st.caption(result["details"])

        if "warning" in result:
            st.warning(result["warning"])
        if "error" in result:
            st.error(result["error"])

        if "answer" in result:
            st.markdown("### Answer")
            st.write(result["answer"])
            st.markdown("### Citations")
            st.write(result.get("cited_ids", []))
            if result.get("rerank_fallback"):
                st.info("Rerank fallback used due to 429 (quota/rate limit).")

        with st.expander("Retrieved (debug)", expanded=False):
            st.write(result.get("retrieved", []))

        if "reranked" in result:
            with st.expander("Reranked (debug)", expanded=False):
                st.write(result.get("reranked", []))

        with st.expander("Telemetry (this run)", expanded=False):
            st.json(run.to_flat_dict())


def ui_tab_ops() -> None:
    st.subheader("Ops Dashboard (SLO view)")
    df = load_telemetry_df()

    if df.empty:
        st.info("No telemetry yet. Run a few queries in the Run tab.")
        return

    time_cols = [c for c in df.columns if c.startswith("timings_ms.")]
    for c in time_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    st.markdown("### Latency (ms)")
    total = df.get("timings_ms.total_ms", pd.Series(dtype=float))
    p50_total = _percentile_ms(total, 50)
    p95_total = _percentile_ms(total, 95)

    m1, m2 = st.columns(2)
    m1.metric("p50 total", f"{p50_total:.0f} ms" if p50_total is not None else "—")
    m2.metric("p95 total", f"{p95_total:.0f} ms" if p95_total is not None else "—")

    st.markdown("### Per-stage p50 / p95")
    rows = []
    for c in time_cols:
        if c == "timings_ms.total_ms":
            continue
        s = df[c]
        rows.append({
            "stage": c.replace("timings_ms.", ""),
            "p50_ms": _percentile_ms(s, 50),
            "p95_ms": _percentile_ms(s, 95),
        })
    stage_df = pd.DataFrame(rows).sort_values("p95_ms", ascending=False)
    st.dataframe(stage_df, use_container_width=True)

    st.markdown("### Reliability rates")
    rates = compute_rates(df)
    if rates:
        rcols = st.columns(3)
        keys = [
            ("error_rate_overall", "Error rate (overall)"),
            ("rerank_fallback_rate", "Rerank fallback rate (429)"),
            ("guardrail_trigger_rate", "Guardrail trigger rate"),
            ("citation_validity_rate", "Citation validity rate"),
        ]
        for i, (k, label) in enumerate(keys):
            val = rates.get(k, 0.0)
            rcols[i % 3].metric(label, f"{val*100:.1f}%")

        err_stage_cols = [k for k in rates.keys() if k.startswith("error_rate_") and k != "error_rate_overall"]
        if err_stage_cols:
            err_rows = []
            for k in err_stage_cols:
                err_rows.append({"stage": k.replace("error_rate_", ""), "rate": rates[k]})
            st.markdown("#### Error rate by stage")
            st.dataframe(pd.DataFrame(err_rows).sort_values("rate", ascending=False), use_container_width=True)

    st.markdown("### Last 20 runs")
    show_cols = [
        "timestamp",
        "run_id",
        "timings_ms.total_ms",
        "errors.error_stage",
        "errors.status_code",
        "flags.rerank_fallback_used",
        "flags.guardrail_triggered",
        "citation_valid",
        "cited_ids_count",
        "params.retrieve_only",
    ]
    existing = [c for c in show_cols if c in df.columns]
    last = df.sort_values("timestamp", ascending=False).head(20)[existing]
    st.dataframe(last, use_container_width=True)


def main() -> None:
    load_env()
    st.set_page_config(page_title="Build 2.5 — RAG Telemetry + Ops", layout="wide")
    st.title("RAG Engine Builder — Build 2.5: Telemetry + Ops Dashboard")
    st.caption(
        "PM-first observability: stage timings, reliability flags, 429 fallback, and SLO-style summaries."
    )

    tab1, tab2 = st.tabs(["Run Q&A", "Ops Dashboard"])
    with tab1:
        ui_tab_run()
    with tab2:
        ui_tab_ops()


if __name__ == "__main__":
    main()
