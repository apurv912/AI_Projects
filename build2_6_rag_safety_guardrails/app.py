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
from pii_redaction import redact_pii
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

    def rate_true(col: str) -> float:
        if col not in df.columns:
            return 0.0
        s = df[col]
        return float((s == True).sum()) / n  # noqa: E712

    err_rate = 0.0
    if "errors.error_stage" in df.columns:
        err_rate = float((df["errors.error_stage"].fillna("") != "").sum()) / n

    return {
        "error_rate_overall": err_rate,
        "rerank_fallback_rate": rate_true("flags.rerank_fallback_used"),
        "guardrail_trigger_rate": rate_true("flags.guardrail_triggered"),
        "citation_validity_rate": rate_true("citation_valid"),
        "injection_detected_rate": rate_true("flags.injection_detected"),
        "refusal_rate": rate_true("flags.refusal_triggered"),
        "grounding_failure_rate": rate_true("flags.grounding_failed"),
        "pii_redaction_rate": rate_true("flags.pii_redacted"),
    }


def _filter_by_threshold(items: list[tuple[str, float, str]], thr: float) -> list[tuple[str, float, str]]:
    return [x for x in items if float(x[1]) >= float(thr)]


def _rerank_to_tuples(
    candidates: list[tuple[str, float, str]],
    ranked_ids: list[str],
) -> list[tuple[str, float, str]]:
    m = {cid: (cid, score, text) for cid, score, text in candidates}
    out = []
    for rid in ranked_ids:
        if rid in m:
            out.append(m[rid])
    for cid, score, text in candidates:
        if cid not in ranked_ids:
            out.append((cid, score, text))
    return out


def _pii_redact_selected(selected: list[tuple[str, float, str]]) -> tuple[list[tuple[str, float, str]], bool]:
    any_redacted = False
    out = []
    for cid, score, text in selected:
        rr = redact_pii(text)
        any_redacted = any_redacted or bool(rr.redacted)
        out.append((cid, score, rr.text))
    return out, any_redacted


def run_pipeline_with_telemetry(
    *,
    doc_text: str,
    question: str,
    top_n: int,
    top_k: int,
    sim_threshold: float,
    rerank_enabled: bool,
    retrieve_only: bool,
    pii_redaction_enabled: bool,
) -> Tuple[TelemetryRun, Dict[str, Any]]:
    total_sw = Stopwatch()

    decision = guardrails.decide_guardrails(question)

    # chunking
    sw = Stopwatch()
    chunks = rag_core.chunk_document(doc_text)
    run = build_run_record(
        doc_text=doc_text,
        num_chunks=len(chunks),
        top_n=top_n,
        top_k=top_k,
        min_score=sim_threshold,
        rerank_enabled=rerank_enabled,
        retrieve_only=retrieve_only,
    )
    run.timings_ms.chunking_ms = sw.ms()
    run.cost.prompt_chars = len(question or "")

    run.flags.injection_detected = bool(decision.injection_detected)
    run.flags.risk_domain = str(decision.risk_domain)

    try:
        client = rag_core.build_client()
        cfg = rag_core.RagConfig(
            chat_model=env_str("GEMINI_CHAT_MODEL", "gemini-1.5-flash"),
            rerank_model=env_str("GEMINI_RERANK_MODEL", "gemini-1.5-flash"),
            embed_model=env_str("GEMINI_EMBED_MODEL", "gemini-embedding-001"),
            top_n=int(top_n),
            top_k=int(top_k),
            sim_threshold=float(sim_threshold),
            cache_dir=Path(".cache"),
        )

        # embed doc
        sw = Stopwatch()
        chunk_ids, chunk_texts, chunk_embs = rag_core.get_chunk_embeddings_cached(client, cfg, doc_text, chunks)
        run.timings_ms.embed_doc_ms = sw.ms()
        run.cost.context_chars = int(sum(len(t) for t in chunk_texts))

        # retrieve (rag_core embeds query inside)
        top, q_embed_ms, retrieve_ms = rag_core.retrieve_top_n(
            client=client,
            cfg=cfg,
            chunk_ids=chunk_ids,
            chunk_texts=chunk_texts,
            chunk_embs=chunk_embs,
            question=question,
        )
        run.timings_ms.embed_query_ms = int(q_embed_ms)
        run.timings_ms.retrieve_ms = int(retrieve_ms)

        # apply similarity threshold (Build 2.1 style)
        retrieved = _filter_by_threshold(top, cfg.sim_threshold)

        # legacy guardrail rate (similarity/no-evidence)
        run.flags.guardrail_triggered = guardrails.legacy_guardrail_triggered(retrieved, cfg.sim_threshold)

        # strict mode: injection => require evidence strength
        if decision.strict_mode and not guardrails.evidence_strength_ok(retrieved, cfg.sim_threshold):
            run.flags.refusal_triggered = True
            run.timings_ms.total_ms = total_sw.ms()
            return run, {
                "answer": guardrails.fail_closed_message(),
                "retrieved": retrieved,
                "refused": True,
                "reason": "weak_evidence_strict_mode",
                "decision": decision.__dict__,
            }

        # risky domain routing: refuse/escalate without giving advice
        if decision.risk_domain != "none":
            run.flags.refusal_triggered = True
            run.timings_ms.total_ms = total_sw.ms()
            return run, {
                "answer": guardrails.build_risky_domain_safe_message(decision.risk_domain),
                "retrieved": retrieved,
                "refused": True,
                "reason": f"risk_domain_{decision.risk_domain}",
                "decision": decision.__dict__,
            }

        if retrieve_only:
            run.timings_ms.total_ms = total_sw.ms()
            return run, {
            "warning": "Retrieval-only mode enabled (quota saver).",
            "retrieved": retrieved,
            "top_ids": [cid for cid, _, _ in retrieved[: cfg.top_k]],
            "top_scores": [float(score) for _, score, _ in retrieved[: cfg.top_k]],
        }


        # rerank (optional)
        selected_candidates = retrieved[: max(cfg.top_n, cfg.top_k)]
        reranked = selected_candidates
        rerank_fallback = False

        if rerank_enabled and selected_candidates:
            ranked_ids, fallback_used, rerank_ms, err_msg = rag_core.llm_rerank(
                client=client, cfg=cfg, question=question, candidates=selected_candidates
            )
            run.timings_ms.rerank_ms = int(rerank_ms)
            rerank_fallback = bool(fallback_used)
            if err_msg:
                run.errors.error_stage = "rerank"
                run.errors.status_code = detect_status_code(Exception(err_msg))
                run.errors.error_message_short = str(err_msg)[:180]
            reranked = _rerank_to_tuples(selected_candidates, ranked_ids)

        run.flags.rerank_fallback_used = bool(rerank_fallback)

        # take top_k for answering
        selected = reranked[: cfg.top_k] if reranked else []

        # optional PII redaction on context
        if pii_redaction_enabled and selected:
            selected, redacted = _pii_redact_selected(selected)
            run.flags.pii_redacted = bool(redacted)
        else:
            run.flags.pii_redacted = False

        # generate (rag_core returns (answer, ms, err))
        ans, gen_ms, gen_err = rag_core.generate_answer(
            client=client,
            cfg=cfg,
            question=question,
            selected=selected,
        )
        run.timings_ms.generate_ms = int(gen_ms)
        run.cost.answer_chars = len(ans or "")

        if gen_err:
            run.errors.error_stage = "generate"
            run.errors.status_code = detect_status_code(Exception(gen_err))
            run.errors.error_message_short = str(gen_err)[:180]
            run.timings_ms.total_ms = total_sw.ms()
            return run, {"error": "Generation failed (see telemetry).", "retrieved": retrieved}

        # citations validate
        swv = Stopwatch()
        cited_ids = citations.extract_cited_ids(ans)
        is_valid, cited_count = citations.validate_citations(ans, selected)
        run.timings_ms.validate_ms = swv.ms()
        run.citation_valid = bool(is_valid)
        run.cited_ids_count = int(cited_count)

        # strict grounding gate (fail closed)
        grounded_ok, reason = guardrails.grounding_verdict(
            answer_text=ans,
            retrieved=selected,
            citation_valid=run.citation_valid,
            strict_mode=True,
            cited_ids=cited_ids,
        )

        # If model says NO_ANSWER -> treat as safe refusal (not a “grounding failure”)
        if (ans or "").strip() == "NO_ANSWER":
            run.flags.refusal_triggered = True
            run.timings_ms.total_ms = total_sw.ms()
            return run, {
                "answer": guardrails.fail_closed_message(),
                "retrieved": retrieved,
                "reranked": selected,
                "refused": True,
                "reason": "model_no_answer",
            }

        if not grounded_ok:
            run.flags.refusal_triggered = True
            run.flags.grounding_failed = True
            run.timings_ms.total_ms = total_sw.ms()
            return run, {
                "answer": guardrails.fail_closed_message(),
                "retrieved": retrieved,
                "reranked": selected,
                "refused": True,
                "reason": reason,
            }

        run.timings_ms.total_ms = total_sw.ms()
        return run, {
            "answer": ans,
            "cited_ids": cited_ids,
            "retrieved": retrieved,
            "reranked": selected,
            "decision": decision.__dict__,
        }

    except Exception as e:
        run.errors.error_stage = run.errors.error_stage or "client_init"
        run.errors.status_code = detect_status_code(e)
        run.errors.error_message_short = str(e)[:180]
        run.timings_ms.total_ms = total_sw.ms()
        return run, {
            "error": "Client/init failed (check GEMINI_API_KEY, model env vars).",
            "details": str(e)[:500],
        }


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
            sim_threshold = st.number_input("sim_threshold", min_value=0.0, max_value=1.0, value=0.15, step=0.01)
        with c4:
            rerank_enabled = st.checkbox("rerank_enabled", value=True)

        retrieve_only = st.checkbox("retrieve_only (quota saver)", value=False)
        pii_redaction_enabled = st.checkbox("pii_redaction (mask emails/phones in context)", value=True)

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
            sim_threshold=float(sim_threshold),
            rerank_enabled=bool(rerank_enabled),
            retrieve_only=bool(retrieve_only),
            pii_redaction_enabled=bool(pii_redaction_enabled),
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

        if result.get("refused"):
            st.info("Safety mode: refused / not enough info (fail-closed).")

        if "cited_ids" in result:
            st.markdown("### Citations")
            st.write(result.get("cited_ids", []))

        with st.expander("Retrieved (debug)", expanded=False):
            st.write(result.get("retrieved", []))

        if "reranked" in result:
            with st.expander("Selected for answer (debug)", expanded=False):
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
        rows.append({"stage": c.replace("timings_ms.", ""), "p50_ms": _percentile_ms(s, 50), "p95_ms": _percentile_ms(s, 95)})
    stage_df = pd.DataFrame(rows).sort_values("p95_ms", ascending=False)
    st.dataframe(stage_df, use_container_width=True)

    st.markdown("### Safety & Reliability rates")
    rates = compute_rates(df)
    if rates:
        cols = st.columns(3)
        items = [
            ("error_rate_overall", "Error rate"),
            ("citation_validity_rate", "Citation validity rate"),
            ("injection_detected_rate", "Injection detected rate"),
            ("refusal_rate", "Refusal rate"),
            ("grounding_failure_rate", "Grounding failure rate"),
            ("pii_redaction_rate", "PII redaction rate"),
            ("rerank_fallback_rate", "Rerank fallback rate"),
            ("guardrail_trigger_rate", "Legacy guardrail rate"),
        ]
        for i, (k, label) in enumerate(items):
            cols[i % 3].metric(label, f"{rates.get(k, 0.0) * 100:.1f}%")

    st.markdown("### Last 20 runs")
    show_cols = [
        "timestamp",
        "run_id",
        "timings_ms.total_ms",
        "errors.error_stage",
        "errors.status_code",
        "flags.injection_detected",
        "flags.risk_domain",
        "flags.pii_redacted",
        "flags.refusal_triggered",
        "flags.grounding_failed",
        "flags.rerank_fallback_used",
        "citation_valid",
        "cited_ids_count",
        "params.retrieve_only",
    ]
    existing = [c for c in show_cols if c in df.columns]
    last = df.sort_values("timestamp", ascending=False).head(20)[existing]
    st.dataframe(last, use_container_width=True)


def main() -> None:
    load_env()
    st.set_page_config(page_title="Build 2.6 — RAG Safety Guardrails", layout="wide")
    st.title("RAG Engine Builder — Build 2.6: Safety + Robustness Guardrails")
    st.caption("Safety-first RAG: injection detection, strict grounding (fail-closed), risky-domain routing, optional PII redaction, and measurable safety rates.")

    tab1, tab2 = st.tabs(["Run Q&A", "Ops Dashboard"])
    with tab1:
        ui_tab_run()
    with tab2:
        ui_tab_ops()


if __name__ == "__main__":
    main()
