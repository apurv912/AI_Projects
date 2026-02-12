import argparse
import csv
import os
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rag_core import (
    RagConfig,
    build_client,
    chunk_document,
    get_chunk_embeddings_cached,
    retrieve_top_n,
    llm_rerank,
    generate_answer,
    extract_citations,
    citation_valid,
    should_no_answer,
)


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_questions(md_path: Path) -> List[Tuple[str, str]]:
    """
    Extracts questions in format: Q##: question text
    Returns list of (qid, question)
    """
    lines = read_text_file(md_path).splitlines()
    out: List[Tuple[str, str]] = []
    for ln in lines:
        ln = ln.strip()
        if ln.startswith("Q") and ":" in ln:
            left, right = ln.split(":", 1)
            qid = left.strip()
            q = right.strip()
            if qid and q:
                out.append((qid, q))
    if not out:
        raise RuntimeError(f"No questions found in {md_path}. Expected lines like: Q01: ...")
    return out


def env_int(name: str, default: int) -> int:
    v = os.environ.get(name, "").strip()
    return int(v) if v else default


def env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "").strip()
    return float(v) if v else default


def env_str(name: str, default: str) -> str:
    v = os.environ.get(name, "").strip()
    return v if v else default


def ensure_dirs(out_dir: Path, cache_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)


def safe_mean(xs: List[float]) -> float:
    return float(statistics.mean(xs)) if xs else 0.0


def pct(numer: int, denom: int) -> float:
    return 100.0 * numer / denom if denom > 0 else 0.0


def write_csv(rows: List[Dict[str, Any]], csv_path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def to_md_report(rows: List[Dict[str, Any]], md_path: Path, doc_name: str) -> None:
    modes = sorted(set(r["mode"] for r in rows))
    total_q = len(set(r["qid"] for r in rows))

    lines: List[str] = []
    lines.append("# Build 2.4 — Eval Scorecard Report")
    lines.append("")
    lines.append(f"- Document: **{doc_name}**")
    lines.append(f"- Questions: **{total_q}**")
    lines.append(f"- Rows (A/B): **{len(rows)}**")
    lines.append("")

    def summarize(mode: str) -> Dict[str, Any]:
        mr = [r for r in rows if r["mode"] == mode]

        # "Answered" means: generation succeeded AND not retrieve-only AND not empty
        answered = [
            r for r in mr
            if r["generation_error"] == "" and r["answer"] not in ("", "RETRIEVE_ONLY")
        ]
        with_cites = [r for r in answered if r["cited_ids"] != "" and r["answer"] != "NO_ANSWER"]
        cite_valid = [r for r in with_cites if r["citation_valid"] == "true"]

        no_answer = [r for r in mr if r["no_answer"] == "true"]
        gen_err = [r for r in mr if r["generation_error"] != ""]
        fallback = [r for r in mr if r["rerank_fallback_used"] == "true"]

        retrieve_ms = [float(r["retrieve_ms"]) for r in mr]
        rerank_ms = [float(r["rerank_ms"]) for r in mr]
        gen_ms = [float(r["generate_ms"]) for r in mr]
        total_ms = [float(r["total_ms"]) for r in mr]

        return {
            "rows": len(mr),
            "answered_rows": len(answered),
            "citation_coverage_rate": pct(len(with_cites), len(answered)),
            "citation_validity_rate": pct(len(cite_valid), len(with_cites)),
            "no_answer_rate": pct(len(no_answer), len(mr)),
            "gen_error_rate": pct(len(gen_err), len(mr)),
            "rerank_fallback_rate": pct(len(fallback), len(mr)),
            "avg_retrieve_ms": safe_mean(retrieve_ms),
            "avg_rerank_ms": safe_mean(rerank_ms),
            "avg_generate_ms": safe_mean(gen_ms),
            "avg_total_ms": safe_mean(total_ms),
        }

    lines.append("## Scorecard (aggregated)")
    lines.append("")
    lines.append("| Mode | Citation validity* | Citation coverage** | No-answer rate | Rerank fallback rate | Avg retrieve ms | Avg rerank ms | Avg generate ms | Avg total ms | Gen error rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for m in modes:
        s = summarize(m)
        lines.append(
            f"| {m} | {s['citation_validity_rate']:.1f}% | {s['citation_coverage_rate']:.1f}% | {s['no_answer_rate']:.1f}% | "
            f"{s['rerank_fallback_rate']:.1f}% | {s['avg_retrieve_ms']:.1f} | {s['avg_rerank_ms']:.1f} | "
            f"{s['avg_generate_ms']:.1f} | {s['avg_total_ms']:.1f} | {s['gen_error_rate']:.1f}% |"
        )

    lines.append("")
    lines.append("*Citation validity is computed only on answered rows that actually include citations (avoids inflating with NO_ANSWER).")
    lines.append("**Citation coverage = % of answered rows that include at least one citation.")
    lines.append("")

    lines.append("## What to look at (PM)")
    lines.append("")
    lines.append("- If **rerank ON** increases citation validity/coverage with acceptable latency, it’s likely worth defaulting ON.")
    lines.append("- If fallback rate is high, rerank may be too unreliable under quota — treat as best-effort enhancement.")
    lines.append("- If no-answer rate looks wrong, tune `RAG_SIM_THRESHOLD` (guardrail strictness).")
    lines.append("")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_eval(doc_path: Path, retrieve_only: bool, max_q: int = 0) -> None:
    root = Path(__file__).parent
    out_dir = root / "outputs"
    cache_dir = root / ".cache"
    ensure_dirs(out_dir, cache_dir)

    cfg = RagConfig(
        chat_model=env_str("GEMINI_CHAT_MODEL", "gemini-2.0-flash"),
        rerank_model=env_str("GEMINI_RERANK_MODEL", "gemini-2.0-flash"),
        embed_model=env_str("GEMINI_EMBED_MODEL", "gemini-embedding-001"),
        top_n=env_int("RAG_TOP_N", 8),
        top_k=env_int("RAG_TOP_K", 4),
        sim_threshold=env_float("RAG_SIM_THRESHOLD", 0.20),
        cache_dir=cache_dir,
    )

    client = build_client()

    doc_text = read_text_file(doc_path)
    chunks = chunk_document(doc_text)
    chunk_ids, chunk_texts, chunk_embs = get_chunk_embeddings_cached(client, cfg, doc_text, chunks)

    questions = load_questions(root / "eval_questions.md")
    if max_q and max_q > 0:
        questions = questions[:max_q]

    rows: List[Dict[str, Any]] = []

    modes = [("OFF", False), ("ON", True)]

    total_runs = len(questions) * len(modes)
    run_idx = 0

    for qid, q in questions:
        for mode_name, rerank_enabled in modes:
            run_idx += 1
            print(f"[{run_idx}/{total_runs}] {qid} mode={mode_name} (retrieve_only={retrieve_only})")

            top_candidates, q_embed_ms, retrieve_ms = retrieve_top_n(
                client, cfg, chunk_ids, chunk_texts, chunk_embs, q
            )
            best_score = float(top_candidates[0][1]) if top_candidates else 0.0

            rerank_ids: List[str] = [cid for cid, _, _ in top_candidates[: cfg.top_n]]
            rerank_fallback_used = False
            rerank_ms = 0.0
            rerank_error = ""

            if rerank_enabled:
                ranked, fb, r_ms, err = llm_rerank(client, cfg, q, top_candidates[: cfg.top_n])
                rerank_ids = ranked
                rerank_fallback_used = fb
                rerank_ms = float(r_ms)
                rerank_error = err or ""

            id_to_item = {cid: (score, text) for cid, score, text in top_candidates}
            selected: List[Tuple[str, float, str]] = []
            for cid in rerank_ids:
                if cid in id_to_item:
                    score, text = id_to_item[cid]
                    selected.append((cid, float(score), str(text)))
                if len(selected) >= cfg.top_k:
                    break

            retrieved_ids = [cid for cid, _, _ in selected]

            no_answer = should_no_answer(best_score, cfg.sim_threshold)

            answer = ""
            generate_ms = 0.0
            generation_error = ""

            if no_answer:
                answer = "NO_ANSWER"
            elif retrieve_only:
                answer = "RETRIEVE_ONLY"
            else:
                ans, g_ms, err = generate_answer(client, cfg, q, selected)
                answer = ans
                generate_ms = float(g_ms)
                generation_error = err or ""
                if answer.strip() == "NO_ANSWER":
                    no_answer = True

            cited = extract_citations(answer) if answer else []
            cit_ok = citation_valid(cited, retrieved_ids)

            total_ms = float(q_embed_ms + retrieve_ms + rerank_ms + generate_ms)

            rows.append(
                {
                    "qid": qid,
                    "question": q,
                    "mode": mode_name,
                    "rerank_enabled": "true" if rerank_enabled else "false",
                    "retrieved_ids": ",".join(retrieved_ids),
                    "best_score": f"{best_score:.4f}",
                    "no_answer": "true" if no_answer else "false",
                    "answer": answer,
                    "cited_ids": ",".join(cited),
                    "citation_valid": "true" if cit_ok else "false",
                    "rerank_fallback_used": "true" if rerank_fallback_used else "false",
                    "rerank_error": rerank_error,
                    "generation_error": generation_error,
                    "q_embed_ms": f"{float(q_embed_ms):.1f}",
                    "retrieve_ms": f"{float(retrieve_ms):.1f}",
                    "rerank_ms": f"{float(rerank_ms):.1f}",
                    "generate_ms": f"{float(generate_ms):.1f}",
                    "total_ms": f"{float(total_ms):.1f}",
                }
            )

    csv_path = out_dir / "eval_results.csv"
    md_path = out_dir / "eval_report.md"

    write_csv(rows, csv_path)
    to_md_report(rows, md_path, doc_path.name)

    print(f"✅ Wrote CSV: {csv_path}")
    print(f"✅ Wrote report: {md_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build 2.4 — RAG Eval Runner (A/B rerank OFF vs ON)")
    p.add_argument("--doc", type=str, required=True, help="Path to document text file (e.g., sample_doc.txt)")
    p.add_argument("--retrieve-only", action="store_true", help="Skip generation (saves quota); still measures retrieval/rerank.")
    p.add_argument("--max-q", type=int, default=0, help="Limit number of questions (0 = all).")
    args = p.parse_args()

    run_eval(Path(args.doc), retrieve_only=bool(args.retrieve_only), max_q=int(args.max_q))


if __name__ == "__main__":
    main()
