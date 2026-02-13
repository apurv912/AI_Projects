from __future__ import annotations

import csv
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_run_id() -> str:
    return uuid.uuid4().hex


def _short(msg: str, max_len: int = 180) -> str:
    msg = (msg or "").strip().replace("\n", " ")
    return msg[:max_len]


@dataclass
class TelemetryParams:
    top_n: int
    top_k: int
    min_score: float
    rerank_enabled: bool
    retrieve_only: bool


@dataclass
class TelemetryFlags:
    guardrail_triggered: bool = False
    rerank_fallback_used: bool = False


@dataclass
class TelemetryTimingsMs:
    chunking_ms: int = 0
    embed_doc_ms: int = 0
    embed_query_ms: int = 0
    retrieve_ms: int = 0
    rerank_ms: int = 0
    generate_ms: int = 0
    validate_ms: int = 0
    total_ms: int = 0


@dataclass
class TelemetryErrors:
    error_stage: str = ""          # e.g. "rerank" / "generate"
    status_code: str = ""          # keep str for CSV stability (e.g. "429")
    error_message_short: str = ""


@dataclass
class TelemetryCostProxy:
    prompt_chars: int = 0
    context_chars: int = 0
    answer_chars: int = 0
    # tokens if available (optional)
    prompt_tokens: str = ""
    context_tokens: str = ""
    answer_tokens: str = ""


@dataclass
class TelemetryRun:
    # identity
    timestamp: str
    run_id: str

    # doc shape
    doc_length_chars: int
    num_chunks: int

    # config
    params: TelemetryParams
    flags: TelemetryFlags = field(default_factory=TelemetryFlags)
    timings_ms: TelemetryTimingsMs = field(default_factory=TelemetryTimingsMs)
    errors: TelemetryErrors = field(default_factory=TelemetryErrors)

    # outcome/quality
    citation_valid: bool = False
    cited_ids_count: int = 0

    # cost proxy
    cost: TelemetryCostProxy = field(default_factory=TelemetryCostProxy)

    def to_flat_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # flatten nested dicts for CSV
        flat: Dict[str, Any] = {}
        for k, v in d.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    flat[f"{k}.{k2}"] = v2
            else:
                flat[k] = v
        return flat


class TelemetryLogger:
    """
    Append-only logger:
    - outputs/telemetry.jsonl : one JSON record per run
    - outputs/telemetry.csv   : flat rows, append mode; writes header if missing
    """

    def __init__(self, outputs_dir: str = "outputs") -> None:
        self.outputs_dir = Path(outputs_dir)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.outputs_dir / "telemetry.jsonl"
        self.csv_path = self.outputs_dir / "telemetry.csv"

    def append(self, run: TelemetryRun) -> None:
        # JSONL append
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(run), ensure_ascii=False) + "\n")

        # CSV append (flat)
        row = run.to_flat_dict()
        write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0

        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)


class Stopwatch:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def ms(self) -> int:
        return int((time.perf_counter() - self._t0) * 1000)


def detect_status_code(exc: Exception) -> str:
    """
    Best-effort extraction for OpenAI SDK style errors.
    Keep as string for CSV stability.
    """
    # common patterns: exc.status_code, exc.response.status_code, "429" in text
    code = ""
    if hasattr(exc, "status_code"):
        try:
            code = str(getattr(exc, "status_code") or "")
        except Exception:
            pass
    if not code and hasattr(exc, "response"):
        try:
            resp = getattr(exc, "response")
            if resp is not None and hasattr(resp, "status_code"):
                code = str(getattr(resp, "status_code") or "")
        except Exception:
            pass
    if not code:
        txt = str(exc)
        if "429" in txt:
            code = "429"
    return code


def build_run_record(
    *,
    doc_text: str,
    num_chunks: int,
    top_n: int,
    top_k: int,
    min_score: float,
    rerank_enabled: bool,
    retrieve_only: bool,
) -> TelemetryRun:
    return TelemetryRun(
        timestamp=utc_iso(),
        run_id=new_run_id(),
        doc_length_chars=len(doc_text or ""),
        num_chunks=num_chunks,
        params=TelemetryParams(
            top_n=top_n,
            top_k=top_k,
            min_score=min_score,
            rerank_enabled=rerank_enabled,
            retrieve_only=retrieve_only,
        ),
    )
