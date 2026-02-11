"""
Build 2.3 — LLM Reranker (PM-first)

Why this exists (PM-level):
- Improves answer quality by re-ordering semantically-close candidates using an LLM judge.
- Must degrade gracefully under quota/rate limits to preserve reliability and user trust.

Contract:
- Input: query + stage-1 candidates (embedding-ranked)
- Output: top-k candidates, in final order + metadata flags (fallback_used)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RerankResult:
    ranked: List[Dict[str, Any]]
    fallback_used: bool
    fallback_reason: Optional[str] = None
    raw_model_output: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None


def _looks_like_rate_limit(err: Exception) -> bool:
    msg = str(err).lower()
    return ("rate limit" in msg) or ("ratelimit" in msg) or ("429" in msg) or ("resource exhausted" in msg) or ("quota" in msg)


def _safe_json_loads(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _extract_json_object(text: str) -> Optional[dict]:
    """
    Gemini/OpenAI-compat may wrap JSON in prose or code fences.
    Robust extraction = fewer brittle failures = higher reliability.
    """
    if not text:
        return None
    t = text.strip()
    if "```" in t:
        t = t.replace("```json", "").replace("```JSON", "").replace("```", "").strip()

    parsed = _safe_json_loads(t)
    if parsed:
        return parsed

    i = t.find("{")
    j = t.rfind("}")
    if i != -1 and j != -1 and j > i:
        return _safe_json_loads(t[i : j + 1].strip())
    return None


class LLMReranker:
    def __init__(
        self,
        client: Any,
        model: str,
        max_candidates_in_prompt: int = 20,
        retry_attempts: int = 2,
        base_backoff_s: float = 1.2,
    ) -> None:
        self.client = client
        self.model = model
        self.max_candidates_in_prompt = max_candidates_in_prompt
        self.retry_attempts = retry_attempts
        self.base_backoff_s = base_backoff_s

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int,
    ) -> RerankResult:
        if not candidates:
            return RerankResult(ranked=[], fallback_used=True, fallback_reason="no_candidates")

        # Cost/latency + correctness guardrail:
        # If reranking can't improve (<=K items), don't spend a rerank call.
        if len(candidates) <= 1 or len(candidates) <= top_k:
            return RerankResult(
                ranked=candidates[:top_k],
                fallback_used=True,
                fallback_reason="insufficient_candidates_for_rerank",
            )

        capped = candidates[: self.max_candidates_in_prompt]
        prompt = self._build_prompt(query=query, candidates=capped, top_k=top_k)

        last_err: Optional[Exception] = None
        for attempt in range(self.retry_attempts + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a precise ranking function. Follow the output schema exactly."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                )

                content = resp.choices[0].message.content or ""
                parsed = _extract_json_object(content)

                if not parsed or "ranked_ids" not in parsed or not isinstance(parsed["ranked_ids"], list):
                    return RerankResult(
                        ranked=candidates[:top_k],
                        fallback_used=True,
                        fallback_reason="bad_rerank_output",
                        raw_model_output=content,
                        usage=getattr(resp, "usage", None) and dict(resp.usage),
                    )

                ranked_ids = [str(x) for x in parsed["ranked_ids"]]
                id_to_cand = {c["id"]: c for c in candidates}

                final_ranked: List[Dict[str, Any]] = []
                for cid in ranked_ids:
                    if cid in id_to_cand:
                        final_ranked.append(id_to_cand[cid])

                seen = {c["id"] for c in final_ranked}
                for c in candidates:
                    if len(final_ranked) >= top_k:
                        break
                    if c["id"] not in seen:
                        final_ranked.append(c)
                        seen.add(c["id"])

                return RerankResult(
                    ranked=final_ranked[:top_k],
                    fallback_used=False,
                    raw_model_output=content,
                    usage=getattr(resp, "usage", None) and dict(resp.usage),
                )

            except Exception as e:
                last_err = e
                if _looks_like_rate_limit(e):
                    return RerankResult(
                        ranked=candidates[:top_k],
                        fallback_used=True,
                        fallback_reason="rate_limited_429",
                    )
                if attempt < self.retry_attempts:
                    time.sleep(self.base_backoff_s * (2 ** attempt))
                    continue

        return RerankResult(
            ranked=candidates[:top_k],
            fallback_used=True,
            fallback_reason=f"rerank_error:{type(last_err).__name__}" if last_err else "rerank_error",
        )

    def _build_prompt(self, query: str, candidates: List[Dict[str, Any]], top_k: int) -> str:
        lines = []
        for c in candidates:
            snippet = (c.get("text") or "").replace("\n", " ").strip()[:320]
            lines.append(f'- id: {c["id"]} | snippet: "{snippet}"')

        schema = {
            "ranked_ids": ["C001", "C002"],
            "notes": "Optional short note. Keep it brief.",
        }

        return (
            "Rank the candidate chunks by relevance to the user query.\n"
            "Return ONLY valid JSON (no markdown) matching this schema:\n"
            f"{json.dumps(schema)}\n\n"
            f"User query:\n{query}\n\n"
            f"Candidates:\n" + "\n".join(lines) + "\n\n"
            "Rules:\n"
            f"- Output exactly {top_k} ids in ranked_ids when possible.\n"
            "- Choose the most directly answer-supporting chunks.\n"
            "- If unsure, prefer chunks with concrete definitions, steps, or facts.\n"
        )
