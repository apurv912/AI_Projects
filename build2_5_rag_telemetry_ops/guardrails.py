from __future__ import annotations

from typing import Any, List

import rag_core


def guardrail_triggered(retrieved: List[Any]) -> bool:
    """
    Policy boundary for "should we refuse / no-answer?"
    Reuses the heuristic from rag_core to keep core stable.
    """
    try:
        return bool(rag_core.should_no_answer(retrieved))
    except Exception:
        # Guardrail should never crash the pipeline; fail-open.
        return False
