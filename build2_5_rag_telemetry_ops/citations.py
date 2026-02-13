from __future__ import annotations

from typing import Any, List, Tuple

import rag_core


def extract_cited_ids(answer: str) -> List[str]:
    """
    Policy boundary: how we parse citation IDs from the answer text.
    """
    try:
        return list(rag_core.extract_citations(answer or ""))
    except Exception:
        return []


def validate_citations(answer: str, reranked: List[Any]) -> Tuple[bool, int]:
    """
    Policy boundary: do citations actually point to retrieved chunk IDs?
    Returns: (citation_valid, cited_ids_count)
    """
    cited = extract_cited_ids(answer)
    try:
        is_valid = bool(rag_core.citation_valid(answer or "", reranked))
    except Exception:
        is_valid = False
    return is_valid, int(len(cited))
