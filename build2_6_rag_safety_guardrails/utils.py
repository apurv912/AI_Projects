from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def load_env() -> None:
    """
    Deterministic .env loading:
    - Prefer .env in THIS build folder (Build 2.6)
    - Fall back to default dotenv search if not found
    """
    here = Path(__file__).resolve().parent
    env_path = here / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path), override=True)
    else:
        load_dotenv(override=True)


def env_str(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default or "").strip()


def read_text_file(path: str, max_chars: int = 250_000) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    txt = p.read_text(encoding="utf-8", errors="ignore")
    return txt[:max_chars]
