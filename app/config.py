"""Application settings.

Every value can be overridden with an environment variable prefixed ``AIDET_``
(e.g. ``AIDET_PORT=9000``), so nothing needs to be edited in code to tune the app.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.getenv(f"AIDET_{name}", default)


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


@dataclass(frozen=True)
class Settings:
    # --- Model -------------------------------------------------------------
    model_name: str = field(default_factory=lambda: _env("MODEL", "Hello-SimpleAI/chatgpt-detector-roberta"))
    device: str = field(default_factory=lambda: _env("DEVICE", "cuda"))
    # Label in the model's id2label that means "AI". Empty = auto-detect.
    ai_label: str = field(default_factory=lambda: _env("AI_LABEL", ""))
    batch_size: int = field(default_factory=lambda: _env_int("BATCH_SIZE", 16))
    max_tokens: int = field(default_factory=lambda: _env_int("MAX_TOKENS", 512))

    # --- Detection guardrails ----------------------------------------------
    min_words: int = field(default_factory=lambda: _env_int("MIN_WORDS", 300))
    max_words: int = field(default_factory=lambda: _env_int("MAX_WORDS", 50_000))
    chunk_words: int = field(default_factory=lambda: _env_int("CHUNK_WORDS", 200))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 50))
    # Scores below this are reported as "Human-Written" with the % hidden.
    human_threshold: float = field(default_factory=lambda: _env_float("HUMAN_THRESHOLD", 0.20))

    # --- Storage -----------------------------------------------------------
    db_path: Path = field(default_factory=lambda: Path(_env("DB_PATH", str(BASE_DIR / "data" / "documents.db"))))
    max_content_bytes: int = field(default_factory=lambda: _env_int("MAX_CONTENT_BYTES", 5_000_000))

    # --- Web / LAN ---------------------------------------------------------
    static_dir: Path = BASE_DIR / "static"
    port: int = field(default_factory=lambda: _env_int("PORT", 8000))
    # Optional shared secret. When set, every device must open the app once
    # with ?token=<value>; a cookie keeps it signed in afterwards.
    access_token: str = field(default_factory=lambda: _env("ACCESS_TOKEN", ""))

    def __post_init__(self) -> None:
        if not 0 <= self.chunk_overlap < self.chunk_words:
            raise ValueError("AIDET_CHUNK_OVERLAP must be >= 0 and smaller than AIDET_CHUNK_WORDS")
        if not 0.0 < self.human_threshold < 1.0:
            raise ValueError("AIDET_HUMAN_THRESHOLD must be between 0 and 1")
