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


# Order matters for the "confirm" ensemble: the first model is the primary detector.
DEFAULT_MODELS = (
    "wasitaigeneratedcom/tropa-mini-ai-text-detector,"
    "desklib/ai-text-detector-v1.01,"
    "ShantanuT01/vanguard-ai-text-detector"
)


def _env_list(name: str, default: str) -> tuple[str, ...]:
    # AIDET_MODEL (singular) is still honoured for backwards compatibility.
    raw = os.getenv(f"AIDET_{name}") or os.getenv("AIDET_MODEL") or default
    return tuple(part.strip() for part in raw.split(",") if part.strip())


@dataclass(frozen=True)
class Settings:
    # --- Models ------------------------------------------------------------
    # One or more Hugging Face detectors (comma-separated). The defaults won a
    # benchmark against 8 alternatives — see benchmarks/README.md.
    models: tuple[str, ...] = field(default_factory=lambda: _env_list("MODELS", DEFAULT_MODELS))
    # How per-model scores are combined for each window:
    #   "confirm" = the first model's score, capped by the strongest of the others
    #               (the primary must flag it AND at least one other must confirm)
    #   "min"     = every model must agree      "mean" = average
    ensemble: str = field(default_factory=lambda: _env("ENSEMBLE", "confirm"))
    device: str = field(default_factory=lambda: _env("DEVICE", "cuda"))
    # Label in a softmax model's id2label that means "AI". Empty = auto-detect.
    ai_label: str = field(default_factory=lambda: _env("AI_LABEL", ""))
    batch_size: int = field(default_factory=lambda: _env_int("BATCH_SIZE", 8))

    # --- Detection guardrails ----------------------------------------------
    min_words: int = field(default_factory=lambda: _env_int("MIN_WORDS", 300))
    max_words: int = field(default_factory=lambda: _env_int("MAX_WORDS", 50_000))
    # 500-word windows were markedly more accurate than 200-word ones in the
    # benchmark (short windows roughly doubled false flags on human writing).
    chunk_words: int = field(default_factory=lambda: _env_int("CHUNK_WORDS", 500))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 150))
    # Finer windows used only to localise the score to sections for highlighting.
    section_window_words: int = field(default_factory=lambda: _env_int("SECTION_WINDOW_WORDS", 150))
    section_window_overlap: int = field(default_factory=lambda: _env_int("SECTION_WINDOW_OVERLAP", 100))
    max_section_windows: int = field(default_factory=lambda: _env_int("MAX_SECTION_WINDOWS", 300))
    # A passage this long (words) scoring >= 80% locally lets the fine-grained score
    # raise the document score, so half-AI documents aren't cleared. 0 disables it.
    partial_run_words: int = field(default_factory=lambda: _env_int("PARTIAL_RUN_WORDS", 150))
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
        if not self.models:
            raise ValueError("AIDET_MODELS must list at least one model")
        if self.ensemble not in ("confirm", "min", "mean"):
            raise ValueError("AIDET_ENSEMBLE must be 'confirm', 'min' or 'mean'")
        if not 0 <= self.section_window_overlap < self.section_window_words:
            raise ValueError("AIDET_SECTION_WINDOW_OVERLAP must be smaller than AIDET_SECTION_WINDOW_WORDS")
        if not 0.0 < self.human_threshold < 1.0:
            raise ValueError("AIDET_HUMAN_THRESHOLD must be between 0 and 1")
