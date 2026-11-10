"""AI-text detection pipeline.

Pipeline for a scan:
  1. Guardrail: refuse texts shorter than ``min_words`` (default 300).
  2. Sliding window: split into overlapping word chunks (default 200 words,
     50-word overlap → a new window every 150 words). The final window is
     anchored to the end of the text so the tail is never dropped or scored
     on a tiny fragment.
  3. Score every chunk with a RoBERTa classifier on the GPU (batched, fp16).
  4. Aggregate: the document score is the mean of the chunk scores.
  5. Threshold: below 20% the result is reported as "Human-Written" and the
     exact numbers are stripped *server-side*, so they never reach the browser.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol

from .config import Settings

log = logging.getLogger("scribescan.detector")


class TextTooShort(Exception):
    def __init__(self, word_count: int, min_words: int):
        super().__init__(f"Text has {word_count} words; at least {min_words} are required.")
        self.word_count = word_count
        self.min_words = min_words


class TextTooLong(Exception):
    def __init__(self, word_count: int, max_words: int):
        super().__init__(f"Text has {word_count} words; the maximum per scan is {max_words}.")
        self.word_count = word_count
        self.max_words = max_words


class ModelNotReady(Exception):
    pass


# --------------------------------------------------------------------------- #
# Pure helpers (no torch needed — unit-tested directly)
# --------------------------------------------------------------------------- #
def sliding_windows(n_words: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """Return [start, end) word spans covering ``n_words`` with the given overlap."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("size must be > 0 and 0 <= overlap < size")
    if n_words <= 0:
        return []
    if n_words <= size:
        return [(0, n_words)]
    stride = size - overlap
    spans = [(start, start + size) for start in range(0, n_words - size + 1, stride)]
    if spans[-1][1] < n_words:
        spans.append((n_words - size, n_words))
    return spans


def tier_for(score: float) -> tuple[str, str]:
    if score >= 0.80:
        return "ai", "AI-Generated"
    if score >= 0.50:
        return "likely_ai", "Likely AI-Generated"
    return "mixed", "Possibly AI-Assisted"


def build_report(
    *,
    chunk_scores: list[float],
    spans: list[tuple[int, int]],
    word_count: int,
    threshold: float,
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate chunk scores and apply the human-threshold redaction."""
    score = sum(chunk_scores) / len(chunk_scores)
    base = {"word_count": word_count, "chunk_count": len(spans), "threshold": threshold, **meta}

    if score < threshold:
        # Deliberately omit every numeric score — the UI must not be able to show one.
        return {**base, "verdict": "human", "label": "Human-Written", "ai_probability": None, "chunks": None}

    verdict, label = tier_for(score)
    return {
        **base,
        "verdict": verdict,
        "label": label,
        "ai_probability": round(score, 4),
        "chunks": [
            {"start_word": s, "end_word": e, "ai_probability": round(p, 4)}
            for (s, e), p in zip(spans, chunk_scores)
        ],
    }


# --------------------------------------------------------------------------- #
# Detector interface + Hugging Face implementation
# --------------------------------------------------------------------------- #
class BaseDetector(Protocol):
    def status(self) -> dict[str, Any]: ...
    def start_loading(self) -> None: ...
    def analyze(self, text: str) -> dict[str, Any]: ...


@dataclass
class _Loaded:
    tokenizer: Any
    model: Any
    device: str
    ai_index: int
    labels: dict[int, str]


class HFDetector:
    """RoBERTa sequence classifier from the Hugging Face Hub."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._state = "idle"  # idle | loading | ready | error
        self._error: str | None = None
        self._loaded: _Loaded | None = None
        self._infer_lock = threading.Lock()  # one GPU job at a time
        self._load_lock = threading.Lock()

    # -- lifecycle -------------------------------------------------------------
    def start_loading(self) -> None:
        """Load in a background thread so the web server is usable immediately."""
        threading.Thread(target=self.load, name="model-loader", daemon=True).start()

    def load(self) -> None:
        with self._load_lock:
            if self._state in ("loading", "ready"):
                return
            self._state = "loading"
        name = self.settings.model_name
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            device = self.settings.device
            if device.startswith("cuda") and not torch.cuda.is_available():
                log.warning(
                    "CUDA was requested but PyTorch can't see a GPU — falling back to CPU. "
                    "Install the CUDA build of PyTorch (see README)."
                )
                device = "cpu"
            dtype = torch.float16 if device.startswith("cuda") else torch.float32

            t0 = time.perf_counter()
            log.info("Loading %s on %s (%s)…", name, device, str(dtype).replace("torch.", ""))
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSequenceClassification.from_pretrained(name, dtype=dtype)
            model.to(device).eval()

            labels = {int(k): str(v) for k, v in model.config.id2label.items()}
            ai_index = self._resolve_ai_index(labels)
            self._loaded = _Loaded(tokenizer, model, device, ai_index, labels)
            self._score_chunks(["Warm-up pass so the first real scan isn't slowed by CUDA initialisation."])
            self._state = "ready"

            gpu = f" on {torch.cuda.get_device_name(0)}" if device.startswith("cuda") else ""
            log.info(
                "Model ready in %.1fs%s. Labels=%s, AI label='%s'",
                time.perf_counter() - t0, gpu, labels, labels[ai_index],
            )
        except Exception as exc:  # surface any load failure in the UI instead of crashing
            log.exception("Failed to load model %s", name)
            self._error = f"{type(exc).__name__}: {exc}"
            self._state = "error"

    def _resolve_ai_index(self, labels: dict[int, str]) -> int:
        wanted = self.settings.ai_label.strip().lower()
        if wanted:
            for idx, lab in labels.items():
                if lab.lower() == wanted:
                    return idx
            raise ValueError(f"AIDET_AI_LABEL='{self.settings.ai_label}' not in model labels {labels}")
        human_like = ("human", "real", "original")
        candidates = [i for i, lab in labels.items() if not any(h in lab.lower() for h in human_like)]
        if len(candidates) == 1:
            return candidates[0]
        # e.g. generic LABEL_0 / LABEL_1 — convention is 1 = machine-generated
        return 1 if 1 in labels else max(labels)

    def status(self) -> dict[str, Any]:
        info: dict[str, Any] = {"state": self._state, "name": self.settings.model_name, "error": self._error}
        if self._loaded:
            info["device"] = self._loaded.device
            info["ai_label"] = self._loaded.labels[self._loaded.ai_index]
        return info

    # -- inference -------------------------------------------------------------
    def _score_chunks(self, chunks: list[str]) -> list[float]:
        import torch

        loaded = self._loaded
        if loaded is None:
            raise ModelNotReady()
        scores: list[float] = []
        bs = self.settings.batch_size
        with self._infer_lock, torch.inference_mode():
            for i in range(0, len(chunks), bs):
                enc = loaded.tokenizer(
                    chunks[i : i + bs],
                    padding=True,
                    truncation=True,
                    max_length=self.settings.max_tokens,
                    return_tensors="pt",
                ).to(loaded.device)
                logits = loaded.model(**enc).logits.float()
                probs = torch.softmax(logits, dim=-1)[:, loaded.ai_index]
                scores.extend(probs.tolist())
        return scores

    def analyze(self, text: str) -> dict[str, Any]:
        if self._state != "ready":
            raise ModelNotReady()
        s = self.settings
        words = text.split()
        n = len(words)
        if n < s.min_words:
            raise TextTooShort(n, s.min_words)
        if n > s.max_words:
            raise TextTooLong(n, s.max_words)

        t0 = time.perf_counter()
        spans = sliding_windows(n, s.chunk_words, s.chunk_overlap)
        chunks = [" ".join(words[a:b]) for a, b in spans]
        scores = self._score_chunks(chunks)
        elapsed_ms = round((time.perf_counter() - t0) * 1000)

        assert self._loaded is not None
        return build_report(
            chunk_scores=scores,
            spans=spans,
            word_count=n,
            threshold=s.human_threshold,
            meta={"elapsed_ms": elapsed_ms, "device": self._loaded.device, "model": s.model_name},
        )
