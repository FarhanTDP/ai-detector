"""AI-text detection pipeline.

Pipeline for a scan:
  1. Guardrail: refuse texts shorter than ``min_words`` (default 300).
  2. Sliding window: split into overlapping word windows (default 500 words,
     150-word overlap). The final window is anchored to the end of the text so
     the tail is never dropped or scored on a tiny fragment. Windows are cut
     from the original text, so paragraph breaks are preserved.
  3. Score every window with each detector model on the GPU (batched, fp16).
  4. Ensemble: per window, combine the models' scores. The default "min" rule
     means *every* model must agree a window is AI, which minimises false
     accusations against human writers.
  5. Aggregate: the document score is the mean of the window scores.
  6. Threshold: below 20% the result is reported as "Human-Written" and the
     exact numbers are stripped *server-side*, so they never reach the browser.
"""

from __future__ import annotations

import bisect
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol

from . import analysis
from .config import Settings

log = logging.getLogger("scribescan.detector")
WORD_RE = re.compile(r"\S+")


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


def window_texts(text: str, size: int, overlap: int) -> tuple[list[tuple[int, int]], list[str]]:
    """Word-span windows plus the matching slices of the *original* text (formatting kept)."""
    words = [m.span() for m in WORD_RE.finditer(text)]
    spans = sliding_windows(len(words), size, overlap)
    return spans, [text[words[a][0]:words[b - 1][1]] for a, b in spans]


def combine(scores: list[float], method: str) -> float:
    """Ensemble several models' AI probabilities for one window."""
    if method == "min" or (method == "confirm" and len(scores) == 2):
        return min(scores)
    if method == "confirm" and len(scores) > 2:
        # scores[0] is the primary detector; at least one other model must back it up.
        return min(scores[0], max(scores[1:]))
    return sum(scores) / len(scores)


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
    model_scores: dict[str, float] | None = None,
    analytics: dict[str, Any] | None = None,
    sections: list[dict[str, Any]] | None = None,
    score: float | None = None,
    partial: bool = False,
) -> dict[str, Any]:
    """Aggregate window scores and apply the human-threshold redaction."""
    if score is None:
        score = sum(chunk_scores) / len(chunk_scores)
    base = {"word_count": word_count, "chunk_count": len(spans), "threshold": threshold,
            "analytics": analytics, **meta}

    if score < threshold:
        # Deliberately omit every numeric score — the UI must not be able to show one.
        return {**base, "verdict": "human", "label": "Human-Written",
                "ai_probability": None, "chunks": None, "model_scores": None, "sections": None}

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
        "model_scores": {k: round(v, 4) for k, v in (model_scores or {}).items()} or None,
        "sections": sections,
        # True when the score was raised because part of the text is strongly AI-like
        # even though the document as a whole reads as mixed.
        "partial": partial,
    }


def longest_run(word_scores: list[float], level: float = 0.8) -> int:
    """Length (in words) of the longest unbroken stretch scoring at or above ``level``."""
    best = run = 0
    for w in word_scores:
        run = run + 1 if w >= level else 0
        best = max(best, run)
    return best


def document_score(window_scores: list[float], word_scores: list[float], partial_run_words: int) -> float:
    """Document-level AI probability.

    Normally the mean of the large-window scores. Large windows blur documents
    that are only partly AI, so if a long unbroken passage is strongly AI-like,
    the fine-grained mean is used when it is higher.
    """
    coarse = sum(window_scores) / len(window_scores)
    if partial_run_words > 0 and word_scores and longest_run(word_scores) >= partial_run_words:
        return max(coarse, sum(word_scores) / len(word_scores))
    return coarse


def section_tier(score: float, threshold: float) -> str:
    return "human" if score < threshold else tier_for(score)[0]


def localise(n_words: int, fine_spans: list[tuple[int, int]], fine_scores: list[float]) -> list[float]:
    """Spread overlapping fine-window scores onto individual words (mean of covering windows)."""
    total = [0.0] * n_words
    count = [0] * n_words
    for (a, b), p in zip(fine_spans, fine_scores):
        for i in range(a, b):
            total[i] += p
            count[i] += 1
    return [t / c if c else 0.0 for t, c in zip(total, count)]


def build_sections(text: str, word_scores: list[float], threshold: float) -> list[dict[str, Any]]:
    """Per-section AI probability plus heuristic style signals and revision tips."""
    starts = [w[0] for w in analysis.word_spans(text)]
    out: list[dict[str, Any]] = []
    for a, b in analysis.split_sections(text):
        lo, hi = bisect.bisect_left(starts, a), bisect.bisect_left(starts, b)
        if hi <= lo:
            continue
        score = sum(word_scores[lo:hi]) / (hi - lo)
        tier = section_tier(score, threshold)
        flagged = tier != "human"
        signals = analysis.section_signals(text[a:b]) if flagged else []
        out.append({
            "index": len(out),
            "start_char": a,
            "end_char": b,
            "word_count": hi - lo,
            "ai_probability": round(score, 4),
            "tier": tier,
            "signals": signals,
            "tips": analysis.GENERIC_TIPS if flagged and not signals else [],
        })
    return out


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    """How to load and read a detector.

    head:   "sequence"  -> AutoModelForSequenceClassification
            "mean_pool" -> transformer backbone + mean pooling + linear head
                           (the desklib family; weights in model.safetensors)
    output: "softmax"   -> P(AI label) from a softmax over labels
            "sigmoid"   -> single logit, P(AI) = sigmoid(logit)
            "not_human" -> multi-class head, P(AI) = 1 - P(class 0 = human)
    """

    short_name: str
    head: str = "sequence"
    output: str = "softmax"
    max_tokens: int = 512


KNOWN_MODELS: dict[str, ModelSpec] = {
    "desklib/ai-text-detector-v1.01": ModelSpec("desklib", "mean_pool", "sigmoid", 768),
    "wasitaigeneratedcom/tropa-mini-ai-text-detector": ModelSpec("tropa-mini", "mean_pool", "not_human", 768),
    "ShantanuT01/vanguard-ai-text-detector": ModelSpec("vanguard", "sequence", "sigmoid", 512),
    "fakespot-ai/roberta-base-ai-text-detection-v1": ModelSpec("fakespot", "sequence", "softmax", 512),
    "Hello-SimpleAI/chatgpt-detector-roberta": ModelSpec("chatgpt-roberta", "sequence", "softmax", 512),
}


def spec_for(repo: str) -> ModelSpec:
    return KNOWN_MODELS.get(repo) or ModelSpec(short_name=repo.split("/")[-1])


# --------------------------------------------------------------------------- #
# Detector interface + Hugging Face implementation
# --------------------------------------------------------------------------- #
class BaseDetector(Protocol):
    def status(self) -> dict[str, Any]: ...
    def start_loading(self) -> None: ...
    def analyze(self, text: str) -> dict[str, Any]: ...


@dataclass
class LoadedModel:
    repo: str
    spec: ModelSpec
    tokenizer: Any
    model: Any
    ai_index: int | None = None  # for softmax outputs
    labels: dict[int, str] | None = None


def _mean_pool_classifier(repo: str, dtype: Any) -> Any:
    """Build a desklib-style detector and load its safetensors weights (no remote code)."""
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel

    class MeanPoolClassifier(torch.nn.Module):
        def __init__(self, config: Any, n_out: int):
            super().__init__()
            self.model = AutoModel.from_config(config)
            self.classifier = torch.nn.Linear(config.hidden_size, n_out)

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            hidden = self.model(input_ids=input_ids, attention_mask=attention_mask)[0]
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            return self.classifier(pooled)

    config = AutoConfig.from_pretrained(repo)
    model = MeanPoolClassifier(config, getattr(config, "detector_num_labels", 1))
    model.load_state_dict(load_file(hf_hub_download(repo, "model.safetensors")), strict=True)
    return model.to(dtype)


class HFDetector:
    """One or more Hugging Face AI-text detectors, ensembled per window."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._state = "idle"  # idle | loading | ready | error
        self._error: str | None = None
        self._models: list[LoadedModel] = []
        self._device: str | None = None
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
        try:
            import torch

            device = self.settings.device
            if device.startswith("cuda") and not torch.cuda.is_available():
                log.warning(
                    "CUDA was requested but PyTorch can't see a GPU - falling back to CPU. "
                    "Install the CUDA build of PyTorch (see README)."
                )
                device = "cpu"
            dtype = torch.float16 if device.startswith("cuda") else torch.float32
            self._device = device

            t0 = time.perf_counter()
            for repo in self.settings.models:
                self._models.append(self._load_one(repo, device, dtype))
            self._score_chunks(["Warm-up pass so the first real scan isn't slowed by CUDA initialisation."])
            self._state = "ready"

            gpu = f" on {torch.cuda.get_device_name(0)}" if device.startswith("cuda") else " on CPU"
            log.info("Detectors ready in %.1fs%s: %s (ensemble=%s)", time.perf_counter() - t0, gpu,
                     ", ".join(m.spec.short_name for m in self._models), self.settings.ensemble)
        except Exception as exc:  # surface any load failure in the UI instead of crashing
            log.exception("Failed to load detector models")
            self._error = f"{type(exc).__name__}: {exc}"
            self._models = []
            self._state = "error"

    def _load_one(self, repo: str, device: str, dtype: Any) -> LoadedModel:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        spec = spec_for(repo)
        log.info("Loading %s (%s) on %s...", spec.short_name, repo, device)
        tokenizer = AutoTokenizer.from_pretrained(repo)
        if spec.head == "mean_pool":
            model = _mean_pool_classifier(repo, dtype)
            loaded = LoadedModel(repo, spec, tokenizer, model)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(repo, dtype=dtype)
            labels = {int(k): str(v) for k, v in model.config.id2label.items()}
            output = "sigmoid" if model.config.num_labels == 1 else spec.output
            spec = ModelSpec(spec.short_name, spec.head, output, spec.max_tokens)
            ai_index = self._resolve_ai_index(labels) if output == "softmax" else None
            loaded = LoadedModel(repo, spec, tokenizer, model, ai_index, labels)
            if ai_index is not None:
                log.info("  %s labels=%s, AI label='%s'", spec.short_name, labels, labels[ai_index])
        loaded.model.to(device).eval()
        return loaded

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
        names = [spec_for(r).short_name for r in self.settings.models]
        info: dict[str, Any] = {
            "state": self._state,
            "name": " + ".join(names),
            "models": list(self.settings.models),
            "ensemble": self.settings.ensemble,
            "error": self._error,
        }
        if self._state == "ready":
            info["device"] = self._device
        return info

    # -- inference -------------------------------------------------------------
    def _score_model(self, m: LoadedModel, texts: list[str]) -> list[float]:
        import torch

        scores: list[float] = []
        bs = self.settings.batch_size
        for i in range(0, len(texts), bs):
            enc = m.tokenizer(texts[i : i + bs], padding=True, truncation=True,
                              max_length=m.spec.max_tokens, return_tensors="pt").to(self._device)
            out = m.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            logits = (out.logits if hasattr(out, "logits") else out).float()
            if m.spec.output == "sigmoid":
                probs = torch.sigmoid(logits[:, 0])
            elif m.spec.output == "not_human":
                probs = 1.0 - torch.softmax(logits, dim=-1)[:, 0]
            else:
                probs = torch.softmax(logits, dim=-1)[:, m.ai_index]
            scores.extend(probs.tolist())
        return scores

    def _score_chunks(self, chunks: list[str]) -> tuple[list[float], dict[str, list[float]]]:
        """Return (ensembled score per chunk, {model: scores per chunk})."""
        import torch

        if not self._models:
            raise ModelNotReady()
        with self._infer_lock, torch.inference_mode():
            per_model = {m.spec.short_name: self._score_model(m, chunks) for m in self._models}
        combined = [combine(list(col), self.settings.ensemble) for col in zip(*per_model.values())]
        return combined, per_model

    def _word_scores(self, text: str, n_words: int) -> list[float]:
        """AI probability per word, from fine overlapping windows (used to localise the score)."""
        s = self.settings
        size = min(s.section_window_words, n_words)
        overlap = min(s.section_window_overlap, size - 1)
        # Keep very long documents fast by widening the stride.
        while len(sliding_windows(n_words, size, overlap)) > s.max_section_windows and overlap > 0:
            overlap = max(0, overlap - 25)
        fine_spans, fine_chunks = window_texts(text, size, overlap)
        fine_scores, _ = self._score_chunks(fine_chunks)
        return localise(n_words, fine_spans, fine_scores)

    def analyze(self, text: str) -> dict[str, Any]:
        if self._state != "ready":
            raise ModelNotReady()
        s = self.settings
        n = len(WORD_RE.findall(text))
        if n < s.min_words:
            raise TextTooShort(n, s.min_words)
        if n > s.max_words:
            raise TextTooLong(n, s.max_words)

        t0 = time.perf_counter()
        spans, chunks = window_texts(text, s.chunk_words, s.chunk_overlap)
        combined, per_model = self._score_chunks(chunks)

        # Fine windows localise the score to individual words/sections.
        word_scores = self._word_scores(text, n)
        score = document_score(combined, word_scores, s.partial_run_words)
        partial = score > sum(combined) / len(combined)
        # Sections are only built when there is something to show: under the
        # human threshold every number is withheld anyway.
        sections = build_sections(text, word_scores, s.human_threshold) if score >= s.human_threshold else None
        elapsed_ms = round((time.perf_counter() - t0) * 1000)

        return build_report(
            score=score,
            partial=partial,
            analytics=analysis.document_stats(text),
            sections=sections,
            chunk_scores=combined,
            spans=spans,
            word_count=n,
            threshold=s.human_threshold,
            model_scores={name: sum(v) / len(v) for name, v in per_model.items()},
            meta={
                "elapsed_ms": elapsed_ms,
                "device": self._device,
                "model": " + ".join(per_model),
                "ensemble": s.ensemble,
            },
        )
