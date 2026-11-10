"""Evaluate the detector exactly as the app runs it, on labelled text files.

    benchmarks/data/ai/*.txt      texts written by AI
    benchmarks/data/human/*.txt   texts written by people

Usage (from the project folder, with the venv active):
    python benchmarks/fetch_human_stories.py          # one-time: 60 human stories
    python benchmarks/evaluate.py                     # current settings
    set AIDET_MODELS=fakespot-ai/roberta-base-ai-text-detection-v1
    python benchmarks/evaluate.py                     # compare another model

Drop your own .txt files into data/ai or data/human to test them.
"""

from __future__ import annotations

import logging
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from app.config import Settings  # noqa: E402
from app.detector import HFDetector, TextTooShort  # noqa: E402


def pct(p: float | None) -> str:
    return "  -  " if p is None else f"{p * 100:4.0f}%"


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    data = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data"
    settings = Settings()
    detector = HFDetector(settings)
    t0 = time.time()
    detector.load()
    status = detector.status()
    if status["state"] != "ready":
        sys.exit(f"Model failed to load: {status['error']}")
    print(f"Models: {status['name']}  (ensemble={settings.ensemble}, windows={settings.chunk_words}/"
          f"{settings.chunk_overlap}, device={status.get('device')}, loaded in {time.time() - t0:.0f}s)\n")

    results: dict[str, list[tuple[str, dict]]] = {"ai": [], "human": []}
    for label in results:
        for f in sorted((data / label).glob("*.txt")):
            try:
                results[label].append((f.stem, detector.analyze(f.read_text(encoding="utf-8"))))
            except TextTooShort as e:
                print(f"  skipped {label}/{f.name}: {e}")

    for label, rows in results.items():
        if not rows:
            continue
        print(f"--- {label.upper()} texts ({len(rows)}) ---")
        for name, r in rows:
            wrong = (label == "ai" and r["verdict"] == "human") or (label == "human" and r["verdict"] != "human")
            models = "  ".join(f"{k} {pct(v)}" for k, v in (r.get("model_scores") or {}).items())
            print(f"  {'X' if wrong else ' '} {name[:28]:28s} {r['label']:22s} {pct(r['ai_probability'])}   {models}")
        print()

    ai, hu = results["ai"], results["human"]
    if ai:
        caught = sum(r["verdict"] != "human" for _, r in ai)
        print(f"AI texts detected (shown as >= {settings.human_threshold:.0%}): {caught}/{len(ai)} ({caught / len(ai):.0%})")
    if hu:
        flagged = sum(r["verdict"] != "human" for _, r in hu)
        strong = sum((r["ai_probability"] or 0) >= 0.5 for _, r in hu)
        print(f"Human texts wrongly flagged:            {flagged}/{len(hu)} ({flagged / len(hu):.0%}),"
              f" of which >= 50%: {strong}")


if __name__ == "__main__":
    main()
