"""Download human-written texts for benchmarking false positives.

All sources predate ChatGPT, so they are reliably human-written. Fetched via the
Hugging Face datasets-server API; the texts are not committed to the repo.

  * stories  - r/WritingPrompts (2018)
  * reviews  - IMDB movie reviews (2011)
  * news     - BBC articles from the XSum dataset (2010-2017)

    python benchmarks/fetch_human_stories.py [count_per_source]
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
import time
import urllib.parse
import urllib.request

OUT = pathlib.Path(__file__).resolve().parent / "data" / "human"
API = "https://datasets-server.huggingface.co/rows"

# (file prefix, dataset, config, split, text column, needs detokenising)
SOURCES = [
    ("wp_test", "euclaise/writingprompts", "default", "test", "story", True),
    ("wp_validation", "euclaise/writingprompts", "default", "validation", "story", True),
    ("imdb", "stanfordnlp/imdb", "plain_text", "test", "text", False),
    ("news", "EdinburghNLP/xsum", "default", "test", "document", False),
]


def detokenize(s: str) -> str:
    """WritingPrompts is pre-tokenised ("word ." / "<newline>"); turn it back into normal prose."""
    s = s.replace("<newline>", "\n")
    s = re.sub(r" ([.,!?;:%)\]}])", r"\1", s)
    s = re.sub(r"([(\[{$]) ", r"\1", s)
    s = re.sub(r" (n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", s, flags=re.I)
    s = s.replace("`` ", '"').replace(" ''", '"').replace("``", '"').replace("''", '"')
    s = re.sub(r'" (.*?) "', r'"\1"', s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    return s.strip()


def clean(s: str) -> str:
    s = re.sub(r"<br\s*/?>", "\n", s)  # IMDB line breaks
    return re.sub(r"[ \t]+", " ", s).strip()


def get_rows(dataset: str, config: str, split: str, offset: int) -> list[dict]:
    query = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                    "offset": offset, "length": 100})
    for attempt in range(4):
        try:
            return json.load(urllib.request.urlopen(f"{API}?{query}", timeout=60))["rows"]
        except Exception:
            time.sleep(2 + 3 * attempt)
    return []


def fetch(prefix: str, dataset: str, config: str, split: str, column: str, detok: bool, want: int) -> int:
    got, offset = 0, 0
    while got < want and offset < 3000:
        rows = get_rows(dataset, config, split, offset)
        offset += 100
        for row in rows:
            raw = row["row"][column]
            text = detokenize(raw) if detok else clean(raw)
            if 380 <= len(text.split()) <= 900:
                (OUT / f"{prefix}_{row['row_idx']}.txt").write_text(text, encoding="utf-8")
                got += 1
                if got >= want:
                    break
    return got


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    OUT.mkdir(parents=True, exist_ok=True)
    for src in SOURCES:
        print(f"{src[0]:14s} saved {fetch(*src, want=n)}")
