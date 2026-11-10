"""Download human-written short stories for benchmarking false positives.

Source: the WritingPrompts dataset (r/WritingPrompts, collected in 2018, so it
predates ChatGPT and is reliably human-written), via the Hugging Face
datasets-server API. The stories are not committed to the repo.

    python benchmarks/fetch_human_stories.py [count_per_split]
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
import urllib.request

OUT = pathlib.Path(__file__).resolve().parent / "data" / "human"
API = "https://datasets-server.huggingface.co/rows?dataset=euclaise/writingprompts&config=default"


def detokenize(s: str) -> str:
    """The dataset is pre-tokenised ("word ." / "<newline>"); turn it back into normal prose."""
    s = s.replace("<newline>", "\n")
    s = re.sub(r" ([.,!?;:%)\]}])", r"\1", s)
    s = re.sub(r"([(\[{$]) ", r"\1", s)
    s = re.sub(r" (n't|'s|'re|'ve|'ll|'d|'m)\b", r"\1", s, flags=re.I)
    s = s.replace("`` ", '"').replace(" ''", '"').replace("``", '"').replace("''", '"')
    s = re.sub(r'" (.*?) "', r'"\1"', s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    return s.strip()


def fetch(split: str, want: int) -> int:
    got, offset = 0, 0
    while got < want and offset < 3000:
        url = f"{API}&split={split}&offset={offset}&length=100"
        rows = json.load(urllib.request.urlopen(url, timeout=60))["rows"]
        offset += 100
        for row in rows:
            story = detokenize(row["row"]["story"])
            if 380 <= len(story.split()) <= 900:
                (OUT / f"wp_{split}_{row['row_idx']}.txt").write_text(story, encoding="utf-8")
                got += 1
                if got >= want:
                    break
    return got


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    OUT.mkdir(parents=True, exist_ok=True)
    total = fetch("test", n) + fetch("validation", n)
    print(f"Saved {total} human stories to {OUT}")
