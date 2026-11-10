"""Text analytics, section splitting and style signals (pure Python, no model).

Two different things live here, and the UI keeps them apart:

  * The AI *probability* of a section comes from the neural detectors.
  * The *style signals* below are transparent heuristics: surface patterns that
    often show up in AI-written prose. They explain what a reader might notice
    and what to revise. They are not the models' internal reasoning, and human
    writing can trigger them too.
"""

from __future__ import annotations

import math
import re
from typing import Any

WORD_RE = re.compile(r"\S+")
TOKEN_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
SENTENCE_RE = re.compile(r"\S[^.!?\n]*(?:[.!?]+[\"”’')\]]*|(?=\n)|$)")
PARAGRAPH_RE = re.compile(r"[^\n]*\S[^\n]*")

# --------------------------------------------------------------------------- #
# Phrase lists
# --------------------------------------------------------------------------- #
STOCK_PHRASES = [
    # essay / article
    "it is important to note", "it's important to note", "it is worth noting", "it's worth noting",
    "it is worth remembering", "in conclusion", "in summary", "to sum up", "in today's world",
    "in today's fast-paced", "in the realm of", "plays a crucial role", "plays a vital role",
    "plays a key role", "a testament to", "stands as a testament", "serves as a reminder",
    "double-edged sword", "ever-evolving", "ever-changing", "rapidly evolving", "a myriad of",
    "a plethora of", "a wide range of", "navigate the complexities", "delve into", "delves into",
    "rich tapestry", "tapestry of", "multifaceted", "paramount", "pivotal", "underscores",
    "foster a sense", "harness the power", "unlock the potential", "at the end of the day",
    "neither inherently good nor", "a balanced approach", "careful consideration",
    "it is crucial to", "it is essential to", "undeniable", "unprecedented",
    "transformative", "significant challenges", "numerous benefits", "meaningful change",
    "a powerful tool", "rests with all of us", "strike a balance",
    # fiction
    "little did", "couldn't help but", "could not help but", "heart pounding", "heart raced",
    "heart was pounding", "heart hammering", "sent shivers", "a shiver ran", "barely above a whisper",
    "let out a breath", "let out a shaky breath", "eyes widened", "for the first time in years",
    "for the first time in", "from that day on", "from that day forward", "something shifted",
    "the weight of", "a sense of", "a mix of", "nestled between", "nestled in", "rolling green hills",
    "vanished into thin air", "as if on cue", "in that moment", "glowing softly", "smiled softly",
    "said softly", "whispered softly", "without hesitating", "without hesitation", "knew what she had to do",
    "knew what he had to do", "black as midnight", "the first rays", "brilliant shades of",
    "swallowing her fear", "swallowing his fear", "something unexpected", "the turning point came",
]
STOCK_RE = re.compile("|".join(re.escape(p) for p in sorted(STOCK_PHRASES, key=len, reverse=True)), re.I)

TRANSITIONS = [
    "however", "moreover", "furthermore", "additionally", "in addition", "consequently", "therefore",
    "thus", "overall", "ultimately", "in conclusion", "finally", "nevertheless", "nonetheless",
    "as a result", "in contrast", "on the other hand", "for instance", "for example", "in summary",
    "importantly", "notably", "similarly", "meanwhile", "eventually", "suddenly", "slowly",
]
TRANSITION_START_RE = re.compile(r"^[\"“']?(?:" + "|".join(re.escape(t) for t in TRANSITIONS) + r")\b[,:]?", re.I)

EXPANDED_FORMS_RE = re.compile(
    r"\b(?:do not|does not|did not|is not|are not|was not|were not|cannot|can not|will not|would not|"
    r"could not|should not|have not|has not|had not|it is|that is|there is|i am|i have|i will|i would|"
    r"you are|we are|they are|he is|she is|let us)\b", re.I)
CONTRACTION_RE = re.compile(r"\b[A-Za-z]+['’](?:t|s|re|ve|ll|d|m)\b")
TRIAD_RE = re.compile(r"\b[\w'’-]+(?: [\w'’-]+){0,3}, [\w'’-]+(?: [\w'’-]+){0,3},? (?:and|or) [\w'’-]+")
NOT_BUT_RE = re.compile(
    r"\bnot (?:just|only|merely|simply)\b[^.!?\n]{0,80}\bbut\b"
    r"|\b(?:is|was|are|were) not [^.;,\n]{2,40}[,;] but\b"
    r"|\b(?:isn't|wasn't|is not|was not) (?:about |just |only )?[^.;,\n]{2,40}[.;,] (?:it's|it is|it was)\b",
    re.I)
MORAL_RE = re.compile(
    r"\b(?:learned that|realized that|realised that|understood that|reminder that|in the end|from that day|"
    r"true \w+ is|is not the absence of|turns out that|it turns out|rests with|depends largely on|"
    r"neither inherently)\b", re.I)
VAGUE_RE = re.compile(
    r"\b(?:various|numerous|several|many people|significant(?:ly)?|important(?:ly)?|crucial|essential|vital|"
    r"potential(?:ly)?|overall|aspects?|factors?|society|individuals|challenges|opportunities|benefits)\b", re.I)
LY_ADVERB_RE = re.compile(r"\b[a-z]{3,}ly\b", re.I)
NOT_ADVERBS = {"only", "family", "early", "likely", "ugly", "holy", "reply", "apply", "supply", "fly", "july",
               "italy", "belly", "silly", "lonely", "lovely", "friendly", "daily", "really", "nearly",
               # sentence connectors are reported by the "transitions" signal instead
               "additionally", "ultimately", "finally", "importantly", "notably", "similarly", "eventually",
               "consequently", "suddenly", "slowly"}
EM_DASH_RE = re.compile(r"—|\s--\s|\s–\s")
SPECIFIC_RE = re.compile(r"\d|(?<=[a-z,;] )[A-Z][a-z]+")


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def word_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in WORD_RE.finditer(text)]


def sentence_spans(text: str, start: int = 0, end: int | None = None) -> list[tuple[int, int]]:
    end = len(text) if end is None else end
    return [(m.start(), m.end()) for m in SENTENCE_RE.finditer(text, start, end) if m.end() > m.start()]


def _count_words(text: str, a: int, b: int) -> int:
    return len(WORD_RE.findall(text, a, b))


def split_sections(text: str, min_words: int = 30, max_words: int = 130, target: int = 75) -> list[tuple[int, int]]:
    """Split text into readable sections of roughly paragraph size.

    Paragraphs are the natural unit. Very long paragraphs are cut at sentence
    boundaries, and runs of tiny paragraphs (dialogue) are merged.
    """
    pieces: list[tuple[int, int]] = []
    for m in PARAGRAPH_RE.finditer(text):
        a, b = m.start() + (len(m.group()) - len(m.group().lstrip())), m.start() + len(m.group().rstrip())
        if _count_words(text, a, b) <= max_words:
            pieces.append((a, b))
            continue
        start, last_end = None, a
        for s, e in sentence_spans(text, a, b):
            start = s if start is None else start
            last_end = e
            if _count_words(text, start, e) >= target:
                pieces.append((start, e))
                start = None
        if start is not None:
            pieces.append((start, last_end))

    sections: list[tuple[int, int]] = []
    buf: tuple[int, int] | None = None
    for a, b in pieces:
        buf = (a, b) if buf is None else (buf[0], b)
        if _count_words(text, *buf) >= min_words:
            sections.append(buf)
            buf = None
    if buf is not None:
        if sections:
            sections[-1] = (sections[-1][0], buf[1])
        else:
            sections.append(buf)
    return sections


# --------------------------------------------------------------------------- #
# Document analytics
# --------------------------------------------------------------------------- #
def _syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    groups = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and not w.endswith(("le", "ee")) and groups > 1:
        groups -= 1
    return max(1, groups)


def _mattr(tokens: list[str], window: int = 50) -> float:
    """Moving-average type-token ratio: vocabulary variety that doesn't depend on text length."""
    if not tokens:
        return 0.0
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    ratios = [len(set(tokens[i:i + window])) / window for i in range(0, len(tokens) - window + 1, 5)]
    return sum(ratios) / len(ratios)


def _sentence_lengths(text: str, a: int = 0, b: int | None = None) -> list[int]:
    return [n for n in (_count_words(text, s, e) for s, e in sentence_spans(text, a, b)) if n > 0]


def _cv(values: list[int]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(var) / mean if mean else 0.0


def document_stats(text: str) -> dict[str, Any]:
    tokens = [t.lower() for t in TOKEN_RE.findall(text)]
    n_words = len(WORD_RE.findall(text))
    lengths = _sentence_lengths(text)
    n_sent = max(1, len(lengths))
    paragraphs = len(PARAGRAPH_RE.findall(text))
    syllables = sum(_syllables(t) for t in tokens)
    n_tok = max(1, len(tokens))
    flesch = 206.835 - 1.015 * (n_tok / n_sent) - 84.6 * (syllables / n_tok)
    per100 = 100 / max(1, n_words)
    stock = [m.group().lower() for m in STOCK_RE.finditer(text)]
    sentences = [text[s:e] for s, e in sentence_spans(text)]
    transitions = sum(1 for s in sentences if TRANSITION_START_RE.match(s))
    return {
        "words": n_words,
        "sentences": len(lengths),
        "paragraphs": paragraphs,
        "reading_time_min": round(n_words / 230, 1),
        "avg_sentence_words": round(sum(lengths) / n_sent, 1) if lengths else 0,
        "longest_sentence_words": max(lengths, default=0),
        "shortest_sentence_words": min(lengths, default=0),
        # 0 = every sentence the same length; human prose is usually 0.5-0.9
        "sentence_variation": round(_cv(lengths), 2),
        "vocabulary_variety": round(_mattr(tokens), 2),
        "avg_word_length": round(sum(len(t) for t in tokens) / n_tok, 1),
        "flesch_reading_ease": round(max(0.0, min(100.0, flesch))),
        "contractions_per_100": round(len(CONTRACTION_RE.findall(text)) * per100, 1),
        "em_dashes_per_100": round(len(EM_DASH_RE.findall(text)) * per100, 1),
        "transition_openers": transitions,
        "stock_phrases": len(stock),
        "stock_phrase_examples": sorted(set(stock))[:8],
    }


# --------------------------------------------------------------------------- #
# Style signals for one section
# --------------------------------------------------------------------------- #
def _signal(key: str, title: str, detail: str, tip: str) -> dict[str, str]:
    return {"key": key, "title": title, "detail": detail, "tip": tip}


def section_signals(section: str) -> list[dict[str, str]]:
    """Heuristic style patterns in a section, each with a concrete revision tip."""
    out: list[dict[str, str]] = []
    n_words = len(WORD_RE.findall(section))
    if n_words == 0:
        return out
    sentences = [section[s:e] for s, e in sentence_spans(section)]
    lengths = _sentence_lengths(section)

    stock = sorted({m.group().lower() for m in STOCK_RE.finditer(section)})
    if stock:
        quoted = ", ".join(f"“{p}”" for p in stock[:4])
        out.append(_signal(
            "stock_phrases", "Stock phrases", f"Contains {quoted}.",
            "Swap each stock phrase for your own wording or a concrete detail only you would include."))

    if len(lengths) >= 4 and _cv(lengths) < 0.38:
        avg = round(sum(lengths) / len(lengths))
        out.append(_signal(
            "uniform_rhythm", "Even sentence rhythm",
            f"{len(lengths)} sentences, nearly all around {avg} words long.",
            "Break the rhythm: follow a long sentence with a very short one, or join two into one."))

    openers = [m.group().strip(" ,:\"“'") for m in map(TRANSITION_START_RE.match, sentences) if m]
    if len(openers) >= 2:
        out.append(_signal(
            "transitions", "Connector-led sentences",
            f"{len(openers)} sentences open with a connector ({', '.join(sorted(set(o.capitalize() for o in openers))[:4])}).",
            "Delete most of these openers. If the order of ideas is clear, the sentences connect on their own."))

    expanded = EXPANDED_FORMS_RE.findall(section)
    if n_words >= 45 and len(expanded) >= 2 and not CONTRACTION_RE.search(section):
        forms = list(dict.fromkeys(e.lower() for e in expanded))[:2]
        out.append(_signal(
            "no_contractions", "No contractions",
            f"Uses full forms ({', '.join(f'“{f}”' for f in forms)}) {len(expanded)} times and never contracts.",
            "Use contractions where you'd say them aloud (don't, it's, I'd), unless the piece must stay formal."))

    triads = TRIAD_RE.findall(section)
    if len(triads) >= 2:
        out.append(_signal(
            "triads", "Lists of three", f"{len(triads)} “A, B, and C” lists, e.g. “{triads[0][:60]}”.",
            "Vary the pattern: keep one list, cut another to its strongest item, or expand one item into its own sentence."))

    if NOT_BUT_RE.search(section):
        m = NOT_BUT_RE.search(section)
        out.append(_signal(
            "not_but", "“Not X, but Y” framing", f"Sets up a contrast: “…{section[max(0, m.start() - 10):m.end() + 25].strip()}…”.",
            "State the point directly. Say what the thing is, without first saying what it isn't."))

    moral = MORAL_RE.search(section)
    if moral:
        out.append(_signal(
            "tidy_conclusion", "Tidy summing-up", f"Spells out the takeaway (“{moral.group().lower()}…”).",
            "Cut the sentence that explains the meaning. End on an image, an action or a line of dialogue and trust the reader."))

    adverbs = [w for w in LY_ADVERB_RE.findall(section) if w.lower() not in NOT_ADVERBS]
    if len(adverbs) >= 4 and len(adverbs) / n_words >= 0.035:
        out.append(_signal(
            "adverbs", "Adverb-heavy", f"{len(adverbs)} -ly adverbs ({', '.join(sorted(set(a.lower() for a in adverbs))[:5])}).",
            "Replace verb + adverb pairs with one precise verb (“walked slowly” → “shuffled”), and drop the rest."))

    vague = VAGUE_RE.findall(section)
    if len(vague) >= 4 and len(vague) / n_words >= 0.04 and not SPECIFIC_RE.search(section):
        out.append(_signal(
            "vague", "General, abstract wording",
            f"{len(vague)} broad terms ({', '.join(sorted(set(v.lower() for v in vague))[:5])}) and no names or numbers.",
            "Anchor each claim: add a named example, a number, a date, or something you saw yourself."))

    dashes = len(EM_DASH_RE.findall(section))
    if dashes >= 2 and dashes * 100 / n_words >= 1.5:
        out.append(_signal(
            "em_dashes", "Frequent dashes", f"{dashes} dashes in {n_words} words.",
            "Keep one. Turn the others into commas, full stops or separate sentences."))

    return out


GENERIC_TIPS = [
    "No single surface pattern stands out here; the models are reacting to how predictable the wording and rhythm are overall.",
    "Rewrite it from memory in your own words instead of editing line by line.",
    "Add something specific that only you know: a name, a number, an odd detail, an opinion.",
    "Read it aloud and change any sentence you wouldn't actually say that way.",
]
