import pytest

from app import analysis
from app.detector import build_sections, combine, document_score, localise, longest_run

AI_ESSAY = (
    "In today's world, technology plays a crucial role in education. It is important to note that "
    "students, teachers, and parents all benefit from these tools. However, there are significant "
    "challenges that deserve careful consideration. Moreover, access is not equal for every student. "
    "Additionally, schools must strike a balance between screens and books. In conclusion, technology "
    "is not just a tool, but a transformative force that will shape society, learning, and work."
)
HUMAN_NOTE = (
    "Got the bike fixed yesterday. The guy near the bus stop charged me 900 rupees, which felt like a "
    "lot until he showed me the chain. Rusted through! I'd been riding on that for months. Mum wasn't "
    "thrilled about the mud on the stairs, but honestly? Worth it."
)


def keys(text):
    return {s["key"] for s in analysis.section_signals(text)}


def test_signals_fire_on_formulaic_text():
    found = keys(AI_ESSAY)
    assert {"stock_phrases", "transitions", "not_but", "triads"} <= found
    for s in analysis.section_signals(AI_ESSAY):
        assert s["title"] and s["detail"] and s["tip"]


def test_signals_stay_quiet_on_casual_text():
    assert keys(HUMAN_NOTE) == set()


def test_no_contractions_signal():
    text = ("I do not think it is going to work. We did not plan for this and there is no budget left. "
            "It is a shame because the team was not told early enough, and we cannot fix that today. "
            "They are tired and I am not sure what happens next.")
    assert "no_contractions" in keys(text)
    assert "no_contractions" not in keys(text.replace("do not", "don't"))


def test_uniform_rhythm_signal():
    even = " ".join("The quick brown fox jumps over the lazy sleeping dog." for _ in range(6))
    assert "uniform_rhythm" in keys(even)


def test_split_sections_merges_dialogue_and_splits_long_paragraphs():
    dialogue = "\n\n".join(['"Hi," she said.', '"Hello."', '"You came."'] + ["word " * 40])
    sections = analysis.split_sections(dialogue)
    assert len(sections) == 1  # tiny paragraphs fold into the next one

    long_para = " ".join(f"This is sentence number {i} of a very long paragraph." for i in range(40))
    parts = analysis.split_sections(long_para)
    assert len(parts) >= 3
    assert parts[0][0] == 0 and parts[-1][1] == len(long_para)
    for a, b in parts:  # every piece ends at a sentence boundary
        assert long_para[a:b].rstrip().endswith(".")


def test_sections_cover_text_in_order_without_overlap():
    text = "\n\n".join(" ".join(f"w{i}" for i in range(n)) + "." for n in (50, 10, 12, 200, 45))
    sections = analysis.split_sections(text)
    assert all(a < b for a, b in sections)
    assert all(prev[1] <= nxt[0] for prev, nxt in zip(sections, sections[1:]))
    covered = sum(len(text[a:b].split()) for a, b in sections)
    assert covered == len(text.split())


def test_document_stats():
    stats = analysis.document_stats(AI_ESSAY + "\n\n" + HUMAN_NOTE)
    assert stats["paragraphs"] == 2
    assert stats["words"] == len((AI_ESSAY + " " + HUMAN_NOTE).split())
    assert stats["stock_phrases"] >= 3 and "in conclusion" in stats["stock_phrase_examples"]
    assert stats["transition_openers"] >= 3
    assert stats["contractions_per_100"] > 0
    assert 0 < stats["vocabulary_variety"] <= 1
    assert 0 <= stats["flesch_reading_ease"] <= 100
    assert stats["sentence_variation"] > 0


def test_confirm_ensemble_needs_primary_plus_one_backer():
    assert combine([0.95, 0.05, 0.90], "confirm") == 0.90   # primary + one other agree
    assert combine([0.95, 0.05, 0.10], "confirm") == 0.10   # nobody backs the primary
    assert combine([0.10, 0.99, 0.99], "confirm") == 0.10   # primary says human
    assert combine([0.9, 0.2], "confirm") == 0.2            # two models: both must agree
    assert combine([0.7], "confirm") == 0.7


def test_localise_averages_overlapping_windows():
    scores = localise(6, [(0, 4), (2, 6)], [1.0, 0.0])
    assert scores == [1.0, 1.0, 0.5, 0.5, 0.0, 0.0]


def test_document_score_catches_partly_ai_text():
    words = [0.02] * 300 + [0.97] * 200  # second part is clearly AI
    assert longest_run(words) == 200
    blurred = [0.10, 0.12]  # large windows average it away
    assert document_score(blurred, words, partial_run_words=150) == pytest.approx(sum(words) / 500)
    assert document_score(blurred, words, partial_run_words=0) == pytest.approx(0.11)
    # scattered high words without a long run do not trigger it
    noisy = ([0.9] * 40 + [0.1] * 60) * 5
    assert document_score(blurred, noisy, partial_run_words=150) == pytest.approx(0.11)


def test_build_sections_attaches_signals_only_to_flagged_sections():
    text = HUMAN_NOTE + "\n\n" + AI_ESSAY
    n_h = len(HUMAN_NOTE.split())
    word_scores = [0.03] * n_h + [0.95] * len(AI_ESSAY.split())
    sections = build_sections(text, word_scores, threshold=0.2)
    assert [s["tier"] for s in sections] == ["human", "ai"]
    human, ai = sections
    assert human["signals"] == [] and human["tips"] == []
    assert ai["signals"] and text[ai["start_char"]:ai["end_char"]] == AI_ESSAY
    assert ai["ai_probability"] == pytest.approx(0.95)


def test_flagged_section_without_patterns_gets_generic_tips():
    text = HUMAN_NOTE
    sections = build_sections(text, [0.9] * len(text.split()), threshold=0.2)
    assert sections[0]["signals"] == [] and len(sections[0]["tips"]) >= 2
