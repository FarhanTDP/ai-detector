import pytest

from app.detector import build_report, sliding_windows, tier_for


def test_short_text_is_single_window():
    assert sliding_windows(120, 200, 50) == [(0, 120)]


def test_windows_overlap_and_cover_tail():
    spans = sliding_windows(300, 200, 50)
    assert spans == [(0, 200), (100, 300)]


@pytest.mark.parametrize("n", [201, 300, 349, 350, 351, 500, 1234, 10_000])
def test_windows_cover_every_word_with_full_size(n):
    spans = sliding_windows(n, 200, 50)
    covered = set()
    for a, b in spans:
        assert b - a == 200
        covered.update(range(a, b))
    assert covered == set(range(n))
    assert spans[0][0] == 0 and spans[-1][1] == n
    # consecutive windows overlap by at least the configured amount
    for (a1, b1), (a2, _) in zip(spans, spans[1:]):
        assert b1 - a2 >= 50


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_window_config(size, overlap):
    with pytest.raises(ValueError):
        sliding_windows(500, size, overlap)


def _report(scores):
    spans = [(i * 150, i * 150 + 200) for i in range(len(scores))]
    return build_report(chunk_scores=scores, spans=spans, word_count=1000, threshold=0.20, meta={})


def test_scores_are_averaged():
    r = _report([0.9, 0.5, 0.1])
    assert r["ai_probability"] == pytest.approx(0.5)
    assert r["verdict"] == "likely_ai"
    assert [c["ai_probability"] for c in r["chunks"]] == [0.9, 0.5, 0.1]


def test_below_threshold_hides_all_numbers():
    r = _report([0.05, 0.30])  # mean 0.175 < 0.20, even though one chunk is 30%
    assert r["verdict"] == "human"
    assert r["label"] == "Human-Written"
    assert r["ai_probability"] is None
    assert r["chunks"] is None


def test_threshold_is_inclusive_for_display():
    r = _report([0.20])
    assert r["verdict"] != "human"
    assert r["ai_probability"] == pytest.approx(0.20)


@pytest.mark.parametrize("score,tier", [(0.2, "mixed"), (0.49, "mixed"), (0.5, "likely_ai"), (0.8, "ai"), (1.0, "ai")])
def test_tiers(score, tier):
    assert tier_for(score)[0] == tier
