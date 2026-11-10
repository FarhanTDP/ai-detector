import pytest

from app.config import Settings
from app.detector import build_report, combine, sliding_windows, spec_for, tier_for, window_texts


def test_window_texts_slice_original_text():
    text = "one two\n\nthree   four five"
    spans, chunks = window_texts(text, size=3, overlap=1)
    assert spans == [(0, 3), (2, 5)]
    assert chunks == ["one two\n\nthree", "three   four five"]


def test_combine():
    assert combine([0.9, 0.3], "min") == 0.3
    assert combine([0.9, 0.3], "mean") == pytest.approx(0.6)


def test_known_and_unknown_model_specs():
    assert spec_for("desklib/ai-text-detector-v1.01").head == "mean_pool"
    assert spec_for("wasitaigeneratedcom/tropa-mini-ai-text-detector").output == "not_human"
    unknown = spec_for("someone/new-detector")
    assert unknown.head == "sequence" and unknown.short_name == "new-detector"


def test_settings_models_from_env(monkeypatch):
    monkeypatch.setenv("AIDET_MODELS", "a/one, b/two")
    assert Settings().models == ("a/one", "b/two")
    monkeypatch.delenv("AIDET_MODELS")
    monkeypatch.setenv("AIDET_MODEL", "legacy/single")  # old variable still works
    assert Settings().models == ("legacy/single",)


def test_settings_reject_bad_ensemble():
    with pytest.raises(ValueError):
        Settings(ensemble="max")


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
