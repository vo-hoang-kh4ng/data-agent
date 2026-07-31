"""A group defined by having none of something must not be described as having a lot.

The 35.8% persona in the real report was named "Nhóm không có khiếu nại nào trong suốt kỳ".
Its own appendix listed `complaint_total_6m` at -100.0%. Its Business Signals said:

    Lịch sử phàn nàn tăng rất mạnh

Both statements came from the same file. The appendix uses `relative_deviation`, which is
signed; the signals use `_ranked_deviations`, which was not:

    dev = abs(val - g_val) / abs(g_val) if g_val != 0 else abs(val) * 100

`old_complaint` at 0.00 against a population mean of 0.59 is -100%, and `abs` turns that
into 1.0 — the same number a +100% rise produces. Everything downstream then read it as a
rise, because a magnitude with no sign has to be called something.

The second branch is worse: when the population mean is 0 the result is `abs(val) * 100`,
which is not a ratio at all. It was then rendered with a percent sign.
"""
import pytest

from triadic_dgm.services.report_generator import ReportGenerator


@pytest.fixture
def generator():
    return ReportGenerator(api_key="x", base_url="http://localhost:1", model_name="m")


def _dev(generator, feature, means, globals_) -> float:
    ranked = {f: d for f, _v, _g, d in generator._ranked_deviations(means, globals_)}
    return ranked[feature]


def test_a_feature_below_the_population_mean_deviates_negatively(generator):
    """The exact numbers from the 22,358-customer persona."""
    dev = _dev(generator, "old_complaint", {"old_complaint": 0.0}, {"old_complaint": 0.59})
    assert dev == pytest.approx(-1.0)


def test_a_feature_above_the_population_mean_deviates_positively(generator):
    dev = _dev(generator, "x", {"x": 1.18}, {"x": 0.59})
    assert dev == pytest.approx(1.0)


def test_a_rise_and_a_fall_of_the_same_size_are_told_apart(generator):
    """The defect in one assertion."""
    up = _dev(generator, "x", {"x": 1.18}, {"x": 0.59})
    down = _dev(generator, "x", {"x": 0.0}, {"x": 0.59})
    assert up != down
    assert up > 0 > down


def test_ranking_still_puts_the_biggest_movement_first(generator):
    """Sorting must go by magnitude — a -100% collapse outranks a +20% drift."""
    ranked = generator._ranked_deviations(
        {"collapse": 0.0, "drift": 1.2}, {"collapse": 0.59, "drift": 1.0})
    assert [f for f, *_ in ranked] == ["collapse", "drift"]


def test_a_zero_population_mean_yields_no_percentage(generator):
    """`abs(val) * 100` was never a ratio, and was rendered with a percent sign."""
    ranked = {f: d for f, _v, _g, d in generator._ranked_deviations({"x": 0.35}, {"x": 0.0})}
    assert ranked["x"] is None


def test_a_negative_population_mean_yields_no_percentage(generator):
    """A quantity that has crossed zero is a signed measure, not a size."""
    ranked = {f: d for f, _v, _g, d in generator._ranked_deviations({"x": 9.74}, {"x": -11.18})}
    assert ranked["x"] is None


def test_a_feature_with_no_usable_ratio_sorts_last_not_first(generator):
    """Otherwise the unrankable feature becomes the persona's headline signal."""
    ranked = generator._ranked_deviations(
        {"unrankable": 0.35, "real": 1.18}, {"unrankable": 0.0, "real": 0.59})
    assert [f for f, *_ in ranked][0] == "real"


def test_a_non_numeric_value_does_not_break_the_ranking(generator):
    ranked = generator._ranked_deviations({"text": "abc", "real": 1.18}, {"real": 0.59})
    assert [f for f, *_ in ranked][0] == "real"


def test_it_matches_the_appendix_calculation(generator):
    """One report, one arithmetic. Two deviation functions is how the two halves disagreed."""
    from triadic_dgm.persona.characterization import relative_deviation

    means = {"a": 0.0, "b": 1.18, "c": 2.5}
    globals_ = {"a": 0.59, "b": 0.59, "c": 1.0}
    for feature, _v, _g, dev in generator._ranked_deviations(means, globals_):
        assert dev == pytest.approx(relative_deviation(means[feature], globals_[feature]))


# --- what a reader ends up seeing ---------------------------------------------------------


def test_a_collapse_is_never_reported_as_a_rise(generator):
    """The sentence the 22,358-customer persona actually printed."""
    signals = generator._get_domain_signals(
        {"feature_means": {"old_complaint": 0.0, "no_complaint_all_period": 1.0}},
        {"old_complaint": 0.59, "no_complaint_all_period": 0.41},
    )
    text = " ".join(str(s) for s in signals)
    assert "lịch sử phàn nàn" not in text.lower() or "tăng" not in text.lower(), text
