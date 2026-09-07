"""Tests for the paired word-vs-digit comparison."""

import pytest

from src.compare_ablation import compare, mcnemar_p


def _rec(qid, model, n, correct):
    return {"question_id": qid, "model": model, "n_people": n,
            "correct": correct, "context_overflow": False}


def test_pairs_only_on_shared_question_ids():
    words = {"a": _rec("a", "M", 6, True), "b": _rec("b", "M", 6, True)}
    digits = {"a": _rec("a", "M", 6, True)}
    df = compare(words, digits)
    assert df["n_paired"].iloc[0] == 1, "unpaired questions must be dropped"


def test_delta_sign_is_digits_minus_words():
    words = {f"q{i}": _rec(f"q{i}", "M", 6, False) for i in range(10)}
    digits = {f"q{i}": _rec(f"q{i}", "M", 6, i < 4) for i in range(10)}
    df = compare(words, digits)
    assert df["delta_digits_minus_words"].iloc[0] == pytest.approx(0.4)
    assert df["acc_digits"].iloc[0] == pytest.approx(0.4)
    assert df["acc_words"].iloc[0] == pytest.approx(0.0)


def test_no_difference_gives_p_of_one():
    words = {f"q{i}": _rec(f"q{i}", "M", 6, i % 2 == 0) for i in range(20)}
    df = compare(words, dict(words))
    assert df["mcnemar_p"].iloc[0] == 1.0
    assert df["delta_digits_minus_words"].iloc[0] == 0.0


def test_a_large_consistent_shift_is_significant():
    words = {f"q{i}": _rec(f"q{i}", "M", 6, False) for i in range(60)}
    digits = {f"q{i}": _rec(f"q{i}", "M", 6, i < 40) for i in range(60)}
    df = compare(words, digits)
    assert df["mcnemar_p"].iloc[0] < 0.001


def test_overflow_records_are_never_paired():
    words = {"a": {**_rec("a", "M", 22, True), "context_overflow": True}}
    digits = {"a": _rec("a", "M", 22, True)}
    from src.compare_ablation import load  # noqa: F401  (documents the filter)
    df = compare({}, digits)
    assert df.empty or df["n_paired"].sum() == 0


@pytest.mark.parametrize("b,c", [(0, 0), (5, 5), (1, 0), (30, 30)])
def test_mcnemar_p_stays_a_probability(b, c):
    assert 0.0 <= mcnemar_p(b, c) <= 1.0
