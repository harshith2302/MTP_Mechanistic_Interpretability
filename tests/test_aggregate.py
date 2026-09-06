"""Tests for the rules that decide whether the headline figure is honest."""

import json
import os

import pandas as pd
import pytest

from src.aggregate import accuracy_table, load_run, taxonomy_table, wilson
from src.chance import NULL_TYPES, chance_table
from src.questions import sample_questions
from src.simulate import generate_story

CFG = {"stories_per_n": 2, "classify": {"chance_baseline_samples": 200}}


def _write(tmp_path, records):
    d = tmp_path / "run" / "ModelA"
    d.mkdir(parents=True)
    with open(d / "N04.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return str(tmp_path / "run")


def _rec(**kw):
    base = dict(model="ModelA", n_people=4, n_timesteps=4, question_type="q",
                context_overflow=False, correct=True, coarse_category="Correct",
                ambiguous=False, question_id="x")
    base.update(kw)
    return base


# --- the rule the whole headline figure rests on ------------------------------
def test_overflow_is_excluded_from_the_accuracy_denominator(tmp_path):
    recs = ([_rec(correct=True, question_id=f"a{i}") for i in range(3)]
            + [_rec(correct=None, context_overflow=True, coarse_category=None,
                    question_id=f"b{i}") for i in range(7)])
    acc = accuracy_table(load_run(_write(tmp_path, recs)))
    row = acc[acc.question_type == "ALL"].iloc[0]
    assert row["n_valid"] == 3, "overflow leaked into the denominator"
    assert row["n_overflow"] == 7
    assert row["accuracy"] == 1.0, "overflow was scored as wrong"


def test_all_overflow_produces_no_plottable_point(tmp_path):
    """A model past its context limit must vanish from the line, not sit at 0."""
    recs = [_rec(correct=None, context_overflow=True, coarse_category=None,
                 question_id=f"b{i}") for i in range(10)]
    acc = accuracy_table(load_run(_write(tmp_path, recs)))
    assert acc.empty or (acc["n_valid"] == 0).all()


def test_format_errors_are_excluded_from_mechanism_denominators(tmp_path):
    recs = ([_rec(correct=False, coarse_category="Format", question_id=f"f{i}")
             for i in range(6)]
            + [_rec(correct=False, coarse_category="Binding",
                    question_type="person_timestep_lookup", question_id=f"m{i}")
               for i in range(4)])
    tax = taxonomy_table(load_run(_write(tmp_path, recs)), CFG, with_chance=False)
    assert set(tax["coarse_category"]) == {"Binding"}
    assert tax.iloc[0]["n_wrong_mechanistic"] == 4
    assert tax.iloc[0]["n_wrong_format"] == 6
    assert tax.iloc[0]["observed_rate"] == 1.0


def test_load_run_tolerates_a_half_written_final_line(tmp_path):
    d = tmp_path / "run" / "ModelA"
    d.mkdir(parents=True)
    with open(d / "N04.jsonl", "w") as f:
        f.write(json.dumps(_rec(question_id="ok")) + "\n")
        f.write('{"model": "ModelA", "n_peo')
    assert len(load_run(str(tmp_path / "run"))) == 1


# --- Wilson interval ----------------------------------------------------------
@pytest.mark.parametrize("k,n", [(0, 20), (20, 20), (10, 20), (1, 3)])
def test_wilson_stays_inside_zero_one(k, n):
    lo, hi = wilson(k, n)
    assert 0.0 <= lo <= hi <= 1.0


def test_wilson_is_not_degenerate_at_the_extremes():
    """The reason we use Wilson and not the normal approximation."""
    lo, hi = wilson(0, 30)
    assert lo == 0.0 and hi > 0.0
    lo, hi = wilson(30, 30)
    assert hi == 1.0 and lo < 1.0


# --- the chance baseline must be a comparable distribution --------------------
def test_chance_category_rate_is_a_distribution():
    story = generate_story(6, 6, 6000)
    tbl = chance_table(story, sample_questions(story, 6000), 300)
    assert tbl, "expected some question types to have a defined null"
    for qtype, cell in tbl.items():
        total = sum(cell["category_rate"].values())
        assert total == pytest.approx(1.0, abs=1e-9), f"{qtype} sums to {total}"
        assert all(0.0 <= v <= 1.0 for v in cell["category_rate"].values())


def test_chance_only_covers_types_with_a_defined_null():
    story = generate_story(6, 6, 6000)
    tbl = chance_table(story, sample_questions(story, 6000), 100)
    assert set(tbl).issubset(NULL_TYPES)
    assert "trajectory" not in tbl and "state_snapshot" not in tbl


def test_corrected_rate_is_nan_when_no_null_covers_the_cell(tmp_path):
    """Never silently correct by zero when the null does not apply."""
    recs = [_rec(correct=False, coarse_category="Binding",
                 question_type="state_snapshot", question_id=f"s{i}")
            for i in range(4)]
    tax = taxonomy_table(load_run(_write(tmp_path, recs)), CFG, with_chance=True)
    assert tax["chance_coverage"].iloc[0] == 0.0
    assert pd.isna(tax["chance_rate"].iloc[0])


def test_binding_chance_falls_as_the_value_range_widens():
    small = generate_story(4, 4, 4000)
    large = generate_story(20, 20, 20000)
    a = chance_table(small, sample_questions(small, 4000), 400)
    b = chance_table(large, sample_questions(large, 20000), 400)
    qa = a["person_timestep_lookup"]["category_rate"].get("Binding", 0)
    qb = b["person_timestep_lookup"]["category_rate"].get("Binding", 0)
    assert qa > qb, "chance binding collisions must be worse at small N"
