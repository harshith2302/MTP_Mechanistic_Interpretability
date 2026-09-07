"""Constructed-failure tests: build an answer that IS an omission / binding swap
/ sign flip by construction, and assert the classifier catches it."""

import pytest

from src.chance import chance_table
from src.classify import COARSE, PRIORITY, classify
from src.grade import grade
from src.questions import sample_questions
from src.replay import _deltas_touching
from src.simulate import generate_story


def _story_and(qtype, n=10, seed=10000):
    story = generate_story(n, n, seed)
    qs = [q for q in sample_questions(story, seed) if q["question_type"] == qtype]
    return story, qs[0]


def _cls(q, pred, story):
    return classify(q, grade(q, pred, story), story)


# --- table sanity -------------------------------------------------------------
def test_every_priority_label_has_a_coarse_category():
    for lab in PRIORITY:
        assert lab in COARSE, f"{lab} missing from COARSE"


def test_correct_answer_is_labelled_correct():
    story, q = _story_and("person_timestep_lookup")
    r = _cls(q, {"answer": q["gold"]["answer"]}, story)
    assert r["label"] == "correct" and not r["ambiguous"]


# --- format never gets a mechanism -------------------------------------------
@pytest.mark.parametrize("pred,want", [
    (None, "format_error"),
    ({"answer": "lots"}, "bad_types"),
    ({"question_type": "argmax_person", "answer": 1}, "wrong_question_type"),
    ({}, "missing_fields"),
])
def test_format_failures_get_no_mechanism(pred, want):
    story, q = _story_and("person_timestep_lookup")
    r = _cls(q, pred, story)
    assert r["label"] == want
    assert r["coarse_category"] == "Format"


# --- constructed structural failures -----------------------------------------
def test_constructed_omission_1():
    """Drop the last narrated transfer touching p: must fire omission_1."""
    story = generate_story(12, 12, 12000)
    for q in sample_questions(story, 12000):
        if q["question_type"] != "person_timestep_lookup":
            continue
        p, t = q["queried_person"], q["queried_timestep"]
        deltas = _deltas_touching(story, p, t)
        if not deltas or deltas[-1] == 0:
            continue
        forged = q["gold"]["answer"] - deltas[-1]
        r = _cls(q, {"answer": forged}, story)
        assert "omission_1" in r["labels_matched"], r
        return
    pytest.skip("no suitable question in this story")


def test_constructed_direction_flip_all():
    story = generate_story(12, 12, 12000)
    for q in sample_questions(story, 12000):
        if q["question_type"] != "person_timestep_lookup":
            continue
        p, t = q["queried_person"], q["queried_timestep"]
        deltas = _deltas_touching(story, p, t)
        if not deltas or sum(deltas) == 0:
            continue
        forged = story["states"][0][p] - sum(deltas)
        r = _cls(q, {"answer": forged}, story)
        assert "direction_flip_all" in r["labels_matched"], r
        return
    pytest.skip("no suitable question")


def test_constructed_over_application():
    story = generate_story(12, 12, 12000)
    for q in sample_questions(story, 12000):
        if q["question_type"] != "person_timestep_lookup":
            continue
        p, t = q["queried_person"], q["queried_timestep"]
        deltas = _deltas_touching(story, p, t)
        if not deltas:
            continue
        for d in deltas:
            if d == 0:
                continue
            r = _cls(q, {"answer": q["gold"]["answer"] + d}, story)
            assert "over_application" in r["labels_matched"], r
            return
    pytest.skip("no suitable question")


def test_constructed_binding_swap():
    """Answer another person's true count at the same t -> binding_person."""
    story = generate_story(12, 12, 12000)
    for q in sample_questions(story, 12000):
        if q["question_type"] != "person_timestep_lookup":
            continue
        p, t = q["queried_person"], q["queried_timestep"]
        others = [story["states"][t][o] for o in story["person_ids"]
                  if o != p and story["states"][t][o] != q["gold"]["answer"]]
        if not others:
            continue
        r = _cls(q, {"answer": others[0]}, story)
        assert "binding_person" in r["labels_matched"], r
        return
    pytest.skip("no suitable question")


def test_constructed_temporal_initial():
    story = generate_story(12, 12, 12000)
    for q in sample_questions(story, 12000):
        if q["question_type"] != "person_timestep_lookup":
            continue
        p, t = q["queried_person"], q["queried_timestep"]
        if t == 0 or story["states"][0][p] == q["gold"]["answer"]:
            continue
        r = _cls(q, {"answer": story["states"][0][p]}, story)
        assert "temporal_initial" in r["labels_matched"], r
        return
    pytest.skip("no suitable question")


def test_constructed_net_delta_sign_error():
    story, q = _story_and("net_delta", 12, 12000)
    g = q["gold"]["answer"]
    if g == 0:
        pytest.skip("gold is zero")
    r = _cls(q, {"answer": -g}, story)
    assert "sign_error" in r["labels_matched"]
    assert r["label"] == "sign_error" and r["coarse_category"] == "Arithmetic"


def test_constructed_snapshot_permutation_is_binding():
    story, q = _story_and("state_snapshot", 12, 12000)
    gold = q["gold"]["counts"]
    names = list(gold)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if gold[names[i]] == gold[names[j]]:
                continue
            sw = dict(gold)
            sw[names[i]], sw[names[j]] = gold[names[j]], gold[names[i]]
            r = _cls(q, {"counts": sw}, story)
            assert "permutation" in r["labels_matched"]
            assert r["label"] == "permutation"
            assert r["coarse_category"] == "Binding"
            return
    pytest.skip("all counts equal")


def test_constructed_duration_included_t0():
    """Answering as if t=0 counted -> included_t0 (Temporal)."""
    from src.questions import _satisfies
    story = generate_story(12, 12, 12000)
    for q in sample_questions(story, 12000):
        if q["question_type"] != "duration_condition":
            continue
        p = q["queried_person"]
        op, th = q["meta"]["operator"], q["meta"]["threshold"]
        with_t0 = sum(_satisfies(story["states"][t][p], op, th) for t in range(13))
        if with_t0 == q["gold"]["answer"]:
            continue
        r = _cls(q, {"answer": with_t0}, story)
        assert "included_t0" in r["labels_matched"], r
        return
    pytest.skip("no suitable question")


def test_constructed_condition_flip():
    from src.questions import _satisfies
    story = generate_story(12, 12, 12000)
    for q in sample_questions(story, 12000):
        if q["question_type"] != "population_condition":
            continue
        op, th, t = q["meta"]["operator"], q["meta"]["threshold"], q["queried_timestep"]
        flip = {">=": "<", "<=": ">", ">": "<=", "<": ">="}[op]
        val = sum(_satisfies(c, flip, th) for c in story["states"][t].values())
        if val == q["gold"]["answer"]:
            continue
        r = _cls(q, {"answer": val}, story)
        assert "condition_flip" in r["labels_matched"], r
        return
    pytest.skip("no suitable question")


def test_constructed_digit_transposition():
    story, q = _story_and("person_timestep_lookup", 25, 25003)
    g = q["gold"]["answer"]
    if g < 10 or str(g)[0] == str(g)[1]:
        pytest.skip("gold not two distinct digits")
    forged = int(str(g)[::-1])
    r = _cls(q, {"answer": forged}, story)
    assert "digit_error" in r["labels_matched"]


def test_trajectory_first_divergence_is_reported_and_classified():
    story, q = _story_and("trajectory", 12, 12000)
    counts = list(q["gold"]["counts"])
    counts[5] = counts[5] + 1
    r = _cls(q, {"counts": counts}, story)
    assert r["first_divergence_index"] == 5
    assert r["label"] != "correct"


def test_ambiguity_is_flagged():
    """An answer matching >1 hypothesis must set ambiguous and keep all labels."""
    story = generate_story(6, 6, 6000)
    hits = 0
    for q in sample_questions(story, 6000):
        if q["question_type"] != "person_timestep_lookup":
            continue
        for a in range(0, story["config"]["total_pencils"] + 1):
            r = _cls(q, {"answer": a}, story)
            if r["ambiguous"]:
                assert len(r["labels_matched"]) > 1
                assert r["label"] in r["labels_matched"]
                hits += 1
    assert hits > 0, "expected some ambiguous answers at N=6"


# --- chance baseline ----------------------------------------------------------
def test_chance_baseline_is_large_at_small_n():
    """The whole point: binding fires constantly by chance when N is small."""
    story = generate_story(4, 4, 4000)
    qs = sample_questions(story, 4000)
    tbl = chance_table(story, qs, n_samples=400)
    fires = tbl["person_timestep_lookup"]["label_fire_rates"]
    binding = fires.get("binding_person", 0.0)
    assert binding > 0.15, f"expected a high chance rate at N=4, got {binding}"


def test_chance_baseline_falls_as_n_grows_relative_to_range():
    story_s = generate_story(4, 4, 4000)
    story_l = generate_story(20, 20, 20000)
    small = chance_table(story_s, sample_questions(story_s, 4000), 400)
    large = chance_table(story_l, sample_questions(story_l, 20000), 400)
    a = small["person_timestep_lookup"]["label_fire_rates"].get("off_by_small", 0)
    b = large["person_timestep_lookup"]["label_fire_rates"].get("off_by_small", 0)
    assert a > b, "off_by_small must be rarer when the value range is wider"


# --- trajectory length mismatch (regression: killed the 2026-09-07 sweep) -----
# grade.py sets first_divergence_index = min(len(pred), len(gold)) when the two
# lists agree on their overlap but differ in length. classify.py then indexes
# BOTH lists with it while guarding only on len(pred), so a prediction LONGER
# than gold raised IndexError -- and because vLLM's EngineCore subprocess
# survives the crash, the Slurm job sat in RUNNING doing nothing instead of
# failing fast. Three of four sweep tasks were lost to this.
@pytest.mark.parametrize("delta,want", [
    (-3, "format_error"),   # shorter than gold
    (-1, "format_error"),
    (3, "format_error"),    # longer than gold -- this is the case that crashed
    (1, "format_error"),
])
def test_trajectory_length_mismatch_is_format_not_a_crash(delta, want):
    story, q = _story_and("trajectory")
    gold = q["gold"]["counts"]
    pred = gold[:delta] if delta < 0 else gold + [7] * delta
    r = _cls(q, {"question_type": "trajectory",
                 "person": q["queried_person"], "counts": pred}, story)
    assert r["label"] == want


def test_trajectory_longer_but_diverging_early_still_gets_a_mechanism():
    """A too-long list that also disagrees inside the overlap must be labelled at
    the divergence point, not swept into format_error."""
    story, q = _story_and("trajectory")
    gold = q["gold"]["counts"]
    pred = gold[:3] + [999] + list(gold[4:]) + [7, 7, 7]
    r = _cls(q, {"question_type": "trajectory",
                 "person": q["queried_person"], "counts": pred}, story)
    assert r["first_divergence_index"] == 3
    assert r["label"] != "format_error"
