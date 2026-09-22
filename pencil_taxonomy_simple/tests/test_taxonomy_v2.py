"""Tests for the v2 taxonomy. The section 5 fixture is the primary one."""

import json
import os
import random

import pytest

from src.taxonomy_v2 import (CATEGORIES, RULES, A1_stale, A2_partial_update,
                             A3_lookahead, A4_boundary_off_by_one, _ctx,
                             chance, classify)

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "taxonomy_v2.json")


def fixture_ctx():
    """Build the section 5 world: N=8, T=8, Ravi at t=5, y=14."""
    d = json.load(open(FIX, encoding="utf-8"))["setup"]
    col = d["ravi_column"]
    # Ravi's column is given; the others only need to be right at t=5, and
    # nowhere else is read for a t=5 question except via A2/A3 on Ravi's column.
    others = {"Sita": 25, "Gita": 2, "Abdul": 30}
    M = {}
    for t in range(d["T"] + 1):
        M[t] = {"Ravi": col[t]}
        for k, v in others.items():
            M[t][k] = v if t == d["t"] else v + 100 + t   # never collide off t=5
    # tot_K must be 41 as the fixture states.
    M[0]["Sita"] = 41 - col[0] - 1
    M[0]["Gita"] = 1
    transfers = [tuple(x) for x in d["transfers"]]
    return _ctx(M, d["t"], "Ravi", transfers, d["cluster"],
                ["Ravi", "Sita", "Gita", "Abdul"], 42)


def test_fixture_setup_is_self_consistent():
    c = fixture_ctx()
    assert c["y"] == 14
    assert c["tot_K"] == 41
    assert c["X"] == [-4, 6, -1]        # transfers at t <= 5 only
    assert c["D"] == [4, 6, 1]


@pytest.mark.parametrize("row", json.load(open(FIX, encoding="utf-8"))["rows"],
                         ids=lambda r: f"a={r['answer']}")
def test_worked_example(row):
    """Every row of the spec's section 5 table."""
    c = fixture_ctx()
    got = classify(row["answer"], c)
    assert got["label_v2"] == row["winner"], (
        f"a={row['answer']}: expected {row['winner']}, got {got['label_v2']} "
        f"(matched {got['labels_matched_v2']})")
    assert sorted(got["labels_matched_v2"]) == sorted(row["fires"]), (
        f"a={row['answer']}: matched {got['labels_matched_v2']}, "
        f"fixture says {row['fires']}")


def test_containment_A1_subset_A2():
    """stale is the s==0 case of partial_update. A regression here silently
    reassigns a whole category."""
    c = fixture_ctx()
    for a in range(0, 42):
        if A1_stale(a, c):
            assert A2_partial_update(a, c), f"A1 fired but not A2 at a={a}"


def test_containment_A4_subset_A2_or_A3():
    c = fixture_ctx()
    for a in range(0, 42):
        if A4_boundary_off_by_one(a, c):
            assert A2_partial_update(a, c) or A3_lookahead(a, c), a


def test_priority_puts_A1_above_A2_and_A4_above_both():
    order = [lab for _, lab, _, _ in RULES]
    assert order.index("stale") < order.index("partial_update")
    assert order.index("boundary_off_by_one") < order.index("partial_update")
    assert order.index("boundary_off_by_one") < order.index("lookahead")


def test_classify_is_pure_and_deterministic():
    c = fixture_ctx()
    a, b = classify(15, c), classify(15, c)
    assert a == b
    assert classify(15, fixture_ctx()) == a      # fresh ctx, same result


def test_every_answer_gets_a_label():
    """`unexplained` is a label, not an absence."""
    c = fixture_ctx()
    for a in range(0, 60):
        r = classify(a, c)
        assert r["labels_matched_v2"], a
        assert r["label_v2"] in CATEGORIES


def test_null_sanity_corrected_near_zero_on_random_answers():
    c = fixture_ctx()
    rows = [{"precheck": None, "answer_space_size": 42, "ctx": c}
            for _ in range(40)]
    ch, cov, nlab = chance(rows, draws=300, seed=1)
    rng = random.Random(7)
    obs = {k: 0 for k in CATEGORIES}
    n = 0
    for _ in range(4000):
        a = rng.randrange(42)
        if a == c["y"]:
            continue
        obs[classify(a, c)["label_v2"]] += 1
        n += 1
    for k in CATEGORIES:
        assert abs(obs[k] / n - ch[k]) < 0.10, k
    assert 0.0 <= cov <= 1.0 and nlab >= 1.0


def test_chance_coverage_is_reported_not_assumed():
    """With 15 rules over a small range the null labels most random draws.
    The test asserts the diagnostic exists and is high -- that IS the finding."""
    c = fixture_ctx()
    rows = [{"precheck": None, "answer_space_size": 42, "ctx": c}]
    _, cov, nlab = chance(rows, draws=500, seed=0)
    assert cov > 0.5, f"coverage {cov} unexpectedly low -- recheck the rules"


# --- integration: these need the sweep on disk, skip if absent ---------------
import glob            # noqa: E402
import hashlib         # noqa: E402
import subprocess      # noqa: E402
import sys             # noqa: E402

from src.util import ROOT   # noqa: E402

RUN = os.path.join(ROOT, "results", "RUN_ID.txt")
HAVE_RUN = os.path.exists(RUN)


def _run_dir():
    rid = open(RUN, encoding="utf-8").read().split("=")[1].strip()
    return os.path.join(ROOT, "results", "raw", f"{rid}_regraded")


@pytest.mark.skipif(not HAVE_RUN, reason="no sweep on disk")
def test_join_integrity_every_ground_truth_matches_M():
    """If this fails the WRONG stories were joined and every v2 label is wrong."""
    from src.taxonomy_v2 import load_stories
    stories = load_stories(os.path.join(ROOT, "data", "stories"))
    n = 0
    for p in glob.glob(os.path.join(_run_dir(), "*.jsonl")):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r["question_type"] != "person_timestep_lookup":
                continue
            s = stories[r["story_id"]]
            assert s["states"][r["target_timestep"]][r["target_person_id"]] == \
                r["ground_truth"], r["question_id"]
            n += 1
    assert n > 0


@pytest.mark.skipif(not HAVE_RUN, reason="no sweep on disk")
def test_v1_tables_reproduce_byte_for_byte():
    """v2 must not disturb v1. src/taxonomy.py is unmodified, so re-running
    aggregate over the (now v2-annotated) records must give identical v1 tables."""
    tdir = os.path.join(ROOT, "results", "tables")
    before = {}
    for name in ("taxonomy.csv", "accuracy.csv", "divergence.csv"):
        p = os.path.join(tdir, name)
        before[name] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    subprocess.check_call(
        [sys.executable, "-m", "src.aggregate", "--run-dir",
         os.path.relpath(_run_dir(), ROOT)], cwd=ROOT,
        stdout=subprocess.DEVNULL, env={**os.environ, "PYTHONPATH": ROOT})
    for name, h in before.items():
        after = hashlib.sha256(
            open(os.path.join(tdir, name), "rb").read()).hexdigest()
        assert after == h, f"{name} changed -- v2 disturbed v1"


@pytest.mark.skipif(not HAVE_RUN, reason="no sweep on disk")
def test_v2_pass_is_idempotent():
    """Running twice produces byte-identical labels."""
    from src.taxonomy_v2 import build_rows, label_rows, load_stories
    stories = load_stories(os.path.join(ROOT, "data", "stories"))
    recs = []
    for p in sorted(glob.glob(os.path.join(_run_dir(), "*.jsonl")))[:1]:
        recs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    a = [(w["question_id"], w["label_v2"], tuple(w["labels_matched_v2"]))
         for w in label_rows(build_rows(recs, stories))]
    b = [(w["question_id"], w["label_v2"], tuple(w["labels_matched_v2"]))
         for w in label_rows(build_rows(recs, stories))]
    assert a == b and len(a) > 0


@pytest.mark.skipif(not HAVE_RUN, reason="no sweep on disk")
def test_no_correct_answer_carries_a_v2_label():
    """Regression: `question_id` is NOT unique across models -- the same 300
    stories are evaluated by all four, so it is
    'N02_s003_person_timestep_lookup' four times. Keying the write-back on it
    alone put one model's label onto every model's record, including correct
    answers. The key must be (model_key, question_id)."""
    bad = 0
    for p in glob.glob(os.path.join(_run_dir(), "*.jsonl")):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r.get("label_v2") and r.get("correct"):
                bad += 1
    assert bad == 0, f"{bad} correct answers carry a v2 label"


@pytest.mark.skipif(not HAVE_RUN, reason="no sweep on disk")
def test_question_id_is_not_unique_across_models():
    """Documents WHY the composite key is needed, so a future refactor that
    'simplifies' it back trips this test instead of silently corrupting labels."""
    seen = {}
    for p in glob.glob(os.path.join(_run_dir(), "*.jsonl")):
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            seen.setdefault(r["question_id"], set()).add(r["model_key"])
    shared = [q for q, m in seen.items() if len(m) > 1]
    assert shared, "expected question_ids shared across models"
