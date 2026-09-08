"""Small, specific tests. Each one covers a bug this class of pipeline actually
produces -- not coverage for its own sake.
"""

import random

import pytest

from src.generator import generate_story
from src.grade import first_divergence, grade_record, is_formulaic
from src.parsing import extract_answer, normalise, parse
from src.questions import questions_for_story
from src.taxonomy import CATEGORIES, chance_table, classify
from src.util import load_config

TYPES = ["initial_state_lookup", "transfer_recall", "person_timestep_lookup",
         "state_snapshot", "trajectory"]


def story(n=6, seed=999):
    return generate_story(n, n, seed, number_word_probability=0.45)


def meta(s):
    return {"M": s["states"], "person_ids": s["person_ids"],
            "transfers": [(e["timestep"], e["giver"], e["receiver"], e["amount"])
                          for e in s["narration"]]}


# --- simulator invariants ----------------------------------------------------
@pytest.mark.parametrize("n", [3, 6, 12])
def test_cluster_sums_are_invariant_and_holdings_never_negative(n):
    s = story(n, 4242)
    T = s["config"]["n_timesteps"]
    for members in s["clusters"].values():
        sums = {sum(s["states"][t][p] for p in members) for t in range(T + 1)}
        assert len(sums) == 1, f"cluster total moved: {sums}"
    for t in range(T + 1):
        assert all(v >= 0 for v in s["states"][t].values())


# --- ground truth against a hand-checked fixture ------------------------------
def test_ground_truth_matches_the_state_matrix():
    """Every type's gold is read straight off M / transfers -- verify, don't trust."""
    s = story(3, 7)
    M, T = s["states"], s["config"]["n_timesteps"]
    qs = {q["question_type"]: q for q in questions_for_story(s, "fix", TYPES)
          if "question_skipped" not in q}
    if "initial_state_lookup" in qs:
        q = qs["initial_state_lookup"]
        assert q["ground_truth"] == M[0][q["target_person_id"]]
    if "person_timestep_lookup" in qs:
        q = qs["person_timestep_lookup"]
        assert q["ground_truth"] == M[q["target_timestep"]][q["target_person_id"]]
    if "trajectory" in qs:
        q = qs["trajectory"]
        assert q["ground_truth"] == [M[t][q["target_person_id"]]
                                     for t in range(T + 1)]
    if "state_snapshot" in qs:
        q = qs["state_snapshot"]
        t = q["target_timestep"]
        assert q["ground_truth"] == {s["id_to_name"][p]: M[t][p]
                                     for p in s["person_ids"]}
    if "transfer_recall" in qs:
        q = qs["transfer_recall"]
        assert any(e["timestep"] == q["target_timestep"]
                   and s["id_to_name"][e["giver"]] == q["target_person"]
                   and e["amount"] == q["ground_truth"] for e in s["narration"])


# --- the sampling constraints the taxonomy depends on -------------------------
@pytest.mark.parametrize("seed", range(20))
def test_person_timestep_lookup_never_equals_the_initial_value(seed):
    """If M[t][P] == M[0][P] the `stale` category is undecidable."""
    s = story(6, 5000 + seed)
    for q in questions_for_story(s, "x", ["person_timestep_lookup"]):
        if "question_skipped" in q:
            continue
        p, t = q["target_person_id"], q["target_timestep"]
        assert s["states"][t][p] != s["states"][0][p]
        assert t >= 1


@pytest.mark.parametrize("seed", range(20))
def test_trajectory_never_picks_a_constant_column(seed):
    s = story(6, 6000 + seed)
    T = s["config"]["n_timesteps"]
    for q in questions_for_story(s, "x", ["trajectory"]):
        if "question_skipped" in q:
            continue
        col = [s["states"][t][q["target_person_id"]] for t in range(T + 1)]
        assert len(set(col)) > 1


# --- parser ------------------------------------------------------------------
@pytest.mark.parametrize("raw,path_has,is_err", [
    ('{"answer": 5}', "direct", False),
    ('```json\n{"answer": 5}\n```', "fence_stripped", False),
    ('Sure! Here it is: {"answer": 5}', "prose_stripped", False),
    ('{"answer": 5', "brace_repaired", False),          # the 9.4% -> 28.4% bug
    ('the answer is five', "no_brace", True),
    ('', "empty", True),
    (None, "no_output", True),
])
def test_parser_branches(raw, path_has, is_err):
    obj, path, err = parse(raw)
    assert err is is_err
    assert path_has in path
    if not err:
        assert obj["answer"] == 5


def test_brace_repair_is_refused_when_the_budget_truncated_the_output():
    """A cut-off answer is incomplete, not merely undelimited -- repairing it
    would invent an answer the model never finished."""
    _, path, err = parse('{"answer": 5', truncated=True)
    assert err is True and "unparseable" in path


# --- normaliser ---------------------------------------------------------------
def test_normalise_coerces_strings_ignores_key_order_and_rejects_non_integers():
    assert normalise("5", "person_timestep_lookup") == 5
    assert normalise(5.0, "person_timestep_lookup") == 5
    assert normalise(5.5, "person_timestep_lookup") is None       # never round
    assert normalise("abc", "person_timestep_lookup") is None
    assert normalise(True, "person_timestep_lookup") is None      # bool is not int
    a = normalise({" Ram": "3", "Sita": 4}, "state_snapshot")
    b = normalise({"sita": 4, "ram": 3}, "state_snapshot")
    assert a == b
    assert normalise([1, "2", 3], "trajectory") == [1, 2, 3]
    assert normalise([1, 2], "trajectory") != [2, 1]              # order matters


def test_strict_and_normalised_grading_are_both_recorded():
    rec = {"question_type": "person_timestep_lookup", "ground_truth": 5,
           "raw_output": '{"answer": "5"}', "truncated": False}
    g = grade_record(rec)
    assert g["correct"] is True
    assert g["correct_strict"] is False
    assert g["normalisation_rescued"] is True


def test_format_error_never_gets_graded_correct():
    g = grade_record({"question_type": "person_timestep_lookup",
                      "ground_truth": 5, "raw_output": "no json here",
                      "truncated": False})
    assert g["format_error"] is True and g["correct"] is False


def test_overflow_rows_are_not_scored():
    g = grade_record({"question_type": "person_timestep_lookup",
                      "ground_truth": 5, "raw_output": None,
                      "context_overflow": True})
    assert g["correct"] is False and g["format_error"] is False
    assert g["parse_path"] == "not_generated"


# --- taxonomy -----------------------------------------------------------------
def test_each_category_fires_on_a_constructed_answer():
    s = story(6, 31337)
    m = meta(s)
    M, ids = s["states"], s["person_ids"]
    T = s["config"]["n_timesteps"]
    cands = [(p, t) for p in ids for t in range(1, T + 1) if M[t][p] != M[0][p]]
    pid, t = cands[0]
    y = M[t][pid]

    lab, _, _ = classify(M[0][pid], y, pid, t, m)
    assert lab == "stale"

    other = [q for q in ids if q != pid and M[t][q] not in (y, M[0][pid])]
    if other:
        lab, _, _ = classify(M[t][other[0]], y, pid, t, m)
        assert lab in ("wrong_person", "wrong_timestep", "single_transfer")

    lab, matched, _ = classify(y + 9999, y, pid, t, m)
    assert lab == "unexplained" and matched == ["unexplained"]


def test_single_transfer_sub_labels():
    s = story(6, 4711)
    m = meta(s)
    M, ids = s["states"], s["person_ids"]
    T = s["config"]["n_timesteps"]
    for pid in ids:
        for t in range(1, T + 1):
            if M[t][pid] == M[0][pid]:
                continue
            deltas = [d for (tt, g, r, amt) in m["transfers"] if tt <= t
                      for d in ([-amt] if g == pid else [amt] if r == pid else [])]
            if not deltas:
                continue
            y = M[t][pid]
            lab, _, sub = classify(y - deltas[0], y, pid, t, m)
            if lab == "single_transfer":
                assert sub in ("missed", "doubled", "sign_flipped")
                return
    pytest.skip("no usable transfer in this fixture")


def test_priority_is_respected_and_all_matches_recorded():
    s = story(6, 8080)
    m = meta(s)
    M, ids = s["states"], s["person_ids"]
    T = s["config"]["n_timesteps"]
    cands = [(p, t) for p in ids for t in range(1, T + 1) if M[t][p] != M[0][p]]
    pid, t = cands[0]
    lab, matched, _ = classify(M[0][pid], M[t][pid], pid, t, m)
    assert lab == matched[0] == "stale"
    assert matched == sorted(set(matched), key=CATEGORIES.index)


def test_chance_null_is_near_zero_on_uniform_random_answers():
    """The corrected rate must vanish when the answers ARE the null."""
    s = story(6, 2024)
    m = meta(s)
    M, ids = s["states"], s["person_ids"]
    T = s["config"]["n_timesteps"]
    pid, t = [(p, tt) for p in ids for tt in range(1, T + 1)
              if M[tt][p] != M[0][p]][0]
    space = sum(M[0][p] for p in s["clusters"]["cluster_1"]) + 1
    rng = random.Random(0)
    wrong = [{"answer_space_size": space, "ground_truth": M[t][pid],
              "target_person_id": pid, "target_timestep": t, "story_meta": m}
             for _ in range(60)]
    chance = chance_table(wrong, draws=200, seed=1)
    observed = {c: 0 for c in CATEGORIES}
    n = 0
    for _ in range(600):
        a = rng.randrange(space)
        if a == M[t][pid]:
            continue
        lab, _, _ = classify(a, M[t][pid], pid, t, m)
        observed[lab] += 1
        n += 1
    for c in CATEGORIES:
        assert abs(observed[c] / n - chance[c]) < 0.12, c


# --- trajectory helpers -------------------------------------------------------
def test_first_divergence_and_formulaic_detection():
    assert first_divergence([1, 2, 9], [1, 2, 3]) == 2
    assert first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert first_divergence([1, 2], [1, 2, 3]) == 2
    assert is_formulaic([3, 3, 3, 3]) is True
    assert is_formulaic([3, 6, 9, 12]) is True
    assert is_formulaic([3, 6, 4, 12]) is False


# --- regrade idempotence ------------------------------------------------------
def test_regrading_twice_is_identical():
    rec = {"question_type": "trajectory", "ground_truth": [1, 2, 3],
           "raw_output": '{"answer": [1, 2, 4]}', "truncated": False}
    a = grade_record(dict(rec))
    b = grade_record(dict(a))
    assert a == b


# --- config sanity ------------------------------------------------------------
def test_only_mistral_omits_the_system_role():
    cfg = load_config()
    flags = {m["key"]: m["use_system_role"] for m in cfg["models"]}
    assert flags["mistral"] is False
    assert all(v for k, v in flags.items() if k != "mistral")
