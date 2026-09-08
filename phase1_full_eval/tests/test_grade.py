"""Unit tests for parse_answer / grade on good, wrong and malformed answers."""

import pytest

from src.grade import as_int, grade, parse_answer
from src.questions import sample_questions
from src.simulate import generate_story

STORY = generate_story(6, 6, 6000)
QS = {q["question_type"]: q for q in sample_questions(STORY, 6000)}


# --- parse_answer -------------------------------------------------------------
def test_parses_plain_json():
    p, s = parse_answer('{"question_type": "x", "answer": 3}')
    assert s == "ok" and p["answer"] == 3


def test_parses_fenced_json():
    p, s = parse_answer('```json\n{"answer": 3}\n```')
    assert s == "ok" and p["answer"] == 3


def test_parses_json_with_prose_around_it():
    p, s = parse_answer('Sure! Here is the answer:\n{"answer": 3}\nHope that helps.')
    assert s == "ok" and p["answer"] == 3


def test_nested_braces_are_balanced_correctly():
    p, s = parse_answer('{"counts": {"A": 1, "B": 2}, "t": 0}')
    assert s == "ok" and p["counts"] == {"A": 1, "B": 2}


def test_brace_inside_string_does_not_confuse_scanner():
    p, s = parse_answer('{"note": "a } brace", "answer": 4}')
    assert s == "ok" and p["answer"] == 4


@pytest.mark.parametrize("text,status", [
    ("", "empty"),
    ("   ", "empty"),
    ("There is no JSON here at all.", "no_json_found"),
    ('{"answer": 3,,}', "invalid_json"),
    ("[1, 2, 3]", "no_json_found"),
])
def test_bad_inputs(text, status):
    p, s = parse_answer(text)
    assert p is None and s == status


def test_truncation_detected_via_finish_reason():
    p, s = parse_answer('{"counts": [1, 2, 3', finish_reason="length")
    assert p is None and s == "truncated"


def test_refusal():
    p, s = parse_answer("I'm sorry, I cannot answer that.")
    assert p is None and s == "refusal"


# --- as_int -------------------------------------------------------------------
@pytest.mark.parametrize("raw,want", [
    (7, 7), ("7", 7), (7.0, 7), (" 7 ", 7), ("+7", 7), ("-7", -7), ("1,024", 1024),
])
def test_as_int_accepts(raw, want):
    assert as_int(raw) == want


@pytest.mark.parametrize("raw", [7.5, "seven", None, [], {}, True, ""])
def test_as_int_rejects(raw):
    assert as_int(raw) is None


# --- grading ------------------------------------------------------------------
def test_correct_scalar():
    q = QS["person_timestep_lookup"]
    r = grade(q, {"question_type": q["question_type"], "answer": q["gold"]["answer"]}, STORY)
    assert r["correct"] and r["status"] == "ok"


def test_correct_scalar_as_string():
    q = QS["person_timestep_lookup"]
    r = grade(q, {"answer": str(q["gold"]["answer"])}, STORY)
    assert r["correct"]


def test_wrong_scalar():
    q = QS["person_timestep_lookup"]
    r = grade(q, {"answer": q["gold"]["answer"] + 3}, STORY)
    assert not r["correct"] and r["status"] == "ok"


def test_wrong_question_type_declared():
    q = QS["person_timestep_lookup"]
    r = grade(q, {"question_type": "argmax_person", "answer": 1}, STORY)
    assert r["status"] == "wrong_question_type"


def test_missing_fields():
    q = QS["person_timestep_lookup"]
    r = grade(q, {"question_type": q["question_type"]}, STORY)
    assert r["status"] == "missing_fields"


def test_bad_types():
    q = QS["person_timestep_lookup"]
    r = grade(q, {"answer": "quite a lot"}, STORY)
    assert r["status"] == "bad_types"


def test_person_value_timesteps_order_insensitive():
    q = QS["person_value_timesteps"]
    r = grade(q, {"timesteps": list(reversed(q["gold"]["timesteps"]))}, STORY)
    assert r["correct"]


def test_person_value_timesteps_reports_missed_and_spurious():
    q = QS["person_value_timesteps"]
    r = grade(q, {"timesteps": q["gold"]["timesteps"] + [99]}, STORY)
    assert not r["correct"] and r["detail"]["spurious"] == [99]


def test_trajectory_correct():
    q = QS["trajectory"]
    r = grade(q, {"counts": list(q["gold"]["counts"])}, STORY)
    assert r["correct"] and r["detail"]["first_divergence_index"] is None


def test_trajectory_first_divergence_index():
    q = QS["trajectory"]
    counts = list(q["gold"]["counts"])
    counts[3] += 1
    r = grade(q, {"counts": counts}, STORY)
    assert not r["correct"] and r["detail"]["first_divergence_index"] == 3


def test_trajectory_short_list_diverges_at_its_end():
    q = QS["trajectory"]
    counts = list(q["gold"]["counts"])[:2]
    r = grade(q, {"counts": counts}, STORY)
    assert r["detail"]["length_mismatch"] and r["detail"]["first_divergence_index"] == 2


def test_state_snapshot_correct():
    q = QS["state_snapshot"]
    r = grade(q, {"counts": dict(q["gold"]["counts"])}, STORY)
    assert r["correct"] and not r["detail"]["conservation_violation"]


def test_state_snapshot_permutation_flagged():
    q = QS["state_snapshot"]
    gold = q["gold"]["counts"]
    names = list(gold)
    swapped = dict(gold)
    # find two people with different counts and swap them
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if gold[names[i]] != gold[names[j]]:
                swapped[names[i]], swapped[names[j]] = gold[names[j]], gold[names[i]]
                r = grade(q, {"counts": swapped}, STORY)
                assert not r["correct"]
                assert r["detail"]["permutation"]
                assert not r["detail"]["conservation_violation"]
                return
    pytest.skip("all counts equal in this story")


def test_state_snapshot_conservation_violation():
    q = QS["state_snapshot"]
    bad = dict(q["gold"]["counts"])
    bad[list(bad)[0]] += 5
    r = grade(q, {"counts": bad}, STORY)
    assert r["detail"]["conservation_violation"]
    assert not r["detail"]["permutation"]


def test_state_snapshot_partial_fraction():
    q = QS["state_snapshot"]
    partial = dict(q["gold"]["counts"])
    keys = list(partial)
    partial[keys[0]] = partial[keys[0]] + 100
    r = grade(q, {"counts": partial}, STORY)
    expected = (len(keys) - 1) / len(keys)
    assert r["detail"]["partial_correct_fraction"] == pytest.approx(expected)


def test_comparison_name_case_insensitive():
    q = QS["pairwise_comparison"]
    r = grade(q, {"answer": q["gold"]["answer"].upper()}, STORY)
    assert r["correct"]


def test_argmax_wrong_name():
    q = QS["argmax_person"]
    r = grade(q, {"answer": "Nobody"}, STORY)
    assert not r["correct"] and r["status"] == "ok"


def test_net_delta_negative_answer():
    q = QS["net_delta"]
    r = grade(q, {"answer": q["gold"]["answer"]}, STORY)
    assert r["correct"]


def test_all_twelve_types_grade_their_own_gold_as_correct():
    """Round-trip: for every type, the gold answer must grade as correct."""
    for n, seed in [(4, 4000), (10, 10000), (20, 20000)]:
        story = generate_story(n, n, seed)
        for q in sample_questions(story, seed):
            parsed = {"question_type": q["question_type"], **q["gold"]}
            r = grade(q, parsed, story)
            assert r["correct"], f"{q['question_type']} failed to grade its own gold"
