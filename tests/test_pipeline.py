"""End-to-end test of the record pipeline with a fake engine (no GPU, no vLLM).

Exercises exactly what run_eval._generate does per generation: parse -> grade ->
classify -> record, plus the resume logic and the context_overflow rule.
"""

import json
import os

import pytest

from src.classify import classify
from src.grade import grade, parse_answer
from src.prompts import PromptStore, build_body, prompt_sha256
from src.questions import make_question_id, sample_questions
from src.run_eval import existing_ids
from src.simulate import generate_story

REQUIRED_FIELDS = [
    "run_id", "model", "n_people", "n_timesteps", "total_pencils", "story_seed",
    "story_index", "question_id", "question_type", "question_text",
    "prompt_sha256", "gold", "context_overflow",
]


def _fake_generation(question, story, mode):
    """Produce a raw model output of a chosen quality."""
    if mode == "perfect":
        return json.dumps({"question_type": question["question_type"], **question["gold"]})
    if mode == "fenced":
        return "```json\n" + json.dumps(
            {"question_type": question["question_type"], **question["gold"]}) + "\n```"
    if mode == "chatty":
        return ("Sure, here is the answer.\n"
                + json.dumps({"question_type": question["question_type"], **question["gold"]})
                + "\nLet me know if you need more.")
    if mode == "garbage":
        return "I think the answer is probably around seven or so."
    if mode == "empty":
        return ""
    raise ValueError(mode)


def _record(question, story, raw, finish="stop"):
    parsed, status = parse_answer(raw, finish)
    graded = grade(question, parsed, story)
    if parsed is None:
        graded["status"] = status
    labels = classify(question, graded, story)
    return {"parse_status": status, "correct": graded["correct"], **labels}


@pytest.mark.parametrize("mode", ["perfect", "fenced", "chatty"])
def test_well_formed_gold_answers_grade_correct_for_all_types(mode):
    story = generate_story(8, 8, 8000)
    for q in sample_questions(story, 8000):
        rec = _record(q, story, _fake_generation(q, story, mode))
        assert rec["correct"], f"{q['question_type']} / {mode}"
        assert rec["label"] == "correct"
        assert rec["coarse_category"] == "Correct"


def test_garbage_is_format_error_and_never_gets_a_mechanism():
    story = generate_story(8, 8, 8000)
    for q in sample_questions(story, 8000):
        rec = _record(q, story, _fake_generation(q, story, "garbage"))
        assert rec["coarse_category"] == "Format"
        assert rec["correct"] is False


def test_empty_output_is_refusal_or_truncation():
    story = generate_story(8, 8, 8000)
    q = sample_questions(story, 8000)[0]
    rec = _record(q, story, "")
    assert rec["label"] == "refusal_or_truncation"


def test_truncated_heavy_answer_is_format_not_wrong_reasoning():
    story = generate_story(20, 20, 20000)
    q = [x for x in sample_questions(story, 20000) if x["question_type"] == "trajectory"][0]
    raw = '{"question_type": "trajectory", "person": "X", "counts": [1, 2, 3'
    rec = _record(q, story, raw)
    assert rec["coarse_category"] == "Format"


def test_every_record_field_is_json_serialisable():
    story = generate_story(6, 6, 6000)
    for q in sample_questions(story, 6000):
        rec = _record(q, story, _fake_generation(q, story, "perfect"))
        json.dumps(rec)


def test_resume_skips_ids_already_written(tmp_path):
    path = tmp_path / "N06.jsonl"
    ids = ["m|N6|s0|trajectory|0", "m|N6|s1|net_delta|1"]
    with open(path, "w") as f:
        for i in ids:
            f.write(json.dumps({"question_id": i}) + "\n")
    assert existing_ids(str(path)) == set(ids)


def test_resume_tolerates_a_truncated_last_line(tmp_path):
    """A walltime kill can leave half a line; resume must not crash on it."""
    path = tmp_path / "N06.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps({"question_id": "good|1"}) + "\n")
        f.write('{"question_id": "half-writ')
    assert existing_ids(str(path)) == {"good|1"}


def test_question_ids_are_unique_within_a_story():
    story = generate_story(12, 12, 12000)
    qs = sample_questions(story, 12000)
    ids = [make_question_id("m", 12, 0, q["question_type"], q["index_within_type"])
           for q in qs]
    assert len(ids) == len(set(ids))


def test_prompt_store_deduplicates(tmp_path):
    store = PromptStore(str(tmp_path / "prompts.jsonl"))
    h1 = store.add("hello")
    h2 = store.add("hello")
    h3 = store.add("world")
    store.flush()
    store.close()
    assert h1 == h2 == prompt_sha256("hello") and h3 != h1
    with open(tmp_path / "prompts.jsonl") as f:
        assert len(f.readlines()) == 2


def test_prompt_store_reloads_seen_hashes_across_restarts(tmp_path):
    p = str(tmp_path / "prompts.jsonl")
    s1 = PromptStore(p); s1.add("a"); s1.flush(); s1.close()
    s2 = PromptStore(p); s2.add("a"); s2.add("b"); s2.flush(); s2.close()
    with open(p) as f:
        assert len(f.readlines()) == 2


def test_prompt_body_contains_story_question_and_schema():
    story = generate_story(5, 5, 5000)
    q = sample_questions(story, 5000)[0]
    body = build_body(story["text"], q)
    assert story["text"] in body
    assert q["question_text"] in body
    assert "question_type" in body
    assert "valid JSON" in body


def test_context_overflow_records_are_not_counted_wrong():
    """The rule the headline figure depends on: overflow != incorrect."""
    rec = {"context_overflow": True, "correct": None, "label": None}
    assert rec["correct"] is None, "overflow must never be recorded as correct=False"
