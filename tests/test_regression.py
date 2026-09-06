"""The gate test: src/simulate.py must reproduce the original generator exactly.

Fixtures under tests/fixtures/ were produced by running the *unmodified*
Pencil_Exchange/simulation.py with data/names.txt and the constants named in
each filename. If this test fails, the refactor changed the random call order —
fix the refactor, never the fixtures.
"""

import json
import os

import pytest

from src.simulate import generate_story

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
NAMES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "names.txt"
)

CASES = [
    ("golden_n2_t2_seed2000", 2, 2, 2000),
    ("golden_n4_t6_seed7", 4, 6, 7),
    ("golden_n10_t10_seed42", 10, 10, 42),
    ("golden_n25_t25_seed25003", 25, 25, 25003),
]


def _load(stem):
    with open(os.path.join(FIXTURES, stem + ".json")) as f:
        gt = json.load(f)
    with open(os.path.join(FIXTURES, stem + ".txt")) as f:
        text = f.read()
    return gt, text


@pytest.mark.parametrize("stem,n,t,seed", CASES)
def test_matches_original_generator(stem, n, t, seed):
    gt, text = _load(stem)
    story = generate_story(n, t, seed, names_path=NAMES)

    assert story["text"] == text, "story text diverged from the original generator"

    expected_states = {
        int(k.split("=")[1]): v for k, v in gt["ground_truth_pencil_counts"].items()
    }
    assert story["states"] == expected_states

    assert story["person_ids"] == gt["persons"]
    assert story["id_to_name"] == gt["id_to_name"]
    assert story["name_to_id"] == gt["name_to_id"]
    assert story["clusters"] == gt["clusters"]
    assert story["config"]["total_pencils"] == gt["config"]["total_pencils"]


@pytest.mark.parametrize("stem,n,t,seed", CASES)
def test_conservation(stem, n, t, seed):
    story = generate_story(n, t, seed, names_path=NAMES)
    total = story["config"]["total_pencils"]
    for state in story["states"].values():
        assert sum(state.values()) == total


@pytest.mark.parametrize("stem,n,t,seed", CASES)
def test_transfers_replay_to_states(stem, n, t, seed):
    """Applying transfers[t] to states[t-1] must land exactly on states[t]."""
    story = generate_story(n, t, seed, names_path=NAMES)
    for step in range(1, t + 1):
        cur = dict(story["states"][step - 1])
        for giver, receiver, amount in story["transfers"][step]:
            cur[giver] -= amount
            cur[receiver] += amount
        assert cur == story["states"][step], f"replay mismatch at t={step}"


@pytest.mark.parametrize("stem,n,t,seed", CASES)
def test_narration_order_matches_text(stem, n, t, seed):
    """narration[] must be the order the reader encounters transfers in text."""
    story = generate_story(n, t, seed, names_path=NAMES)
    text = story["text"]
    # each narrated transfer's giver name must appear at a non-decreasing
    # position when we walk the text left to right consuming sentences.
    cursor = text.index("At timestep") if "At timestep" in text else 0
    for ev in story["narration"]:
        giver = story["id_to_name"][ev["giver"]]
        receiver = story["id_to_name"][ev["receiver"]]
        window = text[cursor:]
        gi, ri = window.find(giver), window.find(receiver)
        assert gi >= 0 and ri >= 0, f"{giver}/{receiver} not found after cursor"
        cursor += min(gi, ri)


def test_determinism():
    a = generate_story(12, 12, 999, names_path=NAMES)
    b = generate_story(12, 12, 999, names_path=NAMES)
    assert a["text"] == b["text"]
    assert a["states"] == b["states"]


def test_seeds_differ():
    a = generate_story(8, 8, 1, names_path=NAMES)
    b = generate_story(8, 8, 2, names_path=NAMES)
    assert a["text"] != b["text"]


@pytest.mark.parametrize("n", [2, 3, 4, 5, 30])
def test_small_and_large_n_do_not_crash(n):
    story = generate_story(n, n, 1000 * n, names_path=NAMES)
    assert len(story["person_ids"]) == n
    assert sum(story["states"][n].values()) == n * 5


def test_number_word_probability_zero_has_no_words():
    story = generate_story(10, 10, 42, names_path=NAMES, number_word_probability=0.0)
    for word in ("one ", "two ", "three ", "eleven ", "twenty"):
        assert word not in story["text"].lower().replace("timestep", "")
