"""What the simplified generator must guarantee.

The load-bearing test is the second one: for N < 4 the ORIGINAL generator also
had a single cluster, so from the same seed the two must agree on initial
holdings, every transfer, and the whole state matrix. That is what lets us say
the simplified task has the same physics and differs only in narration order.
"""
import os
import re
import sys

import pytest

from src.simulation import generate_story

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = os.path.join(os.path.dirname(HERE), "data", "names.txt")
# The original, verbatim copy that pencil_taxonomy ran on.
_ORIG_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                         "pencil_taxonomy", "src")


def _original():
    if _ORIG_DIR not in sys.path:
        sys.path.insert(0, _ORIG_DIR)
    import importlib
    return importlib.import_module("generator").generate_story


def _timestep_order(text):
    return [int(t) for t in re.findall(r"At timestep (\d+),", text)]


@pytest.mark.parametrize("n,t,seed", [(2, 1, 1), (2, 5, 9), (5, 5, 3),
                                      (12, 12, 44), (30, 30, 700030)])
def test_narration_is_strictly_chronological(n, t, seed):
    s = generate_story(n, t, seed, names_path=NAMES)
    order = _timestep_order(s["text"])
    assert order == sorted(order), order
    assert len(order) == len(set(order)), "a timestep is narrated twice"
    # narration list agrees with the text
    assert [e["timestep"] for e in s["narration"]] == sorted(
        e["timestep"] for e in s["narration"])


@pytest.mark.skipif(not os.path.isdir(_ORIG_DIR),
                    reason="original generator not alongside")
@pytest.mark.parametrize("n,t,seed", [(2, 1, 1), (2, 6, 17), (3, 3, 5),
                                      (3, 10, 700003), (3, 30, 8)])
def test_same_physics_as_the_original_for_single_cluster_sizes(n, t, seed):
    """N < 4 => the original also had exactly one cluster. Same seed must give
    the same holdings, the same transfers and the same state matrix."""
    orig = _original()(n, t, seed, names_path=NAMES)
    new = generate_story(n, t, seed, names_path=NAMES)
    assert new["states"] == orig["states"]
    assert new["person_ids"] == orig["person_ids"]
    assert new["id_to_name"] == orig["id_to_name"]
    for step in range(t + 1):
        assert new["transfers"][step] == orig["transfers"][step], step


@pytest.mark.skipif(not os.path.isdir(_ORIG_DIR),
                    reason="original generator not alongside")
def test_the_original_was_not_chronological_at_larger_n():
    """Sanity: the property we removed really was there to remove."""
    orig = _original()(25, 25, 25003, names_path=NAMES)
    order = _timestep_order(orig["text"])
    assert order != sorted(order), "original narrates in order here?"
    new = generate_story(25, 25, 25003, names_path=NAMES)
    assert _timestep_order(new["text"]) == sorted(_timestep_order(new["text"]))


@pytest.mark.parametrize("n,t", [(2, 1), (2, 30), (30, 1), (30, 30), (7, 13)])
def test_conservation_and_replay(n, t):
    s = generate_story(n, t, 1000 * n + t, names_path=NAMES)
    total = s["config"]["total_pencils"]
    cur = dict(s["states"][0])
    for step in range(1, t + 1):
        for g, r, a in s["transfers"][step]:
            assert cur[g] >= a, "gave more than held"
            cur[g] -= a
            cur[r] += a
        assert cur == s["states"][step]
        assert sum(cur.values()) == total
        assert min(cur.values()) >= 0


def test_every_timestep_has_at_least_one_transfer_in_random_mode():
    s = generate_story(4, 20, 3, names_path=NAMES)
    assert all(len(s["transfers"][t]) >= 1 for t in range(1, 21))


def test_fixed_transfers_per_timestep():
    s = generate_story(6, 10, 3, names_path=NAMES, transfers_per_timestep=1)
    assert all(len(s["transfers"][t]) == 1 for t in range(1, 11))
    assert len(s["narration"]) == 10
    s3 = generate_story(6, 10, 3, names_path=NAMES, transfers_per_timestep=3)
    assert all(len(s3["transfers"][t]) == 3 for t in range(1, 11))


def test_clusters_key_is_one_cluster_of_everyone():
    s = generate_story(9, 3, 1, names_path=NAMES)
    assert list(s["clusters"].values()) == [s["person_ids"]]


def test_deterministic_and_seed_sensitive():
    a = generate_story(8, 8, 5, names_path=NAMES)
    b = generate_story(8, 8, 5, names_path=NAMES)
    c = generate_story(8, 8, 6, names_path=NAMES)
    assert a["text"] == b["text"] and a["states"] == b["states"]
    assert c["text"] != a["text"]


def test_number_word_probability_zero_uses_only_digits():
    s = generate_story(8, 8, 5, names_path=NAMES, number_word_probability=0.0)
    body = s["text"].lower()
    for w in (" one ", " two ", " three ", " ten ", "eleven", "twenty"):
        assert w not in body.replace("timestep", "")
