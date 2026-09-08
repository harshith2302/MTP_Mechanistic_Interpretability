"""The five question types and their ground truths.

Everything here is derived from the generator's output -- the state matrix `M`
and the ordered `transfers` list. The generator itself is never modified.

The five types form a ladder in how much state accumulation the answer needs:

    1 initial_state_lookup   nothing -- copy one number stated in the text
    2 transfer_recall        retrieve one narrated event
    3 person_timestep_lookup one accumulated value
    4 state_snapshot         the whole state vector
    5 trajectory             one entity's whole history

Types 1 and 2 are CONTROLS. If they hold up while 3-5 collapse, then long
context and person-to-value binding are not the bottleneck and the failure sits
in the state update. That contrast is the experiment.

Two sampling constraints are load-bearing rather than cosmetic:

  * `person_timestep_lookup` requires M[t][P] != M[0][P]. Without it the correct
    answer is sometimes the initial value, and `stale` (the model never applied
    an update) becomes indistinguishable from a correct answer -- the taxonomy's
    top-priority category would be undecidable.
  * `trajectory` requires a non-constant column, for the same reason.

If a story cannot satisfy a constraint (only possible at very small N) the
question is recorded as skipped with a reason. It is never silently replaced.
"""

import random


def _matrix(story):
    """(T+1) x N state matrix as {t: {person_id: count}}, plus ordered ids."""
    return story["states"], story["person_ids"]


def _cluster_of(story, pid):
    for members in story["clusters"].values():
        if pid in members:
            return members
    return [pid]


def _answer_space_size(story, pid):
    """Plausible answer set for a scalar count question about `pid`.

    Pencils only move inside a cluster, so the largest count `pid` can ever hold
    is its cluster's total; the answer space is 0..that, inclusive. This is the
    range the chance null draws from, so it must be the real bound rather than a
    generous one -- too wide a range makes every category look non-coincidental.
    """
    members = _cluster_of(story, pid)
    return sum(story["states"][0][p] for p in members) + 1


def _transfers_list(story):
    """Flat (t, giver, receiver, amount) list in narration order."""
    return [(e["timestep"], e["giver"], e["receiver"], e["amount"])
            for e in story["narration"]]


# --- the five types ----------------------------------------------------------

def q_initial_state_lookup(story, rng):
    M, ids = _matrix(story)
    pid = rng.choice(ids)
    return {
        "question_type": "initial_state_lookup",
        "target_person": story["id_to_name"][pid],
        "target_person_id": pid,
        "target_timestep": 0,
        "ground_truth": M[0][pid],
        "answer_space_size": _answer_space_size(story, pid),
        "answer_dim": None,
    }


def q_transfer_recall(story, rng):
    tr = _transfers_list(story)
    if not tr:
        return {"question_skipped": "story has no transfers"}
    t, giver, receiver, amount = rng.choice(tr)
    return {
        "question_type": "transfer_recall",
        "target_person": story["id_to_name"][giver],
        "target_person_id": giver,
        "target_receiver": story["id_to_name"][receiver],
        "target_timestep": t,
        "ground_truth": amount,
        "answer_space_size": _answer_space_size(story, giver),
        "answer_dim": None,
    }


def q_person_timestep_lookup(story, rng, max_tries=200):
    M, ids = _matrix(story)
    T = story["config"]["n_timesteps"]
    cands = [(p, t) for p in ids for t in range(1, T + 1) if M[t][p] != M[0][p]]
    if not cands:
        return {"question_skipped": "no (P,t) with M[t][P] != M[0][P]"}
    pid, t = rng.choice(cands)
    return {
        "question_type": "person_timestep_lookup",
        "target_person": story["id_to_name"][pid],
        "target_person_id": pid,
        "target_timestep": t,
        "ground_truth": M[t][pid],
        "answer_space_size": _answer_space_size(story, pid),
        "answer_dim": None,
    }


def q_state_snapshot(story, rng):
    M, ids = _matrix(story)
    T = story["config"]["n_timesteps"]
    t = rng.randint(1, T)
    return {
        "question_type": "state_snapshot",
        "target_person": None,
        "target_person_id": None,
        "target_timestep": t,
        "ground_truth": {story["id_to_name"][p]: M[t][p] for p in ids},
        "answer_space_size": None,     # combinatorial -- never a single integer
        "answer_dim": len(ids),
    }


def q_trajectory(story, rng):
    M, ids = _matrix(story)
    T = story["config"]["n_timesteps"]
    varying = [p for p in ids if len({M[t][p] for t in range(T + 1)}) > 1]
    if not varying:
        return {"question_skipped": "every column is constant"}
    pid = rng.choice(varying)
    return {
        "question_type": "trajectory",
        "target_person": story["id_to_name"][pid],
        "target_person_id": pid,
        "target_timestep": None,
        "ground_truth": [M[t][pid] for t in range(T + 1)],
        "answer_space_size": None,
        "answer_dim": T + 1,
    }


BUILDERS = {
    "initial_state_lookup": q_initial_state_lookup,
    "transfer_recall": q_transfer_recall,
    "person_timestep_lookup": q_person_timestep_lookup,
    "state_snapshot": q_state_snapshot,
    "trajectory": q_trajectory,
}


def questions_for_story(story, story_id, types):
    """One question of each requested type. Deterministic given the story seed."""
    rng = random.Random(story["config"]["seed"] ^ 0x51EED)
    out = []
    for qt in types:
        q = BUILDERS[qt](story, rng)
        q.setdefault("question_type", qt)
        q["question_id"] = f"{story_id}_{qt}"
        q["story_id"] = story_id
        q["story_seed"] = story["config"]["seed"]
        q["N"] = story["config"]["n_people"]
        q["T"] = story["config"]["n_timesteps"]
        out.append(q)
    return out
