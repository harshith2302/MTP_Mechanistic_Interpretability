"""The 12-type question bank.

Every sampler returns both the question string and the exact gold answer,
computed from the ground truth in the story dict. A question whose gold answer
is undefined (tied argmax, non-unique transfer triple, ...) is never emitted —
the sampler resamples or returns None.

Question dict shape:
    {question_type, question_text, schema, gold, queried_person, queried_timestep,
     meta:{...}}
`schema` is the JSON skeleton shown to the model; `gold` is the parsed answer
we compare against.
"""

import argparse
import json
import random

from src.simulate import generate_story

SCALAR_TYPES = [
    "person_timestep_lookup",
    "person_value_timesteps",
    "population_condition",
    "duration_condition",
    "initial_state_lookup",
    "transfer_recall",
    "pairwise_comparison",
    "argmax_person",
    "total_conservation",
    "net_delta",
]
HEAVY_TYPES = ["trajectory", "state_snapshot"]
ALL_TYPES = SCALAR_TYPES + HEAVY_TYPES

CONDITIONS = [(">=", "at least"), ("<=", "at most"), (">", "more than"), ("<", "fewer than")]

_MAX_TRIES = 60


def _satisfies(count, op, threshold):
    return {">=": count >= threshold, "<=": count <= threshold,
            ">": count > threshold, "<": count < threshold}[op]


def _intro_position(story, pid):
    return story["person_ids"].index(pid)


def _last_mention_chars(story, pid):
    """Characters from this person's final mention to the end of the prompt.

    A cheap, tokenizer-free proxy recorded at sampling time; run_eval.py
    overwrites `last_mention_distance_tokens` with the real token count.
    """
    name = story["id_to_name"][pid]
    idx = story["text"].rfind(name)
    return len(story["text"]) - idx if idx >= 0 else -1


def _base(story, qtype, text, schema, gold, person=None, timestep=None, meta=None):
    q = {
        "question_type": qtype,
        "question_text": text,
        "schema": schema,
        "gold": gold,
        "queried_person": person,
        "queried_timestep": timestep,
        "meta": meta or {},
    }
    if person is not None:
        q["meta"]["person_intro_position"] = _intro_position(story, person)
        q["meta"]["last_mention_distance_chars"] = _last_mention_chars(story, person)
    return q


# --- 1. person_timestep_lookup ------------------------------------------------
def q_person_timestep_lookup(story, rng):
    pid = rng.choice(story["person_ids"])
    t = rng.randint(0, story["config"]["n_timesteps"])
    name = story["id_to_name"][pid]
    return _base(
        story, "person_timestep_lookup",
        f"How many pencils did {name} have at timestep {t}?",
        {"question_type": "person_timestep_lookup", "person": "<PERSON_NAME>",
         "timestep": "<INTEGER>", "answer": "<INTEGER>"},
        {"answer": story["states"][t][pid]},
        person=pid, timestep=t,
    )


# --- 2. person_value_timesteps ------------------------------------------------
def q_person_value_timesteps(story, rng):
    """Counts ALL timesteps including t=0 (inherited from verifier.py)."""
    T = story["config"]["n_timesteps"]
    for _ in range(_MAX_TRIES):
        pid = rng.choice(story["person_ids"])
        # bias toward values the person actually holds so the answer is rarely empty
        value = rng.choice([story["states"][t][pid] for t in range(T + 1)])
        gold = [t for t in range(T + 1) if story["states"][t][pid] == value]
        if gold:
            name = story["id_to_name"][pid]
            return _base(
                story, "person_value_timesteps",
                f"At which timesteps did {name} have exactly {value} "
                f"{'pencil' if value == 1 else 'pencils'}? Consider every timestep "
                f"from 0 to {T} (including timestep 0). Return all matching "
                f"timesteps in ascending order.",
                {"question_type": "person_value_timesteps", "person": "<PERSON_NAME>",
                 "pencil_count": "<INTEGER>", "timesteps": ["<INTEGER>", "..."]},
                {"timesteps": gold, "pencil_count": value},
                person=pid,
            )
    return None


# --- 3. population_condition --------------------------------------------------
def q_population_condition(story, rng):
    T = story["config"]["n_timesteps"]
    t = rng.randint(0, T)
    op, phrase = rng.choice(CONDITIONS)
    counts = list(story["states"][t].values())
    threshold = rng.randint(1, max(2, max(counts)))
    gold = sum(_satisfies(c, op, threshold) for c in counts)
    return _base(
        story, "population_condition",
        f"At timestep {t}, how many people had {phrase} {threshold} pencils?",
        {"question_type": "population_condition", "timestep": "<INTEGER>",
         "condition": f"{op}{threshold}", "answer": "<INTEGER>"},
        {"answer": gold, "condition": f"{op}{threshold}"},
        timestep=t, meta={"operator": op, "threshold": threshold},
    )


# --- 4. duration_condition ----------------------------------------------------
def q_duration_condition(story, rng):
    """Counts t>=1 only (inherited from verifier.py). Prompt says so."""
    T = story["config"]["n_timesteps"]
    pid = rng.choice(story["person_ids"])
    op, phrase = rng.choice(CONDITIONS)
    vals = [story["states"][t][pid] for t in range(1, T + 1)]
    threshold = rng.randint(1, max(2, max(vals)))
    gold = sum(_satisfies(v, op, threshold) for v in vals)
    name = story["id_to_name"][pid]
    return _base(
        story, "duration_condition",
        f"During how many timesteps did {name} have {phrase} {threshold} pencils? "
        f"Count only timesteps 1 through {T}; do not count timestep 0.",
        {"question_type": "duration_condition", "person": "<PERSON_NAME>",
         "condition": f"{op}{threshold}", "answer": "<INTEGER>"},
        {"answer": gold, "condition": f"{op}{threshold}"},
        person=pid, meta={"operator": op, "threshold": threshold},
    )


# --- 5. trajectory ------------------------------------------------------------
def q_trajectory(story, rng):
    T = story["config"]["n_timesteps"]
    pid = rng.choice(story["person_ids"])
    name = story["id_to_name"][pid]
    return _base(
        story, "trajectory",
        f"List how many pencils {name} had at every timestep from 0 to {T}, "
        f"in order. The list must have exactly {T + 1} numbers.",
        {"question_type": "trajectory", "person": "<PERSON_NAME>",
         "counts": ["<INTEGER>", "..."]},
        {"counts": [story["states"][t][pid] for t in range(T + 1)]},
        person=pid, timestep=T,
    )


# --- 6. state_snapshot --------------------------------------------------------
def q_state_snapshot(story, rng):
    T = story["config"]["n_timesteps"]
    t = rng.randint(0, T)
    counts = {story["id_to_name"][p]: story["states"][t][p] for p in story["person_ids"]}
    return _base(
        story, "state_snapshot",
        f"At timestep {t}, state how many pencils each person had. "
        f"Include every one of the {story['config']['n_people']} people.",
        {"question_type": "state_snapshot", "timestep": "<INTEGER>",
         "counts": {"<PERSON_NAME>": "<INTEGER>"}},
        {"counts": counts},
        timestep=t,
    )


# --- 7. initial_state_lookup --------------------------------------------------
def q_initial_state_lookup(story, rng):
    pid = rng.choice(story["person_ids"])
    name = story["id_to_name"][pid]
    return _base(
        story, "initial_state_lookup",
        f"How many pencils did {name} hold at timestep 0?",
        {"question_type": "initial_state_lookup", "person": "<PERSON_NAME>",
         "answer": "<INTEGER>"},
        {"answer": story["states"][0][pid]},
        person=pid, timestep=0,
    )


# --- 8. transfer_recall -------------------------------------------------------
def q_transfer_recall(story, rng):
    """Only (giver, receiver, timestep) triples that occur exactly once."""
    seen = {}
    for ev in story["narration"]:
        key = (ev["timestep"], ev["giver"], ev["receiver"])
        seen.setdefault(key, []).append(ev["amount"])
    unique = [(k, v[0]) for k, v in seen.items() if len(v) == 1]
    if not unique:
        return None
    (t, giver, receiver), amount = rng.choice(unique)
    gname, rname = story["id_to_name"][giver], story["id_to_name"][receiver]
    return _base(
        story, "transfer_recall",
        f"At timestep {t}, how many pencils did {gname} give to {rname}?",
        {"question_type": "transfer_recall", "giver": "<PERSON_NAME>",
         "receiver": "<PERSON_NAME>", "timestep": "<INTEGER>", "answer": "<INTEGER>"},
        {"answer": amount},
        person=giver, timestep=t,
        meta={"receiver": receiver, "giver": giver},
    )


# --- 9. pairwise_comparison ---------------------------------------------------
def q_pairwise_comparison(story, rng):
    if story["config"]["n_people"] < 2:
        return None
    T = story["config"]["n_timesteps"]
    t = rng.randint(0, T)
    p, q = rng.sample(story["person_ids"], 2)
    cp, cq = story["states"][t][p], story["states"][t][q]
    if cp > cq:
        ans = story["id_to_name"][p]
    elif cq > cp:
        ans = story["id_to_name"][q]
    else:
        ans = "tie"
    pname, qname = story["id_to_name"][p], story["id_to_name"][q]
    return _base(
        story, "pairwise_comparison",
        f"At timestep {t}, who had more pencils, {pname} or {qname}? "
        f'Answer with the person\'s name, or "tie" if they had the same number.',
        {"question_type": "pairwise_comparison", "timestep": "<INTEGER>",
         "answer": "<PERSON_NAME or tie>"},
        {"answer": ans},
        person=p, timestep=t,
        meta={"other_person": q, "gap": abs(cp - cq),
              "candidates": [pname, qname]},
    )


# --- 10. argmax_person --------------------------------------------------------
def q_argmax_person(story, rng):
    T = story["config"]["n_timesteps"]
    candidates = []
    for t in range(T + 1):
        vals = story["states"][t]
        top = max(vals.values())
        winners = [p for p, v in vals.items() if v == top]
        if len(winners) == 1:
            candidates.append((t, winners[0], top))
    if not candidates:
        return None
    t, pid, top = rng.choice(candidates)
    second = sorted(story["states"][t].values(), reverse=True)[1] if len(story["states"][t]) > 1 else 0
    return _base(
        story, "argmax_person",
        f"At timestep {t}, who held the most pencils?",
        {"question_type": "argmax_person", "timestep": "<INTEGER>",
         "answer": "<PERSON_NAME>"},
        {"answer": story["id_to_name"][pid]},
        person=pid, timestep=t,
        meta={"gap": top - second, "top_count": top},
    )


# --- 11. total_conservation ---------------------------------------------------
def q_total_conservation(story, rng):
    T = story["config"]["n_timesteps"]
    t = rng.randint(0, T)
    return _base(
        story, "total_conservation",
        f"At timestep {t}, what is the total number of pencils held by "
        f"everyone together?",
        {"question_type": "total_conservation", "timestep": "<INTEGER>",
         "answer": "<INTEGER>"},
        {"answer": story["config"]["total_pencils"]},
        timestep=t,
    )


# --- 12. net_delta ------------------------------------------------------------
def q_net_delta(story, rng):
    T = story["config"]["n_timesteps"]
    if T < 1:
        return None
    for _ in range(_MAX_TRIES):
        t1, t2 = sorted(rng.sample(range(T + 1), 2))
        pid = rng.choice(story["person_ids"])
        gold = story["states"][t2][pid] - story["states"][t1][pid]
        name = story["id_to_name"][pid]
        return _base(
            story, "net_delta",
            f"Between timestep {t1} and timestep {t2}, what was {name}'s net "
            f"change in pencils? Use a negative number for a loss.",
            {"question_type": "net_delta", "person": "<PERSON_NAME>",
             "from_timestep": "<INTEGER>", "to_timestep": "<INTEGER>",
             "answer": "<SIGNED INTEGER>"},
            {"answer": gold},
            person=pid, timestep=t2,
            meta={"from_timestep": t1, "to_timestep": t2},
        )
    return None


SAMPLERS = {
    "person_timestep_lookup": q_person_timestep_lookup,
    "person_value_timesteps": q_person_value_timesteps,
    "population_condition": q_population_condition,
    "duration_condition": q_duration_condition,
    "trajectory": q_trajectory,
    "state_snapshot": q_state_snapshot,
    "initial_state_lookup": q_initial_state_lookup,
    "transfer_recall": q_transfer_recall,
    "pairwise_comparison": q_pairwise_comparison,
    "argmax_person": q_argmax_person,
    "total_conservation": q_total_conservation,
    "net_delta": q_net_delta,
}


def sample_questions(story, seed, scalar_per_type=2, heavy_per_type=1, types=None):
    """Sample the full question set for one story. Deterministic in `seed`."""
    rng = random.Random(seed)
    out = []
    for qtype in (types or ALL_TYPES):
        k = heavy_per_type if qtype in HEAVY_TYPES else scalar_per_type
        made = 0
        for attempt in range(k * 8):
            if made >= k:
                break
            q = SAMPLERS[qtype](story, rng)
            if q is None:
                continue
            # avoid emitting the identical question twice for one story
            if any(o["question_text"] == q["question_text"] for o in out):
                continue
            q["index_within_type"] = made
            out.append(q)
            made += 1
    return out


def make_question_id(model, n, story_index, qtype, k):
    return f"{model}|N{n}|s{story_index}|{qtype}|{k}"


def main():
    ap = argparse.ArgumentParser(description="Sample and inspect questions.")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--t", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true",
                    help="print each question, its schema and its gold answer")
    ap.add_argument("--type", help="restrict to one question type")
    args = ap.parse_args()

    story = generate_story(args.n, args.t if args.t is not None else args.n, args.seed)
    qs = sample_questions(story, args.seed,
                          types=[args.type] if args.type else None)
    if args.dry_run:
        print(f"### story N={args.n} T={story['config']['n_timesteps']} "
              f"seed={args.seed} total={story['config']['total_pencils']}")
        print(f"### text ({len(story['text'])} chars)\n{story['text'][:400]}...\n")
    for q in qs:
        print(f"--- {q['question_type']}[{q['index_within_type']}]")
        print(f"Q: {q['question_text']}")
        if args.dry_run:
            print(f"schema: {json.dumps(q['schema'])}")
        print(f"gold: {json.dumps(q['gold'])}")
        if q["meta"]:
            print(f"meta: {json.dumps(q['meta'])}")
        print()
    print(f"total questions: {len(qs)}")


if __name__ == "__main__":
    main()
