"""Pencil Exchange, simplified: no clusters, chronological narration.

The original generator (pencil_taxonomy/src/generator.py) split the people into
clusters and then narrated the clusters' timesteps in an interleaved order, so a
"timestep 1" paragraph could appear after a "timestep 7" paragraph. The task
therefore measured state tracking PLUS narrative reordering. This version removes
the clusters, which removes the reordering: every timestep is narrated in order,
1, 2, 3, ..., and within a timestep the transfers appear in the order they
happened.

Everything else is deliberately unchanged:

  * same name bag, same random initial partition, same transfer sampling
    (a random giver with pencils, a random other receiver, a random amount up to
    the giver's holding), same sentence templates, same number-word mixing;
  * the simulation loop makes its random calls in the SAME ORDER as the original
    with a single cluster, so for N < 4 -- where the original also had one
    cluster -- the two generators produce identical initial states, identical
    transfers and identical state matrices from the same seed. Only the text
    differs. tests/test_simulation.py pins this.

The transfer count per timestep is configurable:
  transfers_per_timestep = "random"  -> randint(1, N), as the original did
  transfers_per_timestep = k (int)   -> exactly k, so T counts transactions
"""

import argparse
import json
import os
import random

from num2words import num2words

DEFAULT_NAMES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "names.txt")

TRANSFER_SENTENCE_TEMPLATES = [
    "{A} gave {n} {pencils} to {B}.",
    "{A} handed {n} {pencils} over to {B}.",
    "{A} transferred {n} {pencils} to {B}.",
    "{A} donated {n} {pencils} to {B}.",
    "{A} passed {n} {pencils} to {B}.",
    "{A} sent {n} {pencils} to {B}.",
    "{A} dropped {n} {pencils} into {B}'s bag.",
    "{A} let {B} have {n} {pencils}.",
]


def load_name_bag(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _random_integer_partition(rng, total_value, parts):
    if total_value < parts:
        raise ValueError(f"Cannot distribute {total_value} pencils among "
                         f"{parts} people: each needs at least 1.")
    cuts = sorted(rng.sample(range(1, total_value), parts - 1))
    out, prev = [], 0
    for c in cuts:
        out.append(c - prev)
        prev = c
    out.append(total_value - prev)
    return out


def _pencil_word(count):
    return "pencil" if count == 1 else "pencils"


def generate_story(n_people, n_timesteps, seed, names_path=DEFAULT_NAMES_PATH,
                   pencils_per_person=5, number_word_probability=0.45,
                   transfers_per_timestep="random"):
    """One story plus its exact ground truth. Same dict shape as the original."""
    if n_people < 2:
        raise ValueError("need at least 2 people for a transfer to exist")
    rng = random.Random(seed)
    total = n_people * pencils_per_person

    def number_token(value):
        return num2words(value) if rng.random() < number_word_probability else str(value)

    name_bag = load_name_bag(names_path)
    if n_people > len(name_bag):
        raise ValueError(f"name bag has {len(name_bag)} names, need {n_people}")
    names = rng.sample(name_bag, n_people)
    ids = [f"A{i}" for i in range(1, n_people + 1)]
    id_to_name = dict(zip(ids, names))
    name_to_id = {v: k for k, v in id_to_name.items()}

    # --- simulation: identical call order to the original with one cluster ---
    current = dict(zip(ids, _random_integer_partition(rng, total, n_people)))
    states = {0: dict(current)}
    events = {t: [] for t in range(n_timesteps + 1)}
    for t in range(1, n_timesteps + 1):
        local = dict(current)
        k = (rng.randint(1, n_people) if transfers_per_timestep == "random"
             else int(transfers_per_timestep))
        for _ in range(k):
            givers = [p for p in ids if local[p] > 0]
            if not givers:
                break
            giver = rng.choice(givers)
            receivers = [p for p in ids if p != giver]
            receiver = rng.choice(receivers)
            amount = rng.randint(1, local[giver])
            local[giver] -= amount
            local[receiver] += amount
            events[t].append((giver, receiver, amount))
        current = local
        states[t] = dict(current)

    for t in range(n_timesteps + 1):
        assert sum(states[t].values()) == total, f"conservation broken at t={t}"

    # --- narration: strictly chronological ---
    narration, paragraphs = [], []
    for t in range(1, n_timesteps + 1):
        if not events[t]:
            continue
        sentences = []
        for giver, receiver, amount in events[t]:
            tpl = rng.choice(TRANSFER_SENTENCE_TEMPLATES)
            sentences.append(tpl.format(A=id_to_name[giver], B=id_to_name[receiver],
                                        n=number_token(amount),
                                        pencils=_pencil_word(amount)))
            narration.append({"timestep": t, "giver": giver, "receiver": receiver,
                              "amount": amount, "narration_index": len(narration)})
        paragraphs.append(f"At timestep {t}, " + " ".join(sentences))

    names_text = ", ".join(id_to_name[p] for p in ids[:-1]) + " and " + id_to_name[ids[-1]]
    holdings = [f"{id_to_name[p]} holds {number_token(states[0][p])} "
                f"{_pencil_word(states[0][p])}" for p in ids]
    intro = (f"There are {number_token(n_people)} people participating in the "
             f"pencil exchange: {names_text}. At timestep 0, their initial pencil "
             f"holdings are as follows: {', '.join(holdings[:-1])} and {holdings[-1]}.")
    text = (intro + " " + " ".join(paragraphs)).replace("\n", " ").strip()

    return {
        "config": {"n_people": n_people, "n_timesteps": n_timesteps, "seed": seed,
                   "total_pencils": total, "pencils_per_person": pencils_per_person,
                   "number_word_probability": number_word_probability,
                   "transfers_per_timestep": transfers_per_timestep,
                   "chronological": True},
        "person_ids": ids,
        "id_to_name": id_to_name,
        "name_to_id": name_to_id,
        # One cluster containing everyone. Kept so downstream code that asks
        # "who can this person trade with?" gets the right answer: everyone.
        "clusters": {"cluster_1": list(ids)},
        "states": states,
        "transfers": {t: list(events[t]) for t in range(n_timesteps + 1)},
        "narration": narration,
        "text": text,
    }


def main():
    ap = argparse.ArgumentParser(description="Generate one chronological story.")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--t", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--transfers-per-timestep", default="random")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    k = a.transfers_per_timestep
    s = generate_story(a.n, a.t, a.seed,
                       transfers_per_timestep=k if k == "random" else int(k))
    if a.json:
        print(json.dumps(s, indent=2, default=str))
    else:
        print(s["text"])
        print(f"\n{len(s['narration'])} transfers over {a.t} timesteps; "
              f"total={s['config']['total_pencils']}")


if __name__ == "__main__":
    main()
