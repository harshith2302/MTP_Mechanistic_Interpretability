# ---------------------------------------------------------------------------
# COPIED VERBATIM -- do not edit. Any change here breaks comparability with the
# experiment this generator came from.
#
#   source : phase1_full_eval/src/simulate.py (same repo)
#   mtime  : 2026-09-06 13:16:02 +0530
#   copied : 2026-09-08
#   sha256 : 222dd6bf1b3dc4b7 (of the source file)
#
# ZERO edits were made. DEFAULT_NAMES_PATH is written relative to this file
# (src/../data/names.txt), so it resolves inside this repo unchanged; data/names.txt
# was copied alongside it. The RNG call order -- which the source repo's
# regression tests pin byte-for-byte -- is therefore untouched by construction.
# ---------------------------------------------------------------------------
"""Pencil Exchange story generator.

Refactor of ``Pencil_Exchange/simulation.py`` into a pure function. The random
call order is preserved *exactly* — every ``rng`` call happens at the same point
in the same sequence as the original module-level script, so
``generate_story(10, 10, 42)`` reproduces the original byte for byte.
See ``tests/test_regression.py``; do not reorder anything in here.
"""

import argparse
import json
import os
import random

from num2words import num2words

DEFAULT_NAMES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "names.txt"
)

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


def _sample_display_names(rng, name_bag, num_people):
    if num_people > len(name_bag):
        raise ValueError(
            f"Name bag has only {len(name_bag)} names, but {num_people} were requested."
        )
    return rng.sample(name_bag, num_people)


def _create_person_clusters(rng, ids):
    n = len(ids)
    if n < 4:
        return [ids[:]]
    max_possible_splits = n // 2 - 1
    num_splits = rng.randint(1, max(1, max_possible_splits))
    split_positions = sorted(rng.sample(range(2, n - 1), min(num_splits, n - 3)))
    clusters, prev = [], 0
    for s in split_positions:
        clusters.append(ids[prev:s])
        prev = s
    clusters.append(ids[prev:])
    return clusters


def _random_integer_partition(rng, total_value, parts):
    if total_value < parts:
        raise ValueError(
            f"Cannot distribute {total_value} pencils among {parts} people: "
            f"each person needs at least 1."
        )
    cut_positions = sorted(rng.sample(range(1, total_value), parts - 1))
    partitions, prev = [], 0
    for cut in cut_positions:
        partitions.append(cut - prev)
        prev = cut
    partitions.append(total_value - prev)
    return partitions


def _pencil_word(count):
    return "pencil" if count == 1 else "pencils"


def generate_story(
    n_people: int,
    n_timesteps: int,
    seed: int,
    names_path: str = DEFAULT_NAMES_PATH,
    pencils_per_person: int = 5,
    number_word_probability: float = 0.45,
) -> dict:
    """Generate one Pencil Exchange story plus its exact ground truth."""
    rng = random.Random(seed)
    total_pencils = n_people * pencils_per_person

    def format_number_token(value):
        return num2words(value) if rng.random() < number_word_probability else str(value)

    name_bag = load_name_bag(names_path)
    display_names = _sample_display_names(rng, name_bag, n_people)

    person_ids = [f"A{i}" for i in range(1, n_people + 1)]
    id_to_name = dict(zip(person_ids, display_names))
    name_to_id = {name: pid for pid, name in id_to_name.items()}

    person_clusters = _create_person_clusters(rng, person_ids)

    initial_distribution = _random_integer_partition(rng, total_pencils, n_people)
    current = {person_ids[i]: initial_distribution[i] for i in range(n_people)}

    # --- simulation -------------------------------------------------------
    states = {0: dict(current)}
    events_by_cluster = [[[] for _ in range(n_timesteps + 1)] for _ in person_clusters]

    for timestep in range(1, n_timesteps + 1):
        updated = dict(current)
        for cluster_index, cluster_members in enumerate(person_clusters):
            local = {p: updated[p] for p in cluster_members}
            num_transfers = rng.randint(1, len(cluster_members))
            for _ in range(num_transfers):
                possible_givers = [p for p in cluster_members if local[p] > 0]
                if not possible_givers:
                    break
                giver = rng.choice(possible_givers)
                possible_receivers = [p for p in cluster_members if p != giver]
                if not possible_receivers:
                    break
                receiver = rng.choice(possible_receivers)
                amount = rng.randint(1, local[giver])
                local[giver] -= amount
                local[receiver] += amount
                events_by_cluster[cluster_index][timestep].append(
                    (giver, receiver, amount)
                )
            for p in cluster_members:
                updated[p] = local[p]
        current = updated
        states[timestep] = dict(current)

    for t in range(n_timesteps + 1):
        total = sum(states[t].values())
        assert total == total_pencils, (
            f"Conservation broken at t={t}: found {total}, expected {total_pencils}."
        )

    # --- narration schedule ----------------------------------------------
    schedulable = [(ci, 1) for ci in range(len(person_clusters))]
    execution_schedule = []
    while schedulable:
        selected = rng.choice(schedulable)
        schedulable.remove(selected)
        execution_schedule.append(selected)
        cluster_index, timestep = selected
        if timestep + 1 <= n_timesteps:
            schedulable.append((cluster_index, timestep + 1))

    # --- text -------------------------------------------------------------
    # narration[] is the reading order of transfers; it is NOT sorted by t,
    # because execution_schedule interleaves clusters. classify.omission_k
    # depends on this being what the reader actually saw.
    narration = []
    timestep_paragraphs = []
    for cluster_index, timestep in execution_schedule:
        cluster_events = events_by_cluster[cluster_index][timestep]
        if not cluster_events:
            continue
        sentences = []
        for giver, receiver, amount in cluster_events:
            template = rng.choice(TRANSFER_SENTENCE_TEMPLATES)
            sentences.append(
                template.format(
                    A=id_to_name[giver],
                    B=id_to_name[receiver],
                    n=format_number_token(amount),
                    pencils=_pencil_word(amount),
                )
            )
            narration.append(
                {
                    "timestep": timestep,
                    "giver": giver,
                    "receiver": receiver,
                    "amount": amount,
                    "narration_index": len(narration),
                }
            )
        timestep_paragraphs.append(
            f"At timestep {timestep}, " + " ".join(sentences)
        )

    names_text = (
        ", ".join(id_to_name[p] for p in person_ids[:-1])
        + " and "
        + id_to_name[person_ids[-1]]
    )
    initial_state_sentences = [
        f"{id_to_name[p]} holds {format_number_token(states[0][p])} "
        f"{_pencil_word(states[0][p])}"
        for p in person_ids
    ]
    initial_state_text = (
        ", ".join(initial_state_sentences[:-1]) + " and " + initial_state_sentences[-1]
    )
    intro_text = (
        f"There are {format_number_token(n_people)} people participating "
        f"in the pencil exchange: {names_text}. "
        f"At timestep 0, their initial pencil holdings are as follows: "
        f"{initial_state_text}."
    )
    text = (intro_text + " " + " ".join(timestep_paragraphs)).replace("\n", " ")

    # transfers[t] in narration order for that t
    transfers = {t: [] for t in range(n_timesteps + 1)}
    for ev in narration:
        transfers[ev["timestep"]].append((ev["giver"], ev["receiver"], ev["amount"]))

    return {
        "config": {
            "n_people": n_people,
            "n_timesteps": n_timesteps,
            "seed": seed,
            "total_pencils": total_pencils,
            "pencils_per_person": pencils_per_person,
            "number_word_probability": number_word_probability,
        },
        "person_ids": person_ids,
        "id_to_name": id_to_name,
        "name_to_id": name_to_id,
        "clusters": {f"cluster_{i + 1}": c for i, c in enumerate(person_clusters)},
        "states": states,
        "transfers": transfers,
        "narration": narration,
        "text": text,
    }


def main():
    ap = argparse.ArgumentParser(description="Generate one Pencil Exchange story.")
    ap.add_argument("--n", type=int, default=10, help="number of people")
    ap.add_argument("--t", type=int, default=None, help="timesteps (default: = --n)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--names", default=DEFAULT_NAMES_PATH)
    ap.add_argument("--pencils-per-person", type=int, default=5)
    ap.add_argument("--number-word-probability", type=float, default=0.45)
    ap.add_argument("--print", action="store_true", help="print the story text")
    ap.add_argument("--json", action="store_true", help="dump the full dict as JSON")
    args = ap.parse_args()

    story = generate_story(
        args.n,
        args.t if args.t is not None else args.n,
        args.seed,
        names_path=args.names,
        pencils_per_person=args.pencils_per_person,
        number_word_probability=args.number_word_probability,
    )
    if args.json:
        print(json.dumps(story, indent=2, default=str))
    else:
        if args.print:
            print(story["text"])
            print()
        print(f"people={args.n} timesteps={story['config']['n_timesteps']} "
              f"total={story['config']['total_pencils']} "
              f"transfers={len(story['narration'])}")


if __name__ == "__main__":
    main()
