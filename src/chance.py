"""Chance-collision baseline.

At small N a uniformly random wrong integer coincides with another person's
count, or with an omission replay, very often -- so an uncorrected taxonomy rate
is meaningless. Every published rate must carry observed - chance beside it.

The null must be measured the SAME WAY as the observed rate or the subtraction
is nonsense. `observed_rate` is the share of wrong answers whose *winning* label
falls in a coarse category, so the null here is also a distribution over winning
categories: it sums to 1 within a question type, and can be compared directly.
(`label_fire_rates` keeps the any-match rates too, but only as a diagnostic --
those overlap and do not form a distribution.)

Nulls by question type:
  integer-valued -> uniform int in [0, total_pencils]   (PROJECT_PLAN §8)
  name-valued    -> uniform over the other people
  everything else (trajectory, state_snapshot, person_value_timesteps)
                 -> NO defined null; those types are reported UNCORRECTED and
                    flagged, never silently corrected by an ill-fitting one.
"""

import random

from src.classify import classify
from src.labels import COARSE

INT_NULL_TYPES = {
    "person_timestep_lookup", "initial_state_lookup", "population_condition",
    "duration_condition", "transfer_recall", "total_conservation", "net_delta",
}
NAME_NULL_TYPES = {"pairwise_comparison", "argmax_person"}
NULL_TYPES = INT_NULL_TYPES | NAME_NULL_TYPES


def _draw(q, story, rng):
    """One random WRONG answer under this question type's null, or None."""
    qtype, gold = q["question_type"], q["gold"]
    if qtype in INT_NULL_TYPES:
        a = rng.randint(0, story["config"]["total_pencils"])
        return None if a == gold["answer"] else {"answer": a}
    if qtype in NAME_NULL_TYPES:
        names = [n for n in story["name_to_id"] if n != gold["answer"]]
        return {"answer": rng.choice(names)} if names else None
    return None


def chance_table(story, questions, n_samples=1000, seed=0):
    """-> {question_type: {"category_rate": {cat: p}, "label_fire_rates": {...},
                           "n_samples": int}}

    `category_rate` is a distribution over winning coarse categories and is the
    one to subtract from an observed rate. Question types without a defined null
    are simply absent from the result.
    """
    from src.grade import grade
    rng = random.Random(seed)
    acc = {}
    for q in questions:
        qtype = q["question_type"]
        if qtype not in NULL_TYPES:
            continue
        cell = acc.setdefault(qtype, {"cats": {}, "labels": {}, "n": 0})
        for _ in range(n_samples):
            pred = _draw(q, story, rng)
            if pred is None:
                continue
            res = classify(q, grade(q, pred, story), story)
            cell["n"] += 1
            cat = res["coarse_category"]
            cell["cats"][cat] = cell["cats"].get(cat, 0) + 1
            for lab in res["labels_matched"]:
                cell["labels"][lab] = cell["labels"].get(lab, 0) + 1
    out = {}
    for qtype, cell in acc.items():
        n = cell["n"]
        if not n:
            continue
        out[qtype] = {
            "category_rate": {c: v / n for c, v in cell["cats"].items()},
            "label_fire_rates": {l: v / n for l, v in cell["labels"].items()},
            "n_samples": n,
        }
    return out


def category_rate_for(chance, qtype, category):
    cell = chance.get(qtype)
    return cell["category_rate"].get(category, 0.0) if cell else None


__all__ = ["chance_table", "category_rate_for", "NULL_TYPES", "COARSE"]
