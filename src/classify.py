"""Failure taxonomy by counterfactual replay.

For each wrong answer we generate counterfactual answers, each the exact result
of one hypothesised corruption of the correct computation, and see which the
model matched. We own the simulator, so every counterfactual is exact rather
than heuristic.

Three rules keep this honest (PROJECT_PLAN §8):
  1. emit EVERY match in `labels_matched`; `label` is the winner by PRIORITY.
  2. every published rate needs the chance-collision baseline (`chance_table`).
  3. a format_error is never assigned a mechanism.
"""

import argparse
import json

from src.labels import COARSE, FORMAT_STATUSES, PRIORITY
from src.replay import (_condition_labels, _deltas_touching, _digit_error,
                        state_value_labels)

_RANK = {lab: i for i, lab in enumerate(PRIORITY)}


def classify(question, graded, story):
    """-> {label, labels_matched, ambiguous, coarse_category, extra}."""
    qtype = question["question_type"]
    status, correct = graded["status"], graded["correct"]
    extra = {}

    if status != "ok":
        lab = FORMAT_STATUSES.get(status, "format_error")
        return _pack([lab], extra)
    if correct:
        return _pack(["correct"], extra)

    a, gold = graded["pred"], graded["gold"]
    states = story["states"]
    T = story["config"]["n_timesteps"]
    labels = []

    if qtype in ("person_timestep_lookup", "initial_state_lookup"):
        labels = state_value_labels(story, question["queried_person"],
                                    question["queried_timestep"], gold["answer"], a)

    elif qtype == "trajectory":
        div = graded["detail"]["first_divergence_index"]
        extra["first_divergence_index"] = div
        extra["recovered"] = graded["detail"].get("recovered")
        extra["error_growth"] = graded["detail"].get("error_growth")
        if graded["detail"]["length_mismatch"] and (div is None or div >= len(a)):
            labels = ["format_error"]
        elif div is not None and div < len(a):
            labels = state_value_labels(story, question["queried_person"], div,
                                        gold["counts"][div], a[div])

    elif qtype == "state_snapshot":
        d, t = graded["detail"], question["queried_timestep"]
        extra["partial_correct_fraction"] = d["partial_correct_fraction"]
        if d["missing_people"]:
            labels.append("missing_fields")
        if d["permutation"]:
            labels.append("permutation")
        if d["conservation_violation"]:
            labels.append("conservation_violation")
        per = {}
        for name, want in gold["counts"].items():
            got = a.get(name.strip().lower())
            if got is None or got == want:
                continue
            pid = story["name_to_id"][name]
            sub = state_value_labels(story, pid, t, want, got)
            per[name] = sub
            labels.extend(sub)
        extra["per_person_labels"] = per

    elif qtype == "population_condition":
        labels = _condition_labels(question, story, a, gold["answer"],
                                   list(states[question["queried_timestep"]].values()))

    elif qtype == "duration_condition":
        pid = question["queried_person"]
        labels = _condition_labels(question, story, a, gold["answer"],
                                   [states[t][pid] for t in range(1, T + 1)],
                                   t0_values=[states[t][pid] for t in range(T + 1)])

    elif qtype == "person_value_timesteps":
        pid, val = question["queried_person"], gold["pencil_count"]
        pred = set(a)
        if pred == set(gold["timesteps"]) - {0}:
            labels.append("excluded_t0")
        for q2 in story["person_ids"]:
            if q2 != pid and sorted(t for t in range(T + 1)
                                    if states[t][q2] == val) == sorted(a):
                labels.append("binding_person")
                break
        extra["missed"] = graded["detail"]["missed"]
        extra["spurious"] = graded["detail"]["spurious"]

    elif qtype == "transfer_recall":
        t, giver = question["queried_timestep"], question["meta"]["giver"]
        recv = question["meta"]["receiver"]
        if a < 0 or a > story["config"]["total_pencils"]:
            labels.append("out_of_range")
        for ev in story["narration"]:
            if ev["timestep"] == t and (ev["giver"], ev["receiver"]) == (recv, giver) \
                    and ev["amount"] == a:
                labels.append("direction_confusion")
                break
        if any(ev["amount"] == a and (ev["giver"], ev["receiver"]) != (giver, recv)
               for ev in story["narration"] if ev["timestep"] == t):
            labels.append("transfer_binding_other")
        if _digit_error(a, gold["answer"]):
            labels.append("digit_error")
        if 0 < abs(a - gold["answer"]) <= 2:
            labels.append("off_by_small")

    elif qtype == "total_conservation":
        labels.append("invariant_violation")
        extra["signed_magnitude"] = a - gold["answer"]
        if _digit_error(a, gold["answer"]):
            labels.append("digit_error")

    elif qtype == "net_delta":
        g = gold["answer"]
        if g != 0 and a == -g:
            labels.append("sign_error")
        pid = question["queried_person"]
        t1, t2 = question["meta"]["from_timestep"], question["meta"]["to_timestep"]
        if any(states[t2][q] - states[t1][q] == a
               for q in story["person_ids"] if q != pid):
            labels.append("binding_person")
        if _digit_error(a, g):
            labels.append("digit_error")
        if 0 < abs(a - g) <= 2:
            labels.append("off_by_small")

    elif qtype in ("pairwise_comparison", "argmax_person"):
        labels.append("comparison_error")
        extra["gap"] = question["meta"].get("gap")
        pid = question["queried_person"]
        pred_id = story["name_to_id"].get(a) if isinstance(a, str) else None
        if pred_id is not None:
            i, j = story["person_ids"].index(pid), story["person_ids"].index(pred_id)
            extra["intro_distance"] = abs(i - j)
            if abs(i - j) == 1:
                labels.append("referent_adjacent")

    return _pack(labels or ["unexplained"], extra)


def _pack(labels, extra):
    seen = [l for i, l in enumerate(labels) if l not in labels[:i]]
    winner = min(seen, key=lambda l: _RANK.get(l, 999)) if seen != ["correct"] else "correct"
    return {
        "label": winner,
        "labels_matched": seen,
        "ambiguous": len(seen) > 1,
        "coarse_category": COARSE.get(winner, "Unexplained"),
        **extra,
    }


def main():
    from src.chance import chance_table
    from src.questions import sample_questions
    from src.simulate import generate_story
    ap = argparse.ArgumentParser(description="Chance-collision baseline for a story.")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--samples", type=int, default=1000)
    args = ap.parse_args()
    story = generate_story(args.n, args.n, args.seed)
    qs = sample_questions(story, args.seed)
    print(json.dumps(chance_table(story, qs, args.samples), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
