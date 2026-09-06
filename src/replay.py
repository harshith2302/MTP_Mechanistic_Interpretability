"""Counterfactual replay: the exact arithmetic behind each taxonomy label.

Every function here answers "what number would a reader produce if they had
made exactly THIS mistake?" — computed from the ground truth, never guessed.
`_deltas_touching` returns deltas in NARRATION order because omission_k means
"dropped the last k transfers the reader saw", not the last k chronologically.
"""

def _deltas_touching(story, pid, t):
    """Signed deltas on `pid` up to timestep t, in NARRATION (reading) order."""
    out = []
    for ev in story["narration"]:
        if ev["timestep"] > t:
            continue
        if ev["giver"] == pid:
            out.append(-ev["amount"])
        elif ev["receiver"] == pid:
            out.append(ev["amount"])
    return out


def _digit_error(a, g):
    """Transposition, single-digit substitution, or a x10 shift.

    Both rules require >=2 digits: for single-digit numbers "differs in one
    digit" is vacuously true of every wrong answer, which would make the label
    fire ~45% of the time by chance and mean nothing.
    """
    sa, sg = str(abs(a)), str(abs(g))
    if len(sa) >= 2 and len(sg) >= 2:
        if sorted(sa) == sorted(sg) and sa != sg:
            return True                                # transposition
        if len(sa) == len(sg) and sum(x != y for x, y in zip(sa, sg)) == 1:
            return True                                # one digit substituted
    if g != 0 and a in (g * 10, g * 100):
        return True                                    # decimal shift up
    return g != 0 and a != 0 and a * 10 == g           # decimal shift down


def state_value_labels(story, pid, t, gold, a):
    """Counterfactual labels for 'how many pencils did p have at t'."""
    labels = []
    states, total = story["states"], story["config"]["total_pencils"]
    T = story["config"]["n_timesteps"]
    if a < 0 or a > total:
        labels.append("out_of_range")

    base = states[0][pid]
    deltas = _deltas_touching(story, pid, t)
    running = base + sum(deltas)

    for k in (1, 2, 3):                                        # omission
        if len(deltas) >= k and running - sum(deltas[-k:]) == a:
            labels.append(f"omission_{k}")
            break
    if any(running + d == a for d in deltas):
        labels.append("over_application")
    if any(running - 2 * d == a for d in deltas):
        labels.append("direction_flip_one")
    if deltas and base - sum(deltas) == a:
        labels.append("direction_flip_all")
    if t >= 1:                                                 # sign on last step
        step = states[t][pid] - states[t - 1][pid]
        if step != 0 and states[t - 1][pid] - step == a:
            labels.append("sign_error_last")

    if any(states[t][q] == a for q in story["person_ids"] if q != pid):
        labels.append("binding_person")
    if t != 0 and states[0][pid] == a:
        labels.append("temporal_initial")
    elif any(0 <= tp <= T and states[tp][pid] == a for tp in (t - 1, t + 1)):
        labels.append("temporal_off_by_one")
    elif any(states[tp][pid] == a for tp in range(T + 1) if tp != t):
        labels.append("temporal_other")

    if _digit_error(a, gold):
        labels.append("digit_error")
    if 0 < abs(a - gold) <= 2:
        labels.append("off_by_small")
    return labels


def _condition_labels(q, story, a, gold, values, t0_values=None):
    """population_condition / duration_condition."""
    from src.questions import _satisfies
    labels, op, th = [], q["meta"]["operator"], q["meta"]["threshold"]
    if a < 0 or a > len(values):
        labels.append("out_of_range")
    flipped = {">=": "<", "<=": ">", ">": "<=", "<": ">="}[op]
    if sum(_satisfies(v, flipped, th) for v in values) == a:
        labels.append("condition_flip")
    boundary = {">=": ">", ">": ">=", "<=": "<", "<": "<="}[op]
    if sum(_satisfies(v, boundary, th) for v in values) == a:
        labels.append("boundary_error")
    if t0_values is not None and sum(_satisfies(v, op, th) for v in t0_values) == a:
        labels.append("included_t0")
    if _digit_error(a, gold):
        labels.append("digit_error")
    if 0 < abs(a - gold) <= 2:
        labels.append("off_by_small")
    return labels
