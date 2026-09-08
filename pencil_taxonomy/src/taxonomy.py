"""Five failure categories, decided by direct comparison against `M`.

Scope: `person_timestep_lookup` answers, and the first divergent element of a
`trajectory` answer. Both are "one accumulated value, for one person, at one
timestep", so one implementation covers both. The other three types are reported
correct/wrong only -- the two controls have no accumulation to mis-perform, and a
whole-vector answer has no single mechanism to attribute. That is a scoping
decision, and it is stated in the report rather than left implicit.

Priority order, for person P at timestep t with truth y and wrong answer a:

  1 stale            a == M[0][P]          -- reported the initial state, no update
  2 wrong_person     a == M[t][Q], Q != P  -- right timestep, wrong column
  3 wrong_timestep   a == M[s][P], s != t  -- right person, wrong row
  4 single_transfer  a is y off by exactly one transfer touching P at time <= t
  5 unexplained      anything else numeric

`format_error` never receives a mechanism.

An answer can satisfy several rules; `label` is the highest-priority match and
`labels_matched` records all of them. The share of wrong answers with more than
one match is the ambiguity rate -- a validity diagnostic, reported, not hidden.
"""

import random

CATEGORIES = ["stale", "wrong_person", "wrong_timestep", "single_transfer",
              "unexplained"]


def _deltas_for(story_meta, pid, upto_t):
    """Signed change to `pid` from each transfer at time <= upto_t."""
    out = []
    for (t, giver, receiver, amount) in story_meta["transfers"]:
        if t > upto_t:
            continue
        if giver == pid:
            out.append(-amount)
        elif receiver == pid:
            out.append(amount)
    return out


def classify(a, y, pid, t, story_meta):
    """-> (label, labels_matched, sub_label or None). `a` must be an int."""
    M = story_meta["M"]                      # {t: {person_id: count}}
    ids = story_meta["person_ids"]
    matched, sub = [], None

    if a == M[0][pid] and y != M[0][pid]:
        matched.append("stale")
    if any(a == M[t][q] for q in ids if q != pid):
        matched.append("wrong_person")
    if any(a == M[s][pid] for s in M if s != t):
        matched.append("wrong_timestep")

    deltas = _deltas_for(story_meta, pid, t)
    for d in deltas:
        if a == y - d:
            matched.append("single_transfer"); sub = "missed"; break
        if a == y + d:
            matched.append("single_transfer"); sub = "doubled"; break
        if a == y - 2 * d:
            matched.append("single_transfer"); sub = "sign_flipped"; break

    if not matched:
        matched = ["unexplained"]
    label = min(matched, key=CATEGORIES.index)
    return label, sorted(set(matched), key=CATEGORIES.index), sub


def chance_table(wrong, draws, seed):
    """Monte Carlo null: what the classifier returns on random in-range answers.

    `wrong` is a list of dicts with keys: answer_space_size, ground_truth,
    target_person_id, target_timestep, story_meta.

    The null must be measured the same way the observation is. `observed` is a
    distribution over WINNING categories, so the null is too -- averaging
    any-match rates instead produces overlapping "rates" that can exceed 1 and
    are not comparable with anything.
    """
    rng = random.Random(seed)
    tally = {c: 0.0 for c in CATEGORIES}
    n = 0
    for w in wrong:
        space = w.get("answer_space_size")
        if not space or space < 2:
            continue
        n += 1
        local = {c: 0 for c in CATEGORIES}
        for _ in range(draws):
            a = rng.randrange(space)
            if a == w["ground_truth"]:
                continue                      # a correct draw is not a failure
            lab, _, _ = classify(a, w["ground_truth"], w["target_person_id"],
                                 w["target_timestep"], w["story_meta"])
            local[lab] += 1
        tot = sum(local.values()) or 1
        for c in CATEGORIES:
            tally[c] += local[c] / tot
    if n == 0:
        return {c: float("nan") for c in CATEGORIES}
    return {c: tally[c] / n for c in CATEGORIES}
