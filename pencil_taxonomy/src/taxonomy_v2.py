"""Extended failure taxonomy (v2) -- a SECOND, offline classification pass.

v1 stays primary. The 5-category taxonomy in src/taxonomy.py was pre-registered
before the sweep and remains the headline result. v2 was specified AFTER the
data existed and is reported as secondary / exploratory. Both label sets live in
the same records. Never present v2 as if it had been pre-registered.

No generation, no GPU. Reads the graded JSONL, joins it back to the stored
stories by `story_id` (the JSONL carries M and transfers but NOT clusters, and
C1/C2 need the cluster partition), and writes v2 labels alongside the v1 ones.

TWO DEPARTURES FROM THE WRITTEN SPEC, both forced by internal inconsistency in
its own worked example. Recorded here rather than silently resolved:

  1. A2 `partial_update` is `0 <= s < t`, not the spec's `0 < s < t`. Section 4.1
     requires A1 subset-of A2 ("stale is the s==0 case") and the section 5
     fixture lists BOTH A1 and A2 firing for a=13 == M[0][P]. Excluding s=0 makes
     the containment test unsatisfiable and contradicts the fixture.
  2. The section 5 row for a=15 lists {A4, A3, A5}. A3 `lookahead` needs
     a == M[s][P] for some s > t=5; those cells are 14, 17, 17, so A3 cannot
     fire. A2 does (M[3] == M[4] == 15). The set is {A4, A2, A5}; the winner is
     unchanged at `boundary_off_by_one`.

Everything else follows the spec exactly, including B3 and B6, which are
included knowing they are weak -- the chance null is what shows that.
"""

import argparse
import glob
import json
import os
import random
import subprocess

from src.grade import first_divergence, is_formulaic
from src.parsing import extract_answer, normalise
from src.util import ROOT

# --- families ---------------------------------------------------------------
FAMILY = {"A": "update", "B": "arithmetic", "C": "reference",
          "D": "artifact", "E": "residual"}


def _ctx(M, t, P, transfers, cluster, person_ids, space):
    """Everything every rule needs, computed once per question."""
    X = []
    D = []
    for (tt, giver, receiver, amount) in transfers:
        if tt > t:
            continue
        if giver == P:
            X.append(-amount); D.append(amount)
        elif receiver == P:
            X.append(amount); D.append(amount)
    return {
        "y": M[t][P], "M": M, "t": t, "P": P, "T": max(M),
        "K": cluster, "person_ids": person_ids,
        "tot_K": sum(M[0][q] for q in cluster),
        "tot_all": sum(M[0].values()),
        "X": X, "D": D, "answer_space_size": space,
        "_transfers": [t_ for t_ in transfers if t_[0] <= t],
    }


# --- Family A: update failures ----------------------------------------------
def A1_stale(a, c):
    return a == c["M"][0][c["P"]]


def A2_partial_update(a, c):
    # 0 <= s < t. See the module docstring: the spec's own containment
    # requirement (A1 subset-of A2) and its fixture both need s=0 included.
    return any(a == c["M"][s][c["P"]] for s in c["M"] if 0 <= s < c["t"])


def A3_lookahead(a, c):
    return any(a == c["M"][s][c["P"]] for s in c["M"] if s > c["t"])


def A4_boundary_off_by_one(a, c):
    M, t, P = c["M"], c["t"], c["P"]
    return ((t - 1 in M and a == M[t - 1][P]) or
            (t + 1 in M and a == M[t + 1][P]))


def A5_transfer_omitted(a, c):
    return any(a == c["y"] - d for d in c["X"])


def A6_transfer_doubled(a, c):
    return any(a == c["y"] + d for d in c["X"])


# --- Family B: arithmetic failures ------------------------------------------
def B1_direction_single(a, c):
    return any(a == c["y"] - 2 * d for d in c["X"])


def B2_direction_global(a, c):
    return a == 2 * c["M"][0][c["P"]] - c["y"]


def B3_arithmetic_slip(a, c):
    # A MAGNITUDE HEURISTIC, NOT A MECHANISM. Without counterfactual replay this
    # cannot separate "applied the right transfers and mis-summed" from "guessed
    # and landed nearby". Report as an upper bound, never as a rate.
    return 0 < abs(a - c["y"]) <= 2


def B4_conservation_total(a, c):
    return a == c["tot_K"] or a == c["tot_all"]


def B5_out_of_range(a, c):
    return a < 0 or a > c["tot_K"]


def B6_digit_error(a, c):
    # Included mainly to demonstrate it is chance -- at one digit, "differs in
    # one position" is true of every wrong answer. No guard is added, on purpose.
    sa, sy = str(abs(a)), str(abs(c["y"]))
    if a == c["y"]:
        return False
    if sorted(sa) == sorted(sy):
        return True
    if len(sa) == len(sy):
        return sum(x != z for x, z in zip(sa, sy)) == 1
    return False


# --- Family C: reference failures -------------------------------------------
def C1_wrong_person_same_cluster(a, c):
    return any(a == c["M"][c["t"]][q] for q in c["K"] if q != c["P"])


def C2_wrong_person_other_cluster(a, c):
    return any(a == c["M"][c["t"]][q] for q in c["person_ids"]
               if q not in c["K"])


def C3_delta_readout(a, c):
    return a in c["D"]


# --- priority ---------------------------------------------------------------
# FIXED BEFORE THE CLASSIFIER WAS RUN AND NEVER TUNED TO THE DATA.
# A rule ranks higher when it matches a smaller set of integers: a more specific
# rule carries more information, so it wins. Ties break by family order A->B->C.
RULES = [
    ("A1", "stale", "A", A1_stale),
    ("B2", "direction_global", "B", B2_direction_global),
    ("B4", "conservation_total", "B", B4_conservation_total),
    ("A4", "boundary_off_by_one", "A", A4_boundary_off_by_one),
    ("A5", "transfer_omitted", "A", A5_transfer_omitted),
    ("A6", "transfer_doubled", "A", A6_transfer_doubled),
    ("B1", "direction_single", "B", B1_direction_single),
    ("C3", "delta_readout", "C", C3_delta_readout),
    ("C1", "wrong_person_same_cluster", "C", C1_wrong_person_same_cluster),
    ("A2", "partial_update", "A", A2_partial_update),
    ("A3", "lookahead", "A", A3_lookahead),
    ("C2", "wrong_person_other_cluster", "C", C2_wrong_person_other_cluster),
    ("B6", "digit_error", "B", B6_digit_error),
    ("B3", "arithmetic_slip", "B", B3_arithmetic_slip),
    ("B5", "out_of_range", "B", B5_out_of_range),
]
CATEGORIES = [lab for _, lab, _, _ in RULES] + ["unexplained"]
LABEL_FAMILY = {lab: fam for _, lab, fam, _ in RULES}
LABEL_FAMILY["unexplained"] = "E"
for _lab in ("format_error", "truncated", "refusal", "degenerate"):
    LABEL_FAMILY[_lab] = "D"

REFUSAL_MARKERS = ("cannot be determined", "cannot be determined from",
                   "insufficient information", "not enough information",
                   "unknown", "not specified", "cannot determine",
                   "unable to determine", "n/a")


def precheck(rec, answer_raw, surface):
    """Family D. Short-circuits before any semantic rule."""
    if rec.get("format_error"):
        return "format_error"
    if rec.get("truncated"):
        return "truncated"
    if answer_raw is None:
        return "refusal"
    if isinstance(answer_raw, str):
        low = answer_raw.strip().lower()
        if any(m in low for m in REFUSAL_MARKERS):
            return "refusal"
    if surface == "trajectory" and isinstance(answer_raw, list):
        norm = normalise(answer_raw, "trajectory")
        if norm is not None and is_formulaic(norm):
            return "degenerate"
    return None


def classify(a, ctx):
    """Pure and deterministic: no I/O, no randomness. The chance null calls this
    a million times, which is only safe because of that."""
    matched = [lab for _, lab, _, fn in RULES if fn(a, ctx)]
    if not matched:
        matched = ["unexplained"]
    label = matched[0]
    return {"label_v2": label, "labels_matched_v2": matched,
            "n_labels_v2": len(matched), "family_v2": LABEL_FAMILY[label]}


# --- driver ------------------------------------------------------------------
def load_stories(path):
    out = {}
    for p in sorted(glob.glob(os.path.join(path, "N*.jsonl"))):
        for line in open(p, encoding="utf-8"):
            s = json.loads(line)
            s["states"] = {int(k): v for k, v in s["states"].items()}
            out[s["story_id"]] = s
    return out


def cluster_of(story, pid):
    for members in story["clusters"].values():
        if pid in members:
            return members
    return [pid]


def build_rows(recs, stories):
    """The two classified surfaces, with join integrity asserted."""
    rows = []
    for r in recs:
        if r.get("context_overflow"):
            continue
        qt = r["question_type"]
        if qt not in ("person_timestep_lookup", "trajectory"):
            continue
        s = stories.get(r["story_id"])
        if s is None:
            raise SystemExit(f"join failed: no story {r['story_id']}")
        M, P = s["states"], r["target_person_id"]
        transfers = [(e["timestep"], e["giver"], e["receiver"], e["amount"])
                     for e in s["narration"]]
        raw = extract_answer(r.get("parsed"), qt) if r.get("parsed") else None

        pre = precheck(r, raw, qt)
        if qt == "person_timestep_lookup":
            t = r["target_timestep"]
            if M[t][P] != r["ground_truth"]:
                raise SystemExit(f"join integrity: {r['question_id']}")
            if r.get("correct"):
                continue
            a = normalise(raw, qt)
        else:
            norm = normalise(raw, "trajectory")
            gt = r["ground_truth"]
            if r.get("correct"):
                continue
            if pre is None and (norm is None or first_divergence(norm, gt) is None):
                pre = pre or "format_error"
            i = first_divergence(norm, gt) if norm else None
            if pre is None and (i is None or i == 0 or i >= len(gt) or i >= len(norm)):
                continue
            t, a = (i, norm[i]) if pre is None else (0, None)
        rows.append({
            "question_id": r["question_id"], "model_key": r["model_key"],
            "N": r["N"], "question_type": qt, "surface": qt,
            "answer": a, "ground_truth": M[t][P] if pre is None else r["ground_truth"],
            "target_person_id": P, "target_timestep": t,
            "answer_space_size": r.get("answer_space_size") or (
                sum(M[0][q] for q in cluster_of(s, P)) + 1),
            "first_divergence_index": t if qt == "trajectory" else None,
            "precheck": pre,
            "ctx": _ctx(M, t, P, transfers, cluster_of(s, P), s["person_ids"],
                        r.get("answer_space_size")),
        })
    return rows


def label_rows(rows):
    for w in rows:
        if w["precheck"]:
            w.update(label_v2=w["precheck"], labels_matched_v2=[w["precheck"]],
                     n_labels_v2=1, family_v2="D")
        else:
            w.update(classify(w["answer"], w["ctx"]))
    return rows


def chance(rows, draws, seed):
    """Same method as v1, over the v2 category set.

    Also returns the two diagnostics that matter more than any single corrected
    rate: what share of RANDOM draws get a mechanistic label at all, and how many
    rules a random draw matches on average.
    """
    rng = random.Random(seed)
    tally = {c: 0.0 for c in CATEGORIES}
    n = 0
    covered = 0.0
    nlab = 0.0
    for w in rows:
        if w["precheck"]:
            continue
        space = w["answer_space_size"]
        if not space or space < 2:
            continue
        n += 1
        local = {c: 0 for c in CATEGORIES}
        cov = lab = 0
        for _ in range(draws):
            a = rng.randrange(space)
            if a == w["ctx"]["y"]:
                continue
            res = classify(a, w["ctx"])
            local[res["label_v2"]] += 1
            lab += res["n_labels_v2"]
            cov += res["label_v2"] != "unexplained"
        tot = sum(local.values()) or 1
        for c in CATEGORIES:
            tally[c] += local[c] / tot
        covered += cov / tot
        nlab += lab / tot
    if n == 0:
        return {c: float("nan") for c in CATEGORIES}, float("nan"), float("nan")
    return ({c: tally[c] / n for c in CATEGORIES}, covered / n, nlab / n)


def classifier_sha():
    try:
        return subprocess.check_output(
            ["git", "-C", ROOT, "log", "-1", "--format=%h", "--",
             "src/taxonomy_v2.py"], text=True,
            stderr=subprocess.DEVNULL).strip() or "uncommitted"
    except Exception:                                        # noqa: BLE001
        return "unknown"


def _v1_label(rec):
    """v1's label, computed on demand.

    v1 classifies inside aggregate.py and never writes labels into the records,
    so there is nothing on the record to join to. Recompute with the UNMODIFIED
    v1 classifier rather than reimplementing it -- the whole point of the
    v1-vs-v2 table is that both sides are the real thing.
    """
    return rec.get("label_v1_cached")


def compute_v1_labels(rows):
    from src.taxonomy import classify as v1_classify
    for w in rows:
        if w["precheck"] or w["answer"] is None:
            w["label_v1_cached"] = None
            continue
        lab, _, _ = v1_classify(w["answer"], w["ctx"]["y"],
                                w["target_person_id"], w["target_timestep"],
                                {"M": w["ctx"]["M"],
                                 "person_ids": w["ctx"]["person_ids"],
                                 "transfers": w["ctx"]["_transfers"]})
        w["label_v1_cached"] = lab
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--stories", default="data/stories")
    ap.add_argument("--n-null", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tables", default="results/tables")
    a = ap.parse_args()

    import pandas as pd

    run = os.path.join(ROOT, a.run_dir)
    stories = load_stories(os.path.join(ROOT, a.stories))
    print(f"[v2] {len(stories)} stories loaded")

    recs, files = [], sorted(glob.glob(os.path.join(run, "*.jsonl")))
    for p in files:
        recs += [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    print(f"[v2] {len(recs)} records from {len(files)} file(s)")

    rows = compute_v1_labels(label_rows(build_rows(recs, stories)))
    sha = classifier_sha()
    print(f"[v2] {len(rows)} classified answers (classifier {sha})")

    # --- write labels back into the records, leaving v1 untouched -----------
    # Key on (model_key, question_id). `question_id` alone is NOT unique -- it is
    # "N02_s003_person_timestep_lookup" for all four models, since the same 300
    # stories are evaluated by every model. Keying on it alone wrote one model's
    # label onto all four records, including correct answers.
    by_qid = {(w["model_key"], w["question_id"]): w for w in rows}
    for p in files:
        out = []
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            w = by_qid.get((r["model_key"], r["question_id"]))
            if w:
                r.update(label_v2=w["label_v2"],
                         labels_matched_v2=w["labels_matched_v2"],
                         n_labels_v2=w["n_labels_v2"], family_v2=w["family_v2"],
                         first_divergence_index=w["first_divergence_index"],
                         taxonomy_version="v2", classifier_sha=sha)
            out.append(r)
        with open(p, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
    print(f"[v2] labels written back into {len(files)} file(s)")

    sem = [w for w in rows if not w["precheck"]]
    tdir = os.path.join(ROOT, a.tables)
    os.makedirs(tdir, exist_ok=True)

    # --- rates + null, per model -------------------------------------------
    rates, diag = [], []
    for mk in sorted({w["model_key"] for w in rows}):
        mine = [w for w in sem if w["model_key"] == mk]
        if not mine:
            continue
        ch, cov, nlab = chance(mine, a.n_null, a.seed)
        n = len(mine)
        for c in CATEGORIES:
            obs = sum(w["label_v2"] == c for w in mine) / n
            rates.append({"model_key": mk, "category": c,
                          "family": LABEL_FAMILY[c], "n": n,
                          "observed": obs, "chance": ch[c],
                          "corrected": obs - ch[c],
                          "interpretable": bool(obs - ch[c] >= 0)})
        amb = sum(w["n_labels_v2"] > 1 for w in mine) / n
        v1 = [w for w in mine if w.get("label_v1_cached")]
        diag.append({"model_key": mk, "n_classified_v2": n,
                     "ambiguity_rate_v2": amb,
                     "mean_labels_v2": sum(w["n_labels_v2"] for w in mine) / n,
                     "chance_coverage_v2": cov, "chance_ambiguity_v2": nlab,
                     "unexplained_v2": sum(w["label_v2"] == "unexplained"
                                           for w in mine) / n,
                     "n_wrong_v1": len(v1)})
    pd.DataFrame(rates).to_csv(
        os.path.join(tdir, "taxonomy_v2_rates.csv"), index=False)
    fam = pd.DataFrame(rates)
    fam = fam[fam.family.isin(list("ABC"))].groupby(
        ["model_key", "family"])[["observed", "chance", "corrected"]].sum().reset_index()
    fam.to_csv(os.path.join(tdir, "taxonomy_v2_families.csv"), index=False)
    pd.DataFrame(diag).to_csv(
        os.path.join(tdir, "taxonomy_v2_diagnostics.csv"), index=False)

    # --- where did each v1 category land in v2? -----------------------------
    cross = {}
    for w in sem:
        k = (w.get("label_v1_cached") or "(not classified in v1)", w["label_v2"])
        cross[k] = cross.get(k, 0) + 1
    pd.DataFrame([{"v1_label": k[0], "v2_label": k[1], "n": v}
                  for k, v in sorted(cross.items())]).to_csv(
        os.path.join(tdir, "taxonomy_v1_vs_v2.csv"), index=False)

    d = pd.DataFrame(diag)
    print(f"\n[v2] ambiguity_rate_v2   {d.ambiguity_rate_v2.mean():.1%}")
    print(f"[v2] chance_coverage_v2 {d.chance_coverage_v2.mean():.1%}  "
          f"<- share of RANDOM draws getting a mechanistic label")
    print(f"[v2] chance_ambiguity_v2 {d.chance_ambiguity_v2.mean():.2f} rules per "
          f"random draw (real answers: {d.mean_labels_v2.mean():.2f})")
    print(f"[v2] unexplained_v2      {d.unexplained_v2.mean():.1%}")
    print("[v2] wrote taxonomy_v2_{rates,families,diagnostics}.csv, "
          "taxonomy_v1_vs_v2.csv")


if __name__ == "__main__":
    main()
