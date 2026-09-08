"""Tables. Every figure has a matching CSV here, so nothing in a figure is
unreproducible.

Denominator rule, applied once and used everywhere: a question counts unless
`context_overflow`. Format errors are wrong answers and stay in; they are also
reported separately so both readings are available.
"""

import argparse
import glob
import json
import math
import os

import pandas as pd

from src.grade import first_divergence, is_formulaic
from src.parsing import extract_answer, normalise
from src.taxonomy import CATEGORIES, chance_table, classify
from src.util import ROOT, load_config


def load(run_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(run_dir, "*.jsonl"))):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                rows.append(json.loads(line))
    return rows


def wilson(k, n, z=1.0):
    """z=1 -> +-1 standard error, which is what the figures band."""
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def accuracy_table(recs):
    out = []
    df = pd.DataFrame([{k: r.get(k) for k in
                        ("model_key", "N", "question_type", "correct",
                         "correct_strict", "format_error", "context_overflow",
                         "truncated", "per_element_accuracy",
                         "normalisation_rescued")} for r in recs])
    for (mk, n, qt), g in df.groupby(["model_key", "N", "question_type"]):
        asked = len(g)
        ov = int(g.context_overflow.sum())
        v = g[~g.context_overflow.astype(bool)]
        n_valid = len(v)
        k = int(v.correct.sum())
        lo, hi = wilson(k, n_valid)
        pe = v.per_element_accuracy.dropna()
        out.append({
            "model_key": mk, "N": n, "question_type": qt,
            "n_asked": asked, "n_overflow": ov, "n_valid": n_valid,
            "n_correct": k,
            "accuracy": k / n_valid if n_valid else float("nan"),
            "se_low": lo, "se_high": hi,
            "accuracy_strict": (int(v.correct_strict.sum()) / n_valid
                                if n_valid else float("nan")),
            "n_rescued": int(v.normalisation_rescued.sum()),
            "format_error_rate": (int(v.format_error.sum()) / n_valid
                                  if n_valid else float("nan")),
            "truncated_rate": (int(v.truncated.sum()) / n_valid
                               if n_valid else float("nan")),
            "coverage": n_valid / asked if asked else 0.0,
            "per_element_accuracy": pe.mean() if len(pe) else float("nan"),
        })
    return pd.DataFrame(out).sort_values(["model_key", "question_type", "N"])


def _taxonomy_rows(recs):
    """Wrong scalar answers eligible for a mechanism label.

    `person_timestep_lookup` answers, plus the first divergent element of a
    `trajectory` answer -- both are one accumulated value for one person at one
    timestep, so one classifier covers both.
    """
    rows = []
    for r in recs:
        if r.get("context_overflow") or r.get("format_error") or r.get("correct"):
            continue
        qt = r["question_type"]
        meta = r.get("story_meta")
        if not meta:
            continue
        M = {int(k): v for k, v in meta["M"].items()}
        meta = {**meta, "M": M}
        if qt == "person_timestep_lookup":
            a = normalise(extract_answer(r.get("parsed"), qt), qt)
            if a is None:
                continue
            rows.append({"model_key": r["model_key"], "N": r["N"],
                         "question_type": qt, "answer": a,
                         "ground_truth": r["ground_truth"],
                         "target_person_id": r["target_person_id"],
                         "target_timestep": r["target_timestep"],
                         "answer_space_size": r.get("answer_space_size"),
                         "story_meta": meta})
        elif qt == "trajectory":
            a = normalise(extract_answer(r.get("parsed"), qt), qt)
            gt = r["ground_truth"]
            if a is None or is_formulaic(a):
                continue
            i = first_divergence(a, gt)
            if i is None or i >= len(gt) or i >= len(a) or i == 0:
                continue
            rows.append({"model_key": r["model_key"], "N": r["N"],
                         "question_type": qt, "answer": a[i],
                         "ground_truth": gt[i],
                         "target_person_id": r["target_person_id"],
                         "target_timestep": i,
                         "answer_space_size": r.get("answer_space_size")
                         or (max(gt) + 2),
                         "story_meta": meta})
    return rows


def taxonomy_table(recs, cfg):
    rows = _taxonomy_rows(recs)
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    for w in rows:
        lab, matched, sub = classify(w["answer"], w["ground_truth"],
                                     w["target_person_id"], w["target_timestep"],
                                     w["story_meta"])
        w["label"], w["labels_matched"], w["sub_label"] = lab, matched, sub
        w["ambiguous"] = len(matched) > 1

    det = pd.DataFrame([{k: w[k] for k in
                         ("model_key", "N", "question_type", "answer",
                          "ground_truth", "label", "sub_label", "ambiguous")}
                        for w in rows])

    tax = []
    for mk, g in det.groupby("model_key"):
        sub = [w for w in rows if w["model_key"] == mk]
        chance = chance_table(sub, cfg["taxonomy"]["chance_draws"],
                              cfg["taxonomy"]["chance_seed"])
        n = len(g)
        for c in CATEGORIES:
            obs = float((g.label == c).mean())
            ch = chance[c]
            tax.append({"model_key": mk, "category": c, "n_wrong": n,
                        "observed": obs, "chance": ch,
                        "corrected": obs - ch,
                        # A negative corrected value means the null over-fires
                        # for that category; it is reported observed-only and is
                        # NOT presented as a rate.
                        "interpretable": bool(obs - ch >= 0),
                        "ambiguity_rate": float(g.ambiguous.mean())})
    return pd.DataFrame(tax), det


def divergence_table(recs, cfg):
    out = []
    by = {}
    for r in recs:
        if r["question_type"] != "trajectory" or r.get("context_overflow"):
            continue
        a = normalise(extract_answer(r.get("parsed"), "trajectory"), "trajectory")
        gt = r["ground_truth"]
        key = (r["model_key"], r["T"])
        d = by.setdefault(key, {"idx": [], "formulaic": 0, "unusable": 0, "n": 0})
        d["n"] += 1
        if a is None or len(a) != len(gt):
            d["unusable"] += 1
            continue
        if is_formulaic(a):
            d["formulaic"] += 1
            continue
        i = first_divergence(a, gt)
        d["idx"].append(len(gt) if i is None else i)
    for (mk, T), d in sorted(by.items()):
        k = len(d["idx"])
        out.append({"model_key": mk, "T": T, "n_answers": d["n"],
                    "n_formulaic_excluded": d["formulaic"],
                    "n_unusable_excluded": d["unusable"], "n_used": k,
                    "mean_first_divergence": (sum(d["idx"]) / k) if k else float("nan"),
                    "kept": k >= cfg["plots"]["min_divergence_cell"]})
    return pd.DataFrame(out)


def normalisation_table(recs):
    df = pd.DataFrame([{k: r.get(k) for k in
                        ("model_key", "question_type", "normalisation_rescued",
                         "context_overflow", "correct", "correct_strict")}
                       for r in recs])
    df = df[~df.context_overflow.astype(bool)]
    g = df.groupby(["model_key", "question_type"]).agg(
        n=("correct", "size"), n_correct=("correct", "sum"),
        n_correct_strict=("correct_strict", "sum"),
        n_rescued=("normalisation_rescued", "sum")).reset_index()
    g["rescued_rate"] = g.n_rescued / g.n
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default="results/tables")
    a = ap.parse_args()
    cfg = load_config()
    recs = load(os.path.join(ROOT, a.run_dir))
    out = os.path.join(ROOT, a.out)
    os.makedirs(out, exist_ok=True)
    print(f"[aggregate] {len(recs)} records, "
          f"{len({r['model_key'] for r in recs})} model(s), "
          f"{sum(bool(r.get('context_overflow')) for r in recs)} overflow")

    acc = accuracy_table(recs)
    acc.to_csv(os.path.join(out, "accuracy.csv"), index=False)
    print(f"  accuracy.csv            {len(acc)} rows")

    tax, det = taxonomy_table(recs, cfg)
    if len(tax):
        tax.to_csv(os.path.join(out, "taxonomy.csv"), index=False)
        det.to_csv(os.path.join(out, "taxonomy_detail.csv"), index=False)
        print(f"  taxonomy.csv            {len(tax)} rows  "
              f"({len(det)} labelled wrong answers)")

    div = divergence_table(recs, cfg)
    div.to_csv(os.path.join(out, "divergence.csv"), index=False)
    print(f"  divergence.csv          {len(div)} rows")

    nor = normalisation_table(recs)
    nor.to_csv(os.path.join(out, "normalisation.csv"), index=False)
    print(f"  normalisation.csv       {len(nor)} rows  "
          f"({int(nor.n_rescued.sum())} answers rescued by normalisation)")


if __name__ == "__main__":
    main()
