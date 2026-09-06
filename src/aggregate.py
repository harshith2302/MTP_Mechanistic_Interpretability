"""raw JSONL -> tidy CSVs in results/tables/.

The one rule that governs every denominator here: context_overflow records are
EXCLUDED, never counted as wrong. n_valid is the accuracy denominator; n_overflow
is reported beside it so a truncated model line is visibly truncated, not zero.
"""

import argparse
import glob
import json
import math
import os

import pandas as pd

from src.chance import chance_table
from src.questions import sample_questions
from src.simulate import generate_story

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_run(run_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(run_dir, "*", "N*.jsonl"))):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue          # tolerate a half-written final line
    if not rows:
        raise SystemExit(f"no records found under {run_dir}")
    df = pd.DataFrame(rows)
    for col in ("context_overflow", "correct", "ambiguous"):
        if col not in df:
            df[col] = None
    df["context_overflow"] = df["context_overflow"].fillna(False).astype(bool)
    return df


def wilson(k, n, z=1.96):
    """95% Wilson score interval — correct at the 0% and 100% ends, unlike normal."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


ACCURACY_COLUMNS = ["model", "n", "question_type", "n_correct", "n_valid",
                    "n_overflow", "n_format_error", "accuracy", "ci_low",
                    "ci_high", "format_error_rate", "ambiguous_rate"]
TAXONOMY_COLUMNS = ["model", "n", "coarse_category", "count",
                    "n_wrong_mechanistic", "n_wrong_format", "observed_rate",
                    "chance_rate", "corrected_rate", "chance_coverage"]


def _frame(rows, columns, sort_by):
    """Empty results are normal: a model past its context limit contributes no
    valid records at all. Return a correctly-shaped empty frame, never crash."""
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values(sort_by)


def accuracy_table(df):
    out = []
    valid = df[~df["context_overflow"]]
    keys = ["model", "n_people", "question_type"]
    for (model, n, qtype), g in valid.groupby(keys):
        ov = df[(df.model == model) & (df.n_people == n)
                & (df.question_type == qtype) & df.context_overflow]
        fmt = int((g["coarse_category"] == "Format").sum())
        k, tot = int(g["correct"].sum()), len(g)
        lo, hi = wilson(k, tot)
        out.append({
            "model": model, "n": n, "question_type": qtype,
            "n_correct": k, "n_valid": tot, "n_overflow": len(ov),
            "n_format_error": fmt,
            "accuracy": k / tot if tot else float("nan"),
            "ci_low": lo, "ci_high": hi,
            "format_error_rate": fmt / tot if tot else float("nan"),
            "ambiguous_rate": float(g["ambiguous"].fillna(False).mean()),
        })
    # the aggregated-over-question-type rows the headline figure uses
    for (model, n), g in valid.groupby(["model", "n_people"]):
        ov = df[(df.model == model) & (df.n_people == n) & df.context_overflow]
        k, tot = int(g["correct"].sum()), len(g)
        lo, hi = wilson(k, tot)
        out.append({
            "model": model, "n": n, "question_type": "ALL",
            "n_correct": k, "n_valid": tot, "n_overflow": len(ov),
            "n_format_error": int((g["coarse_category"] == "Format").sum()),
            "accuracy": k / tot if tot else float("nan"),
            "ci_low": lo, "ci_high": hi,
            "format_error_rate": float((g["coarse_category"] == "Format").mean()),
            "ambiguous_rate": float(g["ambiguous"].fillna(False).mean()),
        })
    return _frame(out, ACCURACY_COLUMNS, ["model", "question_type", "n"])


def _chance_for_n(n, cfg, cache):
    """Chance distribution over winning coarse categories, per question type.

    Averaged over a handful of stories at this N so it reflects the actual value
    range, which is what drives the collision rate.
    """
    if n in cache:
        return cache[n]
    per_type, k = {}, 0
    n_stories = min(5, cfg["stories_per_n"])          # 5 stories is plenty
    for si in range(n_stories):
        seed = 1000 * n + si
        story = generate_story(n, n, seed)
        qs = sample_questions(story, seed)
        tbl = chance_table(story, qs, cfg["classify"]["chance_baseline_samples"], seed)
        for qtype, cell in tbl.items():
            dst = per_type.setdefault(qtype, {})
            for cat, rate in cell["category_rate"].items():
                dst[cat] = dst.get(cat, 0.0) + rate
        k += 1
    cache[n] = {qt: {c: v / k for c, v in d.items()} for qt, d in per_type.items()}
    return cache[n]


def taxonomy_table(df, cfg, with_chance=True):
    """Observed, chance and corrected rates per coarse category.

    Denominator is WRONG, non-overflow, non-format records: a format error is
    reported on its own line and never assigned a mechanism.

    The null is weighted by the question-type mix of the wrong answers actually
    in the cell, and only over question types that HAVE a defined null -- so it
    is measured the same way as the observed rate and the subtraction is
    meaningful. `chance_coverage` is the share of the cell the null covers;
    where it is 0 the corrected rate is NaN, never a silent 0.
    """
    valid = df[~df["context_overflow"]]
    wrong = valid[valid["correct"] == False]  # noqa: E712
    mech = wrong[wrong["coarse_category"] != "Format"]
    cache, out = {}, []
    for (model, n), g in mech.groupby(["model", "n_people"]):
        tot = len(g)
        fmt_n = int((wrong[(wrong.model == model) & (wrong.n_people == n)]
                     ["coarse_category"] == "Format").sum())

        weights, chance = {}, {}
        if with_chance:
            per_type = _chance_for_n(n, cfg, cache)
            counts = g["question_type"].value_counts()
            covered = {q: int(c) for q, c in counts.items() if q in per_type}
            denom = sum(covered.values())
            if denom:
                weights = {q: c / denom for q, c in covered.items()}
                for qtype, w in weights.items():
                    for cat, rate in per_type[qtype].items():
                        chance[cat] = chance.get(cat, 0.0) + w * rate
            coverage = denom / tot if tot else 0.0
        else:
            coverage = 0.0

        for cat, cnt in g["coarse_category"].value_counts().items():
            obs = cnt / tot
            ch = chance.get(cat, 0.0) if coverage > 0 else float("nan")
            out.append({
                "model": model, "n": n, "coarse_category": cat,
                "count": int(cnt), "n_wrong_mechanistic": tot,
                "n_wrong_format": fmt_n,
                "observed_rate": obs, "chance_rate": ch,
                "corrected_rate": obs - ch,
                "chance_coverage": coverage,
            })
    return _frame(out, TAXONOMY_COLUMNS, ["model", "n", "coarse_category"])


def divergence_table(df):
    traj = df[(df.question_type == "trajectory") & (~df["context_overflow"])].copy()
    if "first_divergence_index" not in traj:
        return pd.DataFrame()
    traj = traj[traj["first_divergence_index"].notna()]
    if traj.empty:
        return pd.DataFrame()
    g = traj.groupby(["model", "n_people"])["first_divergence_index"]
    out = g.agg(["mean", "median", "count", "std"]).reset_index()
    return out.rename(columns={"n_people": "n", "mean": "mean_first_divergence",
                               "median": "median_first_divergence",
                               "count": "n_diverged", "std": "std_first_divergence"})


def main():
    import yaml
    ap = argparse.ArgumentParser(description="Aggregate raw JSONL into tables.")
    ap.add_argument("--run-dir", required=True,
                    help="results/raw/<run_id>  (contains one dir per model)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "tables"))
    ap.add_argument("--config", default=os.path.join(ROOT, "configs/experiment.yaml"))
    ap.add_argument("--no-chance", action="store_true",
                    help="skip the chance baseline (fast, but rates are then "
                         "NOT publishable)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    os.makedirs(args.out, exist_ok=True)
    df = load_run(args.run_dir)
    print(f"[aggregate] {len(df)} records, {df['model'].nunique()} model(s), "
          f"{int(df['context_overflow'].sum())} context_overflow")

    for name, table in [
        ("accuracy_by_model_n", accuracy_table(df)),
        ("taxonomy_by_model_n", taxonomy_table(df, cfg, not args.no_chance)),
        ("divergence_by_model_n", divergence_table(df)),
    ]:
        path = os.path.join(args.out, name + ".csv")
        table.to_csv(path, index=False)
        print(f"[aggregate] {path}  ({len(table)} rows)")
    if args.no_chance:
        print("[aggregate] WARNING: taxonomy rates have NO chance baseline. "
              "Do not publish them.")


if __name__ == "__main__":
    main()
