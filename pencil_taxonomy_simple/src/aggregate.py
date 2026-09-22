"""Tables. Every figure has a matching CSV here, so nothing in a figure is
unreproducible.

Two views of the same records:

  * the (N, T) GRID -- accuracy_grid.csv, taxonomy_grid.csv, divergence_grid.csv
  * the N = T DIAGONAL -- accuracy.csv, taxonomy.csv, divergence.csv, in exactly
    the format pencil_taxonomy wrote, so the two experiments can be laid side by
    side. The diagonal has 30 stories per cell for that reason.

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

FIELDS = ("model_key", "N", "T", "question_type", "correct", "correct_strict",
          "format_error", "context_overflow", "truncated", "per_element_accuracy",
          "normalisation_rescued", "answer_mode")


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


def _cell(g):
    asked = len(g)
    ov = int(g.context_overflow.sum())
    v = g[~g.context_overflow.astype(bool)]
    n_valid = len(v)
    k = int(v.correct.sum())
    lo, hi = wilson(k, n_valid)
    pe = v.per_element_accuracy.dropna()
    return {
        "n_asked": asked, "n_overflow": ov, "n_valid": n_valid, "n_correct": k,
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
    }


def accuracy_grid(recs):
    df = pd.DataFrame([{k: r.get(k) for k in FIELDS} for r in recs])
    out = []
    for (mk, n, t, qt), g in df.groupby(["model_key", "N", "T", "question_type"]):
        out.append({"model_key": mk, "N": n, "T": t, "question_type": qt, **_cell(g)})
    # pooled over models, per cell -- what the heatmaps draw
    for (n, t, qt), g in df.groupby(["N", "T", "question_type"]):
        out.append({"model_key": "ALL", "N": n, "T": t, "question_type": qt, **_cell(g)})
    for (n, t), g in df.groupby(["N", "T"]):
        out.append({"model_key": "ALL", "N": n, "T": t, "question_type": "ALL", **_cell(g)})
    for (mk, n, t), g in df.groupby(["model_key", "N", "T"]):
        out.append({"model_key": mk, "N": n, "T": t, "question_type": "ALL", **_cell(g)})
    return pd.DataFrame(out).sort_values(["model_key", "question_type", "N", "T"])


def accuracy_diagonal(recs, cfg):
    """pencil_taxonomy's accuracy.csv: the N == T cells that were given
    `diagonal_stories` (30 per model), so every point has the same footing as
    that experiment's. Grid cells that merely happen to sit on the diagonal
    (5 stories) are left out here; they still appear in accuracy_grid.csv."""
    df = pd.DataFrame([{k: r.get(k) for k in FIELDS} for r in recs])
    # NB: df["T"], never df.T -- that attribute is the transpose.
    diag = set(cfg["grid"].get("diagonal_n_values", []))
    df = df[(df["N"] == df["T"]) & df["N"].isin(diag)]
    out = []
    for (mk, n, qt), g in df.groupby(["model_key", "N", "question_type"]):
        out.append({"model_key": mk, "N": n, "question_type": qt, **_cell(g)})
    return pd.DataFrame(out).sort_values(["model_key", "question_type", "N"])


def _taxonomy_rows(recs):
    """Wrong scalar answers eligible for a mechanism label: person_timestep_lookup
    answers, and the first divergent element of a trajectory answer."""
    rows = []
    for r in recs:
        if r.get("context_overflow") or r.get("format_error") or r.get("correct"):
            continue
        qt = r["question_type"]
        meta = r.get("story_meta")
        if not meta:
            continue
        meta = {**meta, "M": {int(k): v for k, v in meta["M"].items()}}
        base = {"model_key": r["model_key"], "N": r["N"], "T": r["T"],
                "question_type": qt, "target_person_id": r["target_person_id"],
                "story_meta": meta}
        if qt == "person_timestep_lookup":
            a = normalise(extract_answer(r.get("parsed"), qt), qt)
            if a is None:
                continue
            rows.append({**base, "answer": a, "ground_truth": r["ground_truth"],
                         "target_timestep": r["target_timestep"],
                         "answer_space_size": r.get("answer_space_size")})
        elif qt == "trajectory":
            a = normalise(extract_answer(r.get("parsed"), qt), qt)
            gt = r["ground_truth"]
            if a is None or is_formulaic(a):
                continue
            i = first_divergence(a, gt)
            if i is None or i >= len(gt) or i >= len(a) or i == 0:
                continue
            rows.append({**base, "answer": a[i], "ground_truth": gt[i],
                         "target_timestep": i,
                         "answer_space_size": r.get("answer_space_size")
                         or (max(gt) + 2)})
    for w in rows:
        lab, matched, sub = classify(w["answer"], w["ground_truth"],
                                     w["target_person_id"], w["target_timestep"],
                                     w["story_meta"])
        w.update(label=lab, labels_matched=matched, sub_label=sub,
                 ambiguous=len(matched) > 1)
    return rows


def _rates(rows, cfg, key_fn):
    """observed / chance / corrected per category, for each group key_fn gives.
    key_fn returns a dict of grouping columns; grouped on its sorted items."""
    groups = {}
    for w in rows:
        groups.setdefault(tuple(sorted(key_fn(w).items())), []).append(w)
    out = []
    for key_items, sub in sorted(groups.items()):
        key = dict(key_items)
        chance = chance_table(sub, cfg["taxonomy"]["chance_draws"],
                              cfg["taxonomy"]["chance_seed"])
        n = len(sub)
        amb = sum(w["ambiguous"] for w in sub) / n
        for c in CATEGORIES:
            obs = sum(w["label"] == c for w in sub) / n
            out.append({**key, "category": c, "n_wrong": n, "observed": obs,
                        "chance": chance[c], "corrected": obs - chance[c],
                        "interpretable": bool(obs - chance[c] >= 0),
                        "ambiguity_rate": amb})
    return pd.DataFrame(out)


def taxonomy_tables(recs, cfg):
    rows = _taxonomy_rows(recs)
    if not rows:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    det = pd.DataFrame([{k: w[k] for k in
                         ("model_key", "N", "T", "question_type", "answer",
                          "ground_truth", "label", "sub_label", "ambiguous")}
                        for w in rows])
    # diagonal, per model: the pencil_taxonomy comparison
    dn = set(cfg["grid"].get("diagonal_n_values", []))
    diag = _rates([w for w in rows if w["N"] == w["T"] and w["N"] in dn], cfg,
                  lambda w: {"model_key": w["model_key"]})
    # whole grid, per (N, T) cell pooled over models: observed shares only.
    # The chance null is expensive (1000 draws per answer) and cells are thin, so
    # per-cell chance is NOT computed; the per-model grid-wide null is reported
    # alongside as the reference.
    cells = det.groupby(["N", "T", "label"]).size().unstack(fill_value=0)
    cells = cells.reindex(columns=CATEGORIES, fill_value=0)
    cells["n_wrong"] = cells.sum(axis=1)
    for c in CATEGORIES:
        cells[f"{c}_share"] = cells[c] / cells["n_wrong"]
    grid = cells.reset_index()
    return diag, det, grid


def taxonomy_pooled(recs, cfg):
    """Per model over the WHOLE grid, with the chance null."""
    rows = _taxonomy_rows(recs)
    if not rows:
        return pd.DataFrame()
    return _rates(rows, cfg, lambda w: {"model_key": w["model_key"]})


def divergence_table(recs, cfg, diagonal_only):
    by = {}
    for r in recs:
        if r["question_type"] != "trajectory" or r.get("context_overflow"):
            continue
        if diagonal_only and (r["N"] != r["T"] or
                              r["N"] not in cfg["grid"].get("diagonal_n_values", [])):
            continue
        a = normalise(extract_answer(r.get("parsed"), "trajectory"), "trajectory")
        gt = r["ground_truth"]
        key = (r["model_key"], r["N"], r["T"]) if not diagonal_only \
            else (r["model_key"], r["T"])
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
    out = []
    for key, d in sorted(by.items()):
        k = len(d["idx"])
        row = {"model_key": key[0]}
        if diagonal_only:
            row["T"] = key[1]
        else:
            row["N"], row["T"] = key[1], key[2]
        row.update(n_answers=d["n"], n_formulaic_excluded=d["formulaic"],
                   n_unusable_excluded=d["unusable"], n_used=k,
                   mean_first_divergence=(sum(d["idx"]) / k) if k else float("nan"),
                   kept=k >= cfg["plots"]["min_divergence_cell"])
        out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default="results/tables")
    a = ap.parse_args()
    cfg = load_config()
    recs = load(os.path.join(ROOT, a.run_dir))
    out = os.path.join(ROOT, a.out)
    os.makedirs(out, exist_ok=True)
    modes = {r.get("answer_mode", "direct") for r in recs}
    print(f"[aggregate] {len(recs)} records, "
          f"{len({r['model_key'] for r in recs})} model(s), "
          f"{sum(bool(r.get('context_overflow')) for r in recs)} overflow, "
          f"answer_mode={sorted(modes)}")

    def save(df, name):
        df.to_csv(os.path.join(out, name), index=False)
        print(f"  {name:26s} {len(df)} rows")

    save(accuracy_grid(recs), "accuracy_grid.csv")
    save(accuracy_diagonal(recs, cfg), "accuracy.csv")
    diag, det, grid = taxonomy_tables(recs, cfg)
    if len(det):
        save(diag, "taxonomy.csv")
        save(det, "taxonomy_detail.csv")
        save(grid, "taxonomy_grid.csv")
        save(taxonomy_pooled(recs, cfg), "taxonomy_pooled_grid.csv")
    save(divergence_table(recs, cfg, diagonal_only=True), "divergence.csv")
    save(divergence_table(recs, cfg, diagonal_only=False), "divergence_grid.csv")


if __name__ == "__main__":
    main()
