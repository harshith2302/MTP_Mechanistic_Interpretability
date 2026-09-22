"""Side-by-side with pencil_taxonomy on the N = T diagonal.

Reads the sibling experiment's tables directly (results/tables/ in
../pencil_taxonomy), so the comparison is between two published table sets, not
a re-analysis. Two figures:

  figC1  accuracy per question type vs N = T, OLD dashed, NEW solid, models pooled
  figC2  failure taxonomy per model, OLD vs NEW observed bars with each one's
         chance level

What "OLD -> NEW" bundles, stated on both figures because it matters: narration
order (interleaved clusters -> chronological), transfer density (1..N per
timestep -> exactly 1), and the prompt (bare question -> rules stated). At the
same N = T the NEW story has far fewer transfers, so the matched-transfer-count
table in the report is the fairer read of the narration-order effect alone.
"""
import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import pandas as pd                      # noqa: E402

from src.plots import COLOURS, LABEL, ORDER, _save, _style   # noqa: E402
from src.util import ROOT                                   # noqa: E402

OLD_DIR = os.path.join(os.path.dirname(ROOT), "pencil_taxonomy", "results", "tables")
NOTE = ("OLD = interleaved clusters, 1..N transfers per timestep, bare question.  "
        "NEW = chronological, exactly 1 transfer per timestep, rules stated in the "
        "prompt.\nAt the same N = T the NEW story has far fewer transfers "
        "(30 vs ~465 at N = 30), so part of the gap is transfer count, not order — "
        "see the matched-transfer table in REPORT.md.")


def _pooled(acc):
    g = acc.groupby(["question_type", "N"]).agg(k=("n_correct", "sum"),
                                                n=("n_valid", "sum")).reset_index()
    g["accuracy"] = g.k / g.n
    g["se"] = (g.accuracy * (1 - g.accuracy) / g.n).pow(0.5)
    return g


def figC1(old, new, out):
    o, n = _pooled(old), _pooled(new)
    fig, axes = plt.subplots(1, 5, figsize=(19, 4.0), sharey=True)
    for ax, qt in zip(axes, ORDER):
        for df, ls, lab, c in ((o, "--", "OLD (pencil_taxonomy)", "#888"),
                               (n, "-", "NEW (pencil_taxonomy_simple)", "#1f77b4")):
            g = df[df.question_type == qt].sort_values("N")
            ax.plot(g.N, g.accuracy, ls, marker="o", ms=3.5, lw=1.8, color=c, label=lab)
            ax.fill_between(g.N, g.accuracy - g.se, g.accuracy + g.se, color=c,
                            alpha=0.15, lw=0)
        tag = " (CONTROL)" if qt in ORDER[:2] else ""
        ax.set_title(f"{qt}{tag}", fontsize=9.5)
        ax.set_ylim(-0.03, 1.03); ax.set_xlabel("N = T")
        _style(ax)
    axes[0].set_ylabel("accuracy, 4 models pooled (±1 SE)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Same question types, same models, same N = T points: "
                 "non-chronological (OLD) vs chronological (NEW)", fontsize=12, y=1.04)
    fig.text(0.0, -0.10, NOTE, fontsize=8, color="#444")
    fig.tight_layout()
    _save(fig, out, "figC1_diagonal_old_vs_new")


def figC2(old, new, out):
    models = [m for m in ("qwen", "llama", "mistral", "olmo")
              if m in set(old.model_key) and m in set(new.model_key)]
    cats = list(dict.fromkeys(new.category))
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 4.2), sharey=True)
    axes = [axes] if len(models) == 1 else list(axes)
    x = range(len(cats))
    for ax, mk in zip(axes, models):
        go = old[old.model_key == mk].set_index("category").reindex(cats)
        gn = new[new.model_key == mk].set_index("category").reindex(cats)
        ax.bar([i - 0.3 for i in x], go.observed, width=0.28, color="#bbb", label="OLD observed")
        ax.bar([i - 0.3 for i in x], go.chance, width=0.28, facecolor="none",
               edgecolor="#555", hatch="////", lw=0.7, label="OLD chance")
        ax.bar([i + 0.05 for i in x], gn.observed, width=0.28, color=COLOURS.get(mk, "grey"),
               label="NEW observed")
        ax.bar([i + 0.05 for i in x], gn.chance, width=0.28, facecolor="none",
               edgecolor="#111", hatch="////", lw=0.7, label="NEW chance")
        ax.set_xticks([i - 0.12 for i in x])
        ax.set_xticklabels(cats, rotation=35, ha="right", fontsize=8)
        ax.set_title(f"{LABEL.get(mk, mk)}\nn wrong: OLD {int(go.n_wrong.iloc[0])}, "
                     f"NEW {int(gn.n_wrong.iloc[0])}", fontsize=9.5)
        _style(ax)
    axes[0].set_ylabel("share of wrong answers (N = T diagonal)")
    axes[0].legend(fontsize=7.5, frameon=False, ncol=2)
    fig.suptitle("Failure taxonomy, OLD vs NEW, per model — hatched = that experiment's "
                 "chance level", fontsize=12, y=1.03)
    fig.text(0.0, -0.08, NOTE, fontsize=8, color="#444")
    fig.tight_layout()
    _save(fig, out, "figC2_taxonomy_old_vs_new")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=OLD_DIR)
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    if not os.path.isdir(a.old):
        raise SystemExit(f"previous experiment's tables not found at {a.old}")
    new_t = os.path.join(ROOT, a.tables); out = os.path.join(ROOT, a.out)
    old_acc = pd.read_csv(os.path.join(a.old, "accuracy.csv"))
    new_acc = pd.read_csv(os.path.join(new_t, "accuracy.csv"))
    figC1(old_acc, new_acc, out); print("  figC1_diagonal_old_vs_new")
    old_tax = pd.read_csv(os.path.join(a.old, "taxonomy.csv"))
    new_tax = pd.read_csv(os.path.join(new_t, "taxonomy.csv"))
    figC2(old_tax, new_tax, out); print("  figC2_taxonomy_old_vs_new")


if __name__ == "__main__":
    main()
