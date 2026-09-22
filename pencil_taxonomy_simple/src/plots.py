"""Figures, all regenerable from the graded JSONL.

fig1-fig4 are the N = T DIAGONAL, drawn with the same construction, colours
and filenames as pencil_taxonomy's figures so the two experiments can be laid
side by side: the only difference between them is chronological narration
(plus the prompt, if config says so).

fig5-fig7 are the (N, T) GRID: accuracy heatmaps per question type, per model,
and the failure-taxonomy share per cell. On a heatmap, N is the y-axis and T is
the x-axis; a cell that context overflow emptied is masked grey, never drawn as
zero.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

from src.taxonomy import CATEGORIES      # noqa: E402
from src.util import ROOT, load_config   # noqa: E402

COLOURS = {"qwen": "#1f77b4", "llama": "#d62728",
           "mistral": "#2ca02c", "olmo": "#ff7f0e"}
LABEL = {"qwen": "Qwen2.5-7B", "llama": "Llama-3.1-8B",
         "mistral": "Mistral-7B-v0.3", "olmo": "OLMo-2-7B"}
ORDER = ["initial_state_lookup", "transfer_recall", "person_timestep_lookup",
         "state_snapshot", "trajectory"]
SUBTITLE = "chronological narration, no clusters"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


def _keep(df, cfg):
    return df[df.coverage >= cfg["plots"]["min_cell_coverage"]]


def _save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".png"), dpi=160, bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


# --- diagonal: same as pencil_taxonomy ---------------------------------------

def fig1(acc, cfg, out):
    d = _keep(acc, cfg)
    fig, axes = plt.subplots(1, 5, figsize=(19, 3.9), sharey=True)
    for ax, qt in zip(axes, ORDER):
        g = d[d.question_type == qt]
        for mk, gg in g.groupby("model_key"):
            gg = gg.sort_values("N")
            c = COLOURS.get(mk, "grey")
            ax.plot(gg.N, gg.accuracy, "o-", color=c, ms=3.5, lw=1.6,
                    label=LABEL.get(mk, mk))
            ax.fill_between(gg.N, gg.se_low, gg.se_high, color=c, alpha=0.15, lw=0)
        tag = " (CONTROL)" if qt in ORDER[:2] else ""
        ax.set_title(f"{qt}{tag}\nn = 30 per point", fontsize=9)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("N = T")
        _style(ax)
    axes[0].set_ylabel("accuracy (±1 SE)")
    axes[0].legend(fontsize=7.5, frameon=False)
    fig.suptitle(f"Accuracy by question type on the N = T diagonal — {SUBTITLE}",
                 fontsize=12, x=0.5, y=1.04)
    fig.tight_layout()
    _save(fig, out, "fig1_accuracy_by_question_type")


def fig2(acc, cfg, out):
    d = _keep(acc, cfg)
    g = d.groupby(["model_key", "N"]).agg(
        n_correct=("n_correct", "sum"), n_valid=("n_valid", "sum"),
        strict=("accuracy_strict", "mean")).reset_index()
    g["accuracy"] = g.n_correct / g.n_valid
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for mk, gg in g.groupby("model_key"):
        gg = gg.sort_values("N")
        c = COLOURS.get(mk, "grey")
        ax.plot(gg.N, gg.accuracy, "o-", color=c, ms=4, lw=1.8, label=LABEL.get(mk, mk))
        ax.plot(gg.N, gg.strict, "--", color=c, lw=1.0, alpha=0.45)
    ax.set_xlabel("N = T"); ax.set_ylabel("pooled accuracy")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    ax.set_title(f"Pooled accuracy, N = T diagonal — {SUBTITLE}", fontsize=12, loc="left")
    fig.text(0.0, -0.10,
             "Dashed = strict (byte-exact) match. This pooled number is held up by "
             "the two CONTROL types and is NOT a measure of\nresidual state-tracking "
             "skill — see figure 1.", fontsize=8, color="#444")
    fig.tight_layout()
    _save(fig, out, "fig2_accuracy_overall")


def fig3(tax, out, name="fig3_failure_taxonomy", scope="N = T diagonal"):
    if not len(tax):
        return
    models = sorted(tax.model_key.unique())
    cats = list(dict.fromkeys(tax.category))
    fig, axes = plt.subplots(1, len(models), figsize=(4.4 * len(models), 4.1), sharey=True)
    axes = [axes] if len(models) == 1 else list(axes)
    x = range(len(cats))
    for ax, mk in zip(axes, models):
        g = tax[tax.model_key == mk].set_index("category").reindex(cats)
        ax.bar([i - 0.2 for i in x], g.observed, width=0.4,
               color=COLOURS.get(mk, "grey"), label="observed")
        ax.bar([i + 0.2 for i in x], g.chance, width=0.4, facecolor="none",
               edgecolor="#333", hatch="////", lw=0.8, label="chance")
        ax.set_xticks(list(x))
        ax.set_xticklabels(cats, rotation=35, ha="right", fontsize=8)
        ax.set_title(f"{LABEL.get(mk, mk)}\nn wrong = {int(g.n_wrong.iloc[0])}", fontsize=10)
        _style(ax)
    axes[0].set_ylabel("share of wrong answers")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Failure taxonomy, {scope}: observed vs chance — {SUBTITLE}",
                 fontsize=12, x=0.5, y=1.03)
    fig.text(0.0, -0.06,
             "Bars side by side rather than as a difference: where the hatched chance "
             "bar meets or exceeds the observed bar, the null over-fires and\nthe "
             "corrected value is not interpretable. `unexplained` at its chance level "
             "is a result — it caps how much of the failure any mechanism can claim.",
             fontsize=8, color="#444")
    fig.tight_layout()
    _save(fig, out, name)


def fig4(div, out):
    d = div[div.kept] if "kept" in div.columns else div
    if not len(d):
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for mk, g in d.groupby("model_key"):
        g = g.sort_values("T")
        ax.plot(g["T"], g.mean_first_divergence, "o-", ms=4, lw=1.8,
                color=COLOURS.get(mk, "grey"), label=LABEL.get(mk, mk))
    tmax = max(d["T"])
    ax.plot([2, tmax], [2, tmax], "--", color="grey", lw=1.2)
    ax.annotate("perfect tracking (y = T)", xy=(tmax * 0.62, tmax * 0.68), color="grey", fontsize=9)
    ax.set_xlabel("T (timesteps), N = T"); ax.set_ylabel("mean first divergence index")
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    ax.set_title(f"Where tracking first leaves the truth — {SUBTITLE}", fontsize=12, loc="left")
    n_ex = int(d.n_formulaic_excluded.sum() + d.n_unusable_excluded.sum())
    fig.text(0.0, -0.08,
             f"From the trajectory question. {n_ex} answers excluded: formulaic "
             "(constant or arithmetic progression) or the wrong length — both\n"
             "diverge at index 0–1 by construction. Cells with fewer than 5 surviving "
             "answers are dropped.", fontsize=8, color="#444")
    fig.tight_layout()
    _save(fig, out, "fig4_first_divergence")


# --- grid: heatmaps -----------------------------------------------------------

def _grid(df, value, cfg):
    """(N x T) matrix of `value`, NaN where the cell is missing or overflow-emptied.

    Columns are the grid's t_values only: the diagonal-only cells (N = T at T
    values not in the grid) exist for figures 1-4 and would otherwise show up
    as near-empty columns here.
    """
    df = df.copy()
    if "coverage" in df.columns and "plots" in cfg:
        df.loc[df.coverage < cfg["plots"]["min_cell_coverage"], value] = np.nan
    g = cfg.get("grid", {})
    ns = g.get("n_values") or sorted(df.N.unique())
    ts = g.get("t_values") or sorted(df["T"].unique())
    piv = df.pivot_table(index="N", columns="T", values=value, aggfunc="first")
    return piv.reindex(index=ns, columns=ts), ns, ts


def _heat(ax, mat, ns, ts, title, vmin=0, vmax=1, cmap="viridis"):
    m = np.ma.masked_invalid(mat.values.astype(float))
    cm = plt.get_cmap(cmap).copy(); cm.set_bad("#d9d9d9")
    # T is not evenly spaced, so draw cells by index and label them by value.
    im = ax.imshow(m, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap=cm,
                   extent=[-0.5, len(ts) - 0.5, ns[0] - 0.5, ns[-1] + 0.5])
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel("T (transfers)")
    step = 5 if len(ts) > 12 else 1
    idx = list(range(len(ts))) if len(ts) <= 12 else [i for i, t in enumerate(ts) if t % step == 0 or i == 0]
    ax.set_xticks(idx); ax.set_xticklabels([ts[i] for i in idx])
    ax.set_yticks([n for n in ns if n % 5 == 0 or n == ns[0]])
    # mark the N = T cells that fall on the grid
    for i, t in enumerate(ts):
        if t in ns:
            ax.plot(i, t, marker="s", ms=3, mfc="none", mec="white", mew=0.7, alpha=0.8)
    return im


def fig5(accg, cfg, out):
    """Accuracy over the (N, T) grid, one panel per question type + pooled."""
    d = accg[accg.model_key == "ALL"]
    fig, axes = plt.subplots(1, 6, figsize=(24, 4.3))
    for ax, qt in zip(axes, ORDER + ["ALL"]):
        g = d[d.question_type == qt]
        mat, ns, ts = _grid(g, "accuracy", cfg)
        im = _heat(ax, mat, ns, ts,
                   ("all five types pooled" if qt == "ALL" else qt)
                   + (" (CONTROL)" if qt in ORDER[:2] else ""))
        ax.set_ylabel("N (people)" if ax is axes[0] else "")
    fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01, label="accuracy")
    fig.suptitle(f"Accuracy over the (N, T) grid, all four models pooled — {SUBTITLE}",
                 fontsize=12, y=1.02)
    fig.text(0.0, -0.06,
             "Each cell is 5 stories per model (20 answers per type pooled, 100 for "
             "the pooled panel); N = T cells have 30. Grey = overflow-emptied cell "
             "(masked, never 0).\nWhite squares mark the N = T cells drawn in "
             "figures 1–4. T is the number of transfers: exactly one per timestep.",
             fontsize=8, color="#444")
    _save(fig, out, "fig5_accuracy_grid")


def fig6(accg, cfg, out):
    """Pooled-over-types accuracy per model."""
    d = accg[(accg.question_type == "ALL") & (accg.model_key != "ALL")]
    models = [m for m in ("qwen", "llama", "mistral", "olmo") if m in set(d.model_key)]
    fig, axes = plt.subplots(1, len(models), figsize=(4.4 * len(models), 4.3))
    axes = [axes] if len(models) == 1 else list(axes)
    for ax, mk in zip(axes, models):
        mat, ns, ts = _grid(d[d.model_key == mk], "accuracy", cfg)
        im = _heat(ax, mat, ns, ts, LABEL.get(mk, mk))
        ax.set_ylabel("N (people)" if ax is axes[0] else "")
    fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01, label="accuracy (5 types pooled)")
    fig.suptitle(f"Pooled accuracy per model over the (N, T) grid — {SUBTITLE}",
                 fontsize=12, y=1.02)
    fig.text(0.0, -0.06,
             "25 answers per cell (5 stories x 5 types), 150 on the N = T cells. Grey = "
             "context overflow (OLMo-2's 4096-token window), masked rather than scored 0.",
             fontsize=8, color="#444")
    _save(fig, out, "fig6_accuracy_grid_by_model")


def fig7(taxg, cfg, out, min_wrong=4):
    """Failure taxonomy over the grid: share of each category per (N, T) cell."""
    if not len(taxg):
        return
    d = taxg.copy()
    fig, axes = plt.subplots(1, len(CATEGORIES) + 1, figsize=(4.0 * (len(CATEGORIES) + 1), 4.3))
    for ax, c in zip(axes, CATEGORIES):
        g = d.copy()
        g.loc[g.n_wrong < min_wrong, f"{c}_share"] = np.nan
        mat, ns, ts = _grid(g, f"{c}_share", {"grid": cfg["grid"]})
        im = _heat(ax, mat, ns, ts, c, cmap="magma")
        ax.set_ylabel("N (people)" if ax is axes[0] else "")
    mat, ns, ts = _grid(d, "n_wrong", {"grid": cfg["grid"]})
    vmax = float(np.nanmax(mat.values))
    _heat(axes[-1], mat, ns, ts, f"n labelled wrong answers (white = 0, black = {vmax:.0f})",
          vmin=0, vmax=vmax, cmap="Greys")
    fig.colorbar(im, ax=list(axes), fraction=0.012, pad=0.01, label="share of wrong answers")
    fig.suptitle(f"Failure taxonomy over the (N, T) grid, all models pooled — {SUBTITLE}",
                 fontsize=12, y=1.02)
    fig.text(0.0, -0.07,
             "person_timestep_lookup answers plus the first divergent trajectory element. "
             f"Cells with fewer than {min_wrong} labelled wrong answers are masked.\n"
             "Observed shares only — the chance null is reported on the diagonal "
             "(figure 3) and pooled over the grid (taxonomy_pooled_grid.csv), not per cell.",
             fontsize=8, color="#444")
    _save(fig, out, "fig7_taxonomy_grid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    cfg = load_config()
    tdir = os.path.join(ROOT, a.tables)
    out = os.path.join(ROOT, a.out)
    os.makedirs(out, exist_ok=True)

    def table(name):
        p = os.path.join(tdir, name)
        return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()

    acc = table("accuracy.csv")
    if len(acc):
        dropped = acc[acc.coverage < cfg["plots"]["min_cell_coverage"]]
        if len(dropped):
            print(f"[plots] dropped {len(dropped)} overflow-emptied diagonal cell(s)")
        fig1(acc, cfg, out); print("  fig1_accuracy_by_question_type")
        fig2(acc, cfg, out); print("  fig2_accuracy_overall")
    tax = table("taxonomy.csv")
    if len(tax):
        fig3(tax, out); print("  fig3_failure_taxonomy")
    pooled = table("taxonomy_pooled_grid.csv")
    if len(pooled):
        fig3(pooled, out, "fig3g_failure_taxonomy_grid_pooled", "whole (N, T) grid")
        print("  fig3g_failure_taxonomy_grid_pooled")
    div = table("divergence.csv")
    if len(div):
        fig4(div, out); print("  fig4_first_divergence")
    accg = table("accuracy_grid.csv")
    if len(accg):
        fig5(accg, cfg, out); print("  fig5_accuracy_grid")
        fig6(accg, cfg, out); print("  fig6_accuracy_grid_by_model")
    taxg = table("taxonomy_grid.csv")
    if len(taxg):
        fig7(taxg, cfg, out); print("  fig7_taxonomy_grid")
    print(f"[plots] wrote to {out}")


if __name__ == "__main__":
    main()


# --- v2 (secondary / exploratory) -------------------------------------------
FAMILY_NAME = {"A": "A · update", "B": "B · arithmetic", "C": "C · reference",
               "E": "E · residual"}


def fig3b(rates, out):
    """Same construction as fig3, grouped by family with visual separation."""
    models = sorted(rates.model_key.unique())
    # NB: bracket indexing, not `.observed` -- `observed` is also a pandas
    # GroupBy parameter, so attribute access silently returns a bool.
    order = (rates[rates.family != "E"].groupby(["family", "category"])
             ["observed"].mean().reset_index()
             .sort_values(["family", "observed"], ascending=[True, False]))
    cats = list(order.category) + ["unexplained"]
    fams = list(order.family) + ["E"]
    bounds = [i for i in range(1, len(cats)) if fams[i] != fams[i - 1]]

    fig, axes = plt.subplots(len(models), 1, figsize=(13, 3.1 * len(models)),
                             sharex=True)
    axes = [axes] if len(models) == 1 else list(axes)
    x = range(len(cats))
    for ax, mk in zip(axes, models):
        g = rates[rates.model_key == mk].set_index("category").reindex(cats)
        ax.bar([i - 0.2 for i in x], g.observed, width=0.4,
               color=COLOURS.get(mk, "grey"), label="observed")
        ax.bar([i + 0.2 for i in x], g.chance, width=0.4, facecolor="none",
               edgecolor="#333", hatch="////", lw=0.8, label="chance")
        for b in bounds:
            ax.axvline(b - 0.5, color="#bbb", lw=0.9)
        ax.set_ylabel(LABEL.get(mk, mk), fontsize=9)
        _style(ax)
    axes[0].legend(fontsize=8, frameon=False, ncol=2)
    axes[-1].set_xticks(list(x))
    axes[-1].set_xticklabels(cats, rotation=40, ha="right", fontsize=8)
    for b, name in zip([0] + bounds, [FAMILY_NAME.get(fams[0])] +
                       [FAMILY_NAME.get(fams[i]) for i in bounds]):
        axes[0].text(b, axes[0].get_ylim()[1] * 0.94, name, fontsize=8.5,
                     color="#555")
    fig.suptitle(f"Failure taxonomy v2 (SECONDARY / EXPLORATORY), N = T diagonal — {SUBTITLE}",
                 fontsize=12, x=0.5, y=1.0)
    fig.text(0.0, -0.02,
             "v1 (figure 3) is the pre-registered, primary taxonomy. v2 was "
             "specified after the data existed.\nWith 15 rules over a small "
             "integer range the null labels 78% of RANDOM draws, so these "
             "corrected values are differences of large numbers — read them "
             "with heavy scepticism.",
             fontsize=8, color="#444")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3b_taxonomy_v2.png"), dpi=160,
                bbox_inches="tight")
    fig.savefig(os.path.join(out, "fig3b_taxonomy_v2.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig3c(fams, out):
    """Family rollup — the level at which the counts are actually solid."""
    models = sorted(fams.model_key.unique())
    cats = ["A", "B", "C"]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    w = 0.8 / len(models)
    for j, mk in enumerate(models):
        g = fams[fams.model_key == mk].set_index("family").reindex(cats)
        pos = [i - 0.4 + w * (j + 0.5) for i in range(len(cats))]
        ax.bar(pos, g.observed, width=w * 0.92, color=COLOURS.get(mk, "grey"),
               label=LABEL.get(mk, mk))
        ax.bar(pos, g.chance, width=w * 0.92, facecolor="none",
               edgecolor="#333", hatch="////", lw=0.7)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels([FAMILY_NAME[c] for c in cats])
    ax.set_ylabel("share of classified wrong answers")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    _style(ax)
    ax.set_title("v2 family rollup (hatched = chance)", fontsize=12, loc="left")
    fig.text(0.0, -0.07,
             "Look here before the per-category figure: at 15 categories the "
             "counts only really support the family level.\nOnly family A "
             "(update failures) clears its null.",
             fontsize=8, color="#444")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3c_family_rollup.png"), dpi=160,
                bbox_inches="tight")
    fig.savefig(os.path.join(out, "fig3c_family_rollup.pdf"), bbox_inches="tight")
    plt.close(fig)


def main_v2():
    cfg = load_config()
    tdir = os.path.join(ROOT, "results/tables")
    out = os.path.join(ROOT, "results/figures")
    os.makedirs(out, exist_ok=True)
    fig3b(pd.read_csv(os.path.join(tdir, "taxonomy_v2_rates.csv")), out)
    print("  fig3b_taxonomy_v2")
    fig3c(pd.read_csv(os.path.join(tdir, "taxonomy_v2_families.csv")), out)
    print("  fig3c_family_rollup")
