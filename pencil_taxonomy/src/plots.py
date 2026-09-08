"""Four figures, all regenerable from the graded JSONL.

Two rules applied here and stated in every caption:

  * Overflow-emptied cells are DROPPED, never plotted as zero. What survives
    such a cell is the shortest stories and the easiest instances -- a biased
    subsample, not that model's accuracy at that N.
  * Figure 3 plots the observed bar and the chance bar SIDE BY SIDE rather than
    their difference. Showing both makes an over-firing null visible at a glance;
    a single corrected number hides it.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import pandas as pd                      # noqa: E402

from src.util import ROOT, load_config   # noqa: E402

COLOURS = {"qwen": "#1f77b4", "llama": "#d62728",
           "mistral": "#2ca02c", "olmo": "#ff7f0e"}
LABEL = {"qwen": "Qwen2.5-7B", "llama": "Llama-3.1-8B",
         "mistral": "Mistral-7B-v0.3", "olmo": "OLMo-2-7B"}
ORDER = ["initial_state_lookup", "transfer_recall", "person_timestep_lookup",
         "state_snapshot", "trajectory"]


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


def _keep(df, cfg):
    """Drop cells that overflow emptied past the coverage threshold."""
    return df[df.coverage >= cfg["plots"]["min_cell_coverage"]]


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
    fig.suptitle("Accuracy by question type — the two controls need no arithmetic; "
                 "types 3–5 need accumulation", fontsize=12, x=0.5, y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_accuracy_by_question_type.png"),
                dpi=160, bbox_inches="tight")
    fig.savefig(os.path.join(out, "fig1_accuracy_by_question_type.pdf"),
                bbox_inches="tight")
    plt.close(fig)


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
        ax.plot(gg.N, gg.accuracy, "o-", color=c, ms=4, lw=1.8,
                label=LABEL.get(mk, mk))
        ax.plot(gg.N, gg.strict, "--", color=c, lw=1.0, alpha=0.45)
    ax.set_xlabel("N = T"); ax.set_ylabel("pooled accuracy")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    ax.set_title("Pooled accuracy across all five question types", fontsize=12,
                 loc="left")
    fig.text(0.0, -0.10,
             "Dashed line = strict (byte-exact) match; the gap to the solid line is "
             "what normalisation rescues.\nThis pooled number is held up by the two "
             "CONTROL types and is NOT a measure of residual state-tracking skill "
             "— see figure 1.",
             fontsize=8, color="#444")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_accuracy_overall.png"), dpi=160,
                bbox_inches="tight")
    fig.savefig(os.path.join(out, "fig2_accuracy_overall.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig3(tax, out):
    if not len(tax):
        return
    models = sorted(tax.model_key.unique())
    cats = list(dict.fromkeys(tax.category))
    fig, axes = plt.subplots(1, len(models), figsize=(4.4 * len(models), 4.1),
                             sharey=True)
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
        ax.set_title(f"{LABEL.get(mk, mk)}\nn wrong = {int(g.n_wrong.iloc[0])}",
                     fontsize=10)
        _style(ax)
    axes[0].set_ylabel("share of wrong answers")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Failure taxonomy: observed vs chance, pooled over N",
                 fontsize=12, x=0.5, y=1.03)
    fig.text(0.0, -0.06,
             "Bars are shown side by side rather than as a difference: where the "
             "hatched chance bar meets or exceeds the observed bar, the null "
             "over-fires and\nthe corrected value is not interpretable. "
             "`unexplained` at its chance level is a result — it caps how much of "
             "the failure any mechanism can claim.",
             fontsize=8, color="#444")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_failure_taxonomy.png"), dpi=160,
                bbox_inches="tight")
    fig.savefig(os.path.join(out, "fig3_failure_taxonomy.pdf"), bbox_inches="tight")
    plt.close(fig)


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
    ax.annotate("perfect tracking (y = T)", xy=(tmax * 0.62, tmax * 0.68),
                color="grey", fontsize=9)
    ax.set_xlabel("T (timesteps)"); ax.set_ylabel("mean first divergence index")
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    ax.set_title("Where tracking first leaves the truth", fontsize=12, loc="left")
    n_ex = int(d.n_formulaic_excluded.sum() + d.n_unusable_excluded.sum())
    fig.text(0.0, -0.08,
             f"From the trajectory question. {n_ex} answers excluded: formulaic "
             "(a constant list or an arithmetic progression) or the wrong length "
             "— both\ndiverge at index 0–1 by construction. Cells with fewer than "
             "5 surviving answers are dropped.",
             fontsize=8, color="#444")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_first_divergence.png"), dpi=160,
                bbox_inches="tight")
    fig.savefig(os.path.join(out, "fig4_first_divergence.pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--out", default="results/figures")
    a = ap.parse_args()
    cfg = load_config()
    tdir = os.path.join(ROOT, a.tables)
    out = os.path.join(ROOT, a.out)
    os.makedirs(out, exist_ok=True)

    acc = pd.read_csv(os.path.join(tdir, "accuracy.csv"))
    dropped = acc[acc.coverage < cfg["plots"]["min_cell_coverage"]]
    if len(dropped):
        print(f"[plots] dropped {len(dropped)} overflow-emptied cell(s):")
        for r in dropped.itertuples():
            print(f"    {r.model_key} N={r.N} {r.question_type} "
                  f"coverage={r.coverage:.2f}")
    fig1(acc, cfg, out); print("  fig1_accuracy_by_question_type")
    fig2(acc, cfg, out); print("  fig2_accuracy_overall")
    p = os.path.join(tdir, "taxonomy.csv")
    if os.path.exists(p):
        fig3(pd.read_csv(p), out); print("  fig3_failure_taxonomy")
    p = os.path.join(tdir, "divergence.csv")
    if os.path.exists(p):
        fig4(pd.read_csv(p), out); print("  fig4_first_divergence")
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
    fig.suptitle("Failure taxonomy v2 (SECONDARY / EXPLORATORY) — observed vs chance",
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
