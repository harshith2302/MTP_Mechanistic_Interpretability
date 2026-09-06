"""Five figures. No more.

The rule that makes or breaks the headline figure: a model's line ENDS where its
context runs out. OLMo-2 (4096 tokens) overflows from N~=16 for the heavy
question types and entirely by N~=26. Drawing those points as 0% accuracy would
make figure 1 flatly wrong, so overflow points are dropped, not zeroed, and the
truncation is annotated on the plot.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# one consistent colour per model across all five figures
COLOURS = {
    "Qwen2.5-7B-Instruct": "#3b6ea5",
    "Llama-3.1-8B-Instruct": "#c1666b",
    "Mistral-7B-Instruct-v0.3": "#4f9d69",
    "OLMo-2-1124-7B-Instruct": "#d4a017",
}
CATEGORY_ORDER = ["Binding", "Arithmetic", "Digit", "Omission", "Over-application",
                  "Direction", "Referent", "Temporal", "Conservation", "Unexplained"]
MIN_VALID = 5          # do not plot a point backed by fewer than this many records


def _style(ax, xlabel="N = T (people = timesteps)", ylabel=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def _save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] {name}.png / .pdf")


def _truncation_note(ax, acc, model):
    """Annotate where a model's line stops because of context overflow."""
    ov = acc[(acc.model == model) & (acc.n_overflow > 0)]
    if ov.empty:
        return
    first = int(ov["n"].min())
    ax.axvline(first, color=COLOURS.get(model, "grey"), ls=":", lw=1, alpha=0.7)
    ax.annotate(f"{model.split('-')[0]} context limit\n(first overflow at N={first};\n"
                f"overflow excluded, not scored 0)",
                xy=(first, 0.05), xytext=(first + 0.6, 0.16), fontsize=7,
                color=COLOURS.get(model, "grey"))


def _declutter(labels, min_gap):
    """Push overlapping direct labels apart, keeping their vertical order.

    Lines here converge near zero at large N, so unadjusted end-labels land on
    top of each other and the figure becomes unreadable.
    """
    labels = sorted(labels, key=lambda t: t[0])
    for i in range(1, len(labels)):
        if labels[i][0] - labels[i - 1][0] < min_gap:
            labels[i] = (labels[i - 1][0] + min_gap, labels[i][1], labels[i][2])
    return labels


def fig1_headline(acc, out):
    d = acc[(acc.question_type == "ALL") & (acc.n_valid >= MIN_VALID)]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ends = []
    for model, g in d.groupby("model"):
        g = g.sort_values("n")
        c = COLOURS.get(model, None)
        ax.plot(g["n"], g["accuracy"], marker="o", ms=3.5, lw=1.8, color=c, label=model)
        ax.fill_between(g["n"], g["ci_low"], g["ci_high"], color=c, alpha=0.13, lw=0)
        last = g.iloc[-1]
        ends.append((float(last["accuracy"]), model, (float(last["n"]), float(last["accuracy"]))))
        _truncation_note(ax, acc[acc.question_type == "ALL"], model)
    xmax = d["n"].max()
    for y, model, (x0, y0) in _declutter(ends, 0.045):
        c = COLOURS.get(model)
        ax.annotate(model, xy=(x0, y0), xytext=(xmax + 1.2, y), fontsize=8,
                    color=c, va="center",
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.6, alpha=0.55,
                                    shrinkA=2, shrinkB=2))
    _style(ax, ylabel="accuracy (95% Wilson CI)")
    ax.set_ylim(0, 1)
    ax.set_xticks([n for n in sorted(d["n"].unique()) if n % 2 == 0])
    ax.set_title("Pencil Exchange: accuracy collapses as N = T grows", loc="left")
    ax.set_xlim(right=xmax + 9)
    _save(fig, out, "fig1_accuracy_headline")


def fig2_by_question_type(acc, out):
    d = acc[(acc.question_type != "ALL") & (acc.n_valid >= MIN_VALID)]
    types = sorted(d["question_type"].unique())
    ncol = 4
    nrow = -(-len(types) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.5 * nrow),
                             sharex=True, sharey=True)
    for ax, qtype in zip(axes.flat, types):
        for model, g in d[d.question_type == qtype].groupby("model"):
            g = g.sort_values("n")
            ax.plot(g["n"], g["accuracy"], lw=1.5, marker="o", ms=2.5,
                    color=COLOURS.get(model), label=model)
        ax.set_title(qtype, fontsize=8, loc="left")
        ax.set_ylim(0, 1)
        _style(ax, xlabel="", ylabel="")
    for ax in axes.flat[len(types):]:
        ax.set_visible(False)
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Accuracy vs N = T by question type "
                 "(initial_state_lookup and transfer_recall are the controls)",
                 fontsize=10, x=0.01, ha="left")
    fig.supxlabel("N = T", fontsize=9)
    fig.tight_layout()
    _save(fig, out, "fig2_by_question_type")


def fig3_failure_composition(tax, out):
    models = sorted(tax["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.4),
                             sharey=True, squeeze=False)
    cmap = plt.get_cmap("tab10")
    colours = {c: cmap(i % 10) for i, c in enumerate(CATEGORY_ORDER)}
    for ax, model in zip(axes[0], models):
        g = tax[tax.model == model]
        piv = g.pivot_table(index="n", columns="coarse_category",
                            values="corrected_rate", fill_value=0.0)
        cols = [c for c in CATEGORY_ORDER if c in piv.columns]
        bottom = None
        for c in cols:
            ax.bar(piv.index, piv[c].clip(lower=0), bottom=bottom, width=0.85,
                   color=colours[c], label=c, linewidth=0)
            bottom = piv[c].clip(lower=0) if bottom is None else bottom + piv[c].clip(lower=0)
        ax.set_title(model, fontsize=8, loc="left")
        _style(ax, xlabel="N = T")
    axes[0][0].set_ylabel("chance-corrected share of wrong answers")
    # categories that fire BELOW chance are clipped at 0, so bars do not sum to 1
    axes[0][0].text(0.0, -0.30, "Bars are observed minus chance; categories firing "
                    "below chance are clipped at 0, so bars need not sum to 1.",
                    transform=axes[0][0].transAxes, fontsize=7, color="0.35")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, -0.10))
    fig.suptitle("Failure composition, chance-corrected "
                 "(format errors excluded and reported separately)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, out, "fig3_failure_composition")


def fig4_divergence(div, out):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ns = sorted(div["n"].unique())
    ax.plot(ns, ns, ls="--", lw=1, color="0.55")
    ax.annotate("perfect tracking (y = T)", xy=(ns[-1], ns[-1]),
                xytext=(-6, 4), textcoords="offset points", fontsize=8,
                color="0.4", ha="right")
    for model, g in div.groupby("model"):
        g = g.sort_values("n")
        ax.plot(g["n"], g["mean_first_divergence"], marker="o", ms=3.5, lw=1.8,
                color=COLOURS.get(model), label=model)
    _style(ax, xlabel="T (timesteps)", ylabel="mean first divergence index")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Where state tracking breaks, from the trajectory question",
                 loc="left")
    _save(fig, out, "fig4_first_divergence")


def fig5_diagnostics(acc, out):
    d = acc[(acc.question_type == "ALL") & (acc.n_valid >= MIN_VALID)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 3.6), sharex=True)
    for model, g in d.groupby("model"):
        g = g.sort_values("n")
        c = COLOURS.get(model)
        a1.plot(g["n"], g["format_error_rate"], marker="o", ms=3, lw=1.6,
                color=c, label=model)
        a2.plot(g["n"], g["ambiguous_rate"], marker="o", ms=3, lw=1.6, color=c)
    a1.set_title("format error rate", fontsize=9, loc="left")
    a2.set_title("taxonomy ambiguity rate (>1 label matched)", fontsize=9, loc="left")
    for ax in (a1, a2):
        _style(ax)
        ax.set_ylim(0, 1)
    a1.set_ylabel("rate")
    a1.legend(fontsize=8, frameon=False)
    fig.suptitle("Validity diagnostics", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    _save(fig, out, "fig5_diagnostics")


def main():
    ap = argparse.ArgumentParser(description="Build the five figures.")
    ap.add_argument("--tables", default=os.path.join(ROOT, "results", "tables"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "figures"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    acc = pd.read_csv(os.path.join(args.tables, "accuracy_by_model_n.csv"))
    fig1_headline(acc, args.out)
    fig2_by_question_type(acc, args.out)
    fig5_diagnostics(acc, args.out)

    tax_path = os.path.join(args.tables, "taxonomy_by_model_n.csv")
    if os.path.exists(tax_path):
        tax = pd.read_csv(tax_path)
        if not tax.empty:
            fig3_failure_composition(tax, args.out)

    div_path = os.path.join(args.tables, "divergence_by_model_n.csv")
    if os.path.exists(div_path):
        div = pd.read_csv(div_path)
        if not div.empty:
            fig4_divergence(div, args.out)
    print(f"[plots] wrote to {args.out}")


if __name__ == "__main__":
    main()
