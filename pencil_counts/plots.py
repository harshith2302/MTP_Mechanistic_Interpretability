"""Figures.   python plots.py --out results/<model>     -> the four per-model figures
            python plots.py --compare llama qwen ... -> results/figures/fig5_models.png"""
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TASKS = {"people": "number of people", "total": "total number of pencils"}
MODEL = ""
T_COLOURS = {"0": "#1f77b4", "1": "#ff7f0e", "3": "#2ca02c", "5": "#d62728", "8": "#9467bd", "12": "#8c564b"}


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".png"), dpi=160, bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True); ax.set_ylim(0, 1.02)


def best_layer(r, task):
    v = r[(r.task == task) & (r.split == "val") & (r["T"] == "all")]
    return int(v.set_index("layer").accuracy.idxmax())


def at_best(r, task):
    return r[(r.task == task) & (r.split == "test") & (r["T"] == "all") & (r.layer == best_layer(r, task))].iloc[0]


def fig1_layers(r, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, task in zip(axes, TASKS):
        t = r[(r.task == task) & (r.split == "test")]
        for T, c in T_COLOURS.items():
            g = t[t["T"] == T].sort_values("layer")
            ax.plot(g.layer, g.accuracy, "-", color=c, lw=1.2, label=f"{T} timesteps")
        g = t[t["T"] == "all"].sort_values("layer")
        ax.plot(g.layer, g.accuracy, "-", color="black", lw=2.4, label="all stories")
        ax.axhline(g.baseline_majority.mean(), ls="--", color="grey", label="majority class")
        ax.axhline(g.baseline_shuffled.mean(), ls="--", color="red", alpha=0.6, label="shuffled labels")
        ax.set_title(f"{MODEL}: probe for the {TASKS[task]}"); ax.set_xlabel("layer (residual stream after the block)")
        tidy(ax)
    axes[0].set_ylabel("test accuracy"); axes[1].legend(fontsize=8, frameon=False, loc="lower right", ncol=2)
    save(fig, out, "fig1_layers")


def fig2_timesteps(r, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, task in zip(axes, TASKS):
        L = best_layer(r, task)
        t = r[(r.task == task) & (r.split == "test") & (r.layer == L) & (r["T"] != "all")].copy()
        t["T"] = t["T"].astype(int); t = t.sort_values("T")
        ax.plot(t["T"], t.accuracy, "o-", color="#1f77b4", label="probe")
        ax.plot(t["T"], t.model_accuracy, "s-", color="#d62728", label="model's own answer")
        ax.plot(t["T"], t.baseline_text, "^:", color="black", label="text-only features")
        ax.plot(t["T"], t.baseline_majority, "--", color="grey", label="majority class")
        ax.plot(t["T"], t.baseline_shuffled, "--", color="red", alpha=0.6, label="shuffled labels")
        ax.set_xticks(t["T"]); ax.set_xlabel("number of timesteps (transfers) in the story")
        ax.set_title(f"{MODEL}: {TASKS[task]}, probe at layer {L}"); tidy(ax)
    axes[0].set_ylabel("test accuracy"); axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    save(fig, out, "fig2_timesteps")


def fig3_probe_vs_model(r, out):
    fig, ax = plt.subplots(figsize=(7, 4))
    series = [("chance (majority class)", "baseline_majority", "grey"), ("text-only features", "baseline_text", "black"),
              ("model's own answer", "model_accuracy", "#d62728"), ("probe", "accuracy", "#1f77b4")]
    x, w = np.arange(len(TASKS)), 0.2
    for i, (label, col, colour) in enumerate(series):
        vals = [at_best(r, task)[col] for task in TASKS]
        ax.bar(x + (i - 1.5) * w, vals, w, color=colour, label=label)
        for xi, v in zip(x + (i - 1.5) * w, vals):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"{TASKS[t]}\n(probe at layer {best_layer(r, t)})" for t in TASKS])
    ax.set_ylabel("test accuracy"); tidy(ax); ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    ax.set_title(f"{MODEL}: is the answer linearly present, and does the model give it?", pad=24)
    save(fig, out, "fig3_probe_vs_model")


def fig4_confusion(r, p, out):
    L = best_layer(r, "total")
    t = p[(p.task == "total") & (p.layer == L)]
    classes = sorted(set(t.gold) | set(t.pred))
    M = np.zeros((len(classes), len(classes)))
    for g, q in zip(t.gold, t.pred):
        M[classes.index(g), classes.index(q)] += 1
    M /= M.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5, 4.4))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if M[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("probe prediction"); ax.set_ylabel("true total"); plt.colorbar(im, ax=ax, label="fraction of row")
    ax.set_title(f"{MODEL}: total-pencils probe, layer {L}, test set (n = {len(t)})", fontsize=10)
    save(fig, out, "fig4_confusion_total")


def fig5_models(models, out):
    """All models on one pair of axes: the total probe against depth, and against
    story length with each model's own answer beside it."""
    colours = dict(zip(models, ["#1f77b4", "#e07b39", "#2e8b57", "#7b52ab"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for m in models:
        r = pd.read_csv(os.path.join("results", m, "results.csv"), dtype={"T": str})
        t = r[(r.task == "total") & (r.split == "test")]
        g = t[t["T"] == "all"].sort_values("layer")
        axes[0].plot(g.layer / g.layer.max(), g.accuracy, "-", color=colours[m], lw=2, label=m)
        L = best_layer(r, "total")
        d = t[(t.layer == L) & (t["T"] != "all")].copy(); d["T"] = d["T"].astype(int); d = d.sort_values("T")
        axes[1].plot(d["T"], d.accuracy, "o-", ms=4, color=colours[m], label=f"{m}, probe (layer {L})")
        axes[1].plot(d["T"], d.model_accuracy, "s--", ms=4, mfc="none", color=colours[m], label=f"{m}, own answer")
    for ax, xl, ti in ((axes[0], "depth (layer / last layer)", "the total, probed at every depth"),
                       (axes[1], "number of timesteps (transfers) in the story", "the total, at each model's best layer")):
        ax.axhline(0.201, ls="--", color="grey", lw=1, label="chance")
        ax.set_xlabel(xl); ax.set_title(ti, fontsize=11); tidy(ax)
    axes[0].set_ylabel("test accuracy"); axes[0].legend(fontsize=9, frameon=False, loc="lower right")
    axes[1].legend(fontsize=7.5, frameon=False, ncol=2, loc="lower left")
    fig.suptitle("four models, the same 12,000 stories: probe (solid) and the model's own answer (dashed)",
                 fontsize=12, y=1.02)
    save(fig, out, "fig5_models")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--compare", nargs="*", default=None, help="model keys; writes results/figures/fig5_models")
    a = ap.parse_args()
    if a.compare:
        out = os.path.join("results", "figures"); os.makedirs(out, exist_ok=True)
        fig5_models(a.compare, out)
        print(f"[plots] fig5_models -> {out}/  ({', '.join(a.compare)})")
        return
    r = pd.read_csv(os.path.join(a.out, "results.csv"), dtype={"T": str})
    p = pd.read_csv(os.path.join(a.out, "predictions.csv"))
    out = os.path.join(a.out, "figures"); os.makedirs(out, exist_ok=True)
    global MODEL; MODEL = os.path.basename(os.path.normpath(a.out))
    fig1_layers(r, out); fig2_timesteps(r, out); fig3_probe_vs_model(r, out); fig4_confusion(r, p, out)
    print(f"[plots] 4 figures -> {out}/  (best layers: people {best_layer(r, 'people')}, total {best_layer(r, 'total')})")


if __name__ == "__main__":
    main()
