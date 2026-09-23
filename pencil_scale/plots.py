"""Figures.   python plots.py --out results/<model>       the four per-model figures
            python plots.py --compare llama qwen ...   -> results/figures/fig5_models.png"""
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

POS_LABEL = {"first_token": "first token", "allocation_end": "end of the allocations",
             "last_transfer": "end of the last transfer", "question_end": "end of the question",
             "answer": "the answer position"}
PROTO_LABEL = {"random": "random split", "heldout": "totals never trained on",
               "extrapolate": "totals above the training range"}
COL = {"random": "#1f77b4", "heldout": "#e07b39", "extrapolate": "#7b52ab"}


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".png"), dpi=160, bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)


def fig1_map(r, positions, out):
    """Where the total can be read: mean absolute error over position x layer."""
    d = r[r.protocol == "random"]
    M = np.array([[d[(d.position == p) & (d.layer == l)].probe_mae.mean() for l in sorted(d.layer.unique())]
                  for p in positions])
    fig, ax = plt.subplots(figsize=(11, 3.4))
    im = ax.imshow(M, aspect="auto", cmap="viridis_r", origin="lower")
    ax.set_yticks(range(len(positions))); ax.set_yticklabels([POS_LABEL[p] for p in positions])
    ax.set_xlabel("layer (residual stream after the block)")
    i, j = np.unravel_index(np.nanargmin(M), M.shape)
    ax.plot(j, i, "o", ms=9, mfc="none", mec="white", mew=2)
    ha, dx = ("left", 9) if j < M.shape[1] * 0.6 else ("right", -9)
    ax.annotate(f"best: layer {j}, MAE {M[i, j]:.2f}", xy=(j, i), xytext=(dx, 9),
                textcoords="offset points", color="white", fontsize=9, ha=ha)
    plt.colorbar(im, ax=ax, label="mean absolute error (pencils)")
    ax.set_title("how far off a linear read-out of the total is, at every reading position and layer",
                 fontsize=11, loc="left")
    save(fig, out, "fig1_position_layer_map")


def fig2_layers(r, positions, out):
    """The same numbers as curves, with the baselines that matter."""
    d = r[r.protocol == "random"]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    cols = plt.get_cmap("viridis")(np.linspace(0.08, 0.86, len(positions)))
    for c, p in zip(cols, positions):
        g = d[d.position == p].sort_values("layer")
        ax.plot(g.layer, g.probe_mae, "-", lw=2, color=c, label=POS_LABEL[p])
    ax.axhline(d.mean_mae.iloc[0], ls="--", color="grey", lw=1.2, label="predict the mean total")
    ax.axhline(d.text_mae.iloc[0], ls=":", color="black", lw=1.2, label="text-only features")
    ax.axhline(d.model_mae.iloc[0], ls="-.", color="#c0392e", lw=1.4, label="the model's own answer")
    ax.set_xlabel("layer"); ax.set_ylabel("mean absolute error (pencils)"); tidy(ax)
    ax.set_ylim(0, max(d.mean_mae.iloc[0], d.model_mae.iloc[0]) * 1.15)
    ax.legend(fontsize=8.5, frameon=False, ncol=2)
    ax.set_title("random split: lower is better", fontsize=11, loc="left")
    save(fig, out, "fig2_layers")


def fig3_scatter(p, r, out):
    """Predicted against true, per protocol -- the test of whether it is a number."""
    protos = [x for x in ("random", "heldout", "extrapolate") if (p.protocol == x).any()]
    lim = [min(p.gold.min(), p.pred.min()) - 2, max(p.gold.max(), p.pred.max()) + 2]
    fig, axes = plt.subplots(1, len(protos), figsize=(4.2 * len(protos), 4.1), sharey=True)
    for ax, proto in zip(np.atleast_1d(axes), protos):
        d = p[p.protocol == proto]
        ax.plot(lim, lim, "-", color="grey", lw=1, zorder=1)
        ax.scatter(d.gold + np.random.default_rng(0).uniform(-.3, .3, len(d)), d.pred,
                   s=7, alpha=0.35, color=COL[proto], edgecolors="none", zorder=2)
        mean_by = d.groupby("gold").pred.mean()
        ax.plot(mean_by.index, mean_by.values, "o-", ms=3.5, color="black", lw=1.2, zorder=3,
                label="mean prediction")
        row = r[(r.protocol == proto) & (r.position == r[r.protocol == proto].best_position.iloc[0])
                & (r.layer == r[r.protocol == proto].best_layer.iloc[0])].iloc[0]
        ax.set_title(f"{PROTO_LABEL[proto]}\nMAE {row.probe_mae:.2f}, R² {row.probe_r2:.2f}", fontsize=10)
        ax.set_xlabel("true total"); tidy(ax); ax.set_ylim(lim); ax.set_xlim(lim)
    np.atleast_1d(axes)[0].set_ylabel("predicted total")
    np.atleast_1d(axes)[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("a read-out that lands on values it was never trained on is holding a magnitude, "
                 "not a list of answers", fontsize=11, y=1.03)
    save(fig, out, "fig3_predicted_vs_true")


def fig4_bars(r, out):
    """Linear against non-linear, against every baseline, per protocol."""
    protos = [x for x in ("random", "heldout", "extrapolate") if (r.protocol == x).any()]
    series = [("predict the mean", "mean_mae", "grey"), ("text-only features", "text_mae", "black"),
              ("shuffled labels", "shuffled_mae", "#c8c8c8"), ("the model's own answer", "model_mae", "#c0392e"),
              ("linear probe", "probe_mae", "#1f77b4"), ("MLP probe", "mlp_mae", "#2e8b57")]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x, w = np.arange(len(protos)), 0.14
    for i, (label, col, colour) in enumerate(series):
        vals = []
        for proto in protos:
            d = r[r.protocol == proto]
            best = d[(d.position == d.best_position.iloc[0]) & (d.layer == d.best_layer.iloc[0])].iloc[0]
            vals.append(best[col])
        ax.bar(x + (i - 2.5) * w, vals, w, color=colour, label=label)
        for xi, v in zip(x + (i - 2.5) * w, vals):
            ax.text(xi, v + 0.15, f"{v:.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([PROTO_LABEL[p] for p in protos])
    ax.set_ylabel("mean absolute error (pencils)"); tidy(ax)
    ax.legend(fontsize=8, frameon=False, ncol=3)
    ax.set_title("lower is better; the linear and MLP probes read the same layer and position",
                 fontsize=11, loc="left")
    save(fig, out, "fig4_probes_vs_baselines")


MODEL_COL = {"llama": "#d62728", "qwen": "#1f77b4", "mistral": "#2e8b57", "olmo": "#7b52ab"}


def fig5_models(models, cfg, out):
    """All models together: how the read-out error falls with depth at each
    model's best position, and how every protocol compares across models."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    protos = ["random", "heldout", "extrapolate"]
    bars = {p: [] for p in protos}
    for m in models:
        r = pd.read_csv(os.path.join("results", m, "results.csv"))
        d = r[r.protocol == "random"]
        pos = d.loc[d.probe_mae.idxmin(), "position"]
        g = d[d.position == pos].sort_values("layer")
        axes[0].plot(g.layer / g.layer.max(), g.probe_mae, "-", lw=2, color=MODEL_COL[m],
                     label=f"{m} ({POS_LABEL[pos]})")
        for p in protos:
            dp = r[r.protocol == p]
            b = dp[(dp.position == dp.best_position.iloc[0]) & (dp.layer == dp.best_layer.iloc[0])].iloc[0]
            bars[p].append((b.probe_mae, b.model_mae, b.mean_mae))
    axes[0].set_xlabel("depth (layer / last layer)"); axes[0].set_ylabel("mean absolute error (pencils)")
    axes[0].set_ylim(0, 14); tidy(axes[0]); axes[0].legend(fontsize=8.5, frameon=False)
    axes[0].set_title("random split, each model at its own best reading position", fontsize=10.5)
    x, w = np.arange(len(protos)), 0.8 / (len(models) + 1)
    for i, m in enumerate(models):
        axes[1].bar(x + (i - len(models) / 2) * w, [bars[p][i][0] for p in protos], w,
                    color=MODEL_COL[m], label=f"{m}, probe")
        axes[1].plot(x + (i - len(models) / 2) * w, [bars[p][i][1] for p in protos], "_",
                     ms=11, mew=2.4, color="black")
    axes[1].plot([], [], "_", ms=11, mew=2.4, color="black", label="the model's own answer")
    axes[1].axhline(bars["random"][0][2], ls="--", color="grey", lw=1, label="predict the mean")
    axes[1].set_xticks(x); axes[1].set_xticklabels(["random split", "totals never\ntrained on",
                                                    "totals above the\ntraining range"])
    axes[1].set_ylabel("mean absolute error (pencils)"); tidy(axes[1])
    axes[1].set_ylim(0, 19)
    axes[1].legend(fontsize=8, frameon=False, ncol=3, loc="upper left")
    axes[1].set_title("lower is better; bars are the linear probe", fontsize=10.5)
    names = {2: "two", 3: "three", 4: "four"}.get(len(models), str(len(models)))
    fig.suptitle(f"reading the unstated total out of {names} models, on the same 12,000 stories",
                 fontsize=12, y=1.02)
    save(fig, out, "fig5_models")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--compare", nargs="*", default=None, help="model keys; -> results/figures/")
    a = ap.parse_args()
    cfg = yaml.safe_load(open("config.yaml"))
    if a.compare:
        out = os.path.join("results", "figures"); os.makedirs(out, exist_ok=True)
        fig5_models(a.compare, cfg, out)
        print(f"[plots] fig5_models -> {out}/  ({', '.join(a.compare)})")
        return
    r = pd.read_csv(os.path.join(a.out, "results.csv"))
    p = pd.read_csv(os.path.join(a.out, "predictions.csv"))
    out = os.path.join(a.out, "figures"); os.makedirs(out, exist_ok=True)
    positions = [x for x in cfg["positions"] if x in set(r.position)]
    fig1_map(r, positions, out); fig2_layers(r, positions, out)
    fig3_scatter(p, r, out); fig4_bars(r, out)
    best = r[r.protocol == "random"].iloc[0]
    print(f"[plots] 4 figures -> {out}/  (best cell: {best.best_position}, layer {best.best_layer})")


if __name__ == "__main__":
    main()
