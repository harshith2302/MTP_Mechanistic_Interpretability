"""Ridge read-outs of the total at every (position, layer), under three split
protocols, plus a non-linear comparison.   python probe.py [--out results]

  random       70/15/15 by story -- how well the total can be read at all
  heldout      five totals never trained on -- does it interpolate to unseen values?
  extrapolate  train on totals <= extrapolation_train_max, test above -- is there
               a magnitude axis, or a lookup over values the probe has seen?

Writes <out>/results.csv (one row per position, layer and protocol) and
<out>/predictions.csv (per-prompt test predictions at the best cell of each).
"""
import argparse, json, os, re
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from num2words import num2words
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

NUMBER_WORDS = {num2words(v) for v in range(200)}


def text_features(story, n_tokens):
    """Surface counts only; the values of the numbers are deliberately absent."""
    words = re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)?", story)
    names = {w for w in words if w[0].isupper() and w not in ("The", "At")}
    numeric = sum(w.lower() in NUMBER_WORDS for w in words) + len(re.findall(r"\d+", story))
    return [n_tokens, story.count("."), len(names), numeric]


def masks(df, protocol, cfg):
    """-> (train, val, test) boolean arrays for one protocol."""
    y, split = df.gold.values, df.split.values
    if protocol == "random":
        return split == "train", split == "val", split == "test"
    if protocol == "heldout":
        held = np.isin(y, cfg["held_out_totals"])
        return (~held) & (split != "val"), (~held) & (split == "val"), held
    cut = cfg["extrapolation_train_max"]
    return (y <= cut) & (split != "val"), (y <= cut) & (split == "val"), y > cut


def score(pred, y):
    e = np.abs(pred - y)
    ss = ((y - y.mean()) ** 2).sum()
    return {"mae": float(e.mean()), "within1": float((e <= 1).mean()), "exact": float((e < 0.5).mean()),
            "r2": float(1 - ((pred - y) ** 2).sum() / ss) if ss > 0 else float("nan"),
            "corr": float(np.corrcoef(pred, y)[0, 1]) if len(set(pred)) > 1 else 0.0}


def fit_ridge(X, y, tr, va, te, alphas):
    """Standardise on train, pick alpha on val, return (test scores, test predictions)."""
    sc = StandardScaler().fit(X[tr])
    Xtr, Xva, Xte = sc.transform(X[tr]), sc.transform(X[va]), sc.transform(X[te])
    best, best_mae = None, np.inf
    for a in alphas:
        m = Ridge(alpha=a).fit(Xtr, y[tr])
        mae = np.abs(m.predict(Xva) - y[va]).mean() if va.sum() else np.abs(m.predict(Xtr) - y[tr]).mean()
        if mae < best_mae:
            best, best_mae = m, mae
    p = best.predict(Xte)
    return score(p, y[te]), p, float(best.alpha)


def cell(path, rows, pos, layer, y, tr, va, te, alphas):
    X = np.asarray(np.load(path, mmap_mode="r")[rows, pos, layer, :], dtype=np.float32)
    s, p, alpha = fit_ridge(X, y, tr, va, te, alphas)
    return pos, layer, s, p, alpha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    cfg = yaml.safe_load(open("config.yaml")); pc = cfg["probe"]
    prompts = pd.DataFrame([json.loads(l) for l in open(os.path.join(a.out, "prompts.jsonl"))])
    answers = pd.DataFrame([json.loads(l) for l in open(os.path.join(a.out, "answers.jsonl"))])
    story = {s["story_id"]: s["story"] for s in map(json.loads, open(os.path.join(a.out, "data.jsonl")))}
    df = prompts.drop(columns=["prompt", "anchors"]).merge(
        answers[["prompt_id", "n_tokens", "model_answer", "abs_error"]], on="prompt_id").sort_values("prompt_id")
    df = df.reset_index(drop=True)
    path = os.path.join(a.out, "activations.npy")
    n_pos, n_layers = np.load(path, mmap_mode="r").shape[1:3]
    y = df.gold.values.astype(float)
    F = np.array([text_features(story[s], n) for s, n in zip(df.story_id, df.n_tokens)], dtype=np.float32)
    rows_idx = np.arange(len(df))
    out, preds = [], []
    for protocol in ("random", "heldout", "extrapolate"):
        tr, va, te = masks(df, protocol, cfg)
        base_mean = np.full(te.sum(), y[tr].mean())
        s_mean = score(base_mean, y[te])
        s_text, _, _ = fit_ridge(F, y, tr, va, te, pc["alphas"])
        rng = np.random.default_rng(0); ysh = y.copy(); ysh[tr] = rng.permutation(ysh[tr])
        res = Parallel(n_jobs=pc["n_jobs"])(
            delayed(cell)(path, rows_idx, p, l, y, tr, va, te, pc["alphas"])
            for p in range(n_pos) for l in range(n_layers))
        for pos, layer, s, p, alpha in res:
            out.append({"protocol": protocol, "position": cfg["positions"][pos], "layer": layer,
                        "alpha": alpha, **{f"probe_{k}": v for k, v in s.items()},
                        **{f"mean_{k}": v for k, v in s_mean.items()},
                        **{f"text_{k}": v for k, v in s_text.items()},
                        "model_mae": float(df.abs_error.values[te].mean()),
                        "model_exact": float((df.abs_error.values[te] == 0).mean()), "n_test": int(te.sum())})
        best = min(res, key=lambda r: r[2]["mae"])
        pos, layer, s, p, _ = best
        X = np.asarray(np.load(path, mmap_mode="r")[rows_idx, pos, layer, :], dtype=np.float32)
        s_shuf, _, _ = fit_ridge(X, ysh, tr, va, te, pc["alphas"])
        sc = StandardScaler().fit(X[tr])
        mlp = MLPRegressor(hidden_layer_sizes=(pc["mlp_hidden"],), max_iter=pc["mlp_max_iter"],
                           random_state=0).fit(sc.transform(X[tr]), y[tr])
        s_mlp = score(mlp.predict(sc.transform(X[te])), y[te])
        for r in out:
            if r["protocol"] == protocol:
                r["shuffled_mae"] = s_shuf["mae"]
                r["best_position"], r["best_layer"] = cfg["positions"][pos], layer
                r["mlp_mae"], r["mlp_r2"] = s_mlp["mae"], s_mlp["r2"]
        preds += [{"protocol": protocol, "prompt_id": i, "position": cfg["positions"][pos], "layer": layer,
                   "gold": int(g), "pred": float(q), "model_answer": int(m)}
                  for i, g, q, m in zip(df.prompt_id[te], y[te], p, df.model_answer.values[te])]
        print(f"[probe] {protocol:11s} best {cfg['positions'][pos]}/L{layer}: MAE {s['mae']:.2f} "
              f"R2 {s['r2']:.3f} within1 {s['within1']:.2f} | mean-baseline {s_mean['mae']:.2f} "
              f"text {s_text['mae']:.2f} shuffled {s_shuf['mae']:.2f} MLP {s_mlp['mae']:.2f} "
              f"| model {df.abs_error.values[te].mean():.2f}", flush=True)
    pd.DataFrame(out).to_csv(os.path.join(a.out, "results.csv"), index=False)
    pd.DataFrame(preds).to_csv(os.path.join(a.out, "predictions.csv"), index=False)
    print(f"[probe] {len(out)} rows -> {a.out}/results.csv")


if __name__ == "__main__":
    main()
