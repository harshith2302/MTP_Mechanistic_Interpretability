"""One linear probe per (question, layer), with three baselines.
   python probe.py [--out results]     (CPU; ~20 min on 16 cores for the full run)

Reads <out>/prompts.jsonl, data.jsonl, answers.jsonl, activations.npy.
Writes <out>/results.csv     -- task, layer, split, T, accuracy, baseline_majority,
                                baseline_shuffled, baseline_text, model_accuracy, n
   and <out>/predictions.csv -- per-prompt test predictions at each question's best layer
"""
import argparse, json, os, re
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")                       # one thread per fit; fits run in parallel
import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from num2words import num2words
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

NUMBER_WORDS = {num2words(v) for v in range(60)}


def text_features(story, n_tokens):
    """Four surface counts; the values of the numbers are deliberately absent."""
    words = re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)?", story)
    names = {w for w in words if w[0].isupper() and w not in ("The", "At")}
    numeric = sum(w.lower() in NUMBER_WORDS for w in words) + len(re.findall(r"\d+", story))
    return [n_tokens, story.count("."), len(names), numeric]


def fit_predict(X, y, train, pc, shuffle_seed=None):
    """Standardise on train, fit balanced multinomial logistic regression, predict every row."""
    sc = StandardScaler().fit(X[train])
    y_train = y[train].copy()
    if shuffle_seed is not None:
        np.random.default_rng(shuffle_seed).shuffle(y_train)
    clf = LogisticRegression(C=pc["C"], max_iter=pc["max_iter"], class_weight=pc["class_weight"])
    clf.fit(sc.transform(X[train]), y_train)
    return clf.predict(sc.transform(X))


def probe_layer(path, rows, layer, y, train, pc):
    X = np.asarray(np.load(path, mmap_mode="r")[rows, layer, :], dtype=np.float32)
    return layer, fit_predict(X, y, train, pc), fit_predict(X, y, train, pc, shuffle_seed=pc["shuffle_seed"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    cfg = yaml.safe_load(open("config.yaml")); pc = cfg["probe"]
    prompts = pd.DataFrame([json.loads(l) for l in open(os.path.join(a.out, "prompts.jsonl"))])
    answers = pd.DataFrame([json.loads(l) for l in open(os.path.join(a.out, "answers.jsonl"))])
    story = {s["story_id"]: s["story"] for s in map(json.loads, open(os.path.join(a.out, "data.jsonl")))}
    df = prompts.drop(columns="prompt").merge(answers[["prompt_id", "n_tokens", "correct"]], on="prompt_id")
    path = os.path.join(a.out, "activations.npy")
    n_layers = np.load(path, mmap_mode="r").shape[1]
    rows, preds = [], []
    for task in ("people", "total"):
        d = df[df.task == task].reset_index()                     # 'index' = row in activations.npy
        y, train = d.gold.values, (d.split == "train").values
        majority = np.bincount(y[train]).argmax()
        F = np.array([text_features(story[s], n) for s, n in zip(d.story_id, d.n_tokens)], dtype=np.float32)
        p_text = fit_predict(F, y, train, pc)
        res = Parallel(n_jobs=pc["n_jobs"])(
            delayed(probe_layer)(path, d["index"].values, l, y, train, pc) for l in range(n_layers))
        for layer, p, p_shuffled in res:
            for split in ("val", "test"):
                for T in list(cfg["timesteps"]) + ["all"]:
                    m = (d.split == split).values & ((d["T"] == T).values if T != "all" else True)
                    rows.append({"task": task, "layer": layer, "split": split, "T": T,
                                 "accuracy": (p[m] == y[m]).mean(), "baseline_majority": (y[m] == majority).mean(),
                                 "baseline_shuffled": (p_shuffled[m] == y[m]).mean(),
                                 "baseline_text": (p_text[m] == y[m]).mean(),
                                 "model_accuracy": d.correct.values[m].mean(), "n": int(m.sum())})
        r = pd.DataFrame(rows); r = r[(r.task == task) & (r["T"] == "all")]
        best = int(r[r.split == "val"].set_index("layer").accuracy.idxmax())
        p, test = {l: q for l, q, _ in res}[best], (d.split == "test").values
        preds += [{"prompt_id": i, "task": task, "layer": best, "T": t, "gold": int(g), "pred": int(q)}
                  for i, t, g, q in zip(d.prompt_id[test], d["T"][test], y[test], p[test])]
        b = r[(r.split == "test") & (r.layer == best)].iloc[0]
        print(f"[probe] {task}: best layer {best} (val); test accuracy {b.accuracy:.3f} | majority "
              f"{b.baseline_majority:.3f} shuffled {b.baseline_shuffled:.3f} text-only {b.baseline_text:.3f} "
              f"model {b.model_accuracy:.3f}")
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "results.csv"), index=False)
    pd.DataFrame(preds).to_csv(os.path.join(a.out, "predictions.csv"), index=False)
    print(f"[probe] {len(rows)} rows -> {a.out}/results.csv")


if __name__ == "__main__":
    main()
