"""Word-vs-digit comparison: the main sweep against the number_word_probability=0
ablation, on the N values they share.

This is the one clean manipulation in the design. `pencils_per_person` and the
narration order are deliberately held fixed, so a difference here is a surface
number-format effect and not a change in how hard the tracking is
(cf. "Are Arithmetic Heuristic Neurons Form-Invariant?", Papers.md §2).

Paired at the question level: the two runs share `question_id`, because the
seed formula and the sampler are unchanged and only the rendering of numbers
differs. That lets us use McNemar's test rather than comparing two independent
proportions.
"""

import argparse
import glob
import json
import math
import os
from collections import defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(run_dir):
    rows = {}
    for path in glob.glob(os.path.join(run_dir, "*", "N*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("context_overflow"):
                    continue
                rows[r["question_id"]] = r
    return rows


def mcnemar_p(b, c):
    """Exact-ish two-sided McNemar. b, c are the two discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    # normal approximation with continuity correction; exact below n=25
    if n < 25:
        from math import comb
        tail = sum(comb(n, k) for k in range(0, min(b, c) + 1))
        return min(1.0, 2 * tail / (2 ** n))
    # clamp at 0: with b == c the continuity correction goes negative and
    # erfc would return a "probability" above 1
    z = max(0.0, abs(b - c) - 1) / math.sqrt(n)
    return min(1.0, math.erfc(z / math.sqrt(2)))


def compare(words, digits):
    shared = set(words) & set(digits)
    per = defaultdict(lambda: [0, 0, 0, 0])       # both, words_only, digits_only, neither
    for qid in shared:
        w, d = bool(words[qid]["correct"]), bool(digits[qid]["correct"])
        key = (words[qid]["model"], words[qid]["n_people"])
        cell = per[key]
        cell[0 if (w and d) else 1 if w else 2 if d else 3] += 1
    out = []
    for (model, n), (both, w_only, d_only, neither) in sorted(per.items()):
        tot = both + w_only + d_only + neither
        out.append({
            "model": model, "n": n, "n_paired": tot,
            "acc_words": (both + w_only) / tot,
            "acc_digits": (both + d_only) / tot,
            "delta_digits_minus_words": (d_only - w_only) / tot,
            "words_only_correct": w_only, "digits_only_correct": d_only,
            "mcnemar_p": mcnemar_p(w_only, d_only),
        })
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description="Compare the digits ablation to the sweep.")
    ap.add_argument("--words-run", required=True, help="main sweep run dir")
    ap.add_argument("--digits-run", required=True, help="ablation run dir")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "tables",
                                                  "ablation_words_vs_digits.csv"))
    args = ap.parse_args()

    words, digits = load(args.words_run), load(args.digits_run)
    print(f"[ablation] words={len(words)} digits={len(digits)} "
          f"paired={len(set(words) & set(digits))}")
    df = compare(words, digits)
    if df.empty:
        raise SystemExit("no paired question_ids -- check the run directories")
    df.to_csv(args.out, index=False)

    print("\n=== accuracy, words vs digits (paired on question_id) ===")
    piv = df.pivot(index="n", columns="model",
                   values="delta_digits_minus_words")
    print((piv * 100).round(1).to_string())
    print("\n(positive = digits-only prompts were EASIER)")

    print("\n=== pooled over N, per model ===")
    for model, g in df.groupby("model"):
        w = g["words_only_correct"].sum()
        d = g["digits_only_correct"].sum()
        tot = g["n_paired"].sum()
        aw = (g["acc_words"] * g["n_paired"]).sum() / tot
        ad = (g["acc_digits"] * g["n_paired"]).sum() / tot
        print(f"  {model:28s} words {aw:6.1%}  digits {ad:6.1%}  "
              f"delta {ad - aw:+6.1%}  McNemar p={mcnemar_p(w, d):.4f}  (n={tot})")
    print(f"\n[ablation] wrote {args.out}")


if __name__ == "__main__":
    main()
