"""GPU pass over the minimal pairs: keep only the ones worth patching.

A pair is only usable if the model is CAUSALLY SENSITIVE to the one token we
changed. Patching is a recovery measurement -- if the model ignores the transfer
amount, there is nothing to recover and the pair contributes noise dressed up as
a null result.

Three things are measured at the teacher-forced answer position:

  clean_ok    logit(v_clean)   > logit(v_corrupt)  on the clean prompt
  corrupt_ok  logit(v_corrupt) > logit(v_clean)    on the corrupt prompt
  is_digit    the model's top-1 token there is actually a number

`clean_ok and corrupt_ok` is the qualifying criterion: the model tracks the
amount in both directions, so the logit difference is a metric that MOVES.
`is_digit` is a sanity check on the teacher-forced prefix -- if the model wants
to emit something else there, the prefix is wrong and every number downstream
would be meaningless.

Clean and corrupt are token-aligned and equal length, so each pair is run as a
batch of exactly two with no padding -- left-padding a causal model and then
reading position -1 is a classic way to get quietly wrong logits.

MUST be a real file: nnsight 0.7 rebuilds the trace block via
inspect.getsource(), so heredocs and `python -c` fail (see scripts/probe_mech.py).
"""

import argparse
import json
import os
import sys

import torch

ROOT = os.environ.get("PROJ", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results/patching/pairs_N10.jsonl")
    ap.add_argument("--model", default="models/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default="results/patching/pairs_N10_verified.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from nnsight import LanguageModel
    from transformers import AutoTokenizer

    mpath = os.path.join(ROOT, a.model)
    tok = AutoTokenizer.from_pretrained(mpath)
    lm = LanguageModel(mpath, device_map="cuda", dispatch=True, dtype=torch.bfloat16)
    print(f"[verify] loaded {os.path.basename(mpath)}", flush=True)

    pairs = [json.loads(l) for l in open(os.path.join(ROOT, a.pairs), encoding="utf-8")
             if l.strip()]
    if a.limit:
        pairs = pairs[:a.limit]
    print(f"[verify] {len(pairs)} pairs", flush=True)

    kept, n_digit, n_clean, n_corrupt = [], 0, 0, 0
    for i, p in enumerate(pairs):
        tc = tok.encode(str(p["v_clean"]), add_special_tokens=False)[0]
        tx = tok.encode(str(p["v_corrupt"]), add_special_tokens=False)[0]

        # Save four scalars and an argmax, NOT the (2, vocab) slice. Saving the
        # full slice held ~0.6 GB per trace alive across iterations and OOM'd a
        # 44 GB L40S after ~60 pairs (job 305298). Do the indexing inside the
        # trace so only the scalars ever leave it.
        with lm.trace([p["clean_body"], p["corrupt_body"]]):
            lg = lm.output.logits[:, -1, :]
            picked = torch.stack([lg[0, tc], lg[0, tx],
                                  lg[1, tc], lg[1, tx]]).float().save()
            top_id = lg[0].argmax().save()

        c_clean, c_corr, x_clean, x_corr = picked.tolist()
        clean_diff = c_clean - c_corr
        corrupt_diff = x_corr - x_clean
        top = tok.decode(int(top_id.item()))
        is_digit = top.strip().isdigit()
        del picked, top_id
        if (i + 1) % 20 == 0:
            torch.cuda.empty_cache()

        n_digit += is_digit
        n_clean += clean_diff > 0
        n_corrupt += corrupt_diff > 0
        p.update(clean_logit_diff=round(clean_diff, 4),
                 corrupt_logit_diff=round(corrupt_diff, 4),
                 top1_clean=top, top1_is_digit=bool(is_digit),
                 qualifies=bool(clean_diff > 0 and corrupt_diff > 0))
        if p["qualifies"]:
            kept.append(p)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(pairs)}  kept={len(kept)}", flush=True)

    n = len(pairs)
    print(f"\n[verify] top-1 is a digit      {n_digit}/{n} ({n_digit/n:.1%})")
    print(f"[verify] clean  prefers v_clean   {n_clean}/{n} ({n_clean/n:.1%})")
    print(f"[verify] corrupt prefers v_corrupt {n_corrupt}/{n} ({n_corrupt/n:.1%})")
    print(f"[verify] QUALIFY (both)            {len(kept)}/{n} ({len(kept)/n:.1%})")

    out = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for p in kept:
            f.write(json.dumps(p) + "\n")
    print(f"[verify] wrote {out}")

    if n_digit / n < 0.5:
        print("\n[verify] WARNING: the model mostly does not want to emit a number "
              "at the teacher-forced position. Re-check the answer prefix "
              "(try --bare in src.patch_pairs) before running any patching.")


if __name__ == "__main__":
    main()
