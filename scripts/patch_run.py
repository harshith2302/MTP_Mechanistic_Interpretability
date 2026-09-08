"""Activation patching: where does the first state update happen?

Denoising design. The corrupt prompt differs from the clean one at exactly one
token -- the transfer amount -- and the model's answer moves with it. We run the
corrupt prompt while splicing in the CLEAN residual stream at one (layer,
position) at a time, and measure how much of the clean answer comes back:

    recovery = (patched_diff - corrupt_diff) / (clean_diff - corrupt_diff)

where diff = logit(v_clean) - logit(v_corrupt) at the teacher-forced answer
position. 0 = the patch changed nothing, 1 = it fully restored the clean answer.
A (layer, position) with high recovery is carrying the updated count.

Positions are curated rather than swept. A full sweep is n_layers x ~1000
positions x 107 pairs, which is millions of forward passes for mostly
uninformative tokens. The six below are the ones the behavioral result makes
predictions about:

  amount      the changed digit itself -- where the new information enters
  pencils     the token straight after it, the usual landing spot for a
              number's meaning under BPE
  sent_end    end of the transfer sentence -- where a per-sentence summary would
              live if the model builds one
  query_name  the queried person's name in the question
  last        the answer position, where the readout happens
  control     an early token in the t=0 intro, causally irrelevant to the
              amount. This one MUST come out near zero; if it does not, the
              measurement is broken, not interesting.

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

POSITIONS = ["amount", "pencils", "sent_end", "query_name", "last", "control"]


def token_of_char(offsets, ch):
    """First token whose span covers character index `ch` (-1 if none)."""
    for i, (a, b) in enumerate(offsets):
        if a <= ch < b:
            return i
    return -1


def pick_positions(pair, tok):
    """Map the six named sites to token indices in the CORRUPT prompt."""
    body = pair["corrupt_body"]
    enc = tok(body, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    n = len(enc["input_ids"])

    sent = pair["corrupt_sentence"]
    s0 = body.find(sent)
    if s0 < 0:
        return None
    amt = body.find(str(pair["corrupt_amount"]), s0)
    pos = {
        "amount": token_of_char(offs, amt),
        "pencils": token_of_char(offs, body.find("pencils", amt)),
        "sent_end": token_of_char(offs, s0 + len(sent) - 1),
        "last": n - 1,
    }
    q = body.rfind("How many pencils did")
    pos["query_name"] = token_of_char(offs, body.find(pair["person_name"], q))
    # Control: a token inside the t=0 intro, well before the transfer sentence.
    intro = body.find("At timestep 0")
    pos["control"] = token_of_char(offs, intro + 20) if 0 <= intro < s0 else 8
    if any(v < 0 or v >= n for v in pos.values()):
        return None
    return pos, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="results/patching/pairs_N10_verified.jsonl")
    ap.add_argument("--model", default="models/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default="results/patching/patch_N10.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0 = all pairs")
    a = ap.parse_args()

    from nnsight import LanguageModel
    from transformers import AutoTokenizer

    # Autograd is the memory sink here. Each traced forward keeps every
    # intermediate activation alive for a backward pass we never run -- roughly
    # 200 MB per trace at seq~1000, and this loop does 168 traces per pair, which
    # is how jobs 305304/305305 filled 44 GB in under a minute. Nothing in this
    # experiment needs gradients: patching is forward-only.
    torch.set_grad_enabled(False)

    mpath = os.path.join(ROOT, a.model)
    tok = AutoTokenizer.from_pretrained(mpath)
    lm = LanguageModel(mpath, device_map="cuda", dispatch=True, dtype=torch.bfloat16)
    L = lm.config.num_hidden_layers
    print(f"[patch] loaded {os.path.basename(mpath)}, {L} layers", flush=True)

    # Older transformers returned (hidden, ...) from a decoder layer, so the
    # idiom was `.output[0]`. In transformers 5.x the layer returns the tensor
    # itself, and `.output[0]` silently selects BATCH 0 instead -- giving a 2D
    # (seq, hidden) tensor and `IndexError: too many indices` on the next slice.
    # Detect it once rather than pinning a version.
    with lm.trace("probe"):
        probe = lm.model.layers[0].output.save()
    TUPLE_OUT = isinstance(probe, (tuple, list))
    shape = (probe[0] if TUPLE_OUT else probe).shape
    print(f"[patch] layer output is {'a tuple' if TUPLE_OUT else 'a tensor'}, "
          f"hidden shape {tuple(shape)}", flush=True)
    assert len(shape) == 3, f"expected (batch, seq, hidden), got {tuple(shape)}"

    def resid(layer):
        """The (batch, seq, hidden) residual stream proxy for a decoder layer."""
        return layer.output[0] if TUPLE_OUT else layer.output

    pairs = [json.loads(l) for l in open(os.path.join(ROOT, a.pairs), encoding="utf-8")
             if l.strip()]
    if a.limit:
        pairs = pairs[:a.limit]
    print(f"[patch] {len(pairs)} pairs x {L} layers x {len(POSITIONS)} positions "
          f"= {len(pairs)*L*len(POSITIONS)} patched forwards", flush=True)

    out_path = os.path.join(ROOT, a.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fh = open(out_path, "w", encoding="utf-8")
    n_done = 0

    for pi, p in enumerate(pairs):
        got = pick_positions(p, tok)
        if got is None:
            print(f"  [skip] pair {pi}: could not locate all positions", flush=True)
            continue
        pos, n_tok = got
        tc = tok.encode(str(p["v_clean"]), add_special_tokens=False)[0]
        tx = tok.encode(str(p["v_corrupt"]), add_special_tokens=False)[0]
        want = sorted(set(pos.values()))

        # 1. cache the clean residual stream, only at the positions we patch.
        # Bind `cache` OUTSIDE the trace and append into it: nnsight re-executes
        # the block's source, and a name first assigned inside it -- especially by
        # a comprehension, which has its own scope -- never reaches this frame
        # (UnboundLocalError). Mutating a pre-bound list is the safe idiom.
        cache = []
        with lm.trace(p["clean_body"]):
            for l in range(L):
                cache.append(resid(lm.model.layers[l])[:, want, :].save())
        clean_cache = [c.detach() for c in cache]

        # 2. baselines
        with lm.trace([p["clean_body"], p["corrupt_body"]]):
            lg = lm.output.logits[:, -1, :]
            base = torch.stack([lg[0, tc], lg[0, tx], lg[1, tc], lg[1, tx]]).float().save()
        c_c, c_x, x_c, x_x = base.tolist()
        clean_diff, corrupt_diff = c_c - c_x, x_c - x_x
        denom = clean_diff - corrupt_diff
        if abs(denom) < 1e-6:
            continue

        # 3. patch one (layer, position) at a time
        rec = {"story_seed": p["story_seed"], "person_name": p["person_name"],
               "role": p["role"], "v_clean": p["v_clean"], "v_corrupt": p["v_corrupt"],
               "clean_diff": round(clean_diff, 4), "corrupt_diff": round(corrupt_diff, 4),
               "n_tokens": n_tok, "positions": pos, "recovery": {}}
        for name, tpos in pos.items():
            k = want.index(tpos)
            row = []
            for l in range(L):
                src = clean_cache[l][:, k, :]
                with lm.trace(p["corrupt_body"]):
                    resid(lm.model.layers[l])[:, tpos, :] = src
                    lg2 = lm.output.logits[:, -1, :]
                    got2 = torch.stack([lg2[0, tc], lg2[0, tx]]).float().save()
                pc, px = got2.tolist()
                row.append(round(((pc - px) - corrupt_diff) / denom, 4))
                # Each trace materialises a (1, seq, vocab) logits tensor -- about
                # 300 MB at seq~1000 for a 152k vocab. 168 of those per pair is
                # ~50 GB if anything holds a reference, which is exactly how job
                # 305304 died 70 s in. Drop the reference and hand the block back
                # to the allocator every iteration; the sync costs far less than
                # an OOM 3 hours into a run.
                del got2
                torch.cuda.empty_cache()
            rec["recovery"][name] = row
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        n_done += 1
        del clean_cache, cache, base
        torch.cuda.empty_cache()
        if n_done % 5 == 0:
            print(f"  {n_done}/{len(pairs)} pairs done", flush=True)

    fh.close()
    print(f"[patch] wrote {n_done} pairs -> {out_path}")


if __name__ == "__main__":
    main()
