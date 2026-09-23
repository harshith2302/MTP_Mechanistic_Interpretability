"""Residual stream at five reading positions, every layer, plus the model's own
answer.   python extract.py [--out results] [--model llama]    (GPU)

Writes <out>/activations.npy -- float16 [n_prompts, n_positions, n_layers, d_model],
positions in config order -- and <out>/answers.jsonl. Token indices come from the
tokenizer's character offsets, so a position always lands on the sentence it names.
"""
import argparse, json, os, time
import yaml
CFG = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")))
os.environ.setdefault("HF_HOME", os.path.abspath(CFG["hf_cache"]))
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer


def mirror_config(tl_name, path):
    """TransformerLens resolves a hub name with AutoConfig; the GPU node has no
    network, so keep the local config.json where that lookup will find it."""
    if os.path.isdir(tl_name):
        return
    repo = os.path.join(os.environ["HF_HOME"], "hub", "models--" + tl_name.replace("/", "--"))
    snap = os.path.join(repo, "snapshots", "local")
    os.makedirs(snap, exist_ok=True); os.makedirs(os.path.join(repo, "refs"), exist_ok=True)
    if not os.path.exists(os.path.join(snap, "config.json")):
        os.symlink(os.path.abspath(os.path.join(path, "config.json")), os.path.join(snap, "config.json"))
    open(os.path.join(repo, "refs", "main"), "w").write("local")


def load_model(m):
    tok = AutoTokenizer.from_pretrained(m["path"])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mirror_config(m["tl_name"], m["path"])
    hf = AutoModelForCausalLM.from_pretrained(m["path"], dtype=torch.bfloat16)
    model = HookedTransformer.from_pretrained_no_processing(
        m["tl_name"], hf_model=hf, hf_config=AutoConfig.from_pretrained(m["path"]), tokenizer=tok,
        device="cuda", dtype=torch.bfloat16, default_prepend_bos=False)
    del hf
    return tok, model


def token_indices(tok, text, anchors, names):
    """Character offset -> index of the last token ending at or before it."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ends = [e for _, e in enc["offset_mapping"]]
    out = []
    for name in names:
        if name == "first_token":
            out.append(0); continue
        c = anchors[name]
        idx = max([i for i, e in enumerate(ends) if e <= c] or [0])
        out.append(idx)
    return out, len(enc["input_ids"])


@torch.no_grad()
def run_batch(model, tok, batch, seqs, cands, hooks, names):
    enc = tok([p["prompt"] for p in batch], return_tensors="pt", padding=True,
              add_special_tokens=not CFG["use_chat_template"]).to("cuda")
    logits, cache = model.run_with_cache(enc.input_ids, attention_mask=enc.attention_mask,
                                         names_filter=lambda n: n in hooks)
    resid = torch.stack([cache[h] for h in hooks], dim=1)            # [B, layers, seq, d]
    width = enc.input_ids.shape[1]
    acts = np.empty((len(batch), len(names), resid.shape[1], resid.shape[-1]), dtype=np.float16)
    for b, p in enumerate(batch):
        idx, n_tok = token_indices(tok, p["prompt"], p["anchors"], names)
        pad = width - n_tok                                          # left padding
        acts[b] = resid[b, :, [pad + i for i in idx], :].transpose(0, 1).to(torch.float16).cpu().numpy()
    last = logits[:, -1].float()
    lp = torch.log_softmax(last, -1)
    score = torch.stack([lp[:, s[0]] for s in seqs], dim=1)
    for prefix in {tuple(s[:-1]) for s in seqs if len(s) > 1}:       # two-token candidates
        pad_t = torch.tensor(prefix, device="cuda").expand(len(batch), len(prefix))
        nxt = torch.log_softmax(model(torch.cat([enc.input_ids, pad_t], 1),
                                      attention_mask=torch.cat([enc.attention_mask,
                                                                torch.ones_like(pad_t)], 1))[:, -1].float(), -1)
        for j, s in enumerate(seqs):
            if len(s) > 1 and tuple(s[:-1]) == prefix:
                score[:, j] = score[:, j] + nxt[:, s[-1]]
    out = [{"model_answer": int(cands[int(r)]), "top1_token": tok.decode([int(t)]), "n_tokens": int(m.sum())}
           for r, t, m in zip(score.argmax(-1), last.argmax(-1), enc.attention_mask)]
    return acts, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()
    key = a.model or CFG["model"]
    prompts = [json.loads(l) for l in open(os.path.join(a.out, "prompts.jsonl"))]
    t0 = time.time()
    tok, model = load_model(CFG["models"][key])
    L, d, names = model.cfg.n_layers, model.cfg.d_model, CFG["positions"]
    hooks = [f"blocks.{l}.hook_resid_post" for l in range(L)]
    cands = list(range(CFG["total_min"], CFG["total_max"] + 1))
    base = tok(prompts[0]["prompt"], add_special_tokens=False).input_ids
    seqs = [tok(prompts[0]["prompt"] + str(c), add_special_tokens=False).input_ids[len(base):] for c in cands]
    assert all(1 <= len(s) <= 2 for s in seqs)
    print(f"[extract] {key}: {L} layers, d_model {d}, {len(names)} positions, loaded in "
          f"{time.time() - t0:.0f}s; {sum(len(s) > 1 for s in seqs)}/{len(cands)} answers are two tokens",
          flush=True)
    acts = np.lib.format.open_memmap(os.path.join(a.out, "activations.npy"), mode="w+",
                                     dtype=np.float16, shape=(len(prompts), len(names), L, d))
    answers = []
    t0, B = time.time(), CFG["batch_size"]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]["prompt"]))
    for s in range(0, len(order), B):
        bi = order[s:s + B]
        acts_b, out = run_batch(model, tok, [prompts[i] for i in bi], seqs, cands, hooks, names)
        acts[bi] = acts_b
        for i, o in zip(bi, out):
            p = prompts[i]
            answers.append({k: p[k] for k in ("prompt_id", "story_id", "N", "T", "split", "gold")}
                           | o | {"correct": o["model_answer"] == p["gold"],
                                  "abs_error": abs(o["model_answer"] - p["gold"])})
        if (s // B) % 50 == 0:
            print(f"[extract] {len(answers)}/{len(prompts)}  "
                  f"{len(answers) / (time.time() - t0):.1f} prompts/s", flush=True)
    acts.flush()
    answers.sort(key=lambda r: r["prompt_id"])
    with open(os.path.join(a.out, "answers.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in answers)
    err = np.array([r["abs_error"] for r in answers])
    print(f"[extract] model exact {np.mean(err == 0):.3f}, within 2: {np.mean(err <= 2):.3f}, "
          f"mean |error| {err.mean():.2f}")
    print(f"[extract] {len(prompts)} prompts in {time.time() - t0:.0f}s -> {acts.nbytes / 1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
