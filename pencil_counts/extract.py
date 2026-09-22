"""Residual stream at the answer position, every layer, and the model's own
answer from the same forward pass.   python extract.py [--out results]   (GPU)

Reads <out>/prompts.jsonl. Writes <out>/activations.npy -- float16
[n_prompts, n_layers, d_model], hook_resid_post at the final prompt token, in
prompts.jsonl order -- and <out>/answers.jsonl with the model's own answer (the
candidate value 5..9 or 12..36 with the highest total log-probability) and the
unrestricted top-1 token. Llama and OLMo write every candidate as one token, so
there the score is a plain restricted argmax; Qwen and Mistral split 12/18/24/30/36
into two tokens, which costs one extra forward pass per distinct first token.
"""
import argparse, json, os, time
import yaml
CFG = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")))
# HF_HOME must be set before transformers is imported: it reads the cache path once.
os.environ.setdefault("HF_HOME", os.path.abspath(CFG["hf_cache"]))
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer


def mirror_config(cfg, tl_name):
    """TransformerLens looks a hub name up with AutoConfig; the GPU node has no
    network, so point HF_HOME at a cache holding the local config.json."""
    if os.path.isdir(tl_name):
        return
    root = os.environ["HF_HOME"]
    repo = os.path.join(root, "hub", "models--" + tl_name.replace("/", "--"))
    snap = os.path.join(repo, "snapshots", "local")
    os.makedirs(snap, exist_ok=True); os.makedirs(os.path.join(repo, "refs"), exist_ok=True)
    dst = os.path.join(snap, "config.json")
    if not os.path.exists(dst):
        os.symlink(os.path.abspath(os.path.join(cfg["_path"], "config.json")), dst)
    open(os.path.join(repo, "refs", "main"), "w").write("local")


def load_model(cfg):
    m = cfg["_model"]
    tok = AutoTokenizer.from_pretrained(m["path"])
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mirror_config(cfg, m["tl_name"])
    hf = AutoModelForCausalLM.from_pretrained(m["path"], dtype=torch.bfloat16)
    model = HookedTransformer.from_pretrained_no_processing(      # no weight folding or centering
        m["tl_name"], hf_model=hf, hf_config=AutoConfig.from_pretrained(m["path"]), tokenizer=tok,
        device="cuda", dtype=torch.bfloat16, default_prepend_bos=False)
    del hf
    return tok, model


@torch.no_grad()
def run_batch(model, tok, texts, seqs, cands, cfg, hooks):
    """One forward pass -> (acts [B, n_layers, d_model] float16, per-prompt answer dicts).
    `seqs` are the candidates' token sequences; multi-token ones need a second pass."""
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=not cfg["use_chat_template"]).to("cuda")
    logits, cache = model.run_with_cache(enc.input_ids, attention_mask=enc.attention_mask,
                                         names_filter=lambda n: n in hooks)
    acts = torch.stack([cache[h][:, -1] for h in hooks], dim=1).to(torch.float16).cpu().numpy()
    last = logits[:, -1].float()
    lp = torch.log_softmax(last, -1)
    score = torch.stack([lp[:, s[0]] for s in seqs], dim=1)          # first token of each candidate
    for prefix in {tuple(s[:-1]) for s in seqs if len(s) > 1}:       # one extra pass per distinct prefix
        pad = torch.tensor(prefix, device="cuda").expand(len(texts), len(prefix))
        nxt = torch.log_softmax(model(torch.cat([enc.input_ids, pad], 1),
                                      attention_mask=torch.cat([enc.attention_mask,
                                                                torch.ones_like(pad)], 1))[:, -1].float(), -1)
        for j, s in enumerate(seqs):
            if len(s) > 1 and tuple(s[:-1]) == prefix:
                score[:, j] = score[:, j] + nxt[:, s[-1]]
    out = [{"model_answer": int(cands[int(r)]), "top1_token": tok.decode([int(t)]), "n_tokens": int(m.sum())}
           for r, t, m in zip(score.argmax(-1), last.argmax(-1), enc.attention_mask)]
    return acts, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--model", default=None, help="key in config models (default: config model)")
    a = ap.parse_args()
    cfg = CFG
    cfg["_model"] = cfg["models"][a.model or cfg["model"]]; cfg["_path"] = cfg["_model"]["path"]
    prompts = [json.loads(l) for l in open(os.path.join(a.out, "prompts.jsonl"))]
    t0 = time.time()
    tok, model = load_model(cfg)
    L, d = model.cfg.n_layers, model.cfg.d_model
    hooks = [f"blocks.{l}.hook_resid_post" for l in range(L)]
    # How a candidate continues THIS prompt, not how it tokenises on its own:
    # Mistral's tokenizer prefixes a standalone "5" with a space token.
    def continuation(prompt, c):
        base = tok(prompt, add_special_tokens=False).input_ids
        return tok(prompt + str(c), add_special_tokens=False).input_ids[len(base):]
    seqs = {c: continuation(prompts[0]["prompt"], c) for c in cfg["n_people"] + cfg["totals"]}
    for p in prompts[1:4]:
        assert all(continuation(p["prompt"], c) == v for c, v in seqs.items()), "answer tokens vary by prompt"
    assert all(1 <= len(v) <= 2 for v in seqs.values()), seqs
    multi = {c: len(v) for c, v in seqs.items() if len(v) > 1}
    print(f"[extract] {a.model or cfg['model']}: {L} layers, d_model {d}, loaded in {time.time() - t0:.0f}s; "
          f"multi-token candidates: {multi or 'none'}", flush=True)
    acts = np.lib.format.open_memmap(os.path.join(a.out, "activations.npy"), mode="w+",
                                     dtype=np.float16, shape=(len(prompts), L, d))
    answers = [None] * len(prompts)
    t0, B = time.time(), cfg["batch_size"]
    for task, cands in (("people", cfg["n_people"]), ("total", cfg["totals"])):
        idx = sorted([i for i, p in enumerate(prompts) if p["task"] == task],
                     key=lambda i: len(prompts[i]["prompt"]))          # similar lengths -> little padding
        for s in range(0, len(idx), B):
            bi = idx[s:s + B]
            acts_b, out = run_batch(model, tok, [prompts[i]["prompt"] for i in bi],
                                    [seqs[c] for c in cands], cands, cfg, hooks)
            acts[bi] = acts_b
            for i, o in zip(bi, out):
                p = prompts[i]
                answers[i] = {k: p[k] for k in ("prompt_id", "story_id", "task", "N", "total", "T", "split", "gold")}
                answers[i].update(o, correct=o["model_answer"] == p["gold"])
            if (s // B) % 50 == 0:
                done = sum(x is not None for x in answers)
                print(f"[extract] {task} {done}/{len(prompts)}  {done / (time.time() - t0):.1f} prompts/s", flush=True)
    acts.flush()
    with open(os.path.join(a.out, "answers.jsonl"), "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in answers)
    for task in ("people", "total"):
        rows = [r for r in answers if r["task"] == task]
        print(f"[extract] {task}: model accuracy {np.mean([r['correct'] for r in rows]):.3f}, "
              f"unrestricted top-1 is a number {np.mean([r['top1_token'].strip().isdigit() for r in rows]):.3f}")
    print(f"[extract] {len(prompts)} prompts in {time.time() - t0:.0f}s -> {a.out}/activations.npy "
          f"({acts.nbytes / 1e9:.1f} GB)", flush=True)


if __name__ == "__main__":
    main()
