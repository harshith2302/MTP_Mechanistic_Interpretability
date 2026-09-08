"""vLLM sweep for one model. Writes append-only JSONL under results/raw/<run_id>/.

Two things this file must get right, because both are silent when wrong:

  * OVERFLOW. A prompt longer than the model's context is recorded with
    `context_overflow: true` and `raw_output: null`, and is NEVER generated. It
    is excluded from every denominator downstream. Scoring it 0 would invent a
    collapse that did not happen -- OLMo-2's 4096-token window makes this a real
    case around N=22, not a hypothetical.

  * TRUNCATION. If the model emits exactly `max_tokens`, `truncated` is set.
    That is reported separately from `format_error`: a budget that was too small
    is our bug, an unparseable answer is the model's behaviour, and conflating
    them makes the first look like the second.

Grading happens here too, but only as a convenience -- every graded field is
reproducible from `raw_output` by src/regrade.py with no GPU.
"""

import argparse
import datetime as dt
import json
import os

from src.grade import grade_record
from src.prompting import build_prompt, max_tokens_for
from src.util import ROOT, code_version, load_config, model_by_key


def load_questions(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_key")
    ap.add_argument("--config", default=None)
    ap.add_argument("--questions", default="data/questions.jsonl")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-values", type=int, nargs="*", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="build prompts and report token stats, generate nothing")
    a = ap.parse_args()

    cfg = load_config(a.config)
    model = model_by_key(cfg, a.model_key)
    gen = cfg["generation"]
    run_id = a.run_id or dt.datetime.utcnow().strftime("%Y-%m-%dT%H%MZ")

    qs = load_questions(os.path.join(ROOT, a.questions))
    if a.n_values:
        qs = [q for q in qs if q["N"] in a.n_values]
    if a.limit:
        qs = qs[:a.limit]
    print(f"[run] model={model['hf_id']} questions={len(qs)} run_id={run_id}",
          flush=True)

    from transformers import AutoTokenizer
    mpath = os.path.join(ROOT, model["local_path"])
    tok = AutoTokenizer.from_pretrained(mpath)

    prepared = []
    for q in qs:
        prompt, tpl_sha, p_sha = build_prompt(
            tok, q["story_text"], q, model["use_system_role"])
        n_tok = len(tok.encode(prompt))
        mt = max_tokens_for(q, cfg)
        # Budget the ANSWER too: a prompt that fits but leaves no room to reply
        # is still an overflow.
        overflow = (n_tok + mt) > model["max_context"]
        prepared.append({"q": q, "prompt": prompt, "tpl_sha": tpl_sha,
                         "p_sha": p_sha, "n_tok": n_tok, "max_tokens": mt,
                         "overflow": overflow})

    n_ov = sum(p["overflow"] for p in prepared)
    lens = [p["n_tok"] for p in prepared]
    print(f"[run] prompt tokens: min={min(lens)} max={max(lens)} "
          f"| context_overflow={n_ov}/{len(prepared)}", flush=True)
    if a.dry_run:
        for n in sorted({p["q"]["N"] for p in prepared}):
            sub = [p for p in prepared if p["q"]["N"] == n]
            print(f"   N={n:2d}: max_prompt={max(x['n_tok'] for x in sub):6d} "
                  f"overflow={sum(x['overflow'] for x in sub)}/{len(sub)}")
        return

    todo = [p for p in prepared if not p["overflow"]]
    outs = {}
    if todo:
        from vllm import LLM, SamplingParams
        llm = LLM(model=mpath, tokenizer=mpath, dtype=gen["dtype"],
                  tensor_parallel_size=gen["tensor_parallel_size"],
                  max_num_seqs=gen["max_num_seqs"], seed=gen["seed"])
        # Group by max_tokens so the long types get their bigger budget without
        # inflating it for everyone.
        for mt in sorted({p["max_tokens"] for p in todo}):
            batch = [p for p in todo if p["max_tokens"] == mt]
            sp = SamplingParams(temperature=gen["temperature"], top_p=gen["top_p"],
                                seed=gen["seed"], max_tokens=mt)
            res = llm.generate([p["prompt"] for p in batch], sp)
            for p, r in zip(batch, res):
                o = r.outputs[0]
                outs[p["p_sha"], p["q"]["question_id"]] = (
                    o.text, len(o.token_ids), o.finish_reason)

    out_dir = os.path.join(ROOT, a.out_dir or f"results/raw/{run_id}")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{a.model_key}.jsonl")
    n_trunc = 0
    with open(path, "a", encoding="utf-8") as f:
        for p in prepared:
            q = p["q"]
            got = outs.get((p["p_sha"], q["question_id"]))
            raw, n_out, finish = got if got else (None, 0, None)
            truncated = bool(got and finish == "length")
            n_trunc += truncated
            rec = {
                "question_id": q["question_id"], "story_id": q["story_id"],
                "model": model["hf_id"], "model_key": a.model_key,
                "N": q["N"], "T": q["T"], "story_seed": q["story_seed"],
                "question_type": q["question_type"],
                "target_person": q.get("target_person"),
                "target_person_id": q.get("target_person_id"),
                "target_receiver": q.get("target_receiver"),
                "target_timestep": q.get("target_timestep"),
                "ground_truth": q["ground_truth"],
                "answer_space_size": q.get("answer_space_size"),
                "answer_dim": q.get("answer_dim"),
                "story_meta": q["story_meta"],
                "template_sha256": p["tpl_sha"], "prompt_sha256": p["p_sha"],
                "prompt_tokens": p["n_tok"], "output_tokens": n_out,
                "used_system_role": model["use_system_role"],
                "raw_output": raw, "finish_reason": finish,
                "context_overflow": p["overflow"], "truncated": truncated,
                "config": {**{k: gen[k] for k in
                              ("temperature", "top_p", "seed", "dtype",
                               "tensor_parallel_size", "max_num_seqs")},
                           "max_tokens": p["max_tokens"]},
                "run_id": run_id, "code_version": code_version(),
            }
            f.write(json.dumps(grade_record(rec)) + "\n")
    print(f"[run] wrote {len(prepared)} records ({n_ov} overflow, "
          f"{n_trunc} truncated) -> {path}")


if __name__ == "__main__":
    main()
