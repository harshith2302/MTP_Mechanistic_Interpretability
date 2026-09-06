"""vLLM offline batch inference -> append-only raw JSONL.

Resumable: on start, existing question_ids in the target file are skipped.
Flushes after every N, so a walltime kill loses at most one N value.
Prompts that exceed the model's context are recorded with context_overflow=true
and NOT generated -- they are excluded from accuracy denominators downstream,
never counted as wrong (see CLAUDE.md; this is how the headline figure stays
honest for OLMo-2).
"""

import argparse
import datetime as dt
import json
import os
import time
import uuid

import yaml

from src.classify import classify
from src.grade import grade, parse_answer
from src.prompts import PromptStore, build_prompt, prompt_sha256
from src.questions import make_question_id, sample_questions
from src.simulate import generate_story

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def existing_ids(path):
    ids = set()
    if not os.path.exists(path):
        return ids
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    ids.add(json.loads(line)["question_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return ids


def build_work(cfg, model_name, n, tokenizer, store, done):
    """Every (question, prompt) pair for one N value that is not already done."""
    st = cfg["story"]
    qp = cfg["questions_per_story"]
    work = []
    for si in range(cfg["stories_per_n"]):
        seed = 1000 * n + si
        story = generate_story(
            n, n, seed,
            names_path=os.path.join(ROOT, st["names_path"]),
            pencils_per_person=st["pencils_per_person"],
            number_word_probability=st["number_word_probability"],
        )
        for q in sample_questions(story, seed, qp["scalar_types"], qp["heavy_types"]):
            qid = make_question_id(model_name, n, si, q["question_type"],
                                   q["index_within_type"])
            if qid in done:
                continue
            prompt = build_prompt(story["text"], q, tokenizer)
            work.append({
                "qid": qid, "story": story, "question": q, "prompt": prompt,
                "story_index": si, "story_seed": seed,
                "prompt_sha256": store.add(prompt),
            })
    return work


def base_record(run_id, model, cfg, item, n):
    q, story = item["question"], item["story"]
    return {
        "run_id": run_id,
        "model": model["short_name"],
        "model_revision": model.get("revision"),
        "n_people": n, "n_timesteps": n,
        "total_pencils": story["config"]["total_pencils"],
        "story_seed": item["story_seed"], "story_index": item["story_index"],
        "question_id": item["qid"],
        "question_type": q["question_type"],
        "question_text": q["question_text"],
        "queried_person": q["queried_person"],
        "queried_timestep": q["queried_timestep"],
        "person_intro_position": q["meta"].get("person_intro_position"),
        "prompt_sha256": item["prompt_sha256"],
        "gold": q["gold"],
        "meta": q["meta"],
        "decoding": cfg["decoding"],
        "engine_settings": cfg["engine"],
    }


class _EstimatingTokenizer:
    """Stand-in used only by --dry-run before weights are downloaded.

    ~4 chars/token is close enough to size the context budget and decide where
    OLMo-2 will overflow; it is never used for a real generation.
    """

    def __call__(self, text):
        return {"input_ids": [0] * (len(text) // 4 + 1)}


def _load_tokenizer(local_path, allow_estimate=False):
    if os.path.isdir(local_path):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(local_path, local_files_only=True)
    if allow_estimate:
        print(f"[run_eval] WARNING: {local_path} not found; --dry-run is using a "
              f"~4 chars/token ESTIMATE, not the real tokenizer.")
        return _EstimatingTokenizer()
    raise SystemExit(
        f"[run_eval] model path not found: {local_path}\n"
        f"           run scripts/download_models.sh on the LOGIN node first.")


def main():
    ap = argparse.ArgumentParser(description="Run the behavioral sweep for one model.")
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--models-config", default="configs/models.yaml")
    ap.add_argument("--model-index", type=int, required=True)
    ap.add_argument("--n-values", type=int, nargs="*",
                    help="override the config's n_values (smoke runs)")
    ap.add_argument("--stories-per-n", type=int, help="override stories_per_n")
    ap.add_argument("--number-word-probability", type=float,
                    help="ablation override (use 0.0 for the digits-only run)")
    ap.add_argument("--run-id", help="resume into an existing run_id directory")
    ap.add_argument("--tag", default="", help="suffix for the output directory")
    ap.add_argument("--dry-run", action="store_true",
                    help="build prompts and report token stats, load no model")
    args = ap.parse_args()

    cfg = load_cfg(os.path.join(ROOT, args.config))
    models = load_cfg(os.path.join(ROOT, args.models_config))["models"]
    model = models[args.model_index]
    name = model["short_name"]
    if args.stories_per_n:
        cfg["stories_per_n"] = args.stories_per_n
    if args.number_word_probability is not None:
        cfg["story"]["number_word_probability"] = args.number_word_probability
    n_values = args.n_values or cfg["n_values"]

    run_id = args.run_id or (
        dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H%MZ")
        + "_" + uuid.uuid4().hex[:6])
    out_dir = os.path.join(ROOT, cfg["output_root"], "raw",
                           run_id + (f"_{args.tag}" if args.tag else ""), name)
    os.makedirs(out_dir, exist_ok=True)
    store = PromptStore(os.path.join(out_dir, "prompts.jsonl"))
    print(f"[run_eval] model={name} run_id={run_id}\n[run_eval] out={out_dir}")

    local_path = os.path.join(ROOT, model["local_path"])
    tokenizer = _load_tokenizer(local_path, allow_estimate=args.dry_run)
    ctx = model["max_context"]

    llm = sampling = None
    if not args.dry_run:
        from vllm import LLM, SamplingParams
        e, d = cfg["engine"], cfg["decoding"]
        llm = LLM(model=local_path, tokenizer=local_path, dtype=e["dtype"],
                  max_model_len=ctx, gpu_memory_utilization=e["gpu_memory_utilization"],
                  tensor_parallel_size=e["tensor_parallel_size"],
                  max_num_seqs=e["max_num_seqs"], enforce_eager=e["enforce_eager"],
                  seed=d["seed"], trust_remote_code=False)
        sampling = SamplingParams(temperature=d["temperature"], top_p=d["top_p"],
                                  seed=d["seed"], max_tokens=d["max_tokens_scalar"])

    heavy = set(cfg["heavy_types"])
    for n in n_values:
        path = os.path.join(out_dir, f"N{n:02d}.jsonl")
        done = existing_ids(path)
        work = build_work(cfg, name, n, tokenizer, store, done)
        store.flush()
        if not work:
            print(f"[N={n:02d}] nothing to do ({len(done)} already present)")
            continue

        fits, overflow = [], []
        for item in work:
            item["prompt_tokens"] = len(tokenizer(item["prompt"])["input_ids"])
            budget = (cfg["decoding"]["max_tokens_heavy"]
                      if item["question"]["question_type"] in heavy
                      else cfg["decoding"]["max_tokens_scalar"])
            (overflow if item["prompt_tokens"] + budget > ctx else fits).append(item)

        print(f"[N={n:02d}] {len(fits)} to generate, {len(overflow)} context_overflow, "
              f"{len(done)} resumed"
              + (f", max_prompt_tokens={max(i['prompt_tokens'] for i in work)}" if work else ""))

        with open(path, "a", encoding="utf-8") as fh:
            for item in overflow:
                rec = base_record(run_id, model, cfg, item, n)
                rec.update(prompt_tokens=item["prompt_tokens"], context_overflow=True,
                           raw_output=None, output_tokens=None, finish_reason=None,
                           parse_status=None, parsed=None, correct=None,
                           label=None, labels_matched=[], ambiguous=False,
                           coarse_category=None, latency_s=None)
                fh.write(json.dumps(rec) + "\n")

            if fits and not args.dry_run:
                _generate(llm, sampling, cfg, heavy, fits, run_id, model, n, fh)
            fh.flush()
            os.fsync(fh.fileno())
        print(f"[N={n:02d}] wrote {path}")

    store.close()
    print("[run_eval] done")


def _generate(llm, sampling, cfg, heavy, fits, run_id, model, n, fh):
    """Generate in two groups so heavy types get the larger token budget."""
    from vllm import SamplingParams
    d = cfg["decoding"]
    groups = [
        ([i for i in fits if i["question"]["question_type"] not in heavy],
         d["max_tokens_scalar"]),
        ([i for i in fits if i["question"]["question_type"] in heavy],
         d["max_tokens_heavy"]),
    ]
    for items, max_tokens in groups:
        if not items:
            continue
        sp = SamplingParams(temperature=d["temperature"], top_p=d["top_p"],
                            seed=d["seed"], max_tokens=max_tokens)
        t0 = time.time()
        outs = llm.generate([i["prompt"] for i in items], sp)
        elapsed = time.time() - t0
        per = elapsed / len(items)
        for item, out in zip(items, outs):
            comp = out.outputs[0]
            raw, finish = comp.text, comp.finish_reason
            parsed, status = parse_answer(raw, finish)
            graded = grade(item["question"], parsed, item["story"])
            if parsed is None:
                graded["status"] = status
            labels = classify(item["question"], graded, item["story"])
            rec = base_record(run_id, model, cfg, item, n)
            rec.update(
                prompt_tokens=item["prompt_tokens"], context_overflow=False,
                raw_output=raw, output_tokens=len(comp.token_ids),
                finish_reason=finish, parse_status=status, parsed=parsed,
                correct=graded["correct"], grade_detail=graded["detail"],
                latency_s=round(per, 4), **labels,
            )
            fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
