"""Mandatory pre-sweep gate: 50 generations, every raw output written to one
readable file for a human and an automated reader to inspect INDEPENDENTLY
before the sweep.

This is the step that catches the class of bug that cost the source experiment a
badly wrong headline number. It is not a unit test -- the point is to look at
what the model actually emits.
"""
import argparse
import json
import os
import subprocess
import sys

from src.util import ROOT, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--out", default="results/smoke")
    ap.add_argument("--stories", type=int, default=3)
    ap.add_argument("--answer-mode", default=None)
    a = ap.parse_args()
    cfg = load_config()
    out = os.path.join(ROOT, a.out)
    os.makedirs(out, exist_ok=True)

    # The corners and the centre of the grid: (2,1) (2,30) (30,1) (30,30) (10,10),
    # up to --stories stories each, all 5 types  ->  ~55 generations.
    CELLS = [(2, 1), (2, 30), (30, 1), (30, 30), (10, 10)]
    qs = [json.loads(l) for l in
          open(os.path.join(ROOT, "data/questions.jsonl"), encoding="utf-8")]
    keep, seen = [], {}
    for q in qs:
        cell = (q["N"], q["T"])
        if cell not in CELLS:
            continue
        k = (cell, q["story_id"])
        if k not in seen:
            if len([x for x in seen if x[0] == cell]) >= a.stories:
                continue
            seen[k] = True
        keep.append(q)
    sub = os.path.join(out, "questions_smoke.jsonl")
    with open(sub, "w", encoding="utf-8") as f:
        for q in keep:
            f.write(json.dumps(q) + "\n")
    print(f"[smoke] {len(keep)} questions -> {sub}", flush=True)

    cmd = [sys.executable, "-m", "src.run_model", a.model,
           "--questions", os.path.relpath(sub, ROOT),
           "--out-dir", os.path.relpath(out, ROOT), "--run-id", "smoke"]
    if a.answer_mode:
        cmd += ["--answer-mode", a.answer_mode]
    rc = subprocess.call(cmd, cwd=ROOT)
    if rc != 0:
        sys.exit(rc)

    recs = [json.loads(l) for l in
            open(os.path.join(out, f"{a.model}.jsonl"), encoding="utf-8")]
    txt = os.path.join(out, "raw_outputs.txt")
    with open(txt, "w", encoding="utf-8") as f:
        for r in sorted(recs, key=lambda x: (x["N"], x["T"], x["question_type"])):
            f.write("=" * 78 + "\n")
            f.write(f"{r['question_id']}  N={r['N']} T={r['T']} "
                    f"type={r['question_type']} mode={r.get('answer_mode')}\n")
            f.write(f"prompt_tokens={r['prompt_tokens']} "
                    f"max_tokens={r['config']['max_tokens']} "
                    f"output_tokens={r['output_tokens']} "
                    f"finish={r['finish_reason']} truncated={r['truncated']} "
                    f"overflow={r['context_overflow']}\n")
            f.write(f"ground_truth: {json.dumps(r['ground_truth'])[:200]}\n")
            f.write(f"parse_path={r['parse_path']} format_error={r['format_error']} "
                    f"correct={r['correct']} strict={r['correct_strict']}\n")
            f.write("--- raw_output ---\n")
            f.write(repr(r["raw_output"]) + "\n\n")
    print(f"[smoke] raw outputs -> {txt}")
    print("[smoke] READ THAT FILE before starting the sweep.")


if __name__ == "__main__":
    main()
