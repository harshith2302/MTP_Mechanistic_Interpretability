"""Re-grade a finished sweep from its stored raw_output. No GPU needed.

Records keep the model's raw text always, precisely so a grading or taxonomy
bug can be repaired without re-running an allocation. This rebuilds
parse_status / correct / label / labels_matched / coarse_category from
raw_output, leaving generation untouched.

Raw JSONL is append-only, so this NEVER edits a run in place: it writes a new
run directory and records `regraded_from` in every record.
"""

import argparse
import glob
import json
import os
import shutil

from src.classify import classify
from src.grade import grade, parse_answer
from src.questions import sample_questions
from src.simulate import generate_story

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# fields this tool owns; everything else is copied through unchanged
REGRADED = ["parse_status", "parsed", "correct", "grade_detail", "label",
            "labels_matched", "ambiguous", "coarse_category",
            "first_divergence_index", "recovered", "error_growth",
            "partial_correct_fraction", "per_person_labels", "gap",
            "intro_distance", "signed_magnitude", "missed", "spurious"]


def _story_cache():
    cache = {}

    def get(n, seed, npb):
        key = (n, seed, npb)
        if key not in cache:
            if len(cache) > 64:
                cache.clear()
            cache[key] = generate_story(n, n, seed, number_word_probability=npb)
        return cache[key]
    return get


def _question_index(story, seed):
    return {(q["question_type"], q["index_within_type"]): q
            for q in sample_questions(story, seed)}


def regrade_record(rec, get_story, qcache):
    """Return the record with every graded field recomputed from raw_output."""
    if rec.get("context_overflow"):
        return rec
    npb = (rec.get("story", {}) or {}).get("number_word_probability")
    if npb is None:
        npb = rec.get("number_word_probability", 0.45)
    story = get_story(rec["n_people"], rec["story_seed"], npb)

    key = (rec["story_seed"], npb)
    if key not in qcache:
        if len(qcache) > 64:
            qcache.clear()
        qcache[key] = _question_index(story, rec["story_seed"])
    # question_id tail is "<type>|<k>"
    parts = rec["question_id"].split("|")
    q = qcache[key].get((parts[-2], int(parts[-1])))
    if q is None:                       # sampler changed; fall back to the record
        return rec

    parsed, status = parse_answer(rec["raw_output"], rec.get("finish_reason"))
    graded = grade(q, parsed, story, parse_status=status)
    if parsed is None:
        graded["status"] = status
    labels = classify(q, graded, story)

    out = {k: v for k, v in rec.items() if k not in REGRADED}
    out.update(parse_status=status, parsed=parsed, correct=graded["correct"],
               grade_detail=graded["detail"], **labels)
    return out


def main():
    ap = argparse.ArgumentParser(description="Re-grade a sweep from raw_output.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True,
                    help="new run directory; must not already exist")
    args = ap.parse_args()

    src = os.path.abspath(args.run_dir)
    dst = os.path.abspath(args.out_dir)
    if os.path.exists(dst):
        raise SystemExit(f"refusing to overwrite {dst} -- pick a new directory")

    get_story, qcache = _story_cache(), {}
    changed = total = 0
    for path in sorted(glob.glob(os.path.join(src, "*", "N*.jsonl"))):
        rel = os.path.relpath(path, src)
        out_path = os.path.join(dst, rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(path, encoding="utf-8") as fin, \
                open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                rec = json.loads(line)
                new = regrade_record(rec, get_story, qcache)
                new["regraded_from"] = os.path.basename(src)
                if new.get("correct") != rec.get("correct") or \
                        new.get("label") != rec.get("label"):
                    changed += 1
                total += 1
                fout.write(json.dumps(new) + "\n")
        print(f"  {rel}")
    for p in glob.glob(os.path.join(src, "*", "prompts.jsonl")):
        shutil.copy2(p, os.path.join(dst, os.path.relpath(p, src)))
    print(f"[regrade] {total} records, {changed} changed -> {dst}")


if __name__ == "__main__":
    main()
