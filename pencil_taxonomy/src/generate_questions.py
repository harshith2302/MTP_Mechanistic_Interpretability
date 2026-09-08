"""Derive 5 questions per story from M and transfers. Model-independent."""
import argparse
import glob
import json
import os

from src.questions import questions_for_story
from src.util import ROOT, load_config


def story_meta(s):
    """The minimal ground-truth bundle the taxonomy needs, carried per question
    so grading never has to re-run the generator."""
    return {
        "M": {int(t): v for t, v in s["states"].items()},
        "person_ids": s["person_ids"],
        "transfers": [(e["timestep"], e["giver"], e["receiver"], e["amount"])
                      for e in s["narration"]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--stories", default="data/stories")
    ap.add_argument("--out", default="data/questions.jsonl")
    a = ap.parse_args()
    cfg = load_config(a.config)
    types = cfg["question_types"]

    n_q = n_skip = 0
    with open(os.path.join(ROOT, a.out), "w", encoding="utf-8") as out:
        for path in sorted(glob.glob(os.path.join(ROOT, a.stories, "N*.jsonl"))):
            for line in open(path, encoding="utf-8"):
                s = json.loads(line)
                s["states"] = {int(k): v for k, v in s["states"].items()}
                for q in questions_for_story(s, s["story_id"], types):
                    if "question_skipped" in q:
                        n_skip += 1
                        print(f"  [skip] {q['question_id']}: {q['question_skipped']}")
                    q["story_text"] = s["text"]
                    q["story_meta"] = story_meta(s)
                    out.write(json.dumps(q) + "\n")
                    n_q += 1
    print(f"[questions] {n_q} questions ({n_skip} skipped) -> {a.out}")


if __name__ == "__main__":
    main()
