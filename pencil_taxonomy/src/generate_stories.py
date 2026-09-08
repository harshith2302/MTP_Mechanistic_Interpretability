"""Generate the 300 stories once, before any model is loaded.

The SAME stories are used by all four models. That makes a cross-model statement
a statement about the models rather than about which stories each happened to
draw, and it leaves a paired test available later at no cost.
"""
import argparse
import json
import os

from src.generator import generate_story
from src.util import ROOT, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="data/stories")
    a = ap.parse_args()
    cfg = load_config(a.config)
    st = cfg["story"]
    out = os.path.join(ROOT, a.out)
    os.makedirs(out, exist_ok=True)

    total = 0
    for n in cfg["grid"]["n_values"]:
        path = os.path.join(out, f"N{n:02d}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for i in range(cfg["grid"]["stories_per_n"]):
                seed = st["seed_base"] + 1000 * n + i
                s = generate_story(
                    n, n, seed,
                    names_path=os.path.join(ROOT, st["names_path"]),
                    pencils_per_person=st["pencils_per_person"],
                    number_word_probability=st["number_word_probability"])
                s["story_id"] = f"N{n:02d}_s{i:03d}"
                f.write(json.dumps(s) + "\n")
                total += 1
        print(f"  N={n:2d}: {cfg['grid']['stories_per_n']} stories -> {path}")
    print(f"[stories] {total} stories written")


if __name__ == "__main__":
    main()
