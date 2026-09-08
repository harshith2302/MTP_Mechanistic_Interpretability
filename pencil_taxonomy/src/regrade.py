"""Rebuild every graded field from stored `raw_output`. No GPU, seconds.

This is the constraint the whole record schema exists to satisfy: if a grading
rule turns out to be wrong, it is fixed and re-run here, not re-generated on a
GPU. Writes a sibling `<run_id>_regraded/` directory -- raw stays append-only.
"""
import argparse
import glob
import json
import os

from src.grade import grade_record
from src.util import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--in-place", action="store_true",
                    help="overwrite the run dir instead of writing a copy")
    a = ap.parse_args()

    src = os.path.join(ROOT, a.run_dir)
    dst = src if a.in_place else (a.out or src.rstrip("/") + "_regraded")
    os.makedirs(dst, exist_ok=True)
    total = changed = 0
    for path in sorted(glob.glob(os.path.join(src, "*.jsonl"))):
        recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        out = []
        for r in recs:
            before = (r.get("correct"), r.get("format_error"), r.get("parse_path"))
            g = grade_record(dict(r))
            after = (g.get("correct"), g.get("format_error"), g.get("parse_path"))
            changed += before != after
            total += 1
            out.append(g)
        with open(os.path.join(dst, os.path.basename(path)), "w",
                  encoding="utf-8") as f:
            for g in out:
                f.write(json.dumps(g) + "\n")
        print(f"  {os.path.basename(path)}: {len(out)} records")
    print(f"[regrade] {total} records, {changed} changed -> {dst}")


if __name__ == "__main__":
    main()
