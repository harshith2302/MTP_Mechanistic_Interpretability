"""Write each downloaded model's resolved commit sha into configs/models.yaml.

Records in results/raw/ carry model_revision, so the sweep stays reproducible.
Login node only (it reads the local HF cache, not the network).
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolved_sha(local_dir):
    ref = os.path.join(local_dir, ".cache", "huggingface", "download")
    for candidate in (os.path.join(local_dir, ".git"),):
        if os.path.isdir(candidate):
            try:
                return subprocess.check_output(
                    ["git", "-C", local_dir, "rev-parse", "HEAD"], text=True).strip()
            except subprocess.CalledProcessError:
                pass
    # hf download --local-dir leaves per-file metadata; take any recorded commit
    for dirpath, _, files in os.walk(ref if os.path.isdir(ref) else local_dir):
        for f in files:
            if f.endswith(".metadata"):
                with open(os.path.join(dirpath, f)) as fh:
                    first = fh.readline().strip()
                if re.fullmatch(r"[0-9a-f]{40}", first):
                    return first
    return None


def main():
    import yaml
    path = os.path.join(ROOT, "configs", "models.yaml")
    with open(path) as f:
        text = f.read()
    cfg = yaml.safe_load(text)
    for m in cfg["models"]:
        sha = resolved_sha(os.path.join(ROOT, m["local_path"]))
        if not sha:
            print(f"  {m['short_name']}: sha not found (leaving null)", file=sys.stderr)
            continue
        text = re.sub(
            rf"(short_name: {re.escape(m['short_name'])}(?:.|\n)*?revision: )\S+",
            rf"\g<1>{sha}", text, count=1)
        print(f"  {m['short_name']}: {sha}")
    with open(path, "w") as f:
        f.write(text)


if __name__ == "__main__":
    main()
