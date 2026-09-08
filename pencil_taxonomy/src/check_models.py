"""Verify each model is in the HuggingFace cache; download any that are missing.

Login node only -- compute nodes are air-gapped. The token comes from the
HF_TOKEN environment variable and is never read from or written to a file here.
"""
import os
import sys

from src.util import load_config


def main():
    cfg = load_config()
    from src.util import ROOT
    missing = []
    for m in cfg["models"]:
        p = os.path.join(ROOT, m["local_path"])
        shards = ([f for f in os.listdir(p) if f.endswith(".safetensors")]
                  if os.path.isdir(p) else [])
        if os.path.isfile(os.path.join(p, "config.json")) and shards:
            print(f"  [ ok ] {m['key']:8s} {len(shards)} shard(s)  {m['local_path']}")
        else:
            print(f"  [MISS] {m['key']:8s} no config.json/safetensors at {m['local_path']}")
            missing.append(m["hf_id"])
    if missing:
        print("\nMissing; fetch on the LOGIN node (compute nodes have no network):")
        for r in missing:
            print(f"  hf download {r}")
        sys.exit(1)
    print("\nAll four models resolve from cache.")


if __name__ == "__main__":
    main()
