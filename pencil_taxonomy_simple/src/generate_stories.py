"""Generate every story once, before any model is loaded.

The (N, T) grid from config.yaml: `stories_per_cell` stories in every cell and
`diagonal_stories` on the N = T cells, so the diagonal has the same footing as
pencil_taxonomy's N = T sweep. The SAME stories are used by all models, which
makes a cross-model statement a statement about the models rather than about
which stories each happened to draw.

Seed formula (recorded per story): seed_base + 10000*N + 100*T + index.
"""
import argparse
import json
import os

from src.simulation import generate_story
from src.util import ROOT, load_config


def story_seed(cfg, n, t, i):
    return cfg["story"]["seed_base"] + 10000 * n + 100 * t + i


def grid_cells(cfg):
    """-> [(n, t, n_stories)] for every cell: the grid, plus the N = T cells of
    pencil_taxonomy's sweep at `diagonal_stories` each."""
    g = cfg["grid"]
    diag = set(g.get("diagonal_n_values", []))
    cells = {}
    for n in g["n_values"]:
        for t in g["t_values"]:
            cells[(n, t)] = g["stories_per_cell"]
    for n in diag:
        cells[(n, n)] = g["diagonal_stories"]
    return sorted((n, t, k) for (n, t), k in cells.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="data/stories")
    a = ap.parse_args()
    cfg = load_config(a.config)
    st = cfg["story"]
    tpt = st["transfers_per_timestep"]
    tpt = tpt if tpt == "random" else int(tpt)
    out = os.path.join(ROOT, a.out)
    os.makedirs(out, exist_ok=True)

    total, cells = 0, grid_cells(cfg)
    by_n = {}
    for n, t, k in cells:
        by_n.setdefault(n, []).append((t, k))
    for n, tks in sorted(by_n.items()):
        path = os.path.join(out, f"N{n:02d}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for t, k in tks:
                for i in range(k):
                    seed = story_seed(cfg, n, t, i)
                    s = generate_story(
                        n, t, seed,
                        names_path=os.path.join(ROOT, st["names_path"]),
                        pencils_per_person=st["pencils_per_person"],
                        number_word_probability=st["number_word_probability"],
                        transfers_per_timestep=tpt)
                    s["story_id"] = f"N{n:02d}_T{t:02d}_s{i:03d}"
                    f.write(json.dumps(s) + "\n")
                    total += 1
        print(f"  N={n:2d}: {sum(k for _, k in tks):4d} stories over "
              f"{len(tks)} T values -> {path}")
    print(f"[stories] {total} stories in {len(cells)} (N,T) cells; "
          f"transfers_per_timestep={tpt}")


if __name__ == "__main__":
    main()
