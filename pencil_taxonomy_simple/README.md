# pencil_taxonomy_simple

The same state-tracking evaluation as [`../pencil_taxonomy`](../pencil_taxonomy),
with the one thing that made the story hard to *read* removed: the original
generator split people into clusters and narrated the clusters' timesteps out of
order, so a "timestep 1" paragraph could follow a "timestep 7" one. Here there are
no clusters, and the account is strictly chronological. The physics is unchanged —
same names, same initial partition, same transfer sampling, same sentence
templates — and for N < 4 (where the original also had one cluster) the two
generators produce identical states from the same seed; a test pins that.

Two further changes, both recorded in every result record:

- **The prompt states the rules.** The old prompt never said what "at timestep k"
  meant or that transfers are cumulative. The new one does, in every template.
- **Two answer modes.** `direct` (a single JSON object — comparable with
  `pencil_taxonomy`) and `scratchpad` (a timestep-by-timestep tally, then the JSON
  on the last line, parsed as the *last* object). Chosen in `config.yaml`.

And a different sweep: N and T vary **independently** over an (N, T) grid —
N = 2…30, T ∈ {1,2,3,5,8,12,16,20,25,30}, 5 stories per cell — with exactly one
transfer per timestep, so **T is the number of transactions**. The ten N = T cells
`pencil_taxonomy` ran get 30 stories each, so the diagonal figures have exactly
that experiment's footing and can be laid beside them.

**[REPORT.md](REPORT.md) is the write-up.** Headline: accuracy is a function of
T (the number of transfers) and almost not at all of N (the number of people);
one update is solved, the second is where it breaks; and chronological order
fixes retrieval (34% → 98%) but not accumulation.

## Figures

| | file | what |
|---|---|---|
| 1–4 | `fig1…fig4` | the **N = T diagonal**, same construction and filenames as `pencil_taxonomy` — compare directly |
| 5 | `fig5_accuracy_grid` | accuracy heatmaps over (N, T), one per question type, models pooled |
| 6 | `fig6_accuracy_grid_by_model` | pooled accuracy per model over (N, T) |
| 7 | `fig7_taxonomy_grid` | share of each failure category per (N, T) cell |
| C1 | `figC1_diagonal_old_vs_new` | **OLD vs NEW** accuracy per question type on the diagonal, same axes |
| C2 | `figC2_taxonomy_old_vs_new` | **OLD vs NEW** taxonomy per model, each with its own chance level |
| 3b, 3c | `fig3b_taxonomy_v2`, `fig3c_family_rollup` | the secondary 15-category pass, as in `pencil_taxonomy` |

On a heatmap N is the y-axis and T the x-axis; an overflow-emptied cell is masked
grey, never drawn as zero.

## How to run

```bash
# envs/ and models/ are symlinks to ../pencil_taxonomy's (gitignored, 68 GB)
source envs/eval/bin/activate
mkdir -p logs

python -m src.generate_stories        # 1,720 stories over 294 (N,T) cells
python -m src.generate_questions      # 8,600 questions + ground truth
python -m src.smoke --model qwen      # corners + centre of the grid; READ the output
sbatch scripts/run_model.sbatch qwen  # one job per model; extra args pass through:
#   sbatch scripts/run_model.sbatch qwen --answer-mode scratchpad
#   sbatch scripts/run_model.sbatch qwen --diagonal-only
python -m src.regrade   --run-dir results/raw/<run_id>
python -m src.aggregate --run-dir results/raw/<run_id>_regraded
python -m src.plots && python -m src.compare_previous
python -m pytest -q
```

Everything after the sweep runs without a GPU; every graded field is rebuilt from
the stored `raw_output`.

## Layout

```
config.yaml          grid, models, seeds, answer mode, budgets -- every number
src/simulation.py    the cluster-free, chronological generator (NEW)
src/prompting.py     templates + answer modes + hashes
prompts/             5 frozen templates, each stating the rules of the account
src/questions.py     the 5 types; transfer_recall now asks only unique triples
src/taxonomy.py      5 categories + Monte-Carlo chance null (unchanged)
src/aggregate.py     diagonal tables (as before) + grid tables
src/plots.py         fig1-4 diagonal (as before) + fig5-7 grid heatmaps
```
