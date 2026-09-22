# Pencil Exchange, simplified: chronological narration over an (N, T) grid

Four open instruction-tuned models (Qwen2.5-7B, Llama-3.1-8B, Mistral-7B-v0.3,
OLMo-2-7B), 34,400 generations, 2026-09-14. Companion to
[`../pencil_taxonomy`](../pencil_taxonomy/REPORT.md); the same question types,
grader and taxonomy, with three deliberate changes stated in every figure:

| | `pencil_taxonomy` (OLD) | this experiment (NEW) |
|---|---|---|
| narration | clusters, timesteps interleaved out of order | chronological, no clusters |
| transfers per timestep | random, 1..N | exactly 1, so **T = number of transfers** |
| prompt | bare question | rules of the account stated |
| sweep | N = T only | full (N, T) grid, N = 2..30, T ∈ {1,2,3,5,8,12,16,20,25,30} |

The grid has 5 stories per cell; the ten N = T cells the old experiment ran have 30,
so figures 1–4 share an x-axis and a footing with the old ones.

---

## 1. The three findings

**1. Accuracy is a function of T, not N.** The heatmaps (figure 5) are vertical
stripes. Pooled over all N, `person_timestep_lookup` goes 91% → 64% → 58% → 49% →
36% → 27% → 21% → 19% → 15% → 14% as T goes 1 → 2 → 3 → 5 → 8 → 12 → 16 → 20 → 25
→ 30. Pooled over all T, it is 36%, 27%, 36%, 40%, 34%, 40%, 39% as N goes 2, 5,
10, 15, 20, 25, 30 — flat. **The number of people in the story barely matters; the
number of updates is everything.**

**2. One update is solved; two is not.** At T = 1, every type is answered well:
`initial_state_lookup` 92%, `transfer_recall` 100%, `person_timestep_lookup` 91%,
`state_snapshot` 66%, `trajectory` 94%. At T = 2, `person_timestep_lookup` is
already 64%, `state_snapshot` 30%, `trajectory` 47%. By T = 5 the whole-vector
types are at 4% and 10%. The failure begins at the second update.

**3. Chronological order fixes retrieval, not accumulation.** With the story in
order, `transfer_recall` — find one narrated event — goes from 34% (OLD, N=T=30)
to 98%. `person_timestep_lookup` — one accumulated value — roughly triples on the
diagonal, from ~7% to ~20%. But `state_snapshot` and `trajectory` are at the
floor beyond T ≈ 5 **in both experiments**. Reading order was a real cost for
finding things; it was not why the models cannot carry a count through updates.

---

## 2. What the OLD → NEW gap bundles, and the fairer comparison

On the diagonal, NEW is far above OLD (figure C1). But at the same N = T the NEW
story has far fewer transfers — 30 instead of ~465 at N = 30 — so the diagonal
comparison mixes narration order with transfer density. Matching on transfer
count instead, all N pooled:

| transfers in story | `transfer_recall` OLD→NEW | `person_timestep_lookup` OLD→NEW | `state_snapshot` OLD→NEW | `trajectory` OLD→NEW |
|---|---|---|---|---|
| 1–3 | 88 → 99 | 27 → 71 | 22 → 36 | 26 → 55 |
| 4–8 | 55 → 96 | 15 → 40 | 2 → 4 | 0 → 9 |
| 9–15 | 61 → 95 | 17 → 27 | 0 → 1 | 1 → 4 |
| 16–30 | 74 → 93 | 11 → 18 | 1 → 0 | 0 → 1 |

(OLD cells at 1–8 transfers hold only 40–88 answers.) The order + prompt effect
survives the match for retrieval and for the single-value question, and it is
largest when there are few transfers. For the whole-vector questions there is
nothing to rescue at any transfer count.

---

## 3. Accuracy on the N = T diagonal

4 models pooled, 30 stories per model per point:

| type | 2 | 4 | 6 | 8 | 10 | 12 | 16 | 20 | 24 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|
| `initial_state_lookup` | 100 | 99 | 100 | 99 | 99 | 100 | 98 | 93 | 98 | 94 |
| `transfer_recall` | 92 | 92 | 95 | 94 | 98 | 92 | 92 | 97 | 97 | 98 |
| `person_timestep_lookup` | 49 | 25 | 25 | 40 | 30 | 26 | 18 | 18 | 22 | 20 |
| `state_snapshot` | 40 | 12 | 3 | 3 | 2 | 2 | 2 | 0 | 0 | 0 |
| `trajectory` | 63 | 17 | 6 | 5 | 5 | 2 | 2 | 4 | 1 | 0 |

Per model, all cells pooled: Llama 57%, Qwen 54%, OLMo-2 50%, Mistral 47%.
OLMo-2 covers the full grid this time — with one transfer per timestep every
story fits its 4,096-token window, so there are no overflow exclusions anywhere.

---

## 4. Failure taxonomy

`person_timestep_lookup` answers plus the first divergent `trajectory` element,
against the state matrix; five categories fixed before the run; Monte-Carlo chance
null on the same answer space. N = T diagonal, share of wrong answers, observed /
chance (%):

| category | Qwen | Llama | Mistral | OLMo-2 |
|---|---|---|---|---|
| `stale` — reported the initial value, applied no update | **30 / 5** | **35 / 8** | **44 / 6** | 4 / 3 |
| `wrong_person` | 31 / 25 | 29 / 30 | 28 / 28 | **39 / 25** |
| `wrong_timestep` | 12 / 5 | 7 / 4 | 6 / 5 | 9 / 6 |
| `single_transfer` — off by exactly one transfer | **14 / 5** | **18 / 6** | **11 / 5** | **16 / 4** |
| `unexplained` | 13 / 60 | 11 / 53 | 12 / 57 | 33 / 63 |

Three readings:

- **`stale` is the dominant mechanism for three of the four models**, five to
  seven times its chance level. When Qwen, Llama and Mistral get a single
  accumulated value wrong, the most common thing they return is the number as it
  was stated at timestep 0. This is the same finding as the old experiment
  (`stale` 24% vs 8% chance there), sharpened.
- **OLMo-2 fails differently.** Its `stale` rate is at chance; instead it returns
  another person's current count (`wrong_person`, 39% vs 25%). Same task, a
  different failure.
- **`unexplained` is far below chance for every model.** A random in-range integer
  lands in `unexplained` 53–63% of the time; the models' wrong answers land there
  11–33%. Their errors are structured.

The ambiguity rate (an answer matching more than one rule) is 55–61% for three
models and 27% for OLMo-2 — high, as before, because five rules over a small
integer range overlap. `stale` is priority 1 and unaffected; the relative sizes of
the categories below it should be read loosely.

Figure 7 shows each category's observed share per (N, T) cell. `wrong_person`
rises with N, which is what its chance level does too; `wrong_timestep`
concentrates at N = 2, where a person has few distinct counts to collide with.
Those cells carry no per-cell null and are shown as observed shares only.

---

## 5. Validity

| check | result |
|---|---|
| unparseable answers | 60 of 34,400 (0.17%) |
| cut off by the length budget | 44 |
| context overflow | **0** — every story fits every model |
| rescued by normalisation | 943, of which 237 were number *words* |
| re-graded from stored raw text | 235 changed — the number-word fix, below |
| tests passing | see `python -m pytest -q` |

**The number-word rescue.** The new prompt says *"four" and "4" mean the same
thing*, and OLMo-2 took it literally: 237 answers were the number copied back as a
word (`{"answer": "seven"}`). The grader only coerced numeric strings, so these
scored wrong and the `initial_state_lookup` control read ~95% instead of ~100%.
`normalise()` now reads number words (built from `num2words`, the generator's own
mapping). They count as `correct` but not `correct_strict`, and are recorded as
rescued, so the leniency is measured rather than hidden. 235 of the 237 became
correct. All 237 were OLMo-2; no other model ever answered in words.

**Smoke gate.** 75 generations across the grid's corners and centre, read before
the sweep: 75/75 parsed, 0 truncated, and the N = 2 / T = 1 corner was 15/15.

---

## 6. Limitations

- **The OLD → NEW gap bundles three changes** (order, transfer density, prompt).
  Section 2 separates density; order and prompt are not separated from each
  other here. A run with the new prompt on the old generator would do that.
- **T = transfers only because `transfers_per_timestep = 1`.** The old design had
  1..N per timestep; that variant is one line in `config.yaml`.
- **Off-diagonal cells are 5 stories per model** (20 answers per type). The
  heatmaps are for shape, not for reading a single cell.
- **`pencils_per_person = 5` is still fixed**, so total pencils scale with N.
  Since N turned out not to matter, this confound did not bite here.
- **Taxonomy scope** is unchanged: mechanisms are assigned only to single-value
  answers; whole-vector and control types are correct/wrong only.

---

## 7. Reproducing

```bash
source envs/eval/bin/activate                 # symlink to ../pencil_taxonomy/envs
python -m src.generate_stories && python -m src.generate_questions
mkdir -p logs && for m in qwen llama mistral olmo; do sbatch scripts/run_model.sbatch $m; done
python -m src.regrade   --run-dir results/raw/<run_id>
python -m src.aggregate --run-dir results/raw/<run_id>_regraded
python -m src.plots && python -m src.compare_previous
```

Run `2026-09-14T0659Z`: four jobs, 12–14 min each on one L40S. Every record
stores `template_sha256`, `prompt_sha256` and `answer_mode` (`direct`).
