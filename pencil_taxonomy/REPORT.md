# Pencil-Exchange Mini-Eval — short report

Four open ≤8B models read a story about people passing pencils around and answer
questions about who holds what. We generate the stories, so every question has
exactly one correct answer.

**4 models × 300 stories × 5 question types = 6,000 answers.** Greedy decoding,
no constrained JSON. 6–10 minutes per model on one L40S.

---

## The three findings

**1. Reading survives. Accumulating does not. The boundary is sharp.**
A question answered by copying a number written in the text: **97%** correct.
A question needing **one update** to that same number: **10%** correct.
Same stories, same models, same prompt style.

**2. The failure is at the *first* update, not gradual drift.**
Asked for a person's full history, models reproduce the stated starting value and
leave the truth immediately — at position 1–2, regardless of story length.

**3. The dominant identified mechanism is "did nothing".**
The largest real failure category is the model reporting the *initial* value
unchanged. Two independently-built taxonomies agree on this, and on little else.

---

## 1. The task

`N` people, `T` timesteps, pencils passed around, total never changes.

> At timestep 0: Anika holds four pencils, Desmond holds eight, Priya holds 1,
> Gustavo holds 7. At timestep 1, Desmond handed six pencils over to Anika. Anika
> gave seven pencils to Desmond. …

| timestep | Anika | Desmond | Priya | Gustavo |
|---|---|---|---|---|
| 0 | 4 | 8 | 1 | 7 |
| 1 | 3 | 9 | 8 | 0 |
| 2 | 11 | 1 | 8 | 0 |

Two deliberate features: numbers appear as **words and digits** mixed at random,
and the story is **not told in chronological order** — it covers several
timesteps for one group, then jumps back for another. So the task is tracking
*plus* re-sequencing. People are also split into **clusters**, and pencils never
cross a cluster boundary; this matters in §4.3.

**Setup.** Qwen2.5-7B, Llama-3.1-8B, Mistral-7B-v0.3, OLMo-2-7B. Story sizes
`N = T` ∈ {2, 4, 6, 8, 10, 12, 16, 20, 24, 30}, 30 stories each, 5 questions per
story. **The same 300 stories go to all four models**, so cross-model comparisons
are about the models, not about which stories they drew.

---

## 2. The question ladder — the core result

The five question types are a ladder in how much accumulation they need. The
ladder *is* the experiment.

| type | needs | N=2 | N=8 | N=16 | N=30 | **overall** |
|---|---|---|---|---|---|---|
| `initial_state_lookup` *(control)* | nothing — the number is in the text | 100% | 99% | 95% | 92% | **97%** |
| `transfer_recall` *(control)* | recall one narrated event | 80% | 68% | 49% | 34% | **56%** |
| `person_timestep_lookup` | **one** accumulated value | 25% | 6% | 8% | 7% | **10%** |
| `state_snapshot` | the whole state vector | 17% | 0% | 0% | 0% | **2%** |
| `trajectory` | one person's whole history | 19% | 0% | 0% | 0% | **2%** |

**Read the N=8 column.** Same model, same story: *"how many did X have at the
start?"* → 99% correct. *"how many did X have at timestep 4?"* → 6% correct. The
only difference is that one number is written down and the other must be carried
through an update.

The two controls are what make this a result rather than an observation. They
need no arithmetic, so their staying high rules out long context, reading
ability, and person↔value binding as the cause. The failure is in the **state
update** itself.

The whole-object types are at the floor from **N=4**. One accumulation step is
already most of the damage.

---

## 3. Where tracking breaks

For `trajectory` we can ask something sharper than right/wrong: **at what position
does the model's list first depart from the truth?**

Gradual drift would mean that position grows with story length. Immediate failure
means it stays near 1 forever.

**It stays flat at 1.1–1.8 for every story length.** Models reproduce the
timestep-0 value — which is written in the text — then leave the truth at the
very next step.

*Excluded:* formulaic answers (a constant list `[3,3,3,…]` or a counting sequence
`[3,6,9,…]`) and wrong-length lists. These are pattern completion, and diverge at
position 0–1 by construction. 6 of 38 cells fell below the 5-answer minimum and
were dropped.

This measurement uses **no taxonomy at all**, which makes it independent evidence
for the same conclusion §4 reaches by a different route.

---

## 4. What the wrong answers look like

Every wrong answer gets labelled with a hypothesis about what the model did
instead — *reported the starting value*, *used the wrong person's count*, *skipped
a transfer*. This is where the analysis can go badly wrong, so read this box
before the numbers.

> ### What "chance" means
>
> Answers are integers in a small range, say 0–41. A wrong answer will often
> match one of these hypotheses **by pure luck** — some number in that range
> *is* somebody else's count, *is* two away from the truth, *is* the starting
> value.
>
> So we measure the luck. Take each wrong answer, throw it away, replace it with
> a random integer from the same range, run the **identical** classifier, repeat
> 1,000 times. How often does each category fire on pure noise? That is the
> **chance** rate.
>
> **Only `observed − chance` means anything.** A category that fires on 30% of
> real answers and 30% of random ones has told you nothing.
>
> A test that comes back positive on most healthy patients is not a test.

### 4.1 Primary taxonomy — 5 categories, fixed *before* the run

| category | meaning | observed | chance | **corrected** |
|---|---|---|---|---|
| `stale` | reported the initial value; applied no update | 23.7% | 8.4% | **+15.3** ✅ |
| `single_transfer` | off by exactly one transfer involving that person | 19.2% | 13.2% | **+6.0** ✅ |
| `wrong_timestep` | right person, wrong timestep | 15.0% | 14.7% | +0.4 — chance |
| `wrong_person` | different person, right timestep | 24.5% | 31.5% | −7.0 ❌ |
| `unexplained` | matched no rule | 17.5% | 32.2% | −14.7 ❌ |

Two readings only the baseline makes possible:

- **`wrong_person` is the second-largest column and is not a finding.** Random
  numbers match somebody's count **more often** (31.5%) than the models do
  (24.5%). Reported raw, this would have been a confident claim about binding
  failure, pointing the wrong way.
- **`unexplained` at 17.5% against 32.2% chance is good news** — the wrong
  answers are *less* random than random, so they carry structure. This is the
  opposite of the usual worry that a taxonomy is labelling noise.

**One category survives strongly: `stale`.**

### 4.2 Extended taxonomy — 15 categories, designed *after* the run

Reported as **exploratory**, because it was specified once the data existed and
could in principle have been chosen to flatter it. Fifteen rules in three
families, each decidable by direct comparison against the ground-truth matrix —
no simulator replay:

| family | claims | categories |
|---|---|---|
| **A · update** | *which* updates were applied | `stale`, `partial_update`, `lookahead`, `boundary_off_by_one`, `transfer_omitted`, `transfer_doubled` |
| **B · arithmetic** | *how* they were combined | `direction_single`, `direction_global`, `arithmetic_slip`, `conservation_total`, `out_of_range`, `digit_error` |
| **C · reference** | right operation, *wrong referent* | `wrong_person_same_cluster`, `wrong_person_other_cluster`, `delta_readout` |

Examples from the data (`truth` → model's answer):

| category | example | meaning |
|---|---|---|
| `stale` | 4 → **7** | never updated; 7 was the starting value |
| `transfer_omitted` | 8 → **6** | skipped one transfer |
| `direction_global` | 2 → **0** | applied every transfer backwards |
| `delta_readout` | 5 → **3** | reported a *transfer size* instead of a *total* |
| `out_of_range` | 1 → **9** | more pencils than exist in that cluster |

**Result 1 — at family level, only "update" clears its baseline.**

| family | observed | chance | **corrected** |
|---|---|---|---|
| **A · update** | 59.6% | 39.2% | **+20.3** ✅ |
| B · arithmetic | 27.2% | 26.9% | +0.3 — chance |
| C · reference | 8.3% | 12.3% | −4.0 ❌ |

Within family A, `stale` is **+16.0** — against the primary taxonomy's
independent **+15.3**, from a different rule set and a different priority order.
**That agreement is the most trustworthy thing in this section.**

**Result 2 — the expansion mostly bought coincidence.** `unexplained` fell from
17.5% to 5.0%, which looks like a large gain until you ask where those answers
went:

| landed in | share of v1's unexplained |
|---|---|
| `digit_error` | 28.7% |
| stayed `unexplained` | 28.7% |
| `out_of_range` | 23.4% |
| everything else | 19.2% |

Over half went into two categories, and `digit_error`'s corrected rate is
**−5.2** — it fires *below* chance. The drop in `unexplained` is not explanation;
it is absorption by rules loose enough to match almost anything.

**Result 3 — the instrument is close to vacuous, and we can show it.**

| diagnostic | 5-category | 15-category |
|---|---|---|
| answers matching more than one category | 52.9% | **81.6%** |
| **random numbers receiving a mechanism label** | 67.8% | **78.4%** |
| categories matched per *random* number | — | 2.87 |
| categories matched per *real* answer | — | 3.91 |

**78% of completely random integers receive a mechanistic label**, and a random
number matches 2.87 of the 15 categories where a real answer matches 3.91. Every
corrected figure in the extended taxonomy is therefore a difference between two
large numbers. Its per-category ranking should **not** be read as a mechanism
ranking; only the family rollup above is on solid ground.

Per-category observed/chance/corrected rates for all 15 are in
`results/tables/taxonomy_v2_rates.csv`, and the full v1→v2 migration in
`results/tables/taxonomy_v1_vs_v2.csv`.

### 4.3 One clean negative result

The **cluster split** was the main reason for building the extended taxonomy.
Because pencils never cross a cluster boundary, confusing someone with a person in
their *own* group is ordinary local binding failure, while confusing them with
someone who could never have exchanged pencils with them would mean the model
never recovered the cluster structure at all.

It returns **no signal**: `wrong_person_same_cluster` at **−1.2** and
`wrong_person_other_cluster` at **−1.7**, both below chance. The models are not
making either mistake more than luck would produce. A negative result, stated
because it was a pre-declared motivation.

---

## 5. Hypotheses and verdicts

Five candidate explanations for why accumulation fails, and what this evaluation
says about each. Nothing here confirms a *mechanism* — behavioral evidence can
only narrow the field. That is the point of the exercise.

| | hypothesis | verdict |
|---|---|---|
| **H1** | **Binding** — with more people, the model loses track of which value belongs to which person | **Rejected**, by three independent measurements. `initial_state_lookup` is at 97% overall and still 92% at N=30, so binding a person to a value in a 30-person list is intact. `wrong_person` fires **below** chance (−7.0). Family C is **−4.0**. The cluster split (§4.3) shows nothing. |
| **H2** | **Retrieval** — the model looks transfers up at query time instead of maintaining state | **Partly supported, but insufficient.** `transfer_recall` does degrade, 80% → 34%, so retrieving a narrated event genuinely gets harder with story length. But it stays **five times higher** than `person_timestep_lookup` (56% vs 10%). Retrieval decay is real and cannot explain the accumulation collapse. |
| **H3** | **State update** — the running state is not maintained | **Supported — with the correction that it does not drift, it fails at once.** Three independent routes agree: the 97% → 10% control gap (§2); first divergence flat at 1.1–1.8 with no taxonomy involved (§3); and `stale` at +15.3 and +16.0 under two different taxonomies (§4). **This is the conclusion the evaluation supports.** |
| **H4** | **Arithmetic** — the model tracks correctly but computes the sums wrong | **Not supported here — and this design cannot rule it out.** Family B is **+0.3**, indistinguishable from chance, and `digit_error` is below chance. But our arithmetic rules are magnitude heuristics (`\|answer − truth\| ≤ 2`), not mechanism tests: they cannot separate a genuine sum error from a near guess. Read this as *the instrument is too blunt to see arithmetic*, not as evidence against it. **The full evaluation, which tests this hypothesis by counterfactual replay, finds arithmetic to be its largest interpretable component — so the disagreement is most likely about instruments, not models.** |
| **H5** | **Readout** — state is tracked internally but reported wrongly | **Not testable here.** The question type that would probe this — a two-way comparison, where the model must only rank two people rather than emit a number — was dropped from this design to keep the headline free of a 50%-chance question. That was the right call for the accuracy figure and it cost us this hypothesis. Do not claim anything about H5 from this data. |

**Where this points.** Everything narrows to one place: the transition from the
stated `t = 0` state to `t = 1`. `initial_state_lookup` is at ceiling and first
divergence sits at index 1, so the failure is localised to a single step, and it
is the *first* step. The natural next probe is a minimal-pair contrast — a
correct `initial_state_lookup` against a wrong `person_timestep_lookup` at `t = 1`,
same person, same story. The prompts differ by almost nothing and the model
succeeds on one and fails on the other.

**Two hypotheses this design cannot address at all**, and should not be silent
about: whether number *surface form* matters (words vs digits are mixed but never
varied as a controlled manipulation), and whether the **non-chronological
narration** contributes independently of tracking difficulty. Both need their own
experiment.

---

## 6. The figures

| figure | what to look for |
|---|---|
| **`fig1_accuracy_by_question_type.png`** — *the main one* | Leftmost panel pinned at the top across the whole x-axis; the two rightmost flat on the floor from N=4. Same stories. That contrast is the result. |
| **`fig4_first_divergence.png`** | The dashed diagonal is perfect tracking. All four models sit flat near 1, nowhere near it. |
| **`fig3_failure_taxonomy.png`** | Solid bar = observed, hatched = chance. Where the hatching is as tall as the bar, that category is luck. Shown as two bars, not a difference, so this is visible at a glance. |
| **`fig3c_family_rollup.png`** | Family A clearly exceeds its hatching; B and C do not. |
| `fig3b_taxonomy_v2.png` | All 15 categories with family separators. Busy by nature — the rollup above is the honest summary. |
| `fig2_accuracy_overall.png` — *show last* | All types pooled. **Misleading alone** — the number is held up by the two controls, so it is not a measure of tracking skill. |

---

## 7. Validity checks

| check | result |
|---|---|
| unparseable answers | 1.1% |
| answers cut off by the length budget | 0.95% |
| answers rescued by lenient grading | **0** |
| answers changed by re-grading from stored raw text | **0 of 6,000** |
| questions skipped by sampling rules | **0 of 1,500** |
| automated tests passing | 89 |

**Zero rescued** is worth a sentence: we scored every answer twice, byte-exact and
lenient (`"5"` counted as `5`). They agree on all 6,000. The headline owes nothing
to forgiving marking.

**OLMo-2's context limit.** OLMo-2 reads only 4,096 tokens and longer stories
exceed it. Those questions were **never sent** and are **excluded from every
average** — not scored wrong, which would have invented a collapse. Cells where
over half the questions overflowed are dropped, because what survives is a biased
sample of the shortest stories. **OLMo-2's lines end at N=20.**

---

## 8. Limitations

- **`N` mixes two difficulties.** Pencils per person is fixed, so larger `N` means
  larger numbers — arithmetic gets harder at the same time tracking does. Fixing
  the total instead would separate them, and is the obvious next experiment.
- **The story is not chronological**, so the task measures tracking *plus*
  re-sequencing. `N` is not a clean single-variable manipulation.
- **Category overlap is high** — 52.9% and 81.6%. Only the top-priority `stale`
  claim is strong; everything below it is weak.
- **The extended taxonomy labels 78% of random numbers.** Its per-category
  ranking is not a mechanism ranking.
- **The arithmetic hypothesis is untested rather than refuted** (H4), and the
  readout hypothesis is untestable in this design (H5).
- **30 questions per (model, size, type) cell.** Curve *shape* is reliable; small
  differences between models are not. **We claim no model ranking.**
- **Greedy decoding is not bit-identical across batch settings.** Ours were fixed
  and recorded in every answer; aggregates would reproduce elsewhere, individual
  answers would not.

---

## 9. Reproducing

`results/raw/` holds all 6,000 answers as one JSON object each, including the
model's raw text and the full settings used. Re-grading and re-plotting need no
GPU and take seconds. Every figure has a CSV behind it in `results/tables/`.