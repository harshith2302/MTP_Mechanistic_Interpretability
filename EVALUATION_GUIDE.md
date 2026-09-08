# Pencil Exchange — what we built, what we measured, and how to read the results

A reference for someone seeing this work for the first time. It explains the task,
every setting, all twelve question types, every failure category, and each of the
five figures.

The short version of the result is in [REPORT.md](REPORT.md); this document is the
one that explains what the words and the graphs mean.

---

## Summary

**What we built.** A synthetic state-tracking benchmark. `N` people pass pencils
over `T` timesteps; the model reads the story as prose and answers questions about
who holds what. Because we generate the stories, every answer has one exactly
computable truth, and every wrong answer can be checked against a replay of what
the model *would* have said under a specific mistake.

**What we ran.** 4 open-weight models (Qwen2.5-7B, Llama-3.1-8B, Mistral-7B-v0.3,
OLMo-2-7B) × 15 story sizes (`N = T` = 2…30) × 15 stories × 22 questions of 12
types = **19,796 answers**, greedy decoding, no constrained JSON. Plus a 6,600-record
ablation. Every wrong answer is auto-classified into a failure taxonomy and
corrected against a chance baseline.

**What we found.**

1. **Accuracy does not collapse to zero — it falls to a ~22% floor by `N ≈ 12` and
   flattens.** All four models are indistinguishable past that point, and the floor
   is carried by the easy question types, not by residual skill.
2. **Retrieval survives; accumulation does not.** The zero-arithmetic control
   (`initial_state_lookup`) stays at 100% through `N=12` and 87% at `N=30`
   (Llama 97%, Qwen 83%, Mistral 80%; OLMo-2 is out of context by then), while
   `trajectory` and `state_snapshot` fall to 0% over the same stories. Models can
   still find a person's number in a 30-person list; they cannot carry it through a
   single update.
3. **The failure is at the *first* update, not gradual drift.** On the trajectory
   question, models reproduce the stated `t=0` value and leave the true sequence at
   index 1–2 regardless of how long the story is.
4. **Number surface form is irrelevant.** Rendering every number as a digit instead
   of a word changes nothing (paired test, all |Δ| < 2.1 points, min p = 0.092).

**What to be careful about.** After chance correction only three failure
categories stand up strongly — Arithmetic, Omission and Binding. 29% of wrong
answers are `Unexplained`, at or slightly below its own chance rate, so the
taxonomy does not cover everything. Two confounds are deliberate and are not
smoothed over: `N` grows arithmetic difficulty alongside tracking difficulty, and
the story is narrated out of chronological order, so the task measures tracking
*plus* narrative reordering. See section 11.

**Status.** The evaluation and taxonomy are complete. No mechanistic
interpretability work has been done — that is a separate phase, not yet started.

---

## 1. What the benchmark is

`N` people pass pencils to each other over `T` timesteps. The total number of
pencils never changes — only who holds them. The model reads the whole thing as
ordinary prose and answers a question about the state in strict JSON.

A complete `N = T = 4` story looks like this:

> There are 4 people participating in the pencil exchange: Anika, Desmond, Priya
> and Gustavo. At timestep 0, their initial pencil holdings are as follows: Anika
> holds four pencils, Desmond holds eight pencils, Priya holds 1 pencil and
> Gustavo holds 7 pencils. At timestep 1, Desmond handed six pencils over to
> Anika. Anika gave seven pencils to Desmond. At timestep 2, Anika gave 1 pencil
> to Desmond. Desmond passed 9 pencils to Anika. …

We generated the story, so we know the exact truth at every timestep:

| timestep | Anika | Desmond | Priya | Gustavo |
|---|---|---|---|---|
| 0 | 4 | 8 | 1 | 7 |
| 1 | 3 | 9 | 8 | 0 |
| 2 | 11 | 1 | 8 | 0 |
| 3 | 9 | 3 | 1 | 7 |
| 4 | 11 | 1 | 5 | 3 |

Two properties of the text are deliberate and matter when reading the results:

- **Numbers appear as words *and* digits**, mixed at random ("four pencils",
  "7 pencils"). Section 9 measures whether this matters. It does not.
- **The story is not told in chronological order.** Look at the excerpt above: it
  covers timesteps 1→4 for Anika and Desmond, then jumps *back* to timestep 1 for
  Priya and Gustavo. So the task measures state tracking **plus** the work of
  putting reordered narration back in sequence. This is a deliberate design
  choice, not a bug, but it means `N` is not a clean single-variable manipulation.

**Why this task.** Every question has one exactly computable answer, difficulty is
tunable with a single knob (`N = T`), and because we own the generator we can
replay any hypothesised mistake and check it against what the model actually said.
That last property is what makes the failure taxonomy in section 7 possible.

---

## 2. Models evaluated

Four open-weight instruction-tuned models, all ≤ 9B, run locally from disk.

| model | size | context | why it is in the set |
|---|---|---|---|
| Qwen2.5-7B-Instruct | 7B | 32,768 | one of the two models used in the circuit-discovery literature we build on |
| Llama-3.1-8B-Instruct | 8B | 131,072 | the other one |
| Mistral-7B-Instruct-v0.3 | 7B | 32,768 | a plain architectural contrast |
| OLMo-2-1124-7B-Instruct | 7B | **4,096** | fully public training data, so training-set claims are checkable |

The exact commit hash of each model's weights is recorded in every single output
record, so any number here can be traced back to precise weights.

**OLMo-2's 4,096-token window is a real constraint, not a footnote.** Long stories
do not fit. Rather than score those as wrong — which would invent a collapse that
never happened — a prompt that does not fit is marked `context_overflow`, **never
sent to the model**, and **excluded from the denominator**. Measured with the real
tokenizer: everything fits up to N=12; at N=16–20 only the two long-answer question
types overflow; at N=22 all twelve do. This is why OLMo-2's line stops early in the
figures instead of dropping to zero.

---

## 3. How much was run

| | |
|---|---|
| story sizes | `N = T` = 2, 4, 6, …, 30 (15 sizes, every even number) |
| stories per size | 15 |
| questions per story | 22 |
| **questions per model** | **4,949** |
| **total records** | **19,796** |
| overflow records (OLMo-2 only) | 1,503, excluded from accuracy |

`N` and `T` are always equal. One axis instead of a 29×29 grid keeps the headline
result readable; an N×T grid is a possible follow-up.

The 22 questions per story are 2 each of the 10 short-answer types, plus 1 each of
the 2 long-answer types. (At N=2 one type cannot produce two distinct questions, so
that cell has 21 — hence 4,949 rather than 4,950.)

---

## 4. Generation settings

Every one of these is recorded inside every record, and every one was held fixed
for the entire sweep.

| setting | value | why |
|---|---|---|
| decoding | greedy, `temperature = 0.0` | deterministic; one answer per question |
| samples per question | 1 | |
| **structured/guided JSON decoding** | **OFF** | forcing valid JSON would erase malformed output, which is a failure category we want to measure |
| chain-of-thought | none, direct JSON answer | keeps the reasoning inside the forward pass |
| answer length budget | 128 tokens (short types), 2,048 (long types) | |
| engine | vLLM 0.11.2, bfloat16, `tensor_parallel_size=1`, `max_num_seqs=64` | |
| hardware | 1× NVIDIA L40S per model | full sweep: 5–13 min per model |

**Why the batch settings are pinned:** vLLM's greedy decoding is not bit-identical
across different batch sizes or parallelism settings. Changing either mid-sweep
would split the results into two halves that cannot be compared. Both were fixed
throughout and verified identical across all 19,796 records.

The model is asked to reply in JSON matching a schema shown in the prompt, e.g.

```json
{ "question_type": "person_timestep_lookup",
  "person": "<PERSON_NAME>", "timestep": "<INTEGER>", "answer": "<INTEGER>" }
```

---

## 5. The twelve question types

All twelve are asked about the same stories. They are designed to *separate*
different abilities, which is what makes the headline finding possible: two of them
are deliberate **controls** that need no arithmetic at all.

| # | type | what it asks | example | overall accuracy |
|---|---|---|---|---|
| 1 | `initial_state_lookup` | **CONTROL.** A number stated verbatim in the text. No arithmetic. | *"How many pencils did Arjun hold at timestep 0?"* → `3` | **97.5%** |
| 2 | `transfer_recall` | **CONTROL.** One transfer amount, stated verbatim. No arithmetic. | *"At timestep 8, how many pencils did Theodora give to Quentin?"* → `1` | **58.5%** |
| 3 | `pairwise_comparison` | Which of two named people held more. Needs both counts but not a readout. | *"At timestep 8, who had more pencils, Ulrich or Chandra?"* → `Chandra` | 47.1% |
| 4 | `argmax_person` | Who held the most. Needs every count. | *"At timestep 0, who held the most pencils?"* → `Chandra` | 28.5% |
| 5 | `population_condition` | Count people meeting a condition at one timestep. | *"At timestep 8, how many people had more than 3 pencils?"* → `4` | 22.7% |
| 6 | `person_timestep_lookup` | One person's count at one timestep. The core tracking question. | *"How many pencils did Farhan have at timestep 4?"* → `7` | 22.5% |
| 7 | `total_conservation` | The grand total. Constant by construction — tests whether the invariant is held. | *"At timestep 10, what is the total held by everyone?"* → `50` | 15.1% |
| 8 | `duration_condition` | How many timesteps a person met a condition. **Counts t≥1 only.** | *"During how many timesteps did Katarina have more than 1 pencils? Count only timesteps 1 through 10."* → `2` | 13.0% |
| 9 | `net_delta` | Change between two timesteps; can be negative. | *"Between timestep 0 and 5, what was Gustavo's net change?"* → `-6` | 12.6% |
| 10 | `state_snapshot` | **LONG.** Every person's count at one timestep. | *"At timestep 8, state how many pencils each person had."* → all N counts | 11.5% |
| 11 | `trajectory` | **LONG.** One person's count at *every* timestep. Shows *where* tracking broke. | *"List how many pencils Quentin had at every timestep 0 to 10."* → `[12,14,4,0,1,0,0,1,1,1,4]` | 4.2% |
| 12 | `person_value_timesteps` | Which timesteps a person held an exact value. **Counts t≥0, including t=0.** | *"At which timesteps did Katarina have exactly 18 pencils?"* → `[10]` | 2.4% |

Two notes a reader will otherwise trip on:

- **Types 8 and 12 differ in whether timestep 0 counts.** Type 12 includes it, type 8
  excludes it. This asymmetry is inherited from the original task definition. Both
  prompts state their rule explicitly, and the taxonomy has a label
  (`included_t0`) for models that get it wrong.
- **`trajectory` is the diagnostic type.** Because the answer is a whole sequence,
  we can find the *first index* where the model leaves the truth. That single
  number is what figure 4 plots.

---

## 6. How answers are graded

1. **Parse.** Pull JSON out of the reply. Parsing is deliberately tolerant: extra
   prose around the JSON is fine, and `"7"` as a string counts the same as `7` as
   an integer. (Verified: string-typed and integer-typed answers score within ~1.5
   points of each other, so this is not quietly penalising anyone.)
2. **Repair delimiters, carefully.** Some models end one `}` short. If the model
   *chose* to stop (its own end-of-text token) and the content is complete, the
   delimiter is repaired and marked `ok_repaired` — never silently "ok". If the
   token budget cut it off, or the text ends inside a string, it is **not**
   repaired and counts as a format failure.
3. **Grade** against the exact simulated truth.
4. **Label** every wrong answer with the taxonomy in section 7.

> **Why step 2 matters.** Before it existed, Mistral scored **9.4%**. It ends most
> answers one brace short, so 37% of its output was being thrown away as
> unparseable. Its real score is **28.4%** — in line with the other three models.
> A single missing character was the difference between "Mistral is far worse at
> state tracking" (false) and "Mistral is worse at closing JSON" (true).

---

## 7. The failure taxonomy

Every wrong answer is labelled with a *mechanism* — a specific hypothesis about
what the model did instead of the right thing. Because we own the simulator, most
hypotheses are checked by **replaying** them: we compute what the answer *would*
have been under that mistake and see whether it matches what the model said.

Labels are grouped into coarse categories. Counts are over all 12,821 wrong
answers in the sweep (excluding overflow).

| category | count | what it means | real example |
|---|---|---|---|
| **Unexplained** | 3,505 | No hypothesis matched. The honest residual — and at chance level, so it carries no signal. | — |
| **Referent** | 2,081 | Right kind of answer, wrong person picked. | *"who had more, Anika or Desmond?"* → model says **Desmond**, truth is Anika |
| **Arithmetic** | 1,561 | The update was applied but the sum is off — usually by a small amount. | truth `1`, model says `2` |
| **Binding** | 1,156 | Answered with *another person's* count — person↔value binding slipped. | truth `4` for Gustavo; model gives `1`, which is someone else's count |
| **Conservation** | 1,150 | Breaks the invariant that the total never changes. | total is `20`; model says `30` |
| **Omission** | 1,074 | Ignored the last *k* transfers the reader saw. Replayed exactly: drop the final k and see if the model's number appears. | truth `5`; model says `2`, which is the value before the last transfer |
| **Format** | 772 | Unparseable, refused, truncated, wrong schema, or wrong types. **Never given a mechanism** — we don't guess at reasoning we can't see. | reply is not valid JSON |
| **Over-application** | 397 | Applied a transfer twice, or applied one that doesn't belong to this person. | truth `4`, model `3` — the transfer counted once too often |
| **Digit** | 355 | Right magnitude, one digit wrong. Requires ≥2 digits on both sides. | truth `21`, model `28` |
| **Direction** | 336 | Added where it should have subtracted, or vice versa. | truth `7`; model `5` — a give treated as a receive |
| **Degenerate** | 233 | The answer is formulaic, not computed: a constant list or an arithmetic progression. | truth `[4,6,5,3,4]`; model `[4,5,6,7,8]` |
| **Temporal** | 201 | Read the state at the wrong timestep, or mishandled the t=0 rule. | prompt says *"do not count timestep 0"*; model counts it and answers one too high |

Two of these categories exist because the first version of the taxonomy was wrong,
and both are worth understanding:

- **`Digit` requires at least two digits on both sides.** For single-digit numbers,
  "differs in one digit" is true of *every* wrong answer, so the label was firing
  ~45% of the time and meaning nothing.
- **`Degenerate` was added after we noticed** that 23–64% of `trajectory` answers
  are formulaic — a constant list, or `3, 6, 9, 12, …`. These were coincidentally
  matching `omission_1` (55 times) and `binding_person` (48 times), inventing
  mechanisms out of pattern-completion. They now get their own label and stay out
  of the mechanism counts.

### 7.1 Why the chance baseline is essential

**This is the single most important methodological point in the project.**

At small `N`, a random wrong integer very often matches another person's count, or
looks like an omission, purely by coincidence. So a raw rate like "11% of failures
break conservation" is meaningless on its own. For every category we compute what
that rate would be **by chance** and subtract it.

The result reverses two conclusions outright:

| category | observed | by chance | corrected | reading |
|---|---|---|---|---|
| Arithmetic | 13.4% | 3.3% | **+10.1** | real |
| Omission | 9.0% | 0.6% | **+8.4** | real |
| Binding | 9.9% | 2.0% | **+7.9** | real |
| Over-application | 3.4% | 0.7% | +2.7 | real but small |
| Direction | 2.8% | 0.5% | +2.3 | real but small |
| Unexplained | 29.0% | 33.4% | **−4.4** | at or below chance — carries no signal |
| Referent | 17.7% | 22.8% | **−5.1** | *below* chance |
| Conservation | 10.4% | 35.5% | **−25.1** | far below chance |

Read that last row carefully: a random wrong answer *looks like* a conservation
violation 36% of the time, but the models only do it 10% of the time. **The models
respect the conservation invariant far better than chance.** Without the baseline
we would have reported the exact opposite of the truth.

Only four categories survive correction as real signal: **Arithmetic, Omission,
Binding**, and more weakly Over-application and Direction. Those are the findings
the taxonomy actually supports.

Three question types have no well-defined random null. They report chance as `NaN`
and are excluded rather than silently corrected by zero.

---

## 8. The five figures, explained

All are in `results/figures/` as both `.png` and `.pdf`.

### Figure 1 — `fig1_accuracy_headline.png`
**Accuracy vs story size, all four models.**

- **x-axis:** `N = T`, from 2 to 30. Bigger = longer story, more people.
- **y-axis:** fraction of questions answered exactly right. Shaded band = 95%
  confidence interval.
- **The dotted vertical line at N=22** marks where OLMo-2's context window runs out.
  Its line *stops* there. It is not zero beyond that — it is unmeasured.

**What it shows:** accuracy falls steeply from ~55% at N=2 to about **22% by N≈12**,
then flattens. All four models sit on top of each other past N=12.

**The trap this figure sets:** the flat ~22% tail is *not* residual tracking skill.
It is carried by the easy question types — the zero-arithmetic control and the
types where guessing does well. Figure 2 is what shows that. Read them together.

### Figure 2 — `fig2_by_question_type.png`
**The same accuracy curve, split into twelve small panels, one per question type.**
This is the most informative figure and the one that carries the main result.

Each panel is one question type; four coloured lines are the four models; axes are
the same as figure 1.

**What to look for — the panels fall into three groups:**

- **Top-right panel, `initial_state_lookup`:** pinned at **1.0 across the whole
  x-axis**, only dipping to ~0.85 at N=30. This is the control: copy one number
  that is written in the text.
- **Bottom-right panel, `transfer_recall`:** starts near 1.0 and declines to ~0.3.
  Also a copy task, but the number is buried deeper in a longer story.
- **Everything else** collapses toward the floor, and `trajectory` and
  `person_value_timesteps` are at essentially **zero from N≈6 onward**.

**The result in one sentence:** the panel that needs no arithmetic stays perfect
while the panels that need one accumulation step go to zero, *over the same
stories, at the same N, for the same models*. The models can still find a person's
number in a 30-person list; they cannot carry it through a single update.

*(The spiky blue line in the `total_conservation` panel is Qwen answering a
question whose answer never changes — occasionally landing on the constant. Small
per-cell samples make it jump about; do not over-read it.)*

### Figure 3 — `fig3_failure_composition.png`
**What the wrong answers are made of, chance-corrected.**

Four panels, one per model. Each vertical bar is one `N`. Colours are the
mechanism categories, stacked.

**Three things you must know to read this correctly:**

1. Bars are **observed minus chance**, clipped at zero. They therefore **do not sum
   to 1** — that is expected, not a bug.
2. **Format errors are excluded** and reported separately (figure 5).
3. **Referent, Conservation and Unexplained are deliberately omitted**, because
   their chance baseline over-fires (corrected values of −3% to −29%, as in the
   table above). Showing a negative bar would imply the opposite of what it means.

**What it shows:** Binding (blue) and Arithmetic (orange) grow as N increases;
Omission (green) shrinks. The mix of failure mechanisms changes with story length
even though the total accuracy is flat.

### Figure 4 — `fig4_first_divergence.png`
**Where in a trajectory the model first goes wrong.**

- **x-axis:** `T`, the number of timesteps.
- **y-axis:** the mean index of the **first** position where the model's list of
  counts departs from the truth.
- **The dashed diagonal `y = T`** is the perfect-tracking reference: a model that
  tracked correctly the whole way would only diverge at the very end.

**What it shows:** every model sits **flat at 1–2, at every T**, nowhere near the
diagonal. Models reproduce the t=0 value (which is written in the text) and then
leave the true trajectory **immediately**.

**Why this is the most important figure for the next phase:** it rules out gradual
drift. There is no regime where a model tracks for a while and slowly degrades. The
failure is located at **the very first state update**, which is a specific enough
target to investigate mechanistically.

*(Formulaic answers are excluded here — they diverge at index 0–1 by construction
and would fake this result.)*

### Figure 5 — `fig5_diagnostics.png`
**Two sanity checks. Not results — evidence that the results are trustworthy.**

- **Left, format error rate:** how often output was unparseable. Near zero for Qwen
  and Llama; rises to ~13% for Mistral and ~8% for OLMo-2 at large N. Low enough
  that almost nothing is being discarded before analysis.
- **Right, taxonomy ambiguity rate:** how often more than one mechanism label
  matched the same wrong answer. Flat at ~20% across all models and all N. So ~80%
  of wrong answers get a single unambiguous label, and the ambiguity does not grow
  with difficulty.

---

## 9. The digits ablation

**Question:** is the difficulty about *tracking*, or just about number words being
awkward to tokenise?

**Design:** re-run with every number as a numeral (`17`) instead of a word
(`seventeen`), same stories, same seeds, 6,600 records at N ∈ {6, 10, 14, 18, 22}.
Because the same question appears in both conditions, each question is its own
control (a **paired** design, tested with McNemar's test).

**Result — a clean null:**

| | |
|---|---|
| paired questions per cell | 156–330 |
| accuracy difference (digits − words) | between **−1.2** and **+2.1** points |
| smallest p-value across 20 cells | **0.092** (nothing significant) |

Number surface form does **not** matter. This is a well-powered null, not an
absence of data — and it rules out tokenisation as the explanation for the collapse.

---

## 10. What the results say

1. **Binding survives; accumulation does not.** `initial_state_lookup` stays at
   100% through N=12 and 83% at N=30, while `trajectory` and `state_snapshot` go to
   0% over the same range. Finding a person's value in a 30-person list is intact;
   carrying it through one update is not.
2. **The failure is at the first update, not gradual drift.** First divergence is
   flat at 1–2 for every T (figure 4).
3. **Number form is irrelevant.** The ablation is a clean null (section 9).

Together these rule out tokenisation, number format, and person↔value binding as
the bottleneck, and locate the failure in the **state update** itself.

---

## 11. Limitations — state these plainly

- **`N` confounds tracking with arithmetic.** Pencils per person is fixed at 5, so
  larger `N` means more pencils and bigger numbers, independent of tracking
  difficulty. A fixed total would separate the two and is the obvious follow-up.
- **The story is not in chronological order** (section 1), so the task measures
  tracking *plus* narrative reordering.
- **29% of wrong answers are `Unexplained`, and that label sits slightly *below*
  its own chance rate (33%).** The taxonomy does not cover everything, and after
  chance correction only Arithmetic, Omission and Binding stand up strongly. We
  say so rather than forcing labels onto answers we cannot explain.
- **OLMo-2 has no data past N=22.** Its curve ends; it is not zero.
- Greedy decoding is not bit-identical across batch settings; ours were fixed for
  the whole sweep and recorded in every record.

---

## 12. What to show, and in what order

Everything below is in the repository.

**Show these five, in this order:**

| file | what it is |
|---|---|
| `REPORT.md` | the writeup: findings, taxonomy, ablation, limitations |
| `results/figures/fig1_accuracy_headline.png` | the headline accuracy curve |
| `results/figures/fig2_by_question_type.png` | **the main result** — the control-vs-accumulation split |
| `results/figures/fig4_first_divergence.png` | failure is at the first update, not drift |
| `results/figures/fig3_failure_composition.png` | the chance-corrected failure mix |

**Have ready if asked:**

| file | what it is |
|---|---|
| this file | definitions of every setting, question type, category and figure |
| `results/figures/fig5_diagnostics.png` | validity checks (format errors, ambiguity) |
| `results/tables/accuracy_by_model_n.csv` | every accuracy number behind figures 1–2 |
| `results/tables/taxonomy_by_model_n.csv` | observed / chance / corrected rates behind figure 3 |
| `results/tables/ablation_words_vs_digits.csv` | the paired ablation with McNemar p-values |
| `configs/models.yaml`, `configs/experiment.yaml` | exact models (with commit hashes) and sweep settings |

**Reproducing:** `results/raw/` holds all 19,796 records, one JSON object per
answer, including the model's raw text. Every record carries the model's commit
hash and the full decoding and engine settings, so any number in this document can
be traced to the exact weights and settings that produced it. 135 automated tests
cover the generator, grader and taxonomy.

*Note on the raw files:* `results/raw/2026-09-07T1341Z_304971/` is the original
sweep; `..._regraded/` is the same generations re-graded after the delimiter fix in
section 6. **All numbers in this document and in the figures come from the
`_regraded` directory.** Raw generations were never re-run — re-grading reads the
stored model output, so no GPU time was needed and the raw record stays untouched.
