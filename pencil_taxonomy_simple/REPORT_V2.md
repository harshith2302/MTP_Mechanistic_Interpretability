# REPORT V2 — Pencil Exchange, simplified
### PENCIL EXCHANGE EXAMPLE

**SKIP IT IF YOU ALREADY KNOW THE PUZZLE**

FOR N=T=4 (this is story `N04_T04_s000` from the actual sweep)

> There are 4 people participating in the pencil exchange: Ulrich, Jamal,
Fionnuala and Priya. At timestep 0, their initial pencil holdings are as
follows: Ulrich holds seven pencils, Jamal holds 1 pencil, Fionnuala holds 10
pencils and Priya holds two pencils. At timestep 1, Fionnuala handed one pencil
over to Jamal. At timestep 2, Fionnuala sent seven pencils to Ulrich. At
timestep 3, Priya donated one pencil to Jamal. At timestep 4, Ulrich donated
twelve pencils to Priya.
>

Ground Truth Table across the time step :

| timestep | Ulrich | Jamal | Fionnuala | Priya |
| --- | --- | --- | --- | --- |
| 0 | 7 | 1 | 10 | 2 |
| 1 | 7 | 2 | 9 | 2 |
| 2 | 14 | 2 | 2 | 2 |
| 3 | 14 | 3 | 2 | 1 |
| 4 | 2 | 3 | 2 | 13 |

What changed from the previous evaluation (V1):

1. **Numbers are still written as words and as digits, mixed at random.** Same as V1.
2. **The story IS told in chronological order.** V1 split people into clusters and narrated the clusters' timesteps interleaved, so "At timestep 1" could appear after "At timestep 3". Here there are no clusters: timestep 1, then 2, then 3, then 4.
3. **Exactly one transfer per timestep.** V1 drew 1..N transfers per timestep, so a story with N=T=30 had ~465 transfers. Here it has 30. **T is now literally the number of transactions**, and N=2, T=1 is the smallest possible puzzle: one person gives some pencils to the other.
4. **No groups.** Anyone can give to anyone.
5. **The prompt states the rules of the account** (shown below). The V1 prompt never said what "at timestep k" means — before or after that timestep's transfers? — or that counts carry forward. A model that guessed "before" was scored wrong for our ambiguity.
6. **N and T are swept independently**: N = 2..30 (every value) × T ∈ {1, 2, 3, 5, 8, 12, 16, 20, 25, 30}, 5 stories per cell, plus the ten N=T cells V1 ran with 30 stories each so the two are comparable point for point.

Everything else is the same code: the same name bag, the same random initial split, the same way a transfer is drawn (random giver with pencils, random other receiver, random amount up to the giver's holding), the same sentence templates, the same grader, the same taxonomy. For N < 4 (where V1 also had a single cluster) the two generators produce identical states from the same seed — a test pins that.

# The five questions, asked on the story above

Each story is shown to the model in full, once per question type, and the model
answers a single question about it. The answer mode is **direct**: a single JSON
object, no reasoning — the same as V1, so the two are comparable.

### Question-1 `initial_state_lookup` *(control)*

The full prompt, exactly as sent (the rules block is the same in all five):

```
You will read an account of a pencil exchange and answer one question about it exactly.

How the account works:
- At timestep 0, each person's starting number of pencils is stated.
- Each later paragraph, "At timestep k, ...", lists the transfers that happen during
  timestep k, in the order they happen. Paragraphs are in chronological order:
  timestep 1, then 2, then 3, and so on.
- A transfer moves that many pencils from the giver to the receiver: the giver's count
  goes down by that amount and the receiver's count goes up by the same amount.
- A person's count "at timestep k" means their count AFTER every transfer up to and
  including timestep k has been applied. Counts carry forward: a person who is not
  mentioned in a timestep keeps the same count they had before it.
- The total number of pencils never changes.
- Numbers may be written as words or as digits; "four" and "4" mean the same thing.

Account:
<<STORY TEXT>>

Question:
How many pencils did Fionnuala have at timestep 0?

Answer with a single JSON object and nothing else. No explanation, no markdown,
no text before or after it. Use exactly this shape:
{"answer": <integer>}
```

No arithmetic, no tracking - find the name, copy the number. Answer: 10.

### Question 2 `transfer_recall` *(control)*

```
Question:
At timestep 4, how many pencils did Ulrich give to Priya?
```

Find the right sentence in the story and return the number - still no arithmetic. Answer: 12. Only (timestep, giver, receiver) triples that occur exactly once in the story are asked, so the answer is never ambiguous.

### Question 3 `person_timestep_lookup`

```
Question:
How many pencils did Jamal have at timestep 2?
```

It needs the actual holding of one person at one timestep, so it must apply the updates. Answer: 2.

### Question 4 `state_snapshot`

```
Question:
How many pencils did each person have at timestep 2? Include every one of the
4 people, using the names exactly as written in the account.

{"answer": {"<person name>": <integer>, ...}}
```

The whole state vector at one moment. Marked correct only if **every** entry is right. Answer: `{"Ulrich": 14, "Jamal": 2, "Fionnuala": 2, "Priya": 2}`.

### Question 5 `trajectory`

```
Question:
List how many pencils Priya had at each timestep from 0 to 4, in order. The list
must contain exactly 5 numbers: the count at timestep 0, then the count after
timestep 1, after timestep 2, and so on up to timestep 4.

{"answer": [<integer>, ...]}
```

One person's entire history - one whole *column* of the truth table. Marked correct only if every entry is right. Answer: `[2, 2, 2, 1, 13]`.

## Summary of all the questions

| rung | type | what the answer requires | is it in the text? |
| --- | --- | --- | --- | 
| 1 | `initial_state_lookup` | copy a number | **yes** |
| 2 | `transfer_recall` | find one event among many | **yes** |
| 3 | `person_timestep_lookup` | one number, after updates | no - must be computed |
| 4 | `state_snapshot` | all `N` numbers at one time | no - must be computed |
| 5 | `trajectory` | all `T+1` numbers for one person | no - must be computed |

# Results

34,400 generations: 8,600 per model, four models, 294 (N, T) cells. Every model answered every question; there was **no context overflow anywhere** — with one transfer per timestep every story fits even OLMo-2's 4,096-token window, so OLMo-2 has a full row this time.

## On the N = T diagonal (the V1 comparison points, 30 stories per model per cell)

Qwen2.5-7B-Instruct:

| question type | 2 | 4 | 6 | 8 | 10 | 12 | 16 | 20 | 24 | 30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `initial_state_lookup` | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 83% | 93% | 90% |
| `transfer_recall` | 90% | 97% | 100% | 97% | 100% | 97% | 93% | 100% | 100% | 100% |
| `person_timestep_lookup` | 57% | 20% | 30% | 37% | 33% | 33% | 27% | 20% | 23% | 23% |
| `state_snapshot` | 57% | 13% | 3% | 3% | 3% | 3% | 3% | 0% | 0% | 0% |
| `trajectory` | 70% | 17% | 10% | 10% | 7% | 0% | 0% | 3% | 0% | 0% |

Llama-3.1-8B-Instruct:

| question type | 2 | 4 | 6 | 8 | 10 | 12 | 16 | 20 | 24 | 30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `initial_state_lookup` | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 97% | 93% |
| `transfer_recall` | 83% | 83% | 97% | 97% | 100% | 100% | 100% | 100% | 100% | 100% |
| `person_timestep_lookup` | 63% | 33% | 33% | 57% | 37% | 33% | 23% | 40% | 37% | 20% |
| `state_snapshot` | 50% | 20% | 7% | 10% | 7% | 3% | 3% | 0% | 0% | 0% |
| `trajectory` | 87% | 33% | 13% | 10% | 13% | 10% | 3% | 13% | 3% | 0% |

Mistral-7B-Instruct-v0.3:

| question type | 2 | 4 | 6 | 8 | 10 | 12 | 16 | 20 | 24 | 30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `initial_state_lookup` | 100% | 97% | 100% | 97% | 97% | 100% | 90% | 90% | 100% | 97% |
| `transfer_recall` | 93% | 93% | 83% | 83% | 93% | 83% | 80% | 93% | 90% | 93% |
| `person_timestep_lookup` | 17% | 10% | 3% | 10% | 10% | 7% | 13% | 0% | 17% | 13% |
| `state_snapshot` | 10% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| `trajectory` | 40% | 17% | 0% | 0% | 0% | 0% | 3% | 0% | 0% | 0% |

OLMo-2-7B-Instruct (no blanks this time — every story fits):

| question type | 2 | 4 | 6 | 8 | 10 | 12 | 16 | 20 | 24 | 30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `initial_state_lookup` | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 97% |
| `transfer_recall` | 100% | 93% | 100% | 100% | 97% | 90% | 93% | 93% | 97% | 100% |
| `person_timestep_lookup` | 60% | 37% | 33% | 57% | 40% | 30% | 7% | 13% | 13% | 23% |
| `state_snapshot` | 43% | 13% | 3% | 0% | 0% | 3% | 0% | 0% | 0% | 0% |
| `trajectory` | 57% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |

## The new axis: N and T separately

This is what the grid was for. All four models pooled.

Accuracy by **T** (number of transfers), all N pooled:

| question type | T=1 | T=2 | T=3 | T=5 | T=8 | T=12 | T=16 | T=20 | T=25 | T=30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `initial_state_lookup` | 99% | 100% | 100% | 99% | 99% | 99% | 98% | 97% | 97% | 97% |
| `transfer_recall` | 100% | 99% | 99% | 98% | 95% | 95% | 93% | 93% | 93% | 94% |
| `person_timestep_lookup` | **91%** | **64%** | 58% | 49% | 36% | 27% | 21% | 19% | 15% | 14% |
| `state_snapshot` | **66%** | **30%** | 12% | 4% | 2% | 1% | 1% | 0% | 0% | 0% |
| `trajectory` | **94%** | **47%** | 25% | 10% | 7% | 3% | 1% | 1% | 1% | 0% |

Accuracy by **N** (number of people), all T pooled:

| question type | N=2 | N=5 | N=10 | N=15 | N=20 | N=25 | N=30 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `initial_state_lookup` | 100% | 100% | 98% | 98% | 96% | 96% | 97% |
| `transfer_recall` | 94% | 90% | 95% | 95% | 97% | 96% | 99% |
| `person_timestep_lookup` | 36% | 27% | 36% | 40% | 34% | 40% | 39% |
| `state_snapshot` | 24% | 15% | 9% | 12% | 4% | 10% | 4% |
| `trajectory` | 35% | 20% | 13% | 20% | 15% | 21% | 13% |

Three things to read off these two tables:

1. **Accuracy is a function of T, not N.** Down the T table every accumulation question falls monotonically. Across the N table `person_timestep_lookup` is flat at 27–40% from 2 people to 30. The heatmaps (`fig5`) are vertical stripes. **The number of people in the story barely matters; the number of updates is everything.**
2. **One update is solved. The second is where it breaks.** At T=1 the models answer every type well — even the whole 30-person state vector is right 66% of the time, and a person's 2-number trajectory 94%. At T=2, `person_timestep_lookup` is already down to 64% and `state_snapshot` to 30%. V1's headline was "fails even a single update"; with the rules stated in the prompt, that is no longer true. It fails the second.
3. **The two controls hold everywhere**, so nothing below is a reading-comprehension or long-context effect.

## Compared with V1

Same ten N=T points, same four models pooled, V1 → V2:

| type | 2 | 4 | 6 | 8 | 10 | 12 | 16 | 20 | 24 | 30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `initial_state_lookup` | 100→100 | 100→99 | 100→100 | 99→99 | 100→99 | 95→100 | 95→98 | 95→93 | 95→98 | 92→94 |
| `transfer_recall` | 80→92 | 59→92 | 73→95 | 68→94 | 58→98 | 48→92 | 49→92 | 42→97 | 39→97 | **34→98** |
| `person_timestep_lookup` | 25→49 | 16→25 | 12→25 | 6→40 | 8→30 | 7→26 | 8→18 | 7→18 | 6→22 | **7→20** |
| `state_snapshot` | 17→40 | 0→12 | 1→3 | 0→3 | 0→2 | 0→2 | 0→2 | 0→0 | 0→0 | 0→0 |
| `trajectory` | 19→63 | 1→17 | 0→6 | 0→5 | 0→5 | 0→2 | 0→2 | 0→4 | 0→1 | 0→0 |

**A caution before reading this as "chronological order fixed it."** At the same N=T the V2 story has far fewer transfers (30 vs ~465 at N=30), so the diagonal comparison mixes narration order with transfer density. Matching on the number of transfers instead, all N pooled:

| transfers in story | `transfer_recall` V1→V2 | `person_timestep_lookup` V1→V2 | `state_snapshot` V1→V2 | `trajectory` V1→V2 |
| --- | --- | --- | --- | --- |
| 1–3 | 88 → 99 | 27 → 71 | 22 → 36 | 26 → 55 |
| 4–8 | 55 → 96 | 15 → 40 | 2 → 4 | 0 → 9 |
| 9–15 | 61 → 95 | 17 → 27 | 0 → 1 | 1 → 4 |
| 16–30 | 74 → 93 | 11 → 18 | 1 → 0 | 0 → 1 |

(V1 cells at 1–8 transfers hold only 40–88 answers.) So the honest version:

- **Chronological order fixes retrieval.** `transfer_recall` — find one narrated event — goes from 34% to 98% at N=T=30, and stays above 93% at every transfer count. Reading order was a real cost for *finding things*.
- **It roughly doubles single-value accuracy at the same transfer count**, and the gain is largest when there are few transfers (27→71 at 1–3).
- **It does nothing for the whole-vector questions.** `state_snapshot` and `trajectory` are at the floor beyond ~8 transfers in both experiments. Order was not why the models cannot carry a state through updates.
- What V1→V2 still bundles: order and the new prompt are not separated from each other here. Running the new prompt on the V1 generator would do that.

## The primary taxonomy — five categories, fixed before the run

`person_timestep_lookup` answers plus the first divergent element of a `trajectory` answer, against the truth table. N=T diagonal, four models pooled, 1,643 wrong answers:

| category | plain meaning | observed | chance | **difference** |
| --- | --- | --- | --- | --- |
| `stale` | reported the initial value; applied no update at all | 29.7% | 5.6% | **+24.1** — real, and larger than V1's +15.3 |
| `single_transfer` | off by exactly one transfer involving that person | 14.4% | 4.7% | **+9.6** — real |
| `wrong_person` | someone else's count at the right moment | 31.2% | 26.8% | +4.4 — mostly luck |
| `wrong_timestep` | right person, wrong moment | 8.2% | 4.9% | +3.2 — weak |
| `unexplained` | matched none of the rules | 16.6% | 57.9% | **−41.4** |

Per model the picture splits in a way V1 could not see:

| category | Qwen | Llama | Mistral | OLMo-2 |
| --- | --- | --- | --- | --- |
| `stale` (observed / chance) | **30 / 5** | **35 / 8** | **44 / 6** | 4 / 3 |
| `wrong_person` | 31 / 25 | 29 / 30 | 28 / 28 | **39 / 25** |
| `single_transfer` | 14 / 5 | 18 / 6 | 11 / 5 | 16 / 4 |
| `unexplained` | 13 / 60 | 11 / 53 | 12 / 57 | 33 / 63 |

- For **Qwen, Llama and Mistral** the dominant failure is `stale`, at 5–7× its chance level: when they get an accumulated value wrong, the most common thing they return is the number as stated at timestep 0.
- **OLMo-2 fails differently.** Its `stale` rate is at chance. It returns another person's *current* count instead. Same task, a different mechanism.
- **`unexplained` is far below chance for every model** — a random in-range integer lands there 53–63% of the time, the models' wrong answers 11–33%. The errors are structured.

The ambiguity rate (an answer matching more than one rule) is 55–61% for three models and 27% for OLMo-2. `stale` is priority 1 and unaffected by it; the relative sizes of the categories below it should be read loosely.

## Where tracking first breaks

`fig4`. From the `trajectory` question: the index of the first wrong element, cells with fewer than 5 usable answers dropped (formulaic answers — a constant list or an arithmetic progression — and wrong-length lists are excluded, as in V1).

| T | Llama | Qwen | Mistral | OLMo-2 |
| --- | --- | --- | --- | --- |
| 2 | 2.9 | 2.5 | 2.2 | 2.4 |
| 6 | 5.3 | 3.1 | 2.2 | 1.5 |
| 10 | 6.1 | 3.5 | 2.7 | 1.1 |
| 16 | 8.9 | 3.4 | 4.9 | — |
| 20 | **10.9** | — | 3.8 | — |
| 30 | 8.1 | — | — | — |

In V1 every model sat flat near 1 at every T. This is different: **Llama now tracks correctly for roughly the first 5–10 updates** before losing the thread, and its first error moves later as T grows. Qwen and Mistral hold for 2–4 updates. OLMo-2 still loses it at the first or second. Over the whole grid, the first wrong element is at index 1 in 29% of wrong trajectories, index 2 in 25%, index 3 in 13% — a decaying distribution, not a spike at 1. (Cells are blank where fewer than 5 answers survived the exclusions; at large T most trajectory answers are formulaic or the wrong length.)

## Extended taxonomy — the same 15 categories as V1

The definitions and priority order are unchanged from V1 (see V1's report for the rule-by-rule listing). Two notes specific to this data:

- **`wrong_person_other_cluster` is empty by construction** — there are no clusters, so every other person is in the same "cluster". It is kept in the table for comparability and is 0.0%.
- **The null fires far less often here.** With exactly one transfer per timestep, replay-based rules ("off by one transfer", "one transfer reversed") match a much smaller set of integers, so chance is 1–2% per category rather than V1's 5–12%, and `unexplained` under the null is 69% rather than 22%. The corrected values are therefore *more* trustworthy than V1's, not less. This pass is run over the whole grid (8,100 labelled wrong answers), not only the diagonal.

Pooled across the four models:

| # | category | family | observed | chance | **corrected** | |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `stale` | A | 27.9% | 2.0% | **+25.9** | survives clearly |
| 2 | `wrong_person_same_cluster` | C | 17.0% | 7.3% | **+9.8** | survives — driven by OLMo-2 |
| 3 | `direction_global` | B | 9.2% | 1.5% | **+7.7** | survives — ran the story backwards |
| 4 | `boundary_off_by_one` | A | 8.5% | 1.5% | **+7.0** | survives — one timestep early or late |
| 5 | `transfer_omitted` | A | 6.2% | 1.8% | +4.4 | survives |
| 6 | `transfer_doubled` | A | 6.4% | 2.0% | +4.4 | survives |
| 7 | `direction_single` | B | 3.1% | 1.2% | +1.8 | weak |
| 8 | `digit_error` | B | 9.1% | 7.8% | +1.4 | chance — the deliberate control, as designed |
| 9 | `delta_readout` | C | 2.0% | 1.4% | +0.6 | chance |
| 10 | `lookahead` | A | 2.6% | 2.0% | +0.6 | chance |
| 11 | `out_of_range` | B | 0.5% | 0.0% | +0.5 | ⚠ chance is 0 by construction |
| 12 | `partial_update` | A | 0.7% | 0.3% | +0.4 | ⚠ suppressed by priority |
| 13 | `arithmetic_slip` | B | 0.6% | 0.2% | +0.4 | ⚠ suppressed by priority |
| 14 | `wrong_person_other_cluster` | C | 0.0% | 0.0% | 0.0 | empty by construction |
| 15 | `conservation_total` | B | 0.5% | 2.2% | −1.7 | below chance |
| — | `unexplained` | E | 5.6% | 68.8% | **−63.1** | |

| family | claim | observed | chance | **corrected** |
| --- | --- | --- | --- | --- |
| **A · update** | wrong set of transfers applied | 52.3% | 9.6% | **+42.7** ✅ |
| B · arithmetic | right transfers, wrong sum | 23.0% | 13.0% | +10.0 — real but half of it is `direction_global` |
| C · reference | right operation, wrong person | 19.1% | 8.7% | +10.4 — real, and it is OLMo-2 |

Two things changed from V1's table. Family A is still the answer, and larger. But **family B is no longer at chance**: `direction_global` — every transfer applied in the wrong direction — is a clear +7.7 here, where in V1 it was a lead driven by two models. And **family C is no longer below chance**: it is now above it, and that is one model, OLMo-2, returning other people's counts.

## Priority: how a winner is chosen

Unchanged from V1: all matching rules are recorded, one winner is reported, ranked by specificity (a rule that matches fewer integers wins), ties broken A → B → C. Fixed before the classifier was run and not tuned to the data.

## Hypotheses

| | hypothesis | verdict | the evidence in one line |
| --- | --- | --- | --- |
| **H1** | **Binding** — with more people, the model loses track of whose number is whose | **Rejected, more firmly than V1** | Accuracy is flat in N from 2 people to 30 on every question type; reading a stated number is 97% correct with 30 people |
| **H2** | **Retrieval** — it looks transfers up on demand instead of keeping a running state | **Retrieval itself is now solved** | Finding a narrated event is 93–100% at every T once the story is in order. Whatever fails, it is not finding the sentence |
| **H3** | **State update** — the running state is not maintained | **Supported** | Accuracy falls with T alone; "reported the starting value" is the top category at 5× chance for three models; one update works, the second does not |
| **H4** | **Arithmetic** — it tracks fine but adds up wrong | **Partly visible now** | `direction_global` (+7.7) and `direction_single` (+1.8) are sign errors, not tracking errors. Small next to family A, but no longer invisible |
| **H5** | **Readout** — it tracks internally but reports wrongly | **Not testable here** | Same as V1: no comparison question in the design |

**The one new thing this evaluation adds:** V1 could not say whether the models fail because of *how many people* or *how many updates*, because N and T were locked together. They fail because of updates. Thirty people with one transfer is easy; two people with thirty transfers is not.

## Validity

| check | result |
| --- | --- |
| unparseable answers | 60 of 34,400 (0.17%) |
| cut off by the length budget | 44 |
| context overflow | **0** |
| answers rescued by lenient grading | 943 — 237 of them number *words* (below) |
| re-graded from stored raw text | 235 changed, all from the number-word rule |
| tests passing | 119 |

**The number-word rescue.** The new prompt says *"four" and "4" mean the same thing*, and OLMo-2 took it literally 237 times: `{"answer": "seven"}`, the number copied back as a word. The grader only coerced numeric strings, so these scored wrong and OLMo-2's control read 95% instead of 100%. `normalise()` now reads number words (from `num2words`, the generator's own mapping). They count as correct but never as strict-correct, and are recorded as rescued, so the leniency is measured rather than hidden. 235 of the 237 became correct. No other model ever answered in words.

### The figures

| figure | what to look for |
| --- | --- |
| `fig5_accuracy_grid.png` — **the main one** | Six heatmaps, N on the vertical, T on the horizontal. Vertical stripes: colour changes left-to-right with T and barely at all bottom-to-top with N. That is the result. |
| `figC1_diagonal_old_vs_new.png` | V1 (grey, dashed) and V2 (blue) on the same axes, same points. `transfer_recall` jumps to the ceiling; the two rightmost panels stay on the floor in both. |
| `fig1_accuracy_by_question_type.png` | Same construction as V1's main figure, on the diagonal. Compare directly. |
| `fig4_first_divergence.png` | In V1 every line sat flat near 1. Here Llama climbs toward the diagonal for the first ~10 updates. Where the lines end is where fewer than 5 answers survived. |
| `fig3_failure_taxonomy.png` | Solid = observed, hatched = chance. `stale` towers over its hatching for three models; for OLMo-2 it is `wrong_person` instead. |
| `figC2_taxonomy_old_vs_new.png` | V1 and V2 bars side by side per model, each with its own chance level. |
| `fig7_taxonomy_grid.png` | Each category's share per (N, T) cell. `wrong_person` rises with N — as its chance does; `wrong_timestep` concentrates at N=2 where there are few distinct counts to collide with. Observed shares only, no per-cell null. |
| `fig3c_family_rollup.png` | Family A far above its hatching; B and C now modestly above theirs. |
| `fig3b_taxonomy_v2.png` | All fifteen categories. Busy by nature — the family rollup is the honest summary. |
| `fig6_accuracy_grid_by_model.png` | The pooled heatmap per model. Llama's stripes reach further right. |
| `fig2_accuracy_overall.png` — **show last, if at all** | All types pooled into one number. Held up by the two controls; not a measure of tracking. |
