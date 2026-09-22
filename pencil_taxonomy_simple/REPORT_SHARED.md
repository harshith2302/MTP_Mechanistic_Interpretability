# IREPORT 
### PENCIL EXCHANGE EXAMPLE

**SKIP IT IF YOU ALREADY KNOW THE PUZZLE**

FOR N=T=4 

> There are 4 people participating in the pencil exchange: Ximena, Quentin,
Nadia and Ingrid. At timestep 0, their initial pencil holdings are as follows:
Ximena holds 2 pencils, Quentin holds four pencils, Nadia holds two pencils and
Ingrid holds 12 pencils. At timestep 1, Ingrid dropped 5 pencils into Nadia's
bag. Ingrid let Nadia have 1 pencil. At timestep 1, Quentin sent 2 pencils to
Ximena. At timestep 2, Nadia transferred seven pencils to Ingrid. Ingrid
donated three pencils to Nadia. At timestep 3, Ingrid handed 4 pencils over to
Nadia. At timestep 2, Ximena sent one pencil to Quentin. Quentin handed one
pencil over to Ximena. At timestep 4, Ingrid dropped three pencils into Nadia's
bag. At timestep 3, Ximena transferred two pencils to Quentin. At timestep 4,
Quentin gave 1 pencil to Ximena. Ximena sent 1 pencil to Quentin.
> 

Ground Truth Table across the time step : 

| timestep | Ximena | Quentin | Nadia | Ingrid |
| --- | --- | --- | --- | --- |
| 0 | 2 | 4 | 2 | 12 |
| 1 | 4 | 2 | 8 | 6 |
| 2 | 4 | 2 | 4 | 10 |
| 3 | 2 | 4 | 8 | 6 |
| 4 | 2 | 4 | 11 | 3 |
1. **Numbers are written as words and as digits, mixed at random.** 
2. **The story is not told in chronological order. As there are clusters which are independent we can change the chronological order**
3. **Pencils never cross between groups**

# The five questions, asked on the story above

Each story is shown to the model in full, once per question type, and the model
answers a single question about it.

### Question-1 `initial_state_lookup` *(control)*

```
Below is a description of a pencil exchange.

<<STORY TEXT>>

Question:
How many pencils did Ingrid have at timestep 0?

Answer with a single JSON object and nothing else. No explanation, no markdown,
no text before or after it. Use exactly this shape:
{"answer": <integer>}
```

No arithmetic, no tracking  - find the name, copy the number.

### Question 2  `transfer_recall` *(control)*

```
Question:
At timestep 2, how many pencils did Ximena give to Quentin?
```

Find the right sentence in the story and return the correct answer - still no arithmetic

### Question 3 `person_timestep_lookup`

```
Question:
How many pencils did Nadia have at timestep 3?
```

It needs to return the actual holdings of person at the given timestep so needs to do some arithmetic operations

### Question 4  `state_snapshot`

```
Question:
How many pencils did each person have at timestep 4? Include every one
of the 4 people.

{"answer": {"<person name>": <integer>, ...}}
```

The whole state vector at one moment. Marked correct only if **every** entry is right

### Question 5  `trajectory`

```
Question:
List how many pencils Ximena had at each timestep from 0 to 4, in order.
The list must contain exactly 5 numbers.

{"answer": [<integer>, ...]}
```

One person’s entire history - one whole *column* of the truth table. Again marked correct only if every entry is right

## Summary of all the questions

| rung | type | what the answer requires | is it in the text? |
| --- | --- | --- | --- |
| 1 | `initial_state_lookup` | copy a number | **yes**  |
| 2 | `transfer_recall` | find one event among many | **yes**  |
| 3 | `person_timestep_lookup` | one number, after updates | no - must be compute |
| 4 | `state_snapshot` | all `N` numbers at one time | no - must be compute |
| 5 | `trajectory` | all `T+1` numbers for one person | no - must be compute |

# Results

Qwen2.5-7B-Instruct:

| question type | 2 | 4 | 6 | 8 | 10 | 12 | 16 | 20 | 24 | 30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `initial_state_lookup` | 100% | 100% | 100% | 97% | 100% | 90% | 87% | 90% | 90% | 90% |
| `transfer_recall` | 90% | 63% | 70% | 77% | 43% | 63% | 60% | 63% | 37% | 23% |
| `person_timestep_lookup` | 17% | 23% | 7% | 10% | 10% | 3% | 7% | 7% | 7% | 7% |
| `state_snapshot` | 30% | 0% | 3% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| `trajectory` | 20% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |

Llama-3.1-8B-Instruct:

| question type | 2 | 4 | 6 | 8 | 10 | 12 | 16 | 20 | 24 | 30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `initial_state_lookup` | 100% | 100% | 100% | 100% | 100% | 97% | 100% | 97% | 100% | 97% |
| `transfer_recall` | 70% | 50% | 83% | 77% | 60% | 37% | 33% | 30% | 47% | 47% |
| `person_timestep_lookup` | 33% | 23% | 17% | 3% | 13% | 3% | 17% | 13% | 3% | 10% |
| `state_snapshot` | 13% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| `trajectory` | 27% | 3% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |

Mistral-7B-Instruct-v0.3:

| question type | 2 | 4 | 6 | 8 | 10 | 12 | 16 | 20 | 24 | 30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `initial_state_lookup` | 100% | 100% | 100% | 100% | 100% | 100% | 97% | 93% | 97% | 90% |
| `transfer_recall` | 83% | 80% | 57% | 63% | 63% | 50% | 43% | 27% | 27% | 33% |
| `person_timestep_lookup` | 13% | 7% | 7% | 3% | 3% | 3% | 3% | 0% | 3% | 3% |
| `state_snapshot` | 7% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| `trajectory` | 13% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |

OLMo-2-7B-Instruct (blank where the story does not fit in its 4,096-token window):

| question type | 2 | 4 | 6 | 8 | 10 | 12 | 16 | 20 | 24 | 30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `initial_state_lookup` | 100% | 100% | 100% | 100% | 100% | 93% | 97% | 100% | — | — |
| `transfer_recall` | 77% | 43% | 83% | 57% | 63% | 43% | 60% | 48% | — | — |
| `person_timestep_lookup` | 37% | 10% | 17% | 7% | 3% | 17% | 7% | 7% | — | — |
| `state_snapshot` | 17% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | — | — |
| `trajectory` | 17% | 0% | 0% | 0% | 0% | 0% | 0% | 0% | — | — |

## The primary taxonomy five categories, fixed before the run

| category | plain meaning | observed | chance | **difference** |
| --- | --- | --- | --- | --- |
| `stale` | reported the initial value; applied no update at all | 23.7% | 8.4% | **+15.3** — real |
| `single_transfer` | off by exactly one transfer involving that person | 19.2% | 13.2% | **+6.0** — real |
| `wrong_timestep` | right person, wrong moment | 15.0% | 14.7% | +0.4 — luck |
| `wrong_person` | someone else’s count at the right moment | 24.5% | 31.5% | −7.0 — *below* luck |
| `unexplained` | matched none of the rules | 17.5% | 32.2% | −14.7 |

## Extended Taxonomy into categories

### Family A · update failures

The claim: *the model applied the wrong set of transfers.*

#### A1 · `stale` — applied no update at all

> The model reported the value written in the text and never changed it.
> 

```python
a == M[0][P]
```

#### A2 · `partial_update` — stopped part-way

> The model applied some of the updates and then stopped, landing on a state
that was true *earlier* in the story.
> 

```python
any(a == M[s][P] for s in M if 0 <= s < t)
```

#### A3 · `lookahead` — read the future

> The model reported a state from *after* the timestep it was asked about.
> 

```python
any(a == M[s][P] for s in M if s > t)
```

#### A4 · `boundary_off_by_one` — right neighbourhood, wrong step

> The model landed exactly one timestep early or late.
> 

```python
a == M[t-1][P] or a == M[t+1][P]
```

#### A5 · `transfer_omitted` — skipped exactly one event

> The model applied every transfer except one.
> 

```python
any(a == y - d for d in X)
```

#### A6 · `transfer_doubled` — counted one event twice

> The model applied one transfer a second time.
> 

```python
any(a == y + d for d in X)
```

### Family B · arithmetic failures

The claim: *the model had the right transfers and combined them wrongly.*

#### B1 · `direction_single` — applied one transfer backwards

> One event was added when it should have been subtracted, or vice versa.
Getting the sign wrong on one transfer moves you by `2d`, not `d`.
> 

```python
any(a == y - 2*d for d in X)
```

#### B2 · `direction_global` — ran the whole story backwards

> Every transfer applied in the wrong direction, so the answer is the starting
value reflected: `2·M[0][P] − y`.
> 

```python
a == 2*M[0][P] - y
```

#### B3 · `arithmetic_slip` — nearly right

> The answer is close to the truth, suggesting the tracking worked and the sum
slipped.
> 

```python
0 < abs(a - y) <= 2
```

**This rule is labelled in its own source code as a magnitude heuristic, not a
mechanism test**, and the comment above it is worth reading in full:

#### B4 · `conservation_total` — reported a total instead of a holding

> The model answered with the number of pencils in the group, or in the whole
story, rather than the number that person holds.
> 

```python
a == tot_K or a == tot_all
```

#### B5 · `out_of_range` — impossible answer

> More pencils than could exist, or a negative number.
> 

```python
a < 0 or a > tot_K
```

#### B6 · `digit_error` — a digit went wrong

> The answer is a digit permutation of the truth, or differs from it in exactly
one digit position.
> 

```python
sorted(str(a)) == sorted(str(y))  or  (same length and exactly one position differs)
```

### Family C · reference failures

The claim: *the model did the right thing to the wrong person.*

#### C1 · `wrong_person_same_cluster`

```python
any(a == M[t][Q] for Q in K if Q != P)
```

#### C2 · `wrong_person_other_cluster`

```python
any(a == M[t][Q] for Q in all_people if Q not in K)
```

#### C3 · `delta_readout` — reported a transfer size instead of a total

> The model answered with *how many pencils moved* rather than *how many the
person holds*. A confusion between an edge and a node.
> 

```python
a in D
```

## Results

Pooled across the four models 

| # | category | family | observed | chance | **corrected** |  |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `stale` | A | 23.7% | 7.7% | **+16.0** | survives clearly |
| 2 | `transfer_omitted` | A | 17.7% | 11.6% | **+6.1** | survives |
| 3 | `out_of_range` | B | 4.0% | 0.0% | +4.0 | ⚠ chance is 0 by construction  |
| 4 | `direction_global` | B | 8.8% | 5.1% | +3.7 | a lead, driven by 2 of 4 models |
| 5 | `boundary_off_by_one` | A | 7.9% | 6.3% | +1.6 | weak |
| 6 | `partial_update` | A | 1.1% | 0.8% | +0.3 | chance ⚠ suppressed by priority |
| 7 | `arithmetic_slip` | B | 0.4% | 0.3% | +0.1 | chance ⚠ suppressed by priority |
| 8 | `transfer_doubled` | A | 5.9% | 6.2% | −0.3 | chance |
| 9 | `direction_single` | B | 4.1% | 4.7% | −0.5 | chance |
| 10 | `delta_readout` | C | 3.0% | 4.1% | −1.1 | below chance |
| 11 | `wrong_person_same_cluster` | C | 3.1% | 4.3% | −1.2 | below chance |
| 12 | `wrong_person_other_cluster` | C | 2.2% | 3.9% | −1.7 | below chance |
| 13 | `conservation_total` | B | 4.6% | 6.4% | −1.9 | below chance |
| 14 | `lookahead` | A | 3.4% | 6.7% | −3.3 | below chance |
| 15 | `digit_error` | B | 5.2% | 10.3% | −5.2 | below chance — the deliberate control |
| — | `unexplained` | E | 5.0% | 21.6% | −16.6 |  |

| family | claim | observed | chance | **corrected** |
| --- | --- | --- | --- | --- |
| **A · update** | wrong set of transfers applied | 59.6% | 39.2% | **+20.3** ✅ |
| B · arithmetic | right transfers, wrong sum | 27.2% | 26.9% | +0.3 — chance |
| C · reference | right operation, wrong person | 8.3% | 12.3% | −4.0 ❌ |

## Priority: how a winner is chosen

Almost every wrong answer matches **several** rules at once. The classifier
records all of them (`labels_matched_v2`) but reports one winner (`label_v2`),
so the fifteen rates form a distribution that sums to 1.

The tie-break rule, from the source:

> *A rule ranks higher when it matches a smaller set of integers: a more
specific rule carries more information, so it wins. Ties break by family order
A → B → C.* **Fixed before the classifier was run and never tuned to the
data.**
> 

The order:

```
A1 stale  →  B2 direction_global  →  B4 conservation_total  →  A4 boundary_off_by_one
   →  A5 transfer_omitted  →  A6 transfer_doubled  →  B1 direction_single
   →  C3 delta_readout  →  C1 wrong_person_same_cluster  →  A2 partial_update
   →  A3 lookahead  →  C2 wrong_person_other_cluster  →  B6 digit_error
   →  B3 arithmetic_slip  →  B5 out_of_range  →  unexplained
```

|  | hypothesis | verdict | the evidence in one line |
| --- | --- | --- | --- |
| **H1** | **Binding** — with more people, the model loses track of whose number is whose | **Rejected** | Reading a stated number is still 92% correct with 30 people, and “used the wrong person’s count” fires *below* chance |
| **H2** | **Retrieval** — it looks transfers up on demand instead of keeping a running state | **Real, but too small** | Finding a narrated event does get harder (80% → 34%), yet it stays 5× better than computing a value (56% vs 10%) |
| **H3** | **State update** — the running state is simply never maintained | **Supported** | The 97% → 10% gap, the first error landing at position 1, and “reported the starting value” topping both taxonomies — three independent routes, same answer |
| **H4** | **Arithmetic** — it tracks fine but adds up wrong | **Untested, not refuted** | Our arithmetic rules only ask “is the answer within 2?”, which cannot tell a real slip from a lucky near-miss. The instrument is too blunt to see arithmetic either way |
| **H5** | **Readout** — it tracks internally but reports wrongly | **Not testable here** | The question that would probe it (*“who has more, A or B?”*) was dropped from the design. We have no evidence in either direction |

### The figures

| figure | what to look for |
| --- | --- |
| `fig1_accuracy_by_question_type.png` — **the main one** | Leftmost panel pinned at the top all the way across; the two rightmost flat on the floor from `N=4`. Same stories throughout. That contrast is the result. |
| `fig4_first_divergence.png` | The dashed diagonal is what perfect tracking would look like. All four models sit flat near 1, nowhere near it. |
| `fig3_failure_taxonomy.png` | Solid bar = observed, hatched bar = chance. Where the hatching is as tall as the bar, that category is luck. Drawn as two bars rather than one difference so this is visible at a glance. |
| `fig3c_family_rollup.png` | Family A clearly exceeds its hatching; families B and C do not. |
| `fig3b_taxonomy_v2.png` | All fifteen categories. Busy by nature — the family rollup is the honest summary. |
| `fig2_accuracy_overall.png` — **show last, if at all** | All question types pooled into one number. **Misleading on its own**, because the number is held up by the two controls, so it is not a measure of tracking ability. |
