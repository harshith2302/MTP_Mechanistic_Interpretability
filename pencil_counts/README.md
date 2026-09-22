# Is the total in there?

A story describes several people passing pencils to one another. It never says how
many people are taking part, and it never says how many pencils there are — the
total is a sum you would have to compute from the starting amounts. We freeze a
model, run 12,000 such stories through it once each, read the residual stream at
the exact token where the model is about to answer, and train a linear classifier
on those vectors to see whether the two numbers are recoverable. Then we compare
that with what the model itself answers.

Four models see exactly the same 12,000 stories: Llama-3.1-8B-Instruct,
Qwen2.5-7B-Instruct, Mistral-7B-Instruct-v0.3 and OLMo-2-1124-7B-Instruct.

## The task

A story (5–9 people, 12–36 pencils, 0–12 timesteps, one transfer per timestep):

> The following people are participating in the pencil exchange: Harriet, Nadia,
> Bartholomew, Valentina, Desmond and Oleg. At timestep 0, their initial pencil
> holdings are as follows: Harriet holds one pencil, Nadia holds 2 pencils,
> Bartholomew holds one pencil, Valentina holds 7 pencils, Desmond holds 2 pencils
> and Oleg holds 5 pencils. At timestep 1, Nadia let Harriet have 1 pencil. At
> timestep 2, Oleg gave four pencils to Nadia. At timestep 3, Desmond transferred 1
> pencil to Nadia. At timestep 4, Nadia dropped 3 pencils into Desmond's bag. At
> timestep 5, Harriet transferred two pencils to Desmond. At timestep 6, Valentina
> dropped four pencils into Desmond's bag. At timestep 7, Bartholomew let Valentina
> have one pencil. At timestep 8, Desmond donated four pencils to Harriet.

Here the answers are 6 people and 18 pencils; neither number appears in the text,
in digits or in words. Numbers are mixed between digits and words at random, the
narration is chronological, and every person is named — no codes, no counts.

Each story becomes two prompts. The story and the question are the user turn, and
`Answer: ` is prefilled into the assistant turn, so the model's very next token is
the answer:

```
{story}
Question: How many people are participating in the pencil exchange?
Answer: 
```

with the second asking *How many pencils are there in total?*.

## The two probes

At that final token we take the residual stream after every block. For each
question and each layer we fit one multinomial logistic regression with balanced
class weights on standardised activations — 64 or 66 probes per model — training on
70 % of the stories and reporting accuracy on the held-out 15 %. Stories with every
number of timesteps are pooled for training; accuracy is then reported broken down
by timestep count.

- **people probe** — how many people are participating (5 classes).
- **total probe** — how many pencils exist in total (5 classes).

## The three baselines

- **Majority class** — always answer the most common label. Chance, ≈ 0.20.
- **Shuffled labels** — the identical probe trained on permuted labels. Catches
  anything that would look like signal with the labels destroyed.
- **Text-only** — the same classifier given four surface features and nothing from
  the model: prompt length in tokens, number of sentences, number of distinct
  capitalised name tokens, and number of numeric mentions. The *values* of the
  numbers are deliberately not features.

The text-only baseline is perfect for the people question — you can count people by
counting names — and near chance (0.18–0.25) for the total, because no surface
feature encodes a sum. That asymmetry is expected and correct: it makes the people
probe a check that the pipeline works, and the total probe the actual result.

## Headline numbers

Test split, 1,725 held-out stories per question per model. Probe at the layer with
the best validation accuracy; chance is 0.207 (people) and 0.201 (total), and
shuffled labels land within 0.03 of chance everywhere.

| | people probe | its layer | the model | | total probe | its layer | the model |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | 0.999 | 8 / 32 | 1.000 | | **0.946** | 19 / 32 | 0.606 |
| Qwen2.5-7B | 1.000 | 21 / 28 | 0.848 | | **0.947** | 22 / 28 | 0.758 |
| Mistral-7B-v0.3 | 1.000 | 10 / 32 | 0.807 | | **0.951** | 18 / 32 | 0.475 |
| OLMo-2-7B | 0.999 | 0 / 32 | 0.503 | | **0.939** | 20 / 32 | 0.567 |

The four models agree on the total probe to within 0.012 while their own answers
range over 0.28 — the representation is far more uniform than the behaviour.

## The figures

Each model has its own set in `results/<model>/figures/`.

**1. Accuracy against layer** — the headline, because it shows the whole curve and
hides nothing behind a choice of layer. The people count is available almost
immediately (OLMo solves it from the embedding itself). The total climbs from
0.51–0.70 at layer 0 to 0.94–0.96 and then flattens: every model builds the sum
over the first half to two-thirds of its depth, rather than reading it off the
input.

**2. Accuracy against story length** — the probe for the total loses six to eight
points across twelve transfers; the models lose far more, and by different amounts:
Mistral 0.91 → 0.07, Llama 0.86 → 0.20, OLMo 0.74 → 0.37, Qwen 0.87 → 0.54. The
widening gap is the result of this experiment.

**3. Probe against model** — chance ≈ 0.20, surface features ≈ 0.20, then the model,
then the probe. The probe bar is essentially the same height for all four models;
the model bar is not.

**4. Where the total probe is wrong** — 85 to 105 errors in 1,725 predictions, and
almost all of them land one step away on the 12/18/24/30/36 scale (mean error 6.1
for every model). The representation carries an approximate magnitude, not noise.
The models' own errors are larger — mean 6.4 to 9.1 — and lopsided: Llama, Mistral
and Qwen over-estimate (73–90 % of their errors), OLMo under-estimates.

**5. All four models on one pair of axes** (`results/figures/fig5_models.png`,
written by `python plots.py --compare llama qwen mistral olmo`) — the left panel
plots the total probe against depth, so networks of different lengths can be
compared: four curves that rise and then converge on the same band. The right panel
puts each model's probe beside its own answer, and the two fans apart — the solid
lines stay together near 0.9 while the dashed ones separate and, for Mistral, fall
to chance and below.

A caveat on all of this: *the probe is trained on thousands of examples while the
model answers zero-shot, so the claim is that a quantity is linearly decodable at
the answer position, not that the model "knows but won't say".*

Two notes on reading the models' answers. The candidate values are single tokens for
Llama and OLMo but Qwen and Mistral split 12/18/24/30/36 into two, so the model's
answer is uniformly scored as the candidate with the highest total log-probability —
for the single-token models that is exactly a restricted argmax over the candidates.
And asked for the total, Llama never begins with a number at all: its unrestricted
first token is "To" (*To find the total…*) on 100 % of prompts, where OLMo answers
with a digit every time and Qwen and Mistral do so on 92 % and 76 %. The habit of
reasoning out loud is Llama's, not a property of the task.

## How to run it

```bash
MODEL=llama bash run.sh                             # generate -> tests -> extract -> probes -> figures
MODEL=qwen N=500 bash run.sh                        # a 500-story pilot for another model
python plots.py --compare llama qwen mistral olmo   # the cross-model figure, once all four have run
```

Extraction needs a GPU; on a Slurm cluster run one job per model:

```bash
sbatch --gres=gpu:1 --cpus-per-task=16 --mem=96G --time=06:00:00 --wrap "MODEL=olmo bash run.sh"
```

Weights are loaded from local paths set in `config.yaml`; nothing is downloaded, and
each model's config is mirrored into `hf_cache/` so the compute node needs no
network. A run takes one to one and a half hours and writes 4.8–6.3 GB of
activations into `results/<model>/`; the two-token models cost an extra forward pass
per candidate prefix, which is what makes them the slower ones.
