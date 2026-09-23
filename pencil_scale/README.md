# Is it a number, and is it only there?

`pencil_counts` showed that the total number of pencils in a story — a sum the text
never states — can be read out of a frozen model's residual stream with a linear
probe, at about 95 % accuracy, in models that answer the same question correctly
only 48–76 % of the time. That experiment left two questions open, and this one is
built to answer them.

**Was it a number, or a five-way choice?** There were only five possible totals
(12/18/24/30/36), so a probe could have learned five regions rather than a
quantity. Here the total is **every integer from 10 to 60** and the read-out is
**ridge regression**, scored by how far off it is, not by whether it picks the
right class.

**Was it only at one place, and only linear?** `pencil_counts` read a single token —
the one where the model is about to answer — with a linear probe. A quantity held
elsewhere, or held non-linearly, would have looked absent. Here the residual
stream is read at **five positions × every layer**, and at the best cell an **MLP**
is fitted beside the linear probe.

## The task

Unchanged in wording from the earlier experiments, so results stay comparable:
5–9 named people, holdings given in a mix of digits and words, one transfer per
timestep, 0/1/3/5/8/12 timesteps, chronological narration, and the total never
stated anywhere in the text.

```
The following people are participating in the pencil exchange: Harriet, Nadia,
Bartholomew, Valentina, Desmond and Oleg. At timestep 0, their initial pencil
holdings are as follows: Harriet holds one pencil, Nadia holds 2 pencils, ...
At timestep 8, Desmond donated four pencils to Harriet.
Question: How many pencils are there in total?
Answer: 
```

One prompt per story, 12,000 stories, the story and question as the user turn and
`Answer: ` prefilled into the assistant turn so the next token is the answer.

## The five reading positions

| position | the token at |
|---|---|
| `first_token` | the very start of the prompt — a control that should carry nothing |
| `allocation_end` | the end of the initial-holdings sentence, before any transfer |
| `last_transfer` | the end of the final transfer sentence |
| `question_end` | the end of the question, before the answer prefix |
| `answer` | the final token — what `pencil_counts` used |

Positions are located through the tokenizer's character-offset map, so each one
lands on the sentence it names rather than on a guessed index.

## The three protocols

The split is what turns "can it read the total" into "is the total a number".

| protocol | train | test | what it shows |
|---|---|---|---|
| `random` | 70 % of stories | 15 % | how well the total can be read at all |
| `heldout` | every story whose total is not 17, 23, 31, 43 or 52 | only those five totals | does the read-out **interpolate** to values it never saw? |
| `extrapolate` | totals ≤ 45 | totals > 45 | is there a magnitude axis that continues beyond the training range? |

A five-way classifier cannot pass `heldout` or `extrapolate` — it has no output for
a value it was never trained on. A linear direction that encodes magnitude can.

## The baselines

Every probe row carries four references, all scored the same way (mean absolute
error, in pencils):

- **predict the mean** — the error of ignoring the input entirely;
- **shuffled labels** — the same ridge fit on permuted training totals, which
  catches anything that would look like signal in 4,096 dimensions;
- **text-only** — ridge on four surface counts and nothing from the model: prompt
  length in tokens, sentences, distinct capitalised names, numeric mentions. No
  surface feature encodes a sum, so this should stay near the mean baseline;
- **the model's own answer** — the candidate total (10–60) it gives the highest
  probability, from the same forward pass.

The MLP probe (one hidden layer) is fitted at the best position and layer of each
protocol, and is the test of whether a linear read-out was hiding anything.

## The figures

1. `fig1_position_layer_map.png` — error across all five positions × every layer,
   with the best cell marked. This is the figure that answers "is it only at the
   answer position?".
2. `fig2_layers.png` — the same as curves, one per position, against the
   predict-the-mean, text-only and model's-own-answer lines.
3. `fig3_predicted_vs_true.png` — predicted against true total for each protocol.
   Points on the diagonal for totals the probe never trained on are the evidence
   that the representation is numeric.
4. `fig4_probes_vs_baselines.png` — linear probe and MLP against every baseline,
   per protocol.

## How to run it

```bash
MODEL=llama bash run.sh                            # generate -> tests -> extract -> probes -> figures
MODEL=llama N=500 OUT=results/pilot bash run.sh    # a 500-story pilot
```

Extraction needs a GPU; on the cluster run it as one job:

```bash
sbatch --partition=a40 --qos=a40 --account=25m0834 --gres=gpu:1 --cpus-per-task=16 \
       --mem=96G --time=06:00:00 --wrap "MODEL=llama bash run.sh"
```

Weights load from the local paths in `config.yaml`; nothing is downloaded, and each
model's config is mirrored into `hf_cache/` so the compute node needs no network.
Caching five positions at every layer costs about 16 GB per model for the full
12,000 stories (0.7 GB for the pilot), and 8 data-integrity tests guard the stories:
conservation of pencils, one transfer per timestep, every number in the text
accounted for as a holding, an amount or a timestep label, and every reading
position landing where it claims to.

`STATUS.md` records where the experiment currently stands.
