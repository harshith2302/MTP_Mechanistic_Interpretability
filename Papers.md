# Mechanistic Interpretability — Compact Reading List for Pencil Exchange

**For:** Sharath  
**From:** Harshith  

**Already shared:**  
- *Do Language Models Track Entities Across State Changes?* — https://arxiv.org/abs/2605.30233  
- *(How) Do Language Models Track State?* — https://arxiv.org/abs/2503.02854  

---

## 1. Core Papers — Most Relevant

| Paper | Main idea | Why useful |
|---|---|---|
| **How do Language Models Bind Entities in Context?** ([2310.17191](https://arxiv.org/abs/2310.17191)) | Finds **binding-ID** representations linking entities to attributes. | Directly maps to `Person ↔ pencil count`; especially relevant as **N increases**. |
| **Fine-Tuning Enhances Existing Mechanisms** ([2402.14811](https://arxiv.org/abs/2402.14811)) · [Code](https://github.com/Nix07/finetuning) | Identifies entity-tracking circuits; fine-tuning strengthens existing mechanisms. | Good circuit-discovery/patching template. |
| **Representational Analysis of Binding in LMs** ([2409.05448](https://arxiv.org/abs/2409.05448)) | Finds a low-rank ordering/binding subspace controlling entity-attribute association. | Helps diagnose **right value, wrong person** failures. |
| **Discovering Variable Binding Circuitry with Desiderata** ([2307.03637](https://arxiv.org/abs/2307.03637)) · [Code](https://github.com/Nix07/binding-circuit-discovery) | Automatically finds small causal binding circuits. | Useful method for locating person/value binding components. |
| **Data-driven Circuit Discovery for Interpretability of LMs** ([2605.09129](https://arxiv.org/abs/2605.09129)) · [Code](https://github.com/Ziyu-Yao-NLP-Lab/data-driven-circuit-discovery) | Tested on **Qwen2.5-7B-Instruct** and **Llama-3.1-8B-Instruct**. | Very practical because both fit our **≤9B** limit. |
| **How Do Transformers Learn Variable Binding in Symbolic Programs?** ([2505.20896](https://arxiv.org/abs/2505.20896)) | Residual stream behaves like addressable memory; attention routes variable values. | Pencil Exchange resembles repeated variable updates. |
| **Language Models Use Lookbacks to Track Beliefs** ([2505.14685](https://arxiv.org/abs/2505.14685)) · [Code](https://github.com/Nix07/belief_tracking) | Models may retrieve earlier state changes at query time instead of maintaining full state. | Strong hypothesis for failures as **T increases**. |
| **Mixing Mechanisms: How LMs Retrieve Bound Entities In-Context** ([2510.06182](https://arxiv.org/abs/2510.06182)) | Entity retrieval mixes positional, lexical and pointer-like mechanisms. | Useful for studying binding/retrieval degradation with **N**. |

---

## 2. Arithmetic — Separate Tracking from Math Errors

| Paper | Main idea | Why useful |
|---|---|---|
| **Arithmetic Without Algorithms** ([2410.21272](https://arxiv.org/abs/2410.21272)) | Arithmetic relies on sparse **heuristic neurons**, not one general algorithm. | Predicts arithmetic errors may cluster by operand/result patterns. |
| **Mechanistic Interpretation of Arithmetic Reasoning** ([EMNLP 2023](https://aclanthology.org/2023.emnlp-main.435/)) · [Code](https://github.com/alestolfo/lm-arithmetic) | Localises arithmetic information flow through attention + MLPs. | Separates **correct tracking + wrong arithmetic** from state failure. |
| **Are Arithmetic Heuristic Neurons Form-Invariant?** ([2607.16693](https://arxiv.org/abs/2607.16693)) | Tests whether the same arithmetic neurons work across symbols, prose and code. | Highly relevant because Pencil Exchange arithmetic is expressed in **prose**. |
| **Pre-trained LLMs Use Fourier Features to Compute Addition** ([2406.03445](https://arxiv.org/abs/2406.03445)) | Studies internal numerical representations used for addition. | Useful if arithmetic becomes the dominant failure mode. |

---

## 3. Probing the Hidden State

| Paper | Why useful |
|---|---|
| **Emergent World Representations — Othello-GPT** ([2210.13382](https://arxiv.org/abs/2210.13382)) · [Code](https://github.com/likenneth/othello_world) | Canonical template for decoding a hidden world state and causally editing it. |
| **Emergent World Models in Chess-Playing LMs** ([2403.15498](https://arxiv.org/abs/2403.15498)) · [Code](https://github.com/adamkarvonen/chess_llm_interpretability) | Clean structured-state probing/intervention setup. |
| **Monitoring Latent World States with Propositional Probes** ([2406.19501](https://arxiv.org/abs/2406.19501)) | Could probe propositions such as `HasPencils(Alice,17)` after each transaction. |
| **Revisiting the Othello World Model Hypothesis** ([2503.04421](https://arxiv.org/abs/2503.04421)) | Useful caution against overclaiming from probe accuracy alone. |

**Useful experiment:** probe the model after every transaction.  
If the internal state becomes wrong → **state-update failure**.  
If the internal state is correct but the final answer is wrong → **arithmetic/readout failure**.

---

## 4. Latent-Reasoning Models

| Paper | Why relevant |
|---|---|
| **Coconut — Training LLMs to Reason in a Continuous Latent Space** ([2412.06769](https://arxiv.org/abs/2412.06769)) · [Code](https://github.com/facebookresearch/coconut) | Candidate framework for our latent-reasoning model class. |
| **Are Latent Reasoning Models Easily Interpretable?** ([2604.04902](https://arxiv.org/abs/2604.04902)) · [Code](https://github.com/connordilgren/are-lrms-easily-interpretable) | Directly studies interpretability of latent reasoning models. |
| **Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer** ([2507.02199](https://arxiv.org/abs/2507.02199)) · [Code](https://github.com/wenquanlu/huginn-latent-cot) | Studies **Huginn-3.5B**, which fits our GPU limit. |
| **A Survey on Latent Reasoning** ([2507.06203](https://arxiv.org/abs/2507.06203)) | Useful map for choosing latent-reasoning models systematically. |

---

## 5. Closest State-Tracking Benchmarks

| Benchmark | Relation to Pencil Exchange |
|---|---|
| **Entity Tracking / Boxes** ([2305.02363](https://arxiv.org/abs/2305.02363)) | Tracks objects across state changes; mostly categorical rather than numeric. |
| **Tracking Shuffled Objects — BIG-Bench** ([Code](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/tracking_shuffled_objects)) | People repeatedly exchange objects; very close structure but no arithmetic. |
| **Exploring State Tracking Capabilities of LLMs** ([2511.10457](https://arxiv.org/abs/2511.10457)) | **HandSwap** varies update depth; close to our **T-axis**, but tracks object identity. |
| **CausalToM** ([2505.14685](https://arxiv.org/abs/2505.14685)) | Sequential character/object state changes + mechanistic analysis. |
| **ProPara** ([Code](https://github.com/allenai/propara)) | Intermediate entity-state annotations across procedural text. |
| **TRIP** ([Paper](https://aclanthology.org/2021.findings-emnlp.422/)) | Dense intermediate physical-state annotations; useful for taxonomy design. |
| **OAKS / OAKS-BABI** ([2603.07392](https://arxiv.org/abs/2603.07392)) · [Code](https://github.com/kaistAI/OAKS) | Changing facts with tracking/counting/comparison queries. |
| **BABILong** ([2406.10149](https://arxiv.org/abs/2406.10149)) · [Code](https://github.com/booydar/babilong) | Useful later for distractor/long-context variants. |

### What Pencil Exchange adds

> **Multiple entities (N) + changing integer states + arbitrary transfers + coupled updates + arithmetic + increasing transactions (T).**

Example: `Alice gives Bob 7 pencils`

- `Alice ← Alice − 7`
- `Bob ← Bob + 7`

with total pencil count conserved.

---

## 6. Crossword

| Paper | Why useful |
|---|---|
| **CrossWordBench** ([2504.00043](https://arxiv.org/abs/2504.00043)) · [Code](https://github.com/SeanLeng1/CrossWordBench) · [Dataset](https://huggingface.co/datasets/HINT-lab/CrossWordBench) | Ready controllable crossword generator + published baselines on 20+ models. |
| **TopoBench** ([2603.12133](https://arxiv.org/abs/2603.12133)) · [Code](https://github.com/mayug/topobench-benchmark) | Good example of separating **representation/constraint extraction** errors from reasoning errors. |

---

## 7. Proposed Failure Taxonomy

| Failure | Diagnostic | Likely mechanism |
|---|---|---|
| **Binding** | Correct amount, wrong person | Entity/value binding |
| **Arithmetic** | Correct transaction, wrong `+/-` | Arithmetic circuit |
| **Digit** | e.g. `47 → 74` | Number representation/output |
| **Omission** | Transaction skipped | Retrieval/attention |
| **Over-application** | Transaction counted twice | Update control |
| **Direction** | Giver/receiver reversed | Relation binding |
| **Referent** | Wrong person resolved | Binding + position |
| **Temporal** | Earlier correct state returned | State indexing/retrieval |
| **Format** | Unparseable response | Exclude from reasoning failures |

---

## 8. Models Under ~9B

**Start with instruction-tuned:**
- **Qwen2.5-7B-Instruct**
- **Llama-3.1-8B-Instruct**
- **OLMo-2-7B-Instruct**

Qwen2.5-7B and Llama-3.1-8B are especially attractive because published circuit-discovery work already uses them.

**Later:**
- Reasoning models ≤9B
- Latent reasoning: **Huginn-3.5B** and/or a Coconut checkpoint

---

## 9. Research Flow

**Behavioral evaluation → Failure taxonomy → Select failure regime → Mechanistic analysis**

1. Plot **Accuracy vs N**, **Accuracy vs T**, and an **N × T heatmap**.
2. Label failures using the taxonomy above.
3. Pick a region with a useful mixture of success/failure.
4. Test:
   - **Binding hypothesis:** increasing N corrupts person/value binding.
   - **Retrieval hypothesis:** transactions are retrieved only at query time.
   - **State-update hypothesis:** locate the first incorrect intermediate state.
   - **Arithmetic hypothesis:** tracking is correct but `+/-` fails.
   - **Readout hypothesis:** correct state is present internally despite a wrong output.

---

## Priority Reading Order

1. **How do Language Models Bind Entities in Context?**
2. **Fine-Tuning Enhances Existing Mechanisms**
3. **Data-driven Circuit Discovery**
4. **Arithmetic Without Algorithms**
5. **Are Arithmetic Heuristic Neurons Form-Invariant?**
6. **Monitoring Latent World States with Propositional Probes**
7. **Language Models Use Lookbacks to Track Beliefs**
8. **How Do Transformers Learn Variable Binding in Symbolic Programs?**
9. **Coconut**
10. **Are Latent Reasoning Models Easily Interpretable?**

### Main question

> **When a transformer must maintain entity-bound numerical states over increasing N and T, does failure arise from binding, retrieval, state update, arithmetic, or readout?**
