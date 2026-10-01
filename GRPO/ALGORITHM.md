# GRPO setup: computation and reward

This file specifies exactly what the current GRPO run computes: the per-step pipeline,
the formulas used, the reward function, and the hyperparameters. It describes the code
as it is, including its known weaknesses. For how to run it, see
[README.md](README.md) and the project root [README](../README.md).

Where each part lives:

- advantages: rLLM `trainer/algorithms/rl_algo.py`;
- losses: verl 0.8.0 `workers/utils/losses.py` and `trainer/ppo/core_algos.py`;
- reward: `src/labelcot/reward.py`;
- rollout and scoring: `src/labelcot/workflow.py`;
- hyperparameters: `src/entrypoint.sh`.

---

## 1. Models

The starting point, `qwen3-8b-sft-v3`, is a **full-parameter** SFT checkpoint with no
adapter. **GRPO adds a new LoRA adapter** at the start of the run on top of the full SFT
weights. The adapter's B matrix starts at zero, so at step 0 the policy behaves exactly
like the SFT model. GRPO then trains only this adapter.

| role | model | trained | notes |
|---|---|---|---|
| policy (actor) | full-parameter SFT checkpoint `qwen3-8b-sft-v3` + a new LoRA adapter added by GRPO (rank 64, alpha 32, all linear layers) | the GRPO adapter only | SFT weights stay frozen, kept in fp32 by FSDP |
| reference policy | the same model with the LoRA adapter switched off | no | used only for the KL loss term; costs no extra memory |
| rollout engine | vLLM, serving the base weights + the current adapter | — | adapter re-synced after every step |

`LORA_RANK=0` switches to full-parameter training. All weights are then updated, verl keeps
a separate frozen copy of the model as the reference policy, and vLLM receives the full
weights after every update.

There is **no critic (value model)** and **no learned reward model**. The baseline for
each answer is the mean reward of the other answers to the same question, and the reward
is the rule function in section 4.

## 2. One training step

With `B = 8` questions per step and group size `G = 8`:

1. **Sample questions.** Draw 8 questions from `GRPO/data/train.jsonl`, shuffled with a
   fixed seed. The prompt is the bare question rendered with the SFT chat template, ending
   at `<|im_start|>assistant\n`.
2. **Roll out.** vLLM generates **8 independent answers per question**, 64 in total, at
   temperature 1.0 with no top-p/top-k and at most 8,192 new tokens.
3. **Score.** `score_completion` turns each answer into a scalar reward `R` (section 4).
4. **Group.** The 64 answers are grouped by question id: 8 groups of 8.
5. **Advantages.** Group-normalised advantages (section 3.1), one number per answer,
   applied to every token of that answer.
6. **Old log-probs.** The actor recomputes `log π_old(a_t | s_t)` for every response token.
   These values, not vLLM's, are the ratio denominator.
7. **Reference log-probs.** The actor with its adapter disabled computes
   `log π_ref(a_t | s_t)`.
8. **Actor update** with the clipped policy loss plus KL loss (sections 3.2 and 3.3), in
   `B / 4 = 2` mini-batches of 4 questions (32 answers each), 1 epoch: two optimizer
   updates per step.
9. **Sync.** Push the new LoRA weights to vLLM.
10. **Periodic work.**
    - Validation on the 102 held-out questions (1 answer each) at step 0, every 20 steps,
      and at the end.
    - A checkpoint every 20 steps and after the last step.

## 3. Formulas

Notation for one question `q` with answers `o_1 … o_G` and rewards `R_1 … R_G`, where
answer `i` has tokens `a_{i,1} … a_{i,T_i}`:

- `ε_low = 0.2`, `ε_high = 0.28`: asymmetric clip;
- `c = 3.0`: dual-clip bound;
- `β = 0.001`: KL coefficient.

### 3.1 Group-relative advantage

```
μ = mean(R_1 … R_G)
σ = std(R_1 … R_G)                (population std, ddof = 0)

A_i = (R_i − μ) / (σ + 1e-6)      if NORM_ADV_BY_STD=true   (standard GRPO, default)
A_i =  R_i − μ                    if NORM_ADV_BY_STD=false  (Dr. GRPO)

A_{i,t} = A_i                     for every token t of answer i
```

- A group in which all `G` rewards are equal gets `A = 0` for every answer and contributes
  no policy gradient. This happens, for example, when all 8 answers are truncated.
- There is no discounting and no per-token credit assignment. Every token of an answer
  shares that answer's advantage.

Checked locally with the real rLLM code: one group with rewards `[1.0, 0.0, 0.6, 1.0]`
(perfect, cut off, wrong answer, perfect) gets advantages `[0.855, −1.588, −0.122, 0.855]`.

### 3.2 Policy loss (clipped, dual clip, asymmetric)

```
ρ_{i,t} = exp(log π_θ(a_{i,t} | s_{i,t}) − log π_old(a_{i,t} | s_{i,t}))

L_{i,t} = max( −A_i · ρ , −A_i · clip(ρ, 1 − ε_low, 1 + ε_high) )                     if A_i ≥ 0
L_{i,t} = min( max(−A_i · ρ , −A_i · clip(ρ, 1 − ε_low, 1 + ε_high)) , −A_i · c )      if A_i < 0
```

- The higher upper bound (`ε_high = 0.28`, "clip-higher" from DAPO) lets the probability
  of good low-probability tokens rise faster, which counters entropy collapse.
- The dual clip caps the loss of very unlikely tokens that have negative advantage.

### 3.3 KL loss to the SFT policy (k3 estimator)

```
x_{i,t}  = log π_ref(a_{i,t} | s_{i,t}) − log π_θ(a_{i,t} | s_{i,t})
kl_{i,t} = exp(x) − x − 1          (clamped to [−10, 10])

L = token-mean(L_{i,t}) + β · token-mean(kl_{i,t})
```

- Token-mean averages over every response token in the mini-batch, so longer answers are
  not down-weighted per token.
- There is no entropy bonus.
- Unlike the classic PPO setup, the KL term is in the **loss**, not in the reward. rLLM's
  unified trainer supports only this form. Set `KL_BETA=0` to drop it.

## 4. The reward `R`

Implemented in `src/labelcot/reward.py`. It uses rules only. Every question carries its
own **annotated label sequence** (the labels of the reference trace's spans, from
`bespoke-v2`) and its reference answer.

### 4.1 Parsing a response

```
response = "<think>\n" thought "</think>" answer
```

- `closed`: `</think>` appears in the response.
- `truncated`: vLLM stopped at the 8,192-token limit.
- **Thought.** The text before the first `</think>`, with the leading `<think>` removed.
- **Answer.** The text after it.
- **Steps.** The thought split on blank lines; empty paragraphs are dropped. The SFT
  targets are written this way: one `[label] text` paragraph per annotated span.
- **A step's label.** The tag at the very start of the paragraph if it is one of the
  eight below. Otherwise the step is untagged; a bracketed tag outside the vocabulary
  also counts as *invented*.

```
planning_next_step  restating_problem  recalling_knowledge  logical_deduction
reflecting          verifying          correcting_itself    concluding
```

### 4.2 The three terms

Let `g` be the list of generated step labels (untagged steps excluded), `n` the number of
steps, and `ref` the annotated label list.

```
tag_rate        = |g| / n                                          (0 if n = 0)

hist_sim        = 1 − ½ · Σ_label | count_g(label)/|g| − count_ref(label)/|ref| |
seq_sim         = difflib.SequenceMatcher(collapse(g), collapse(ref)).ratio()
label_alignment = ½ · (hist_sim + seq_sim)                         (0 if g or ref is empty)

answer_correct  = closed ∧ checkable(gold) ∧ match(last \boxed{…} in the answer, gold)
```

- `hist_sim` is one minus the total-variation distance between the two label
  distributions: does the answer use the tags in the same **proportions** as the
  annotation?
- `seq_sim` compares the **order** of tags. `collapse` merges consecutive repeats
  (`[a, a, b, a] → [a, b, a]`); Ratcliff/Obershelp ratio = 2 · matched / (|x| + |y|).
- `match` is the scorer from `../../train/evaluate/eval_math500.py`:
  - LaTeX normalisation;
  - then exact string equality, or numeric equality within 1e-6;
  - multiple-choice golds (a single letter A–E) compare only the chosen letter, so
    `\textbf{(A)}` matches `A`.
- Only the answer after `</think>` is searched; a `\boxed{}` inside the thought is working,
  not an answer.

### 4.3 Combining them

Weights (env vars `REWARD_W_TAG`, `REWARD_W_ALIGN`, `REWARD_W_CORRECT`):
`w_tag = 0.3`, `w_align = 0.3`, `w_correct = 0.4`.

```
if truncated or not closed:
    R = 0
elif checkable(gold):
    R = w_tag · tag_rate + w_align · label_alignment + w_correct · answer_correct
else:
    R = (w_tag · tag_rate + w_align · label_alignment) / (w_tag + w_align)
```

`R` is always in [0, 1].

`checkable(gold)` is false when string matching cannot judge the reference answer. That
is the case when any of the following holds:
- it is empty;
- it is longer than 40 characters;
- it contains `\text` or `\mbox`;
- it contains a run of four or more letters outside LaTeX commands, i.e. prose.

117 of 5,042 training answers (2.3 %) are not checkable.

### 4.4 Worked examples

For a response with 20 steps, 19 of them tagged, hist_sim 0.80 and seq_sim 0.50
(label_alignment 0.65):

| case | R |
|---|---|
| correct answer | 0.3·0.95 + 0.3·0.65 + 0.4·1 = **0.88** |
| wrong answer | 0.3·0.95 + 0.3·0.65 + 0 = **0.48** |
| prose answer (not checkable) | (0.285 + 0.195) / 0.6 = **0.80** |
| no `</think>`, or hit 8,192 tokens | **0** |
| an SFT target scored against its own annotation | **1.0** (all 5,042 training targets) |

### 4.5 What gets logged

Per answer, averaged per step as `batch/<name>` (training) and
`val/bespoke_labeled_cot/<name>` (validation):

- `reward`, `closed_think`, `truncated`;
- `n_steps`, `tag_rate`, `invented_tag_rate`;
- `label_dist_sim` (= hist_sim), `label_seq_sim`, `label_alignment`;
- `answer_checkable`, and `answer_correct` (checkable questions only);
- `share_<label>` for each of the 8 labels: the generated label mix.

rLLM additionally logs `reward/policy/{mean,std,min,max}` and
`advantage/policy/{mean,std}`. An episode counts as correct (`is_correct`, which feeds
`val/bespoke_labeled_cot/pass@1`) only when `answer_correct` is true.

## 5. Hyperparameters

| group | setting | value | config key (env var) |
|---|---|---|---|
| data | questions per step | 8 | `rllm.data.train_batch_size` (`TRAIN_BATCH_SIZE`) |
| | answers per question (group size) | 8 | `rllm.rollout.n` (`GROUP_SIZE`) |
| | max prompt / response tokens | 2,048 / 8,192 | `rllm.data.max_prompt_length` / `max_response_length` |
| | train / validation questions | 5,042 / 102 (SFT held-out ids) | `GRPO/data/*.jsonl` |
| rollout | train sampling | T = 1.0, top-p 1.0, top-k off | `rllm.rollout.train.*` |
| | validation sampling | 1 answer, T = 0.6, top-p 0.95, top-k 20 | `rllm.rollout.val.*`, `rllm.rollout.n_val` |
| | vLLM GPU share while generating | 0.7 | `actor_rollout_ref.rollout.gpu_memory_utilization` |
| advantage | estimator | GRPO, std-normalised | `rllm.algorithm.adv_estimator`, `norm_adv_by_std_in_grpo` (`NORM_ADV_BY_STD`) |
| loss | clip ε_low / ε_high, dual clip c | 0.2 / 0.28, 3.0 | `rllm.algorithm.eps_clip` / `eps_clip_high` (`CLIP_LOW` / `CLIP_HIGH`) |
| | aggregation, entropy bonus | token-mean, 0 | `rllm.algorithm.loss_agg_mode`, `actor.entropy_coeff` |
| | KL | k3 in the loss, β = 0.001 | `rllm.algorithm.kl_beta` (`KL_BETA`), `actor.kl_loss_type` |
| actor | GRPO LoRA adapter rank / alpha / targets (on the full SFT weights) | 64 / 32 / all linear | `actor_rollout_ref.model.{lora_rank,lora_alpha,target_modules}` |
| | learning rate, schedule | 1e-5, constant | `actor_rollout_ref.actor.optim.lr` (`ACTOR_LR`) |
| | optimizer | AdamW, β = (0.9, 0.999), weight decay 0.01, grad clip 1.0 | verl defaults |
| | questions per optimizer update, epochs | 4 (so 2 updates per step), 1 | `actor.ppo_mini_batch_size` (`PPO_MINI_BATCH_SIZE`), `actor.ppo_epochs` |
| | tokens per micro-batch | 12,288 (dynamic batching) | `actor.ppo_max_token_len_per_gpu` |
| schedule | step cap, epochs | 200 steps (1,600 questions, 12,800 answers), 1 | `rllm.trainer.total_batches` (`TOTAL_TRAINING_STEPS`), `rllm.trainer.total_epochs` |
| | validate / checkpoint every | 20 steps (+ end) | `rllm.trainer.test_freq` / `rllm.trainer.save_freq` |

## 6. Known properties and open questions

**Measured on the data.** Scoring each question's annotation against a *random other*
question's annotation already gives `label_alignment` 0.58 on average: hist_sim 0.73,
seq_sim 0.43. The label terms therefore span only a small range compared with right vs.
wrong (0.4).

The following follow from the definitions and still have to be observed in a run (step-0
validation, or `rllm.trainer.val_only=true`, gives the SFT baseline for every term):

- **Group normalisation amplifies small differences.** This cuts both ways:
  - When all 8 answers to a question are equally right or wrong, only the label terms
    differ, and dividing by the small group σ scales those differences up to unit-size
    advantages. That is the intended signal for "follow the annotation".
  - The same scaling also amplifies noise.
  - `NORM_ADV_BY_STD=false` turns the scaling off.
- **Uniform groups are wasted compute.** All-truncated or otherwise identical groups have
  zero advantage. `batch/truncated` shows how often it happens.
- **`tag_rate` is probably close to saturated.** According to the model card, the SFT
  model already used the tags in all 500 MATH-500 outputs.
- **The label terms can be satisfied cheaply.** Nothing checks that a tag fits its
  paragraph, and length costs nothing until the 8,192 limit. The policy could relabel
  steps or add filler steps.
- **One reference path.** `seq_sim` compares against the single annotated trace, so a
  different valid solution path loses points.
- **Truncation scores 0.** This pushes the model to finish, but also biases it toward
  shorter reasoning.
- **Sparse, sequence-level credit.** Every token of an answer gets the same advantage;
  there is no per-step signal for individual tags.

Open design question: what "follow the annotated labels" should mean.

- **(a) Format and label mix/order match the annotation.** This is what is implemented
  now.
- **(b) Each tag is semantically correct for its step.** This needs a label judge, for
  example a classifier trained on the ~200k labelled spans in `bespoke-v2`.
- **(c) Controllable generation.** The prompt specifies a label plan or allowed label set,
  and the reward checks compliance.
