# GRPO setup: computation and reward

This file specifies exactly what the GRPO configuration **V1** computes: the per-step
pipeline, the formulas, the reward function, every hyperparameter, and what the V1 run
measured. It describes the code as it is, including its known weaknesses. For how to run
it, see [README.md](README.md) and the project root [README](../README.md).

Versions: V1 is the first fixed configuration; earlier runs were trials (§7). A later
version gets its own column in §5, with the settings it changes in bold, and its results
in §6. Sections 1-4 describe V1; V2 changes the data and the sampling only (§5, and the
note on sampling in §6.2), V3 also the response length. **Any change to the configuration is recorded in §5 in the
same change.**

Where each part lives:

- advantages: rLLM `trainer/algorithms/rl_algo.py`;
- losses: verl 0.8.0 `workers/utils/losses.py` and `trainer/ppo/core_algos.py`;
- reward: `src/labelcot/reward.py`;
- rollout and scoring: `src/labelcot/workflow.py`;
- hyperparameters: `src/entrypoint.sh` (code defaults) and `.env` (the values of a run,
  listed in §5). `.env.example` is a template whose defaults are not V1's (§5.7).

---

## 1. Models

The starting point is the full-parameter SFT checkpoint
`Lucas-SJY/qwen3-8b-sft-bespoke` at revision `v3` (Qwen3-8B fine-tuned on the labelled
bespoke-v2 traces). V1 trains **all of its parameters**; there is no adapter.

| role | model | trained | notes |
|---|---|---|---|
| policy (actor) | the SFT checkpoint | all parameters | FSDP2 sharded over 2 GPUs; fp32 weights, gradients and Adam state (~130 GB) live in host RAM (§5.4) |
| reference policy | a separate frozen copy of the SFT checkpoint | no | used only for the KL loss term; CPU-offloaded |
| rollout engine | vLLM, one replica per GPU, serving the current policy weights | — | receives all weights after every step |

There is **no critic (value model)** and **no learned reward model**. The baseline for
each answer is the mean reward of the other answers to the same question, and the reward
is the rule function in section 4.

## 2. One training step

With `B = 8` questions per step and group size `G = 16`:

1. **Sample questions.** Draw 8 questions from `GRPO/data/rest_grpo/train.jsonl` (4,776
   questions), shuffled with seed 42. The prompt is the task's single user turn (the
   question, which already asks for a `\boxed{}` final answer), rendered with the
   checkpoint's chat template in thinking mode and ending at `<|im_start|>assistant\n`.
2. **Roll out.** vLLM generates **16 independent answers per question**, 128 in total, at
   temperature 1.0 with no top-p/top-k and at most 8,192 new tokens. An answer with an
   invalid tag is regenerated (up to 3 times, §4.3). From V4, a training answer whose
   thought gets stuck on one step type (3 consecutive paragraphs with the same label,
   over 200 tokens together) is stopped while it is written and sampled again, up to 2
   times; a third attempt that still has such a run is dropped from training like a
   cut-off answer (§5.2).
3. **Score.** `score_completion` turns each answer into a scalar reward `R` (section 4).
4. **Group.** The 128 answers are grouped by question id: 8 groups of 16. Answers cut off
   at the 8,192-token limit are dropped here (rLLM's compact filtering): they get no
   advantage and no gradient, and a group keeps only its finished answers. A step in
   which every answer was cut off is skipped.
5. **Advantages.** Group-normalised advantages (section 3.1), one number per answer,
   applied to every token of that answer.
6. **Old log-probs.** The actor recomputes `log π_old(a_t | s_t)` for every response token.
   These values, not vLLM's, are the ratio denominator.
7. **Reference log-probs.** The frozen reference model computes `log π_ref(a_t | s_t)`.
8. **Actor update** with the clipped policy loss plus KL loss (sections 3.2 and 3.3), in
   `B / 4 = 2` mini-batches of 4 questions (up to 64 answers each), 1 epoch: two
   optimizer updates per step. Each mini-batch is processed in micro-batches of up to
   32,768 tokens per GPU; the Adam step runs on the CPU.
9. **Sync.** The updated weights are copied into vLLM before the next rollout.
10. **Periodic work.**
    - Validation on the 526 questions of `rest_grpo/validation.jsonl` (1 answer each)
      at step 0, every 20 steps, and at the end.
    - A checkpoint (weights only, no Adam state) every 10 steps and after the last step.

200 steps draw 1,600 of the 4,776 training questions, so no question is seen twice.

## 3. Formulas

Notation for one question `q` with answers `o_1 … o_G` and rewards `R_1 … R_G`, where
answer `i` has tokens `a_{i,1} … a_{i,T_i}`:

- `ε_low = 0.2`, `ε_high = 0.28`: asymmetric clip;
- `c = 3.0`: dual-clip bound;
- `β = 0.01`: KL coefficient.

### 3.1 Group-relative advantage

```
μ = mean(R_1 … R_G)
σ = std(R_1 … R_G)                (population std, ddof = 0)

A_i = (R_i − μ) / (σ + 1e-6)      if NORM_ADV_BY_STD=true   (standard GRPO, V1)
A_i =  R_i − μ                    if NORM_ADV_BY_STD=false  (Dr. GRPO)

A_{i,t} = A_i                     for every token t of answer i
```

- A group in which all `G` rewards are equal gets `A = 0` for every answer and contributes
  no policy gradient. Cut-off answers are not in the group at all (§2 step 4), so `μ` and
  `σ` are over the finished answers only; a group left with one answer also gets `A = 0`.
- There is no discounting and no per-token credit assignment. Every token of an answer
  shares that answer's advantage.

Example: a V1 group of 16 with 12 right and 4 wrong answers, all with valid tags
(`R = 1.0` and `0.2`), has `μ = 0.8`, `σ = 0.346`, so the right answers get `A = +0.577`
and the wrong ones `A = −1.732`. (Checked locally with the real rLLM code: a group with
rewards `[1.0, 0.0, 0.6, 1.0]` gets advantages `[0.855, −1.588, −0.122, 0.855]`.)

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
  unified trainer supports only this form. `KL_BETA=0` drops it.
- On the ~90 % of groups that have zero advantage, the KL term is the only gradient, so
  it pulls those answers back toward the SFT policy. V1 uses `β = 0.01`
  because at 0.001 the label mix drifted (§7).

## 4. The reward `R`

Implemented in `src/labelcot/reward.py`. It uses rules only, and **no reference trace**:
a response is scored on whether its tags are valid and on its final answer. A task needs
only its prompt and its reference answer.

### 4.1 Parsing a response

```
response = "<think>\n" thought "</think>" answer
```

- `closed`: `</think>` appears in the response.
- `truncated`: vLLM stopped at the 8,192-token limit. Such a response is not scored for
  training: the episode ends as `MAX_RESPONSE_LENGTH_EXCEEDED` and the trainer drops it
  (§2 step 4). Its `R = 0` below is only a placeholder.
- **Thought.** The text before the first `</think>`, with the leading `<think>` removed.
- **Answer.** The text after it.
- **Steps.** The thought split on blank lines; empty paragraphs are dropped. The SFT
  targets are written this way: one `[label] text` paragraph per annotated span.
- **A step's label.** The tag at the very start of the paragraph if it is one of the
  eight below. Otherwise the step is untagged.
- **An invalid tag.** A paragraph that opens with something tag-shaped that is not one of
  the eight: a bracketed single word of 3+ letters, digits, `_` or `-` (`[thinking]`,
  `[Planning_Next_Step]`, `[self-check]`), or a spaced spelling of a label
  (`[planning next step]`). Other bracketed text (`[1, 2]`, `[x for x in xs]`,
  `[Step 1]`) is content. Only the thought is checked.

```
planning_next_step  restating_problem  recalling_knowledge  logical_deduction
reflecting          verifying          correcting_itself    concluding
```

### 4.2 The terms

Let `g` be the list of generated step labels (untagged steps excluded), `n` the number of
steps, and `p` the target label mix `LABEL_PRIOR`.

```
label_valid    = |g| > 0  ∧  no invalid tag                                 (0 or 1)

answer_correct = closed ∧ checkable(gold) ∧ match(final answer after </think>, gold)

tag_rate       = |g| / n                                (logged, not rewarded; 0 if n = 0)
label_mix      = 1 − ½ · Σ_label | count_g(label)/|g| − p(label) |  (logged, not rewarded)
```

- `label_valid` is the label term of the reward: 1 when the thought has at least one tag
  and every tag is one of the eight labels (§4.1), 0 when it has no tag at all or any tag
  outside the eight. Untagged paragraphs between tagged ones are allowed. It does not
  look at which labels are used, how often, or in what order.
- `label_mix` (logged only) is one minus the total-variation distance between the
  response's label shares and the target mix `p`: does the thought use the eight labels
  in the proportions of a typical annotated trace?
- The target mix `p` is the mean label share of one annotated trace over all 5,144
  bespoke-v2 traces (201,250 spans), from `src/compute_label_prior.py`:

  | label | logical_deduction | reflecting | planning_next_step | verifying | restating_problem | concluding | recalling_knowledge | correcting_itself |
  |---|---|---|---|---|---|---|---|---|
  | share | 25.3 % | 20.0 % | 15.9 % | 14.3 % | 8.7 % | 7.4 % | 7.2 % | 1.2 % |

  Averaging per trace rather than pooling all spans describes a typical trace, which is
  what one response is compared with; pooling would let the longest traces dominate.
- `match` judges only the final answer, never the reasoning:
  - Hugging Face `math_verify` (0.9.0) reads the last `\boxed{}` after `</think>`, or the
    final stated answer when there is none, and decides mathematical equivalence:
    `0.5 = 1/2 = \frac{1}{2}`, `3,840 = 3840`, `135^\circ = 135`, units dropped,
    `(x+1)^2 = x^2+2x+1`, intervals and tuples.
  - The string matcher of `../../train/evaluate/eval_math500.py` (LaTeX normalisation,
    then exact string or numeric equality within 1e-6, on the last `\boxed{}`) also
    counts as a match, so nothing it accepted is lost.
  - Multiple-choice golds (a single letter A–E) compare only the chosen letter, so
    `\textbf{(A)}` and `\textbf{(B) } 12` work; math_verify rejects the latter.
  - math_verify times out with `signal.alarm`, which only works in the main thread. In a
    worker thread the timeout is off and the input is capped (last 4,000 characters,
    parsed expressions up to 400 characters).
- Only the answer after `</think>` is searched; a `\boxed{}` inside the thought is working,
  not an answer.

### 4.3 Combining them

Weights (env vars `REWARD_W_LABEL`, `REWARD_W_CORRECT`): `w_label = 0.2`,
`w_correct = 0.8`. `tag_rate` and `label_mix` are not part of the reward; they are
computed and logged so the tag format and the label mix stay visible.

Before scoring, the workflow applies the format rule: a response with an invalid tag is
discarded and sampled again with the same prompt and sampling parameters, up to
`MAX_LABEL_RETRIES` (3) extra times. Only the kept response becomes the trajectory; the
discarded ones are neither scored nor trained on. The same holds in validation. A
response that still has an invalid tag after the retries is scored normally, with
`label_valid` = 0.

```
if truncated:
    not trained on (dropped from the batch; R = 0 is a placeholder)
elif not closed:
    R = 0
elif checkable(gold):
    R = w_label · label_valid + w_correct · answer_correct
else:
    R = label_valid
```

`R` is always in [0, 1].

`checkable(gold)` is false when the reference answer cannot be judged automatically, by
math_verify or by string matching. That is the case when any of the following holds:
- it is empty (e.g. coding questions, whose reference solution is a program);
- it is longer than 40 characters;
- it contains `\text` or `\mbox`;
- it contains a run of four or more letters outside LaTeX commands, i.e. prose.

rest_grpo keeps only questions with one checkable answer (the others went to
`no_answer/`, see [`../data_distrib.md`](../data_distrib.md)): all 5,302 of its answers
pass the check, so in V1 the `R = label_valid` branch never applies.

### 4.4 Worked examples

| case | label_valid | R |
|---|---|---|
| valid tags, correct answer | 1 | 0.2 + 0.8 = **1.0** |
| valid tags, wrong answer | 1 | **0.2** |
| every step `[logical_deduction]`, correct | 1 | **1.0** (the label mix is not rewarded) |
| some paragraphs untagged, the rest valid, correct | 1 | **1.0** |
| no tags at all, correct | 0 | **0.8** |
| an invalid tag left after the retries, correct | 0 | **0.8** |
| no `</think>` (finished without closing the thought) | — | **0** |
| hit 8,192 tokens | — | not trained on |

All 5,042 bespoke-v2 SFT targets have valid tags and score 1.0 with their own answers.
The annotated traces score `label_mix` 0.80 on average against the target (10th
percentile 0.71, median 0.81, 90th percentile 0.88), which is what the logged
`label_mix` can be compared with.

### 4.5 What gets logged

Per answer, averaged per step as `batch/<name>` (training) and `val/rest_grpo/<name>` (V2: `val/omni_math/<name>`)
(validation; the middle part is the task's `data_source`):

- `reward` (over the answers that are trained on, i.e. without the cut-off ones),
  `closed_think`, `truncated`;
- `reward_label` and `reward_correct`: the two terms of `reward` (w_label · label_valid and
  w_correct · answer_correct, both 0 for an answer that never closes its thought), over
  the same answers, so the two add up to `reward`;
- `n_steps`, `tag_rate`, `invented_tag_rate`;
- `label_retries` (regenerations this answer needed) and `invalid_label` (1 if it still had
  an invalid tag after the last retry);
- from V4, `repeat_retries` (attempts stopped for a repeated-label run),
  `repeat_discarded_tokens` (tokens those attempts had generated) and `repeat_in_final` (1 if
  the kept answer still has such a run);
- from V4, `loop` / `loop_partial` (1 if the answer ends in a repetition loop / partly
  repeats, `loop_label`) and `loop_among_cut` (the same flag, reported for cut-off answers
  only, so its batch mean is the share of them that loop);
- `label_valid` and `label_mix`;
- `answer_checkable`, and `answer_correct` (checkable questions only; a cut-off answer
  counts as wrong) and `answer_correct_finished` (the checkable answers that were not cut
  off);
- `share_<label>` for each of the 8 labels: the generated label mix.

rLLM additionally logs `reward/policy/{mean,std,min,max}` and
`advantage/policy/{mean,std}` (both over the answers left after dropping the cut-off ones),
`batch/policy/fractions/{effective,too_easy,too_hard}` (the share of groups whose rewards
differ, are all high, are all low),
`groups/num_trajs_before_filter` / `groups/num_trajs_after_filter` (answers generated, and
answers trained on) and `batch/termination_reason/max_response_length_exceeded`. An
episode counts as correct (`is_correct`, which feeds `val/rest_grpo/pass@1`) only when
`answer_correct` is true.

From V4, five wandb sections gather what to watch per training step, one line chart per
key (`labelcot/patches.py`, copied at the end of each step):

- `reward_breakdown/{total,format,correctness}`: `batch/reward`, `batch/reward_label`,
  `batch/reward_correct`, over the answers not cut off (format + correctness = total);
- `rollout_accuracy/{all,finished}`: `batch/answer_correct` (a cut-off answer counting as
  wrong) and `batch/answer_correct_finished`;
- `rollout_length/{rollouts,cut_off_count,cut_off_share,mean_tokens,max_tokens,min_tokens}`:
  rollouts generated, how many and what share hit the response limit (`batch/truncated`
  times the rollouts), and verl's `response_length/*` over every rollout, a cut-off one
  counting at the limit (verl's own `response_length/clip_ratio` stays at 0 and is not used);
- `loop/{loop,partial,among_cut_off}`: `batch/loop`, `batch/loop_partial`,
  `batch/loop_among_cut` (`labelcot.reward.loop_label`, the rules of the V3 loop analyses);
- `retry/{repeat_retries,repeat_discarded_tokens,repeat_in_final,label_retries,masked}`;
  `masked` is the share of rollouts dropped because their last attempt still had a
  repeated-label run (`batch/termination_reason/max_turns_exceeded`).

Validation gets the per-answer ones as `val/omni_math/<name>`. They replace the companion
monitor Job V3 ran beside the trainer (logging to a wandb run `<run>-monitor`, since
removed); V3's own run has none of them.

## 5. Hyperparameters

Every setting of each version. A new version adds a column, with the values it changes
from the previous one in bold; "same" means as in the previous column.

- **V1** was trained as run `qwen3-8b-v3-full-offload-2gpu-rest-grpo-v2` (wandb
  `lx5ib7gm`): the run name predates the version numbering; its `v3` is the SFT
  checkpoint's revision.
- **V2** (run `qwen3-8b-v3-full-offload-2gpu-omni-math-V2`) changes the data and the
  sampling: Omni-MATH instead of rest_grpo, whose ~80 % SFT accuracy left 90 % of the
  groups without a gradient, and Qwen3's thinking-mode sampling (T 0.6, top-p 0.95,
  top-k 20, min-p 0) with a presence penalty of 1.0 for training and validation, since
  greedy and long sampled answers of the SFT model ran into repetition loops (§6.2).
  Stopped after 9 of 200 steps (wandb `ygeidjrz`): 51 % of the answers hit 8,192
  tokens and were dropped, so on average only 1.3 of the 8 groups per step had a
  gradient, and 3 of the 9 steps had none (§6.1).
- **V3** (run `qwen3-8b-v3-full-offload-2gpu-omni-math-V3`) is V2 with 12,288 response
  tokens instead of 8,192 (the MATH-500 evaluation budget), started again from the SFT
  checkpoint.
- **V4** (not run yet) is V3 with a retry for answers stuck on one step type: during
  training, a run of at least 3 consecutive paragraphs opening with the same label, over
  200 tokens together, stops the answer while it is generated, and it is sampled again up
  to 2 times; a last attempt that still has such a run is dropped from training like a
  cut-off answer (it ends as `max_turns_exceeded`, which compact filtering masks; the
  single-turn workflow has no other use for that reason). On V3's steps 100 to 125 this
  rule would have stopped 9 % of the answers (6 % of the right ones, 13 % of the cut-off
  ones), mostly runs of `[reflecting]`. Answers are generated in 1,024-token chunks for it,
  so vLLM's presence penalty covers only the current chunk. Response tokens stay at 12,288;
  the micro-batch drops from 32,768 to 16,384 tokens per GPU, halving the activation memory
  (V3 peaked at 53.4 GB allocated per GPU). With CPUOffloadPolicy each micro-batch pays ~40 s
  to stream the weights in and the gradients out, and V3's actor update already took 729 s of
  its 1,410 s step, so twice as many micro-batches should add roughly 700 s per step.
  Checkpoints every 5 steps instead of 10, since V3 lost 9 steps to a preemption.

### 5.1 Model and data

| setting | V1 | V2 | V3 | V4 | config key (env var) |
|---|---|---|---|---|---|
| starting checkpoint | `Lucas-SJY/qwen3-8b-sft-bespoke` @ `v3` (commit `e24173c`), downloaded once into `HF_HOME` on the RL PVC | same | same | same | `actor_rollout_ref.model.path` (`POLICY_MODEL_PATH`, `POLICY_MODEL_REVISION`) |
| what is trained | all parameters (no LoRA) | same | same | same | `actor_rollout_ref.model.lora_rank` (`LORA_RANK=0`) |
| reference policy (KL) | a separate frozen copy of the SFT checkpoint, CPU-offloaded | same | same | same | `actor_rollout_ref.ref.*` |
| train / validation tasks | `GRPO/data/rest_grpo/`, 4,776 / 526 (math 4,560 / 495, science 216 / 31) | **`GRPO/data/omni_math/`, 3,153 / 350: Omni-MATH without its first 20 problems and without uncheckable or multi-value answers** | same | same | `TRAIN_FILE` / `VAL_FILE` |
| held out, never trained on | `no_answer/` (no checkable answer), `manual_inspection/` (10 questions) | **Omni-MATH's first 20 problems (the evaluation slice), 781 uncheckable and 124 multi-value answers** | same | same | `src/prepare_rest_grpo.py` (V1), `src/prepare_omni_math.py` (V2) |
| prompt | the task's user turn, the checkpoint's chat template, thinking on | same (`Return your final response within \boxed{}.` + the problem) | same | same | `src/labelcot/workflow.py` |
| max prompt / response tokens | 2,048 / 8,192 | same | 2,048 / **12,288** | same | `rllm.data.max_prompt_length` / `max_response_length` (`MAX_PROMPT_LENGTH`, `MAX_RESPONSE_LENGTH`) |
| seed | 42 | same | same | same | `rllm.data.seed` (`SEED`) |

### 5.2 Sampling and reward

| setting | V1 | V2 | V3 | V4 | config key (env var) |
|---|---|---|---|---|---|
| questions per step | 8 | same | same | same | `rllm.data.train_batch_size` (`TRAIN_BATCH_SIZE`) |
| answers per question (group size) | 16 | same | same | same | `rllm.rollout.n` (`GROUP_SIZE`) |
| answers per step | 128 | same | same | same | |
| train sampling | T 1.0, top-p 1.0, top-k off | **T 0.6, top-p 0.95, top-k 20, min-p 0, presence penalty 1.0** | same | same, but **the presence penalty covers only the current 1,024-token chunk** | `rllm.rollout.train.*` (`TRAIN_TEMPERATURE`, `TRAIN_TOP_P`, `TRAIN_TOP_K`, `TRAIN_MIN_P`, `TRAIN_PRESENCE_PENALTY`) |
| validation sampling | 1 answer, T 0.6, top-p 0.95, top-k 20 | 1 answer, T 0.6, top-p 0.95, top-k 20, **min-p 0, presence penalty 1.0** | same | same | `rllm.rollout.val.*` (`VAL_*`, as for training), `rllm.rollout.n_val` |
| reward | 0.2 · label_valid + 0.8 · answer_correct | same | same | same | `REWARD_W_LABEL` / `REWARD_W_CORRECT` |
| regenerations for an invalid tag | up to 3 | same | same | same | `MAX_LABEL_RETRIES` |
| regenerations for a repeated-label run | none | same | same | **training only: 3+ consecutive paragraphs with one label, over 200 tokens together, stop the answer (checked after every 1,024-token chunk); up to 2 new attempts; a last attempt that still has a run is dropped from training like a cut-off answer** | `src/labelcot/workflow.py` (`REPEAT_LABEL_RETRIES`, `REPEAT_MIN_PARAGRAPHS`, `REPEAT_MIN_TOKENS`, `GEN_CHUNK_TOKENS`, `REPEAT_MASK_FINAL`), `rllm.compact_filtering.mask_max_turns_exceeded` |
| answers cut off at the response limit | dropped from training, as are failed or timed-out episodes; validation counts them as wrong | same | same, at **12,288** tokens | same | `rllm.compact_filtering.*` (`MASK_TRUNCATED=true`) |

### 5.3 Objective and optimizer

| setting | V1 | V2 | V3 | V4 | config key (env var) |
|---|---|---|---|---|---|
| advantage | GRPO, std-normalised per group | same | same | same | `rllm.algorithm.adv_estimator`, `norm_adv_by_std_in_grpo` (`NORM_ADV_BY_STD=true`) |
| clip ε_low / ε_high, dual clip c | 0.2 / 0.28, 3.0 | same | same | same | `rllm.algorithm.eps_clip` / `eps_clip_high` (`CLIP_LOW` / `CLIP_HIGH`), `actor.clip_ratio_c` |
| loss aggregation, entropy bonus | token-mean, 0 | same | same | same | `rllm.algorithm.loss_agg_mode`, `actor.entropy_coeff` |
| KL to the reference | k3 (`low_var_kl`) in the loss, β 0.01 | same | same | same | `rllm.algorithm.kl_beta` (`KL_BETA`), `actor.kl_loss_type` |
| learning rate | 2e-6 | same | same | same | `actor_rollout_ref.actor.optim.lr` (`ACTOR_LR`) |
| schedule | constant, no warmup | same | same | same | verl defaults |
| optimizer | AdamW, β (0.9, 0.999), weight decay 0.01, gradient clipping at norm 1.0 | same | same | same | verl defaults |
| questions per optimizer update / updates per step | 4 / 2 | same | same | same | `actor.ppo_mini_batch_size` (`PPO_MINI_BATCH_SIZE`) |
| epochs over each batch | 1 | same | same | same | `actor.ppo_epochs` |

### 5.4 Compute

| setting | V1 | V2 | V3 | V4 | config key (env var) |
|---|---|---|---|---|---|
| GPUs | 2 × A100-80GB on one node, normal priority | same | same | same | `trainer.n_gpus_per_node` (`N_GPUS`, `GPU_TYPE`, `GPU_PRIORITY_CLASS`) |
| sharding / offload | FSDP2 with CPUOffloadPolicy: weights, gradients and Adam state stay in pinned host RAM; each layer is copied to the GPU only while it is computed; Adam steps on the CPU | same | same | same | `actor.strategy=fsdp2`, `fsdp_config.offload_policy=True` (`FSDP_CPU_OFFLOAD=true`) |
| precision | fp32 master weights, bf16 compute | same | same | same | verl defaults |
| tokens per micro-batch per GPU (update and log-prob passes), dynamic batching | 32,768 | same | same | **16,384** (the longest rollout, 2,048 + 12,288 tokens, still fits one) | `actor.ppo_max_token_len_per_gpu`, `*.log_prob_max_token_len_per_gpu` (`PPO_MAX_TOKEN_LEN`) |
| gradient checkpointing, padding removal | on, on | same | same | same | `actor_rollout_ref.model.*` |
| vLLM | one replica per GPU, tensor parallel 1, async, asleep (level 2) while the actor trains | same | same | same | `actor_rollout_ref.rollout.*` |
| vLLM share of the GPU while generating | 0.8 | same | same | same | `rollout.gpu_memory_utilization` (`ROLLOUT_GPU_MEM_UTIL`) |
| CPU threads per training worker (CPU Adam) | 9 | same | same | same | `OMP_NUM_THREADS` |
| pod | 20 CPU, 288Gi RAM, 24Gi `/dev/shm` | same | same | same | `POD_CPU`, `POD_MEMORY`, `k8s/job.yaml` |
| Ray object store | 16 GB | same | same | same | `RAY_OBJECT_STORE_GB` |
| image patch | verl's `save_checkpoint` does not move CPUOffloadPolicy weights to the GPU (it crashed otherwise) | same | same | same | `docker/patch_verl.py` |

### 5.5 Schedule and checkpoints

| setting | V1 | V2 | V3 | V4 | config key (env var) |
|---|---|---|---|---|---|
| steps, epochs | 200 (1,600 questions, a third of the training set), 1 | 200 (1,600 questions, half of the training set), 1 | same | same | `rllm.trainer.total_batches` (`TOTAL_TRAINING_STEPS`), `total_epochs` (`EPOCHS`) |
| validation | at step 0, every 20 steps, at the end | same | same | same | `rllm.trainer.test_freq` (`TEST_FREQ`), `val_before_train` (`VAL_BEFORE_TRAIN`) |
| checkpoint every | 10 steps (+ the last step) | same | same | **5 steps** (+ the last step), so a preempted run loses at most 4 steps | `rllm.trainer.save_freq` (`SAVE_FREQ`) |
| checkpoint contents | model + extra, no Adam state (~33 GB; a resumed run restarts Adam's moments) | same | same | same | `actor.checkpoint.save_contents` (`CKPT_SAVE_OPTIMIZER=false`) |
| checkpoints kept | the latest only (the previous one is deleted after the next is written) | same | same | same | `trainer.max_actor_ckpt_to_keep=1` |

### 5.6 Evaluation after training

| setting | V1 | where |
|---|---|---|
| export | the last checkpoint merged into a Hugging Face model, tokenizer and chat template from the SFT revision | `src/export_policy.py` (`../run.sh export`) |
| manual inspection | the 10 `manual_inspection/` questions, 4 answers each, validation sampling, scored like training; vLLM on 1 × RTX A6000 or L40 | `src/answer_tasks.py` (`../run.sh answer`, `ANSWER_GPU_TYPE`) |
| MATH-500 | by default training's sampling (T 0.6, top-p 0.95, top-k 20, min-p 0, presence penalty 1.0; the results below were greedy, `EVAL_TEMPERATURE=0`), 12,288 new tokens, otherwise as `../train`'s evaluation; transformers on 1 × RTX A6000 (`EVAL_GPU_TYPE=l40` for an L40); scored by string match, Math-Verify and the LLM judge (`qwen3-small`) | `../evaluate/` (`run_eval.sh`, `k8s/eval-job.yaml`) |

### 5.7 Switches not used by V1

`.env.example` and the code defaults describe a different setup: a LoRA adapter (rank 64,
alpha 32, all linear layers, `ACTOR_LR=1e-5`) on one 48 GB L40, FSDP1 offloading during
rollout, `GROUP_SIZE=8`, `KL_BETA=0.001`, `SAVE_FREQ=20`, `ROLLOUT_GPU_MEM_UTIL=0.7` and
12,288 tokens per micro-batch. Other switches the code supports: `MASK_TRUNCATED=false`
(cut-off answers trained on with `R = 0`), `NORM_ADV_BY_STD=false` (Dr. GRPO),
`CKPT_SAVE_OPTIMIZER=true` (checkpoints with Adam state, ~100 GB each).

## 6. Results and known properties

### 6.1 The V1 run

Validation on the 526 rest_grpo questions (1 answer each, T 0.6):

| step | pass@1 | cut off | `reflecting` | `verifying` | `logical_deduction` | steps per thought | `label_mix` |
|---|---|---|---|---|---|---|---|
| 0 (SFT) | 80.6 % | 15.2 % | 22.3 % | 11.4 % | 21.3 % | 20.8 | 0.748 |
| 40 | 79.7 % | 18.4 % | 27.0 % | 14.1 % | 18.9 % | 22.6 | 0.734 |
| 80 | 79.8 % | 18.3 % | 25.4 % | 12.9 % | 18.3 % | 25.8 | 0.697 |
| 120 | 78.3 % | 19.8 % | 20.4 % | 11.1 % | 22.9 % | 25.2 | 0.736 |
| 160 | 76.8 % | 21.3 % | 19.9 % | 9.0 % | 21.8 % | 15.3 | 0.714 |
| 200 | 78.5 % | 18.8 % | 21.7 % | 12.5 % | 21.5 % | 26.1 | 0.732 |

Training: 11.2 min per step (~37 h for 200 steps). 106 of the 200 steps had at least one
group with a nonzero advantage; on average 9.5 % of the groups did
(`batch/policy/fractions/effective`). The per-token KL to the reference rose to 3.5e-3 at
step 100 and 6.0e-3 at step 200.

Reading: V1 did not improve accuracy (80.6 % → 78.5 %; the 2.1-point drop is about one
standard error on 526 questions, but there is no gain), and more answers hit the 8,192-token limit
(15.2 % → 18.8 %). The label mix stayed close to the SFT model's, unlike the trial run
with `β = 0.001` (§7).

MATH-500 (`../evaluate`, greedy, 12,288 new tokens, RTX A6000, the SFT baseline's setup):

| | SFT v3 | V1 step 200 | only SFT / only V1 correct (sign test) |
|---|---|---|---|
| accuracy, string match | 81.4 % | 82.6 % | 22 / 28 (p = 0.48) |
| accuracy, Math-Verify | 85.2 % | 86.2 % | 23 / 28 (p = 0.58) |
| accuracy, LLM judge | 84.6 % | 85.8 % | 23 / 29 (p = 0.49) |
| cut off at 12,288 tokens | 64 | 60 | |
| mean / median completion tokens | 3,764 / 2,095 | 3,943 / 2,294 | |

All three scorers agree: the gain of about one point is within noise. Of the 22 problems
only SFT solves (string match), V1 loses 19 by running out of tokens; of the 28 only V1
solves, the SFT model had run out on 23.
Results: `../evaluate/results/qwen3-8b-v3-full-offload-2gpu-rest-grpo-v2-global_step_200/`.

### 6.1b The V2 run, stopped

Validation at step 0 on the 350 Omni-MATH questions (SFT model, V2 sampling): pass@1
45.7 %, 45.1 % cut off at 8,192 tokens. Training steps 1-9 (8 questions x 16 answers):

| | mean | range |
|---|---|---|
| answers correct (cut-off counts as wrong) | 39.4 % | 22.7-51.6 % |
| answers cut off | 51 % | 39-66 % |
| groups left after dropping cut-off answers (of 8) | 5.4 | 4-7 |
| groups with a gradient (of 8) | 1.3 | 0-4; none in steps 2, 3, 9 |
| per-token KL to the reference | rising | 3.0e-4 -> 1.3e-3 |
| `offpolicy/log_ppl_abs_diff` | 0.023 | 0.021-0.026 |
| minutes per step | 15.0 | 13.2-17.1 |

The groups that were left were mostly all right: the model usually solves what it
finishes, and fails by running out of tokens. V3 raises the limit to 12,288.

### 6.2 Known properties

**Measured on the data.** Every annotated SFT target has valid tags, so for the SFT
model `label_valid` starts at (or near) 1 and the label term is mostly constant. The
learning signal then comes almost entirely from answer correctness; the label term
(0.2) mainly guards against losing the tag format (no tags, or invented tags).

- **Most groups carry no signal.** The SFT model already answers ~80 % of the questions
  correctly, so for most questions all 16 answers are right (or all wrong) and the group
  has zero advantage; in V1 only 9.5 % of the groups got a gradient from the policy
  loss. Filtering out questions the SFT model always (or never) solves would spend the
  compute on the informative ones.
- **Group normalisation amplifies small differences.** This cuts both ways:
  - When all answers to a question are equally right or wrong, only the label terms
    differ, and dividing by the small group σ scales those differences up to unit-size
    advantages. With a binary label term this only happens when some answers lose their
    tags; otherwise the group is uniform and contributes nothing.
  - The same scaling also amplifies noise.
  - `NORM_ADV_BY_STD=false` turns the scaling off.
- **Truncation is neither rewarded nor penalised.** Dropping cut-off answers (DAPO's
  overlong filtering) avoids punishing reasoning that was on track but long, and avoids
  a push toward shorter reasoning. Nothing pushes the model to finish within 8,192 tokens
  either, and in V1 the share of cut-off validation answers rose from 15.2 % to 18.8 %.
  A soft overlong penalty (DAPO) would counter that.
- **`tag_rate` is not rewarded.** According to the model card, the SFT model already
  used the tags in all 500 MATH-500 outputs; `batch/tag_rate` and
  `batch/invented_tag_rate` show whether that holds during training.
- **The label term is easy to satisfy.** One valid tag anywhere in the thought earns it.
  Nothing checks that a tag fits its paragraph, which labels are used, how often, or in
  what order; a thought tagged `[logical_deduction]` throughout scores the same as a
  natural mix. Only the KL term keeps the mix near the SFT model's; `batch/label_mix`
  and the `share_<label>` metrics show whether it drifts.
- **Untagged paragraphs are not penalised.** As long as at least one tag is there and
  none is invalid, untagged paragraphs cost nothing (`tag_rate` shows how many there
  are). An answer with no tags at all loses the label term: if it is right it scores
  0.8, still well above a well-tagged wrong answer (0.2).
  A group whose answers are all untagged and equally right gets no signal towards the
  tags. Enforcing the tag format by rule at generation time (constrained decoding) would
  close this without a reward term.
- **Regeneration hides the format errors from training.** Rejected samples are never
  trained on, so the policy gets no gradient against invalid tags (only a response that
  is still invalid after the last retry loses the label term); the rule keeps the
  output valid, but the underlying rate does not fall by itself. `batch/label_retries`
  shows that rate, and each retry is a full extra generation (up to 8,192 tokens), so a
  high rate also slows every step. Only invalid tags trigger a retry; untagged paragraphs
  are allowed.
- **Training is more lenient than the offline evaluation.** `answer_correct` accepts
  math_verify equivalences that `../../train/evaluate/eval_math500.py` (string and number
  matching only) rejects, so training accuracy can read higher than offline accuracy on
  the same answers.
- **The reference answers are R1's.** A rest_grpo answer is the final answer of
  DeepSeek-R1's solution in Bespoke-Stratos, not an independently verified ground truth.
- **Sampler-only settings make training slightly off-policy (V2).** The actor scores the
  rollouts at the sampling temperature (rLLM passes it to verl's log-prob pass), but
  top-p, top-k, min-p and the presence penalty change only vLLM's sampler: the tokens
  come from a truncated, penalised distribution while the PPO ratio and the KL use the
  untruncated policy. rLLM's verl engine takes only temperature, top-p and top-k from
  `rllm.rollout.*`; min-p and the presence penalty reach vLLM through the workflow
  (`train_grpo.py`, `extra_sampling_params`). `offpolicy/log_ppl_abs_diff` in wandb shows
  how far vLLM's and the actor's log-probs drift apart.
- **Sparse, sequence-level credit.** Every token of an answer gets the same advantage;
  there is no per-step signal for individual tags.
- **Only the last checkpoint is kept.** With `max_actor_ckpt_to_keep=1` an earlier,
  better step cannot be recovered after the run.

Open design question: what "follow the annotated labels" should mean.

- **(a) Format only.** This is what is implemented now: tags must exist and be valid;
  which labels are used is left to the policy. Earlier designs rewarded the label mix,
  first against each question's reference trace (mix and order), then against the
  corpus-wide average mix.
- **(b) Each tag is semantically correct for its step.** This needs a label judge, for
  example a classifier trained on the ~200k labelled spans in `bespoke-v2`.
- **(c) Controllable generation.** The prompt specifies a label plan or allowed label set,
  and the reward checks compliance.

## 7. Before V1

Trial runs that led to V1; none is a version.

| run | setup | outcome |
|---|---|---|
| 1-GPU runs on `grpo_try` (9 questions) | pipeline tests: reward, truncation masking, full-parameter training with FSDP2 CPU offload, checkpointing | the step-10 checkpoint crash that `docker/patch_verl.py` fixes; few steps with any signal |
| `qwen3-8b-v3-full-offload-2gpu-rest-grpo-v1` (wandb `b06lttf5`) | as V1 except: 12 questions × 8 answers per step (96), 6 questions per update, `ACTOR_LR=1e-6`, `KL_BETA=0.001` | pass@1 80.8 % → 81.2 %, cut off 15.4 % → 15.2 %; but the label mix drifted: `reflecting` 21.7 % → 36.1 %, `verifying` 11.6 % → 6.3 %, `label_mix` 0.744 → 0.682; KL 4.6e-3 at step 200; 7.9 min per step; 120 / 200 steps with signal, 8.3 % of groups |

V1 changed from the second trial: `KL_BETA` 0.001 → 0.01 (the label drift), group size
8 → 16 (more groups that mix right and wrong answers), `ACTOR_LR` 1e-6 → 2e-6 (accuracy
had not moved), 8 questions per step (128 answers) and 4 per update.
