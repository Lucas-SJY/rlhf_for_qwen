# GRPO datasets

All task files share one format, one JSON object per line:
`id`, `data_source`, `prompt` (one user turn holding the question, as in SFT),
`question` and `answer` (the reference final answer; `""` when there is none). No
reference reasoning is included: the reward judges only the label format and the final
answer. The image bakes in `GRPO/data/`, so a changed task set needs `./run.sh image`.

## Overview

| Set | Location | Train | Validation | Built by | Notes |
|---|---|---|---|---|---|
| bespoke-v2 | `GRPO/data/{train,validation}.jsonl` | 5,042 | 102 | `GRPO/src/prepare_data.py` | the SFT questions themselves; default when `TRAIN_FILE` is unset |
| grpo_try | `GRPO/data/grpo_try/` | 6 | 3 | `GRPO/src/prepare_grpo_try.py` | 9 hand-picked questions, for pipeline tests |
| **rest_grpo** | `GRPO/data/rest_grpo/` | **4,776** | **526** | `GRPO/src/prepare_rest_grpo.py` | Bespoke-Stratos questions not used in SFT, one checkable answer each; **used by the current runs** |
| no_answer | `no_answer/` | 5,623 | 631 | `GRPO/src/prepare_rest_grpo.py` | split off from rest_grpo: no checkable answer; not trained on (gitignored) |
| manual_inspection | `manual_inspection/` | 10 | – | `GRPO/src/prepare_rest_grpo.py` | held out from rest_grpo training for reading; never trained on |

## rest_grpo

### Source

Bespoke-Stratos-17k has 16,710 samples (`sample_000000` .. `sample_016709`), each a
question with a DeepSeek-R1 reasoning trace and solution. The SFT set
`../train/data_labeled_2` uses 5,144 of them; the harbor dataset
`../jianhong_harbor/harbor/datasets/bespoke-stratos-rest` holds exactly the other 11,566
(`<id>/environment/trajectory.json`). rest_grpo is built from those 11,566 only, and
every id in `../train/data_labeled_2/*.jsonl` is excluded explicitly as well, so it
cannot overlap the SFT data (0 overlap).

### Build steps

1. Read every `trajectory.json`; the question is `question`, the answer is the content of
   the last `\boxed{}` in `solution`. All 11,566 files pass the format check.
2. Split by question, seeded (42): 10 % validation, i.e. 10,409 train / 1,157 validation.
3. Move to `no_answer/`, keeping the split, every question the reward cannot score on
   correctness:
   - **no answer**: the coding questions ("Generate an executable Python function ...")
     and a few science questions, whose solution has no `\boxed{}`;
   - **answer not checkable**: the answer fails `answer_is_checkable` (empty, longer than
     40 characters, contains `\text` / `\mbox`, or is prose), e.g. proofs or `\text{No}`;
   - **several answers**: the solution boxes more than one distinct value (several
     roots, multi-part questions), so the last `\boxed{}` alone is an incomplete answer.
4. Hold out 10 random training questions (seeded) in `manual_inspection/`.
5. Read every written file back and check its format, and that no id appears twice.

### Where every question went

| Destination | Train | Validation | Total |
|---|---|---|---|
| **rest_grpo** (one checkable answer) | **4,776** | **526** | **5,302** |
| manual_inspection | 10 | – | 10 |
| no_answer: no answer (coding) | 4,852 | 543 | 5,395 |
| no_answer: no answer (science) | 12 | 3 | 15 |
| no_answer: answer not checkable (math) | 213 | 27 | 240 |
| no_answer: answer not checkable (science) | 497 | 55 | 552 |
| no_answer: several answers (math) | 48 | 3 | 51 |
| no_answer: several answers (science) | 1 | – | 1 |
| **Total** | **10,409** | **1,157** | **11,566** |

### Composition of rest_grpo

| Question type | Train | Validation |
|---|---|---|
| Math ("Return your final response within \boxed{}. ...") | 4,560 | 495 |
| Science (physics, chemistry, ...) | 216 | 31 |
| **Total** | **4,776** | **526** |

The 10 held-out questions in `manual_inspection/` are all math: `sample_001293`,
`sample_005756`, `sample_006111`, `sample_006202`, `sample_006463`, `sample_007240`,
`sample_007450`, `sample_007728`, `sample_009137`, `sample_010366`.
`manual_inspection/tasks.jsonl` holds them in the task format and
`manual_inspection/trajectories/<id>.json` a copy of each upstream trajectory.

### About the answers

- An answer is the final answer of DeepSeek-R1's solution, not an independent ground
  truth. Bespoke-Stratos reportedly kept only solutions that matched the original
  problems' answers, so they are very likely right, but that filter is the only check;
  nothing here verified them again.
- During training the final answer after `</think>` is compared with it by the string
  matcher of `../train/evaluate` and, failing that, by Hugging Face `math_verify`
  (equivalence such as `0.5 = 1/2`); multiple-choice letters are compared directly.
- `answer_is_checkable` only judges the form of a reference answer (short, no prose), not
  whether it is right.

### Before the separation

The distribution after step 2, before anything was moved out:

| Answer type | Train | Validation | Total |
|---|---|---|---|
| Checkable answer | 4,835 | 529 | 5,364 |
| Answer present but not checkable (prose) | 710 | 82 | 792 |
| No answer (coding questions) | 4,864 | 546 | 5,410 |
| **Total** | **10,409** (90%) | **1,157** (10%) | **11,566** |

Of the 5,364 checkable ones, 52 boxed several distinct values and moved to `no_answer/`
afterwards, and 10 were held out for manual inspection.

## Rebuilding

```bash
python3 GRPO/src/prepare_rest_grpo.py               # rest_grpo/, no_answer/, manual_inspection/
python3 GRPO/src/prepare_rest_grpo.py --check-only  # format check of the existing files
./run.sh image                                      # bake the new files into the image
```

The output is deterministic (seed 42): rerunning it reproduces the same files.
