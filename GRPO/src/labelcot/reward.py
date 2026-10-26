"""Rule-based reward for "reason in labelled steps".

The model writes its thought as blank-line separated paragraphs, each opening with one
of eight tags, e.g. ``[logical_deduction] 196 = 2^2 * 7^2 ...``. No reference trace is
used: a completion is scored only on whether its tags are valid and on its final answer.

    label    1 if the thought has at least one tag and every tag is one of the eight
             labels, else 0 (no tags at all, or any tag outside the eight)
    correct  the final answer after </think> matches the reference answer; only the final
             answer is judged, with math_verify's equivalence (0.5 = 1/2 = \\frac{1}{2})

    reward = w_label * label + w_correct * correct      (0.2 and 0.8)

Untagged paragraphs between tagged ones are allowed. tag_rate (the share of paragraphs
that open with a valid tag) and label_mix (how closely the label shares match the
average annotated trace, LABEL_PRIOR) are logged as diagnostics, not rewarded.

A completion that never closes its thought gets 0. One cut off by the length limit is
not scored at all: the trainer drops it from the batch (compact filtering, see
entrypoint.sh), so its reward field is only a placeholder and it is left out of the
logged reward. A tag outside the eight labels also makes the workflow regenerate the completion
(invalid_labels below); one that still has an invalid tag after the retries is scored
normally, i.e. without the label half.
When the reference answer cannot be checked automatically (prose, proofs, code), the
correctness term is dropped and the reward is the label term alone, so it stays in
[0, 1] either way.

The two terms are logged separately (reward_label + reward_correct = reward, over the
answers that get a reward), next to answer_correct (every checkable answer, a cut-off one
counting as wrong) and answer_correct_finished (the checkable answers that were not cut off).

A thought stuck on one step type, a run of paragraphs that all open with the same label
(repeated_label_run below), makes the workflow stop the completion and sample it again;
that is a rule of the workflow as well, not a reward term.

Every answer is also checked for a repetition loop at its end (loop_label below, logged as
loop, loop_partial and loop_among_cut, not rewarded).

Stdlib only apart from math_verify (Hugging Face), so it can be unit-tested on a laptop
without torch or rLLM. Without math_verify (Python < 3.10) the answer check falls back to
the string matcher alone; the image always has it.
"""

from __future__ import annotations

import re
import threading
import zlib
from collections import Counter
from dataclasses import asdict, dataclass, field
from fractions import Fraction

# Canonical label vocabulary, identical to ../train/src/prepare_data.py.
LABELS = (
    "planning_next_step",
    "restating_problem",
    "recalling_knowledge",
    "logical_deduction",
    "reflecting",
    "verifying",
    "correcting_itself",
    "concluding",
)
_LABEL_SET = frozenset(LABELS)

# Target label mix for the logged label_mix diagnostic: the mean share of each label in one annotated
# trace, over all 5,144 bespoke-v2 traces (201,250 spans). Recompute with
# GRPO/src/compute_label_prior.py. Normalised below, so the rounding does not matter.
_RAW_PRIOR = {
    "logical_deduction": 0.2525,
    "reflecting": 0.1999,
    "planning_next_step": 0.1589,
    "verifying": 0.1427,
    "restating_problem": 0.0874,
    "concluding": 0.0742,
    "recalling_knowledge": 0.0722,
    "correcting_itself": 0.0121,
}
LABEL_PRIOR = {label: _RAW_PRIOR[label] / sum(_RAW_PRIOR.values()) for label in LABELS}

# A step opens with "[tag]" at the very start of its paragraph.
# What counts as a tag at the start of a paragraph: a bracketed single word of 3+
# letters, digits, "_" or "-" ("[thinking]", "[Planning_Next_Step]", "[self-check]"), or
# a spaced spelling of one of the eight labels ("[planning next step]"). Other bracketed
# text, such as "[1, 2]", "[x for x in xs]" or "[Step 1]", is ordinary content.
_TAG_AT_START = re.compile(r"^\[([A-Za-z][A-Za-z0-9_\-]{2,}|[A-Za-z][A-Za-z ]+[A-Za-z])\]")


def _opening_tag(paragraph: str) -> str | None:
    match = _TAG_AT_START.match(paragraph)
    if match is None:
        return None
    name = match.group(1)
    if " " in name and "_".join(name.lower().split()) not in _LABEL_SET:
        return None
    return name


_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class RewardWeights:
    # The final answer carries most of the reward; valid labels the rest.
    label: float = 0.2
    correct: float = 0.8


@dataclass
class RewardBreakdown:
    reward: float
    closed: bool
    truncated: bool
    n_steps: int
    tag_rate: float
    invented_tag_rate: float
    label_valid: bool
    label_mix: float
    answer_checkable: bool
    answer_correct: bool
    label_counts: dict[str, int] = field(default_factory=dict)
    # Regenerations the workflow needed before the completion had no invalid tag.
    label_retries: int = 0
    # Completions the workflow stopped and sampled again for a repeated-label run, the
    # tokens those discarded attempts had generated, and whether the kept completion still
    # has such a run (the last attempt is never stopped).
    repeat_retries: int = 0
    repeat_discarded_tokens: int = 0
    repeat_in_final: bool = False
    # The two parts of the reward: label_reward + correct_reward = reward.
    label_reward: float = 0.0
    correct_reward: float = 0.0
    # How the answer ends (loop_label): "loop", "partial" or "none".
    loop: str = "none"

    def metrics(self) -> dict[str, float]:
        """Flat float metrics for rLLM's per-episode logging (averaged per batch)."""
        out = {
            "closed_think": float(self.closed),
            "truncated": float(self.truncated),
            "n_steps": float(self.n_steps),
            "tag_rate": self.tag_rate,
            "invented_tag_rate": self.invented_tag_rate,
            "invalid_label": float(self.invented_tag_rate > 0),
            "label_retries": float(self.label_retries),
            "repeat_retries": float(self.repeat_retries),
            "repeat_discarded_tokens": float(self.repeat_discarded_tokens),
            "repeat_in_final": float(self.repeat_in_final),
            "label_valid": float(self.label_valid),
            "label_mix": self.label_mix,
            "answer_checkable": float(self.answer_checkable),
            "loop": float(self.loop == "loop"),
            "loop_partial": float(self.loop == "partial"),
        }
        # Reported for cut-off answers only, so the batch mean is the share of them that loop.
        if self.truncated:
            out["loop_among_cut"] = float(self.loop == "loop")
        # A cut-off completion is dropped from training, so its placeholder reward would
        # only drag the logged mean down.
        if not self.truncated:
            out["reward"] = self.reward
            out["reward_label"] = self.label_reward
            out["reward_correct"] = self.correct_reward
        # Only reported where it means something, so the batch mean is accuracy on
        # checkable questions rather than being diluted by prose answers. answer_correct
        # counts a cut-off answer as wrong; answer_correct_finished leaves those out.
        if self.answer_checkable:
            out["answer_correct"] = float(self.answer_correct)
            if not self.truncated:
                out["answer_correct_finished"] = float(self.answer_correct)
        total = sum(self.label_counts.values())
        if total:
            for label in LABELS:
                out[f"share_{label}"] = self.label_counts.get(label, 0) / total
        return out

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def split_completion(text: str) -> tuple[str, str, bool]:
    """Split a completion into (thought, answer, closed).

    The prompt ends at ``<|im_start|>assistant\\n``, so the model opens its own
    ``<think>`` block. Only the text after ``</think>`` counts as the answer; the
    thought routinely contains a ``\\boxed{}`` of its own.
    """
    if "</think>" in text:
        thought, _, answer = text.partition("</think>")
        closed = True
    else:
        thought, answer, closed = text, "", False
    thought = thought.strip()
    if thought.startswith("<think>"):
        thought = thought[len("<think>") :]
    return thought.strip(), answer.strip(), closed


def _scan_labels(thought: str) -> tuple[list[str | None], list[str]]:
    """Return (label per paragraph, invalid tags in order of appearance)."""
    labels: list[str | None] = []
    invalid: list[str] = []
    for paragraph in _PARAGRAPH_SPLIT.split(thought):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        name = _opening_tag(paragraph)
        if name is None:
            labels.append(None)
        elif name in _LABEL_SET:
            labels.append(name)
        else:
            labels.append(None)
            invalid.append(name)
    return labels, invalid


def parse_step_labels(thought: str) -> tuple[list[str | None], int]:
    """Return (label per paragraph, number of invented tags).

    A paragraph maps to its label when it opens with a known tag, to None when it has
    no tag or an unknown one. Unknown tags are counted separately.
    """
    labels, invalid = _scan_labels(thought)
    return labels, len(invalid)


def _compression_ratio(text: str) -> float:
    data = text.encode()
    return len(zlib.compress(data)) / max(1, len(data))


def loop_label(text: str) -> str:
    """How an answer ends: "loop" (it repeats itself), "partial" (it circles, or enumerates
    case after case) or "none".

    Normal reasoning compresses to about 0.3 of its size; a verbatim loop to well under 0.12.
    loop: the last 6,000 characters compress below 0.12, repeat one paragraph 5+ times or keep
    under 30 % distinct lines, or the last 2,000 characters compress below 0.15 (a loop that
    started late). partial: the looser versions (0.2, 3+, 60 %), the last 2,000 characters
    below 0.25, or 20 %+ of the closing 12-word runs already written earlier in the answer.
    Calibrated on V3's validation answers: no finished answer compressed its last 2,000
    characters below 0.24.
    """
    tail = text[-6000:]
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", tail) if p.strip()][-30:]
    top = Counter(paragraphs).most_common(1)[0][1] if paragraphs else 0
    lines = [line.strip() for line in tail.splitlines() if line.strip()][-60:]
    distinct = len(set(lines)) / max(1, len(lines))
    ratio, ratio_end = _compression_ratio(tail), _compression_ratio(text[-2000:])
    if ratio < 0.12 or top >= 5 or distinct < 0.3 or ratio_end < 0.15:
        return "loop"
    words = text.split()
    start = max(12, len(words) - len(tail.split()))
    seen = {tuple(words[i:i + 12]) for i in range(0, start - 11)}
    grams = [tuple(words[i:i + 12]) for i in range(start, len(words) - 11)]
    repeated = 0
    for gram in grams:
        repeated += gram in seen
        seen.add(gram)
    if ratio < 0.2 or top >= 3 or distinct < 0.6 or ratio_end < 0.25 or (grams and repeated / len(grams) >= 0.2):
        return "partial"
    return "none"


def repeated_label_run(thought: str, count_tokens, min_paragraphs: int = 3,
                       min_tokens: int = 200) -> tuple[str, int, int] | None:
    """The first run of at least min_paragraphs consecutive paragraphs opening with the same
    label whose paragraphs together exceed min_tokens tokens, as (label, paragraphs, tokens).

    count_tokens(text) gives a paragraph's length in tokens (the workflow passes the policy's
    tokenizer). An untagged paragraph, or one with a tag outside the eight labels, ends a run.
    The last paragraph may still be unfinished; its tokens only add to a run. None when no
    run qualifies.
    """
    run_label, run_paragraphs, run_tokens = None, 0, 0
    for paragraph in _PARAGRAPH_SPLIT.split(thought):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        name = _opening_tag(paragraph)
        label = name if name in _LABEL_SET else None
        if label is not None and label == run_label:
            run_paragraphs += 1
            run_tokens += count_tokens(paragraph)
        else:
            run_label = label
            run_paragraphs, run_tokens = (1, count_tokens(paragraph)) if label is not None else (0, 0)
        if run_label is not None and run_paragraphs >= min_paragraphs and run_tokens > min_tokens:
            return run_label, run_paragraphs, run_tokens
    return None


def invalid_labels(text: str) -> list[str]:
    """Tags outside the eight labels that open a paragraph of the thought.

    The rule-based format check: LabeledCoTWorkflow regenerates a completion that has
    any, and score_completion gives no label credit to one that still has any after the
    retries. Untagged paragraphs are allowed; only wrong tags fail.
    """
    thought, _, _ = split_completion(text)
    return _scan_labels(thought)[1]


def label_mix_similarity(generated: list[str], target: dict[str, float] | None = None) -> float:
    """1 - total variation distance between the generated label shares and ``target``.

    1.0 means the thought uses the eight labels in exactly the target proportions; a
    thought with no valid tag scores 0.
    """
    if not generated:
        return 0.0
    target = target or LABEL_PRIOR
    counts, n = Counter(generated), len(generated)
    tvd = 0.5 * sum(abs(counts[label] / n - target.get(label, 0.0)) for label in LABELS)
    return max(0.0, 1.0 - tvd)


# ---------------------------------------------------------------------------
# string matcher -- same normalisation as ../train/evaluate/eval_math500.py. It is the
# fallback and the multiple-choice path of answer_is_correct below.
# ---------------------------------------------------------------------------


def extract_boxed(text: str) -> str | None:
    """Return the content of the last \\boxed{...}, matching braces properly."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        rest = text[idx + len("\\boxed") :].strip()
        return rest.split()[0] if rest else None
    depth, out = 0, []
    for ch in text[i:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        out.append(ch)
    return None


_STRIP = [
    (r"\\left", ""), (r"\\right", ""), (r"\\!", ""), (r"\\,", ""), (r"\\;", ""), (r"\\ ", " "),
    (r"\\dfrac", r"\\frac"), (r"\\tfrac", r"\\frac"), (r"\\cdot", "*"), (r"\\times", "*"),
    (r"\^\{\\circ\}", ""), (r"\^\\circ", ""), (r"\\%", ""), (r"%", ""),
    (r"\\\$", ""), (r"\$", ""), (r"\\text\{([^}]*)\}", r"\1"), (r"\\mbox\{([^}]*)\}", r"\1"),
]


def normalize(ans: str | None) -> str:
    if ans is None:
        return ""
    s = ans.strip()
    for pat, rep in _STRIP:
        s = re.sub(pat, rep, s)
    s = s.replace(" ", "").replace("\n", "")
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\frac(\d)(\d)", r"(\1)/(\2)", s)
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    s = s.rstrip(".")
    s = re.sub(r"^\{(.*)\}$", r"\1", s)
    s = re.sub(r"(\d),(\d\d\d)", r"\1\2", s)
    return s.lower()


def to_number(s: str):
    try:
        return float(Fraction(s))
    except (ValueError, ZeroDivisionError):
        pass
    m = re.fullmatch(r"\(?(-?[\d.]+)\)?/\(?(-?[\d.]+)\)?", s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(s)
    except ValueError:
        return None


_CHOICE = re.compile(r"^\(?([a-e])\)?(?![a-z0-9])")


def _choice_letter(ans: str) -> str | None:
    """Letter of a multiple-choice answer such as 'A', '(B)', '\\textbf{(C)} 12'."""
    s = normalize(ans)
    s = re.sub(r"\\(textbf|mathrm|mathbf|text)", "", s).replace("{", "").replace("}", "")
    m = _CHOICE.match(s)
    return m.group(1) if m else None


def is_correct(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    if re.fullmatch(r"\s*[A-Ea-e]\s*", gold):
        return _choice_letter(pred) == gold.strip().lower()
    p, g = normalize(pred), normalize(gold)
    if p == g:
        return True
    pn, gn = to_number(p), to_number(g)
    if pn is not None and gn is not None:
        return abs(pn - gn) < 1e-6
    return False


# ---------------------------------------------------------------------------
# final-answer check with Hugging Face math_verify
# ---------------------------------------------------------------------------

try:  # needs Python >= 3.10; the image always has it (Dockerfile import check)
    from math_verify import parse as _mv_parse
    from math_verify import verify as _mv_verify
except ImportError:  # e.g. a laptop on Python 3.9: only the string matcher above is used
    _mv_parse = _mv_verify = None

MATH_VERIFY_AVAILABLE = _mv_parse is not None
_MV_TIMEOUT_S = 5
_MAX_ANSWER_CHARS = 4000
_MAX_PARSED_CHARS = 400
_MULTIPLE_CHOICE = re.compile(r"\s*[A-Ea-e]\s*")


def _mv_timeout() -> int | None:
    """math_verify times out with signal.alarm, which only works in the main thread.

    rLLM may score inside a worker thread; there the timeout is off and the input size
    is capped instead (see answer_is_correct).
    """
    return _MV_TIMEOUT_S if threading.current_thread() is threading.main_thread() else None


def _math_verify_match(answer: str, gold: str) -> bool:
    timeout = _mv_timeout()
    try:
        gold_parsed = _mv_parse("\\boxed{" + gold + "}", parsing_timeout=timeout)
        pred_parsed = _mv_parse(answer[-_MAX_ANSWER_CHARS:], parsing_timeout=timeout)
        if not gold_parsed or not pred_parsed:
            return False
        # Without a timeout, a huge expression could stall sympy's simplification.
        if any(len(str(x)) > _MAX_PARSED_CHARS for x in pred_parsed):
            return False
        return bool(_mv_verify(gold_parsed, pred_parsed, timeout_seconds=timeout))
    except Exception:  # timeouts and sympy errors on odd input count as "not matched"
        return False


def answer_is_correct(answer: str, gold: str) -> bool:
    """Whether the final answer (the text after </think>) equals the reference answer.

    Only the final answer is judged, never the reasoning. math_verify decides
    equivalence (0.5 = 1/2 = \\frac{1}{2}, 3,840 = 3840, 135^\\circ = 135, units dropped)
    and reads the last \\boxed{}, or the final stated answer when there is none. The
    string matcher above still counts as a match too, so nothing it accepted is lost.
    Multiple-choice golds (a single letter) use only the letter match, because
    math_verify rejects AMC-style answers such as "\\textbf{(B) } 12".
    """
    legacy = is_correct(extract_boxed(answer), gold)
    if legacy or _MULTIPLE_CHOICE.fullmatch(gold) or not MATH_VERIFY_AVAILABLE:
        return legacy
    return _math_verify_match(answer, gold)


def answer_is_checkable(gold: str) -> bool:
    """Whether string matching can judge this reference answer.

    About 2 % of bespoke-v2 answers are prose ("\\text{P and Q cannot both be
    true.}") or long statements to be proven. Those cannot be scored by matching, so
    the correctness term is dropped for them instead of adding a constant 0.
    """
    g = (gold or "").strip()
    if not g or len(g) > 40:
        return False
    if "\\text" in g or "\\mbox" in g:
        return False
    # Four or more consecutive letters outside a LaTeX command means words, i.e. prose.
    return re.search(r"[A-Za-z]{4,}", re.sub(r"\\[A-Za-z]+", "", g)) is None


# ---------------------------------------------------------------------------
# reward
# ---------------------------------------------------------------------------


def score_completion(text: str, task: dict, truncated: bool, weights: RewardWeights | None = None) -> RewardBreakdown:
    """Score one completion against its task (needs only task['answer'])."""
    w = weights or RewardWeights()
    thought, answer, closed = split_completion(text)

    step_labels, invented = parse_step_labels(thought)
    n_steps = len(step_labels)
    tagged = [label for label in step_labels if label is not None]
    tag_rate = len(tagged) / n_steps if n_steps else 0.0
    invented_rate = invented / n_steps if n_steps else 0.0

    # The label half: some tag, and none outside the eight labels.
    label_valid = bool(tagged) and not invented
    label_mix = label_mix_similarity(tagged)

    gold = str(task.get("answer") or "")
    checkable = answer_is_checkable(gold)
    correct = bool(closed and checkable and answer_is_correct(answer, gold))

    label_term = correct_term = 0.0
    if truncated:
        # Placeholder: the trainer masks length-truncated episodes out of the batch, so
        # this value never reaches an advantage (MASK_TRUNCATED=false trains it as 0).
        pass
    elif not closed:
        pass
    elif checkable:
        label_term = w.label * float(label_valid)
        correct_term = w.correct * float(correct)
    else:
        # No correctness term: the label term alone, which keeps R in [0, 1].
        label_term = float(label_valid) if w.label > 0 else 0.0
    reward = label_term + correct_term

    return RewardBreakdown(
        reward=float(reward),
        closed=closed,
        truncated=truncated,
        n_steps=n_steps,
        tag_rate=tag_rate,
        invented_tag_rate=invented_rate,
        label_valid=label_valid,
        label_mix=label_mix,
        answer_checkable=checkable,
        answer_correct=correct,
        label_counts=dict(Counter(tagged)),
        label_reward=float(label_term),
        correct_reward=float(correct_term),
        loop=loop_label(text),
    )
