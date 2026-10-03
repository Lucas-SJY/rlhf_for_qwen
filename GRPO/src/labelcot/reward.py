"""Rule-based reward for "reason in labelled steps, the way the annotation does".

The SFT model writes its thought as blank-line separated paragraphs, each opening with
one of eight tags, e.g. ``[logical_deduction] 196 = 2^2 * 7^2 ...``. Every training
question also carries the human/LLM annotation of the reference trace, i.e. the
sequence of labels the reference reasoning used. The reward scores a completion on:

    tag      fraction of thought paragraphs that open with a valid tag
    align    how closely the generated label sequence follows the annotated one:
             mean of (1 - total variation distance between the two label histograms)
             and a sequence similarity of the run-length collapsed label sequences
    correct  the final answer after </think> matches the reference answer; only the final
             answer is judged, with math_verify's equivalence (0.5 = 1/2 = \\frac{1}{2})

    reward = w_align * align + w_correct * correct      (0.5 each)

tag is computed and logged as a diagnostic, but is not part of the reward.

A completion that never closes its thought, or is cut off by the length limit, gets 0.
A tag outside the eight labels is a format violation, not a reward term: the workflow
regenerates such a completion (invalid_labels below), and one that still has an invalid
tag after the retries gets 0 as well.
When the reference answer cannot be checked automatically (prose, proofs, code), the
correctness term is dropped and the reward is the alignment term alone, so it stays in
[0, 1] either way.

Stdlib only apart from math_verify (Hugging Face), so it can be unit-tested on a laptop
without torch or rLLM. Without math_verify (Python < 3.10) the answer check falls back to
the string matcher alone; the image always has it.
"""

from __future__ import annotations

import difflib
import re
import threading
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
    # Label alignment and answer correctness weigh 0.5 each. tag_rate is computed and
    # logged, but not rewarded.
    align: float = 0.5
    correct: float = 0.5


@dataclass
class RewardBreakdown:
    reward: float
    closed: bool
    truncated: bool
    n_steps: int
    tag_rate: float
    invented_tag_rate: float
    label_dist_sim: float
    label_seq_sim: float
    label_alignment: float
    answer_checkable: bool
    answer_correct: bool
    label_counts: dict[str, int] = field(default_factory=dict)
    # Regenerations the workflow needed before the completion had no invalid tag.
    label_retries: int = 0

    def metrics(self) -> dict[str, float]:
        """Flat float metrics for rLLM's per-episode logging (averaged per batch)."""
        out = {
            "reward": self.reward,
            "closed_think": float(self.closed),
            "truncated": float(self.truncated),
            "n_steps": float(self.n_steps),
            "tag_rate": self.tag_rate,
            "invented_tag_rate": self.invented_tag_rate,
            "invalid_label": float(self.invented_tag_rate > 0),
            "label_retries": float(self.label_retries),
            "label_dist_sim": self.label_dist_sim,
            "label_seq_sim": self.label_seq_sim,
            "label_alignment": self.label_alignment,
            "answer_checkable": float(self.answer_checkable),
        }
        # Only reported where it means something, so the batch mean is accuracy on
        # checkable questions rather than being diluted by prose answers.
        if self.answer_checkable:
            out["answer_correct"] = float(self.answer_correct)
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


def invalid_labels(text: str) -> list[str]:
    """Tags outside the eight labels that open a paragraph of the thought.

    The rule-based format check: LabeledCoTWorkflow regenerates a completion that has
    any, and score_completion gives 0 to one that still has any after the retries.
    Untagged paragraphs are allowed; only wrong tags fail.
    """
    thought, _, _ = split_completion(text)
    return _scan_labels(thought)[1]


def collapse_runs(labels: list[str]) -> list[str]:
    """Merge consecutive repeats: [a, a, b, a] -> [a, b, a]."""
    out: list[str] = []
    for label in labels:
        if not out or out[-1] != label:
            out.append(label)
    return out


def histogram_similarity(generated: list[str], reference: list[str]) -> float:
    """1 - total variation distance between the two label distributions."""
    if not generated or not reference:
        return 0.0
    gen, ref = Counter(generated), Counter(reference)
    n_gen, n_ref = len(generated), len(reference)
    tvd = 0.5 * sum(abs(gen[l] / n_gen - ref[l] / n_ref) for l in LABELS)
    return max(0.0, 1.0 - tvd)


def sequence_similarity(generated: list[str], reference: list[str]) -> float:
    """Ratcliff/Obershelp similarity of the run-length collapsed label sequences."""
    if not generated or not reference:
        return 0.0
    matcher = difflib.SequenceMatcher(None, collapse_runs(generated), collapse_runs(reference), autojunk=False)
    return matcher.ratio()


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

    About a fifth of bespoke-v2 answers are prose ("\\text{P and Q cannot both be
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
    """Score one completion against its task (needs task['ref_labels'] and task['answer'])."""
    w = weights or RewardWeights()
    thought, answer, closed = split_completion(text)

    step_labels, invented = parse_step_labels(thought)
    n_steps = len(step_labels)
    tagged = [label for label in step_labels if label is not None]
    tag_rate = len(tagged) / n_steps if n_steps else 0.0
    invented_rate = invented / n_steps if n_steps else 0.0

    reference = [label for label in (task.get("ref_labels") or []) if label in _LABEL_SET]
    dist_sim = histogram_similarity(tagged, reference)
    seq_sim = sequence_similarity(tagged, reference)
    alignment = 0.5 * (dist_sim + seq_sim)

    gold = str(task.get("answer") or "")
    checkable = answer_is_checkable(gold)
    correct = bool(closed and checkable and answer_is_correct(answer, gold))

    if truncated or not closed or invented:
        # Cut off, no closed thought, or a tag outside the eight labels (the workflow
        # regenerates those; one that still has one after the retries is not rewarded).
        reward = 0.0
    elif checkable:
        reward = w.align * alignment + w.correct * float(correct)
    else:
        # No correctness term: the alignment term alone, which keeps R in [0, 1].
        reward = alignment if w.align > 0 else 0.0

    return RewardBreakdown(
        reward=float(reward),
        closed=closed,
        truncated=truncated,
        n_steps=n_steps,
        tag_rate=tag_rate,
        invented_tag_rate=invented_rate,
        label_dist_sim=dist_sim,
        label_seq_sim=seq_sim,
        label_alignment=alignment,
        answer_checkable=checkable,
        answer_correct=correct,
        label_counts=dict(Counter(tagged)),
    )
