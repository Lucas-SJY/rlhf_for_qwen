"""Rule-based reward for "reason in labelled steps, the way the annotation does".

The SFT model writes its thought as blank-line separated paragraphs, each opening with
one of eight tags, e.g. ``[logical_deduction] 196 = 2^2 * 7^2 ...``. Every training
question also carries the human/LLM annotation of the reference trace, i.e. the
sequence of labels the reference reasoning used. The reward scores a completion on:

    tag      fraction of thought paragraphs that open with a valid tag
    align    how closely the generated label sequence follows the annotated one:
             mean of (1 - total variation distance between the two label histograms)
             and a sequence similarity of the run-length collapsed label sequences
    correct  the final \\boxed{} answer after </think> matches the reference answer

    reward = w_tag * tag + w_align * align + w_correct * correct

A completion that never closes its thought, or is cut off by the length limit, gets 0.
When the reference answer is free-form prose (a proof statement) and cannot be checked
by string matching, the correctness term is dropped and the two label terms are
renormalised, so the reward stays in [0, 1] either way.

Stdlib only, so it can be unit-tested on a laptop without torch or rLLM.
"""

from __future__ import annotations

import difflib
import re
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
_TAG_AT_START = re.compile(r"^\[([A-Za-z_]+)\]")
_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class RewardWeights:
    tag: float = 0.3
    align: float = 0.3
    correct: float = 0.4


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

    def metrics(self) -> dict[str, float]:
        """Flat float metrics for rLLM's per-episode logging (averaged per batch)."""
        out = {
            "reward": self.reward,
            "closed_think": float(self.closed),
            "truncated": float(self.truncated),
            "n_steps": float(self.n_steps),
            "tag_rate": self.tag_rate,
            "invented_tag_rate": self.invented_tag_rate,
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


def parse_step_labels(thought: str) -> tuple[list[str | None], int]:
    """Return (label per paragraph, number of invented tags).

    A paragraph maps to its label when it opens with a known tag, to None when it has
    no tag or an unknown one. Unknown tags are counted separately.
    """
    labels: list[str | None] = []
    invented = 0
    for paragraph in _PARAGRAPH_SPLIT.split(thought):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        match = _TAG_AT_START.match(paragraph)
        if match is None:
            labels.append(None)
        elif match.group(1) in _LABEL_SET:
            labels.append(match.group(1))
        else:
            labels.append(None)
            invented += 1
    return labels, invented


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
# answer checking -- same normalisation as ../train/evaluate/eval_math500.py, so the
# RL signal and the offline evaluation agree on what "correct" means
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
    correct = bool(closed and checkable and is_correct(extract_boxed(answer), gold))

    if truncated or not closed:
        reward = 0.0
    else:
        label_part = w.tag * tag_rate + w.align * alignment
        if checkable:
            reward = label_part + w.correct * float(correct)
        else:
            denom = w.tag + w.align
            reward = label_part / denom if denom > 0 else 0.0

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
