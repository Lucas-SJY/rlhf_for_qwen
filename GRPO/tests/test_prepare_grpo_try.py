"""Unit tests for the grpo_try task builder. Stdlib only:

    python3 -m unittest discover -s tests -v
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from labelcot.reward import score_completion  # noqa: E402
from prepare_grpo_try import (  # noqa: E402
    build_task,
    check_jsonl,
    check_outputs,
    check_sample,
    check_task,
    extract_boxed,
    main,
)

QUESTION = "Return your final response within \\boxed{}. What is 4 + 5?"
SAMPLE = {
    "id": "sample_000001",
    "question": QUESTION,
    "spans": [
        {"label": "restating_problem", "text": "We need 4 + 5."},
        {"label": "logical_deduction", "text": "4 + 5 = 9.\n\nThat is all."},
        {"label": "not_a_label", "text": "Dropped."},
        {"label": "concluding", "text": "  "},
        {"label": "concluding", "text": "So the answer is 9."},
    ],
}


def write_solution(root: Path, sample_id: str, question: str, solution: str) -> None:
    env = root / sample_id / "environment"
    env.mkdir(parents=True)
    (env / "trajectory.json").write_text(json.dumps({"question": question, "solution": solution}))


class BuildTaskTest(unittest.TestCase):
    def test_labels_prefix_their_span_and_answer_comes_from_solution(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_solution(Path(tmp), "sample_000001", QUESTION, "Adding gives $\\boxed{9}$.")
            task = build_task(SAMPLE, tmp, "grpo_try")
        self.assertEqual(task["ref_labels"], ["restating_problem", "logical_deduction", "concluding"])
        self.assertEqual(
            task["ref_cot"],
            "[restating_problem] We need 4 + 5.\n\n"
            "[logical_deduction] 4 + 5 = 9.\nThat is all.\n\n"
            "[concluding] So the answer is 9.",
        )
        self.assertEqual(task["answer"], "9")

    def test_reference_trace_scores_full_reward(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_solution(Path(tmp), "sample_000001", QUESTION, "Adding gives $\\boxed{9}$.")
            task = build_task(SAMPLE, tmp, "grpo_try")
        reference = f"<think>\n{task['ref_cot']}\n</think>\n\n{task['solution']}"
        self.assertAlmostEqual(score_completion(reference, task, truncated=False).reward, 1.0)

    def test_no_answer_when_solution_missing_or_for_another_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(build_task(SAMPLE, tmp, "grpo_try")["answer"], "")
            write_solution(Path(tmp), "sample_000001", "Another question?", "$\\boxed{1}$")
            task = build_task(SAMPLE, tmp, "grpo_try")
        self.assertEqual((task["answer"], task["solution"]), ("", ""))

    def test_extract_boxed_keeps_nested_braces(self):
        self.assertEqual(extract_boxed("so $\\boxed{\\dfrac{1}{13}}$."), "\\dfrac{1}{13}")
        self.assertIsNone(extract_boxed("def solve(): pass"))


def valid_task():
    with tempfile.TemporaryDirectory() as tmp:
        write_solution(Path(tmp), "sample_000001", QUESTION, "Adding gives $\\boxed{9}$.")
        return build_task(SAMPLE, tmp, "grpo_try")


class FormatCheckTest(unittest.TestCase):
    def test_input_structure_errors_and_span_warnings(self):
        errors, _ = check_sample([1, 2], "sample_000009.json")
        self.assertIn("top level is list", errors[0])

        bad = {"id": "sample_000009", "question": "q?",
               "spans": [{"label": "concluding", "start": "5"}, "oops"]}
        errors, _ = check_sample(bad, "sample_000009.json")
        self.assertTrue(any("missing key 'text'" in e for e in errors))
        self.assertTrue(any("start: expected int, got str" in e for e in errors))
        self.assertTrue(any("spans[1]: expected an object" in e for e in errors))

        errors, warnings = check_sample(SAMPLE, "sample_000002.json")
        self.assertEqual(errors, [])
        self.assertTrue(any("does not match the file name" in w for w in warnings))
        self.assertTrue(any("unknown label 'not_a_label'" in w for w in warnings))
        self.assertTrue(any("text: empty" in w for w in warnings))

    def test_built_task_passes_and_tampered_tasks_fail(self):
        task = valid_task()
        self.assertEqual(check_task(task, "t"), [])
        cases = {
            "ref_cot": dict(task, ref_cot=task["ref_cot"].replace("[concluding]", "[verifying]")),
            "answer": dict(task, answer="10"),
            "expected list": dict(task, ref_labels="concluding"),
            "unexpected keys": dict(task, extra=1),
        }
        for needle, tampered in cases.items():
            with self.subTest(needle):
                self.assertTrue(any(needle in e for e in check_task(tampered, "t")))

    def test_jsonl_reports_bad_lines_and_ids_across_splits(self):
        task = valid_task()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "train.jsonl").write_text(json.dumps(task) + "\n" + '{"id": 1,\n' + "\n")
            tasks, errors = check_jsonl(out / "train.jsonl")
            self.assertEqual(len(tasks), 1)
            self.assertTrue(any(":2: invalid JSON" in e for e in errors))
            self.assertTrue(any(":3: blank line" in e for e in errors))

            (out / "train.jsonl").write_text(json.dumps(task) + "\n")
            (out / "validation.jsonl").write_text(json.dumps(task) + "\n")
            _, errors = check_outputs(out)
            self.assertTrue(any("appears in both train and validation" in e for e in errors))

    def test_bad_input_file_is_skipped_or_stops_the_run_with_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "in").mkdir()
            for sid in ("sample_000001", "sample_000003"):
                (root / "in" / f"{sid}.json").write_text(json.dumps(dict(SAMPLE, id=sid)))
            (root / "in" / "sample_000002.json").write_text('{"id": "sample_000002",')
            argv = ["prepare_grpo_try.py", "--input-dir", str(root / "in"), "--solutions-dir", "",
                    "--split-from", "", "--val-ratio", "0.5", "--output-dir", str(root / "out")]

            stderr = io.StringIO()
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(stderr):
                main()
            self.assertIn("sample_000002.json: invalid JSON", stderr.getvalue())
            counts, errors = check_outputs(root / "out")
            self.assertEqual((counts, errors), ({"train": 1, "validation": 1}, []))

            with mock.patch.object(sys, "argv", argv + ["--strict"]), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as stop:
                    main()
            self.assertIn("--strict", str(stop.exception.code))


if __name__ == "__main__":
    unittest.main()
