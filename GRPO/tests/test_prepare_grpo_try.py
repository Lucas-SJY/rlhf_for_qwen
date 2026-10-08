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
    "spans": [{"label": "logical_deduction", "text": "4 + 5 = 9."}],
}


def write_solution(root: Path, sample_id: str, question: str, solution: str) -> None:
    env = root / sample_id / "environment"
    env.mkdir(parents=True)
    (env / "trajectory.json").write_text(json.dumps({"question": question, "solution": solution}))


def valid_task():
    with tempfile.TemporaryDirectory() as tmp:
        write_solution(Path(tmp), "sample_000001", QUESTION, "Adding gives $\\boxed{9}$.")
        return build_task(SAMPLE, tmp, "grpo_try")


class BuildTaskTest(unittest.TestCase):
    def test_only_prompt_question_and_answer_are_kept(self):
        task = valid_task()
        self.assertEqual(task, {
            "id": "sample_000001",
            "data_source": "grpo_try",
            "prompt": [{"role": "user", "content": QUESTION}],
            "question": QUESTION,
            "answer": "9",
        })

    def test_no_answer_when_solution_missing_or_for_another_question(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(build_task(SAMPLE, tmp, "grpo_try")["answer"], "")
            write_solution(Path(tmp), "sample_000001", "Another question?", "$\\boxed{1}$")
            self.assertEqual(build_task(SAMPLE, tmp, "grpo_try")["answer"], "")

    def test_reward_needs_nothing_but_the_task(self):
        text = "<think>\n[logical_deduction] 4 + 5 = 9.\n\n[concluding] So 9.\n</think>\n\n$\\boxed{9}$"
        b = score_completion(text, valid_task(), truncated=False)
        self.assertTrue(b.answer_correct and b.label_valid)
        self.assertAlmostEqual(b.reward, 1.0)

    def test_extract_boxed_keeps_nested_braces(self):
        self.assertEqual(extract_boxed("so $\\boxed{\\dfrac{1}{13}}$."), "\\dfrac{1}{13}")
        self.assertIsNone(extract_boxed("def solve(): pass"))


class FormatCheckTest(unittest.TestCase):
    def test_input_errors_and_warnings(self):
        errors, _ = check_sample([1, 2], "sample_000009.json")
        self.assertIn("top level is list", errors[0])

        errors, _ = check_sample({"id": "sample_000009", "question": 5}, "sample_000009.json")
        self.assertTrue(any("question: expected str, got int" in e for e in errors))
        errors, _ = check_sample({"id": "sample_000009"}, "sample_000009.json")
        self.assertTrue(any("missing key 'question'" in e for e in errors))

        errors, warnings = check_sample(SAMPLE, "sample_000002.json")
        self.assertEqual(errors, [])
        self.assertTrue(any("does not match the file name" in w for w in warnings))

    def test_built_task_passes_and_tampered_tasks_fail(self):
        task = valid_task()
        self.assertEqual(check_task(task, "t"), [])
        cases = {
            "unexpected keys": dict(task, ref_labels=["concluding"]),
            "missing key 'prompt'": {k: v for k, v in task.items() if k != "prompt"},
            "expected list": dict(task, prompt="What is 4 + 5?"),
            "the last turn must be": dict(task, prompt=[{"role": "user", "content": "Another question?"}]),
            "unknown role": dict(task, prompt=[{"role": "robot", "content": QUESTION}]),
            "expected str": dict(task, answer=9),
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
                    "--val-ids", "sample_000003", "--output-dir", str(root / "out")]

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
