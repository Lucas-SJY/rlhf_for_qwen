"""Unit tests for the rest_grpo task builder. Stdlib only:

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

from prepare_grpo_try import check_outputs  # noqa: E402
from prepare_rest_grpo import boxed_answers, check_source, main  # noqa: E402

MATH_Q = "Return your final response within \\boxed{}. What is 4 + 5?"
CODE_Q = "Generate an executable Python function generated from the given prompt. Print 1."


def write_source(root: Path, sample_id: str, question: str, solution: str) -> None:
    env = root / sample_id / "environment"
    env.mkdir(parents=True)
    record = {"id": sample_id, "model": "DeepSeek-R1", "question": question,
              "thought_trace": "...", "segments": [], "solution": solution}
    (env / "trajectory.json").write_text(json.dumps(record))


def run_main(argv: list[str]) -> str:
    stdout = io.StringIO()
    with mock.patch.object(sys, "argv", ["prepare_rest_grpo.py", *argv]), contextlib.redirect_stdout(stdout), \
            contextlib.redirect_stderr(io.StringIO()):
        main()
    return stdout.getvalue()


def read_tasks(out: Path) -> dict[str, list[dict]]:
    return {name: [json.loads(line) for line in (out / f"{name}.jsonl").read_text().splitlines()]
            for name in ("train", "validation")}


class BuildTest(unittest.TestCase):
    def test_tasks_keep_only_prompt_question_and_answer_and_skip_sft_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_source(root / "src", "sample_000001", MATH_Q, "So the sum is $\\boxed{9}$.")
            write_source(root / "src", "sample_000002", MATH_Q, "$\\boxed{4}$")
            write_source(root / "src", "sample_000003", MATH_Q, "$\\boxed{9}$")
            (root / "sft").mkdir()
            (root / "sft" / "train.jsonl").write_text(json.dumps({"id": "sample_000003", "messages": []}) + "\n")
            out = root / "out"
            run_main(["--source-dir", str(root / "src"), "--exclude-from", str(root / "sft"),
                      "--output-dir", str(out), "--no-answer-dir", str(root / "na"), "--val-ratio", "0.5"])

            tasks = {t["id"]: t for split in read_tasks(out).values() for t in split}
            self.assertEqual(sorted(tasks), ["sample_000001", "sample_000002"])
            self.assertEqual(tasks["sample_000001"], {
                "id": "sample_000001",
                "data_source": "rest_grpo",
                "prompt": [{"role": "user", "content": MATH_Q}],
                "question": MATH_Q,
                "answer": "9",
            })
            self.assertEqual(check_outputs(out), ({"train": 1, "validation": 1}, []))

    def test_questions_without_a_checkable_answer_go_to_the_no_answer_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(10):
                write_source(root / "src", f"sample_{i:06d}", f"{MATH_Q} ({i})", f"$\\boxed{{{i}}}$")
            # No \boxed{} in a coding solution: no answer. A prose answer cannot be checked.
            write_source(root / "src", "sample_000010", CODE_Q, "def f():\n    print(1)")
            write_source(root / "src", "sample_000011", MATH_Q, "$\\boxed{\\text{No such number exists}}$")
            # Two roots boxed: the last \boxed{} alone would be an incomplete answer.
            write_source(root / "src", "sample_000012", MATH_Q, "$\\boxed{2}$ and $\\boxed{-2}$")
            # The same value boxed twice is still one answer.
            write_source(root / "src", "sample_000013", MATH_Q, "So $\\boxed{5}$. Hence $\\boxed{5}$.")
            argv =["--source-dir", str(root / "src"), "--exclude-from", "", "--output-dir", str(root / "out")]

            run_main(argv + ["--no-answer-dir", ""])
            together = {name: {t["id"] for t in split} for name, split in read_tasks(root / "out").items()}

            run_main(argv + ["--no-answer-dir", str(root / "na")])
            kept, moved = read_tasks(root / "out"), read_tasks(root / "na")
            moved_ids = {t["id"] for split in moved.values() for t in split}
            self.assertEqual(moved_ids, {"sample_000010", "sample_000011", "sample_000012"})
            self.assertTrue(all(t["answer"] and t["id"] not in moved_ids for split in kept.values() for t in split))
            # Separating them does not change which split a question is in.
            for name in ("train", "validation"):
                self.assertEqual({t["id"] for t in kept[name] + moved[name]}, together[name])

    def test_split_is_seeded_and_ten_percent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(20):
                write_source(root / "src", f"sample_{i:06d}", f"{MATH_Q} ({i})", f"$\\boxed{{{i}}}$")
            splits = []
            for out in ("a", "b"):
                run_main(["--source-dir", str(root / "src"), "--exclude-from", "", "--output-dir", str(root / out),
                          "--no-answer-dir", str(root / f"na_{out}")])
                splits.append(read_tasks(root / out))
            self.assertEqual((len(splits[0]["train"]), len(splits[0]["validation"])), (18, 2))
            self.assertEqual(splits[0], splits[1])


class BoxedAnswersTest(unittest.TestCase):
    def test_every_box_is_read_with_nested_braces(self):
        self.assertEqual(boxed_answers("$\\boxed{\\frac{1}{2}}$ or $\\boxed{3}$, not \\boxed 7; $\\boxed{x^{2}}$"),
                         ["\\frac{1}{2}", "3", "x^{2}"])
        self.assertEqual(boxed_answers("def f(): pass"), [])


class FormatCheckTest(unittest.TestCase):
    def test_source_errors_and_warnings(self):
        errors, _ = check_source([1], "x", "sample_000001")
        self.assertIn("top level is list", errors[0])
        errors, _ = check_source({"id": "sample_000001", "question": "Q?"}, "x", "sample_000001")
        self.assertTrue(any("missing key 'solution'" in e for e in errors))
        errors, _ = check_source({"id": "sample_000001", "question": " ", "solution": ""}, "x", "sample_000001")
        self.assertTrue(any("question: empty" in e for e in errors))
        errors, warnings = check_source({"id": "sample_000009", "question": "Q?", "solution": ""}, "x", "sample_000001")
        self.assertEqual(errors, [])
        self.assertTrue(any("does not match the directory name" in w for w in warnings))

    def test_bad_source_file_is_skipped_or_stops_the_run_with_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in (1, 2):
                write_source(root / "src", f"sample_00000{i}", MATH_Q, "$\\boxed{9}$")
            bad = root / "src" / "sample_000003" / "environment"
            bad.mkdir(parents=True)
            (bad / "trajectory.json").write_text('{"id": "sample_000003",')
            argv = ["--source-dir", str(root / "src"), "--exclude-from", "", "--output-dir", str(root / "out"),
                    "--no-answer-dir", str(root / "na")]
            run_main(argv)
            counts, errors = check_outputs(root / "out")
            self.assertEqual((sum(counts.values()), errors), (2, []))
            with self.assertRaises(SystemExit) as stop:
                run_main(argv + ["--strict"])
            self.assertIn("--strict", str(stop.exception.code))


if __name__ == "__main__":
    unittest.main()
