"""Unit tests for prepare_omni_math.py on a small local export (no Hub access). Stdlib only:

    python3 -m unittest discover -s tests -v
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from prepare_omni_math import clean_answer, top_level_values  # noqa: E402


def row(problem, answer, solution="", difficulty=5.0):
    return {"problem": problem, "answer": answer, "solution": solution, "difficulty": difficulty,
            "domain": "[]", "source": "test"}


class AnswerTest(unittest.TestCase):
    def test_wrappers_are_removed(self):
        self.assertEqual(clean_answer("\\[\n\\boxed{2, 3, 7}\n\\]"), "2, 3, 7")
        self.assertEqual(clean_answer("$\\frac{1}{2}$"), "\\frac{1}{2}")
        self.assertEqual(clean_answer(" 42 "), "42")

    def test_lists_versus_single_values(self):
        self.assertEqual(top_level_values("2, 3"), ["2", "3"])
        self.assertEqual(top_level_values("(2, 1), (3, 1)"), ["(2, 1)", "(3, 1)"])
        for single in ("(2, 4, 6)", "3,840", "1,234,567", "\\min(m, n)", "[0, 1)"):
            self.assertEqual(top_level_values(single), [single])


class BuildTest(unittest.TestCase):
    def test_build(self):
        rows = [row(f"eval {i}", str(i)) for i in range(3)]                 # evaluation slice
        rows += [row("What is 6 * 7?", "42"), row("Half of one?", "$\\frac{1}{2}$"),
                 row("Prove it.", "\\text{Yes}"),                         # not checkable
                 row("Roots?", "2, 3"),                                  # several values
                 row("Two parts?", "5", "so \\boxed{5} and \\boxed{6}")]  # solution boxes two
        rows += [row(f"Sum {i}?", str(i)) for i in range(10)]
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "omni.jsonl"
            src.write_text("".join(json.dumps(r) + "\n" for r in rows))
            out = Path(tmp) / "out"
            res = subprocess.run([sys.executable, str(SRC / "prepare_omni_math.py"), "--input", str(src),
                                  "--output-dir", str(out), "--exclude-first", "3"],
                                 capture_output=True, text=True, cwd=SRC)
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertIn("12 checkable, 3 evaluation slice, 1 answer not checkable, 2 several answers", res.stdout)
            tasks = [json.loads(l) for f in ("train", "validation") for l in (out / f"{f}.jsonl").read_text().splitlines()]
            self.assertEqual(len(tasks), 12)
            by_id = {t["id"]: t for t in tasks}
            self.assertNotIn("omni_0000", by_id)
            self.assertEqual(by_id["omni_0004"]["answer"], "\\frac{1}{2}")
            self.assertTrue(by_id["omni_0003"]["question"].startswith("Return your final response within \\boxed{}. "))
            self.assertEqual(sum(1 for _ in (out / "validation.jsonl").read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
