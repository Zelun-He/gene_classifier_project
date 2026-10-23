"""Unified entry point and rendering of the actual archived baseline experiment."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts import run_evaluation as runner

ROOT = Path(__file__).resolve().parents[1]


class EvaluationRunnerTests(unittest.TestCase):
    def test_table_contains_both_settings_and_every_predeclared_baseline(self):
        text = runner.render_table(ROOT / "evaluation/baselines-v1", {"status": "passed"})
        for value in ("809 genes", "55.75 [52.60, 58.98]", "56.50 [53.01, 59.94]", "54.47 [50.87, 57.68]",
                      "Class prior", "3-mer logistic", "partitions.json", "checkpoint-916"):
            self.assertIn(value, text)

    def test_render_rejects_failed_audit(self):
        with self.assertRaisesRegex(ValueError, "audited"):
            runner.render_table(ROOT / "evaluation/baselines-v1", {"status": "failed"})

    def test_existing_output_is_rejected_before_fitting(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                runner.main(["baselines", "--sequence-dir", tmp, "--output", tmp])
            self.assertEqual(error.exception.code, 2)

    def test_cli_help_works_without_importing_encoder_runtime(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts/run_evaluation.py"), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("baselines,frozen,verify", result.stdout)


if __name__ == "__main__":
    unittest.main()
