"""Recovery must retain exact bytes and reject missing or changed Git objects."""

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import recover_label_sources as recovery


class SourceRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.payloads = {"cosmic": b"original,csv\r\n", "oncokb": b"original\ttsv\n"}
        self.specs = {
            source: {"path": source + (".csv" if source == "cosmic" else ".tsv"),
                     "sha256": hashlib.sha256(payload).hexdigest()}
            for source, payload in self.payloads.items()
        }

    def test_recovery_preserves_bytes_and_records_unknown_original_dates(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "sources"
            with patch.object(recovery, "SOURCES", self.specs), patch.object(
                recovery, "read_historical_blob", side_effect=list(self.payloads.values())
            ) as read:
                manifest = recovery.recover_sources(output, Path(temp))
            self.assertEqual(read.call_count, 2)
            self.assertEqual(manifest, json.loads((output / "recovery_manifest.json").read_text()))
            self.assertEqual(manifest["source_commit"], recovery.SOURCE_COMMIT)
            for source, spec in self.specs.items():
                self.assertEqual((output / Path(spec["path"]).name).read_bytes(), self.payloads[source])
                self.assertEqual(manifest["sources"][source]["retrieved_on"], "unknown")
                self.assertEqual(manifest["sources"][source]["release"], "unreported-original-export")
                self.assertIn(recovery.SOURCE_COMMIT, manifest["sources"][source]["provenance"])

    def test_changed_bytes_do_not_publish_partial_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "sources"
            with patch.object(recovery, "SOURCES", self.specs), patch.object(
                recovery, "read_historical_blob", return_value=b"changed"
            ), self.assertRaisesRegex(ValueError, "SHA-256"):
                recovery.recover_sources(output, Path(temp))
            self.assertFalse(output.exists())

    def test_missing_second_source_does_not_publish_first_source(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "sources"
            with patch.object(recovery, "SOURCES", self.specs), patch.object(
                recovery, "read_historical_blob", side_effect=[self.payloads["cosmic"], ValueError("missing")]
            ), self.assertRaisesRegex(ValueError, "missing"):
                recovery.recover_sources(output, Path(temp))
            self.assertFalse(output.exists())

    def test_existing_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "sources"
            output.mkdir()
            sentinel = output / "keep"
            sentinel.write_bytes(b"keep")
            with self.assertRaisesRegex(ValueError, "already exists"):
                recovery.recover_sources(output, Path(temp))
            self.assertEqual(sentinel.read_bytes(), b"keep")

    def test_missing_git_history_has_actionable_error(self):
        result = subprocess.CompletedProcess(args=[], returncode=128, stdout=b"", stderr=b"missing")
        with patch.object(recovery.subprocess, "run", return_value=result), self.assertRaisesRegex(
            ValueError, "shallow clones"
        ):
            recovery.read_historical_blob(Path("/repo"), "source.csv")


if __name__ == "__main__":
    unittest.main()
