"""Scientific data-integrity checks; no ML packages or network required."""

from contextlib import redirect_stderr, redirect_stdout
import csv
import hashlib
import io
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts import prepare_gene_labels as audit


FIXTURES = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).resolve().parents[1]


def observation(source="cosmic", gene="TEST", roles=("OG",), identifier="123", record=1, issues=(), ensembl=""):
    return audit.Observation(source, f"{source}:{record}", record, gene, identifier, ensembl, roles, issues, {"original": "kept"})


class LabelPolicyTests(unittest.TestCase):
    def test_exclusive_classes_and_legacy_label_mapping(self):
        for roles, category, label in [
            (("OG",), "oncogene_only", "OG"),
            (("TSG",), "tumor_suppressor_only", "TSG"),
        ]:
            with self.subTest(roles=roles):
                row = audit.build_audit([observation(roles=roles)])[0]
                self.assertEqual((row["category"], row["label"], row["included"]), (category, label, True))

    def test_explicit_dual_role_is_never_forced_into_binary_class(self):
        row = audit.build_audit([observation(roles=("OG", "TSG"))])[0]
        self.assertEqual(row["category"], "dual_role")
        self.assertFalse(row["included"])
        self.assertEqual(row["label"], "")

    def test_dual_plus_single_remains_excluded(self):
        row = audit.build_audit([
            observation(roles=("OG", "TSG")), observation(source="oncokb")
        ])[0]
        self.assertEqual(row["category"], "dual_role")
        self.assertFalse(row["included"])

    def test_opposing_annotations_have_no_source_or_row_priority(self):
        rows = [observation(), observation(source="oncokb", roles=("TSG",))]
        for ordering in itertools.permutations(rows):
            result = audit.build_audit(ordering)[0]
            self.assertEqual(result["category"], "unresolved")
            self.assertIn("opposing_exclusive_annotations", json.loads(result["exclusion_reasons"]))
            self.assertFalse(result["included"])

    def test_same_source_disagreement_is_also_excluded(self):
        row = audit.build_audit([observation(), observation(roles=("TSG",), record=2)])[0]
        self.assertEqual(row["category"], "unresolved")

    def test_dual_annotation_does_not_hide_opposing_exclusive_evidence(self):
        row = audit.build_audit([
            observation(), observation(roles=("TSG",), record=2),
            observation(source="oncokb", roles=("OG", "TSG")),
        ])[0]
        self.assertEqual(row["category"], "unresolved")

    def test_duplicate_support_does_not_create_duplicate_genes(self):
        result = audit.build_audit([observation(), observation(record=2)])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["observation_count"], 2)
        self.assertTrue(result[0]["included"])

    def test_source_without_role_is_not_an_opposing_role(self):
        row = audit.build_audit([observation(), observation(source="oncokb", roles=())])[0]
        self.assertTrue(row["included"])

    def test_incomplete_annotation_is_quarantined_even_with_other_support(self):
        row = audit.build_audit([
            observation(), observation(source="oncokb", issues=("incomplete_role_annotation",))
        ])[0]
        self.assertEqual(row["category"], "unresolved")
        self.assertFalse(row["included"])

    def test_unannotated_is_not_a_disagreement_or_negative_training_class(self):
        row = audit.build_audit([observation(roles=())])[0]
        self.assertEqual(row["category"], "unannotated")
        self.assertFalse(row["included"])

    def test_missing_id_is_preserved_but_excluded(self):
        row = audit.build_audit([observation(identifier="")])[0]
        self.assertEqual(row["identity_status"], "symbol_only")
        self.assertEqual(row["category"], "oncogene_only")
        self.assertFalse(row["included"])
        self.assertIn("missing_stable_id", json.loads(row["exclusion_reasons"]))

    def test_aliases_merge_only_with_matching_source_id(self):
        result = audit.build_audit([
            observation(gene="OLD"), observation(source="oncokb", gene="NEW")
        ])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["gene"], "NEW")
        self.assertEqual(json.loads(result[0]["symbols"]), ["NEW", "OLD"])
        self.assertEqual(result[0]["gene_id"], "ncbigene:123")

    def test_symbol_collision_quarantines_entire_transitive_group(self):
        rows = [observation(gene="A"), observation(source="oncokb", gene="A", identifier="456"),
                observation(gene="B", identifier="456", record=2)]
        for ordering in itertools.permutations(rows):
            result = audit.build_audit(ordering)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["identity_status"], "conflict")
            self.assertTrue(result[0]["gene_id"].startswith("ambiguous:"))
            self.assertFalse(result[0]["included"])
            self.assertEqual(json.loads(result[0]["entrez_ids"]), ["123", "456"])

    def test_ensembl_only_id_can_support_identity(self):
        row = audit.build_audit([observation(identifier="", ensembl="ENSG00000000001")])[0]
        self.assertTrue(row["included"])
        self.assertEqual(row["gene_id"], "ensembl:ENSG00000000001")

    def test_conflicting_ensembl_ids_are_excluded(self):
        row = audit.build_audit([
            observation(ensembl="ENSG00000000001"),
            observation(source="oncokb", ensembl="ENSG00000000002"),
        ])[0]
        self.assertFalse(row["included"])


class SourceParsingTests(unittest.TestCase):
    def read_text(self, text, source):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input"
            path.write_text(text, encoding="utf-8")
            return audit.read_source(path, source)

    def test_boolean_values_and_missingness(self):
        for raw in ("TRUE", "true", " Yes ", "1"):
            self.assertIs(audit.parse_bool(raw), True)
        for raw in ("FALSE", "false", " No ", "0"):
            self.assertIs(audit.parse_bool(raw), False)
        for raw in ("", "NA", "N/A", "null"):
            self.assertIsNone(audit.parse_bool(raw))
        with self.assertRaises(ValueError):
            audit.parse_bool("probably")

    def test_cosmic_roles_use_tokens_not_substrings(self):
        self.assertEqual(audit.cosmic_roles("TSG, oncogene;fusion"), (("OG", "TSG"), ()))
        self.assertEqual(audit.cosmic_roles("Fusion"), ((), ()))
        self.assertEqual(audit.cosmic_roles("not-an-oncogene"), ((), ("unrecognized_role_annotation",)))
        self.assertEqual(audit.cosmic_roles(""), ((), ("missing_role_annotation",)))
        self.assertEqual(audit.cosmic_roles(", ; | "), ((), ("missing_role_annotation",)))

    def test_bom_whitespace_and_original_fields_are_preserved(self):
        rows, metadata = self.read_text(
            '\ufeffGene Symbol,Entrez GeneId,Role in Cancer,Note\n test ,00123.0,"TSG,oncogene",original\n', "cosmic"
        )
        self.assertEqual(rows[0].gene, "TEST")
        self.assertEqual(rows[0].entrez_id, "123")
        self.assertEqual(rows[0].raw["Gene Symbol"], " test ")
        self.assertEqual(rows[0].raw["Note"], "original")
        self.assertEqual(metadata["input_records"], 1)

    def test_missing_flag_is_unknown_not_false(self):
        rows, _ = self.read_text(
            "Hugo Symbol\tIs Oncogene\tIs Tumor Suppressor Gene\nTEST\tTrue\t\n", "oncokb"
        )
        self.assertEqual(rows[0].roles, ("OG",))
        self.assertEqual(rows[0].issues, ("incomplete_role_annotation",))

    def test_ensembl_versions_normalize_but_raw_remains_available(self):
        rows, _ = self.read_text(
            "Gene Symbol,Ensembl Gene ID,Role in Cancer\nTEST,ENSG00000000001.3,TSG\n", "cosmic"
        )
        self.assertEqual(rows[0].ensembl_id, "ENSG00000000001")
        self.assertEqual(rows[0].raw["Ensembl Gene ID"], "ENSG00000000001.3")

    def test_malformed_inputs_fail_instead_of_silently_dropping_rows(self):
        bad_inputs = [
            ("Gene Symbol\nTEST\n", "cosmic", "missing columns"),
            ("Gene Symbol,Gene Symbol,Role in Cancer\nA,A,TSG\n", "cosmic", "duplicate column"),
            ("Gene Symbol,Role in Cancer\nA,TSG,extra\n", "cosmic", "wrong number"),
            ("Gene Symbol,Role in Cancer\nA\n", "cosmic", "wrong number"),
            ("Gene Symbol,Role in Cancer\n,TSG\n", "cosmic", "gene symbol"),
            ("Gene Symbol,Entrez GeneId,Role in Cancer\nA,12.5,TSG\n", "cosmic", "invalid Entrez"),
            ("Gene Symbol,Role in Cancer\n", "cosmic", "no data"),
            ("Hugo Symbol\tIs Oncogene\tIs Tumor Suppressor Gene\nA\tmaybe\tFalse\n", "oncokb", "oncokb record 1"),
        ]
        for text, source, message in bad_inputs:
            with self.subTest(text=text):
                with self.assertRaisesRegex(ValueError, message):
                    self.read_text(text, source)


class AuditIntegrationTests(unittest.TestCase):
    def args(self, output):
        return [
            "--cosmic", str(FIXTURES / "cosmic_synthetic.csv"),
            "--oncokb", str(FIXTURES / "oncokb_synthetic.tsv"),
            "--cosmic-release", "synthetic-v1", "--cosmic-retrieved-on", "2026-10-23",
            "--oncokb-release", "synthetic-v1", "--oncokb-retrieved-on", "2026-10-23",
            "--dataset-kind", "synthetic", "--output-dir", str(output),
        ]

    def run_audit(self, output):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(audit.main(self.args(output)), 0)

    def test_cli_outputs_counts_provenance_and_only_eligible_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "audit"
            self.run_audit(output)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["dataset_kind"], "synthetic")
            counts = summary["counts"]
            self.assertEqual((counts["input_records"], counts["identity_groups"]), (20, 11))
            self.assertEqual((counts["included_genes"], counts["excluded_groups"]), (5, 6))
            self.assertEqual(counts["included_labels"], {"OG": 3, "TSG": 2})
            self.assertEqual(sum(counts["categories"].values()), counts["identity_groups"])
            self.assertEqual(counts["included_source_presence"], {"cosmic": 1, "cosmic+oncokb": 3, "oncokb": 1})
            self.assertEqual(counts["included_label_support"], {"cosmic": 1, "cosmic+oncokb": 3, "oncokb": 1})
            with (output / "gene_labels.csv").open(newline="") as handle:
                labels = list(csv.DictReader(handle))
            self.assertEqual(len(labels), 5)
            self.assertEqual({row["gene"] for row in labels}, {
                "SYNTH_OG", "SYNTH_TSG", "SYNTH_ALIAS_NEW", "SYNTH_COSMIC_ONLY", "SYNTH_ONCOKB_ONLY"
            })
            annotations = [json.loads(line) for line in (output / "source_annotations.jsonl").read_text().splitlines()]
            self.assertEqual(len(annotations), 20)
            self.assertTrue(all(row["raw"]["Fixture note"] == "synthetic only" for row in annotations))
            with (output / "gene_manifest.csv").open(newline="") as handle:
                manifest = list(csv.DictReader(handle))
            self.assertEqual(sum(int(row["observation_count"]) for row in manifest), 20)
            record_ids = [record for row in manifest for record in json.loads(row["record_ids"])]
            self.assertEqual(len(set(record_ids)), 20)
            for name, metadata in summary["artifacts"].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), metadata["sha256"])
            for source, path in [("cosmic", FIXTURES / "cosmic_synthetic.csv"), ("oncokb", FIXTURES / "oncokb_synthetic.tsv")]:
                self.assertEqual(summary["sources"][source]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertEqual(summary["sources"][source]["release"], "synthetic-v1")

    def test_repeated_runs_are_byte_identical_and_existing_run_is_protected(self):
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / "one", Path(temp) / "two"
            self.run_audit(first)
            self.run_audit(second)
            for path in first.iterdir():
                self.assertEqual(path.read_bytes(), (second / path.name).read_bytes())
            before = (first / "summary.json").read_bytes()
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                audit.main(self.args(first))
            self.assertEqual(caught.exception.code, 2)
            self.assertEqual(before, (first / "summary.json").read_bytes())

    def test_source_presence_is_not_reported_as_label_agreement(self):
        rows = [observation(), observation(source="oncokb", roles=())]
        with tempfile.TemporaryDirectory() as temp:
            summary = audit.write_audit(Path(temp) / "audit", rows, audit.build_audit(rows), {}, "synthetic")
        self.assertEqual(summary["counts"]["included_source_presence"], {"cosmic+oncokb": 1})
        self.assertEqual(summary["counts"]["included_label_support"], {"cosmic": 1})

    def test_bad_inputs_produce_no_partial_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "audit"
            args = self.args(output)
            args[1] = str(Path(temp) / "missing.csv")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                audit.main(args)
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(output.exists())

    def test_cli_runs_from_another_working_directory_without_ml_dependencies(self):
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run(
                [sys.executable, "-S", str(ROOT / "scripts" / "prepare_gene_labels.py"), *self.args(Path(temp) / "audit")],
                cwd=temp, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("synthetic audit: 5 included genes", result.stdout)

    def test_invalid_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            for flag, value in [("--cosmic-release", " "), ("--oncokb-retrieved-on", "2026-02-30")]:
                args = self.args(Path(temp) / "audit")
                args[args.index(flag) + 1] = value
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                    audit.main(args)
                self.assertEqual(caught.exception.code, 2)

    def test_unknown_retrieval_dates_are_explicit_and_not_replaced_with_today(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "audit"
            args = self.args(output)
            for source in ("cosmic", "oncokb"):
                args[args.index(f"--{source}-retrieved-on") + 1] = "unknown"
                args[args.index(f"--{source}-release") + 1] = "unreported-original-export"
                args.extend([f"--{source}-provenance", "https://example.org/immutable-source"])
            with redirect_stdout(io.StringIO()):
                self.assertEqual(audit.main(args), 0)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(len(summary["metadata_warnings"]), 4)
            for source in ("cosmic", "oncokb"):
                self.assertEqual(summary["sources"][source]["retrieved_on"], "unknown")
                self.assertEqual(summary["sources"][source]["provenance"], "https://example.org/immutable-source")


if __name__ == "__main__":
    unittest.main()
