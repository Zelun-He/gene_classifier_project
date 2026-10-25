"""Leakage, transitivity, nested feasibility, and fitting-boundary tests."""
import hashlib
import json
from pathlib import Path
import unittest

from sklearn.preprocessing import StandardScaler
from scripts import evaluation_protocol as p
from scripts.audit_checkpoint import audit


class HomologyTests(unittest.TestCase):
    def test_components_are_transitive_and_order_independent(self):
        a = p.components(["a", "b", "c", "d"], [("a", "b"), ("c", "b")])
        b = p.components(["d", "c", "b", "a"], [("b", "c"), ("b", "a")])
        self.assertEqual(a, b)
        self.assertEqual(a["a"], a["c"])
        self.assertNotEqual(a["a"], a["d"])

    def test_exact_count_threshold_and_bidirectional_coverage(self):
        ids = {"a": "geneA", "b": "geneB"}
        good = "a\tb\t30\t100\t1\t80\t100\t21\t100\t100\t1e-5\t60"
        self.assertEqual(p.similarity_edges([good], ids), [("geneA", "geneB")])
        self.assertFalse(p.similarity_edges([good.replace("\t30\t", "\t29\t")], ids))
        self.assertFalse(p.similarity_edges([good.replace("\t21\t100", "\t22\t100")], ids))
        self.assertFalse(p.similarity_edges([good.replace("1e-5", "1.01e-5")], ids))

    def test_either_direction_and_no_rounded_identity_boundary(self):
        row = "b\ta\t2996\t10000\t1\t10000\t10000\t1\t10000\t10000\t0\t100"
        self.assertFalse(p.similarity_edges([row], {"a": "A", "b": "B"}))
        row = row.replace("2996", "3000")
        self.assertEqual(p.similarity_edges([row], {"a": "A", "b": "B"}), [("A", "B")])

    def test_unknown_and_invalid_data_fail(self):
        with self.assertRaises(ValueError):
            p.components(["a"], [("a", "b")])
        with self.assertRaises(ValueError):
            p.components(["a", "a"], [])
        with self.assertRaises(ValueError):
            p.similarity_edges(["a\tb\t0\t0\t1\t2\t2\t1\t2\t2\t0\t1"], {"a": "A", "b": "B"})


class NestedTests(unittest.TestCase):
    def setUp(self):
        self.labels = {f"g{i:03}": "OG" if i % 2 else "TSG" for i in range(60)}
        self.groups = {g: str(int(g[1:]) // 3) for g in self.labels}

    def manifest(self, grouped=True):
        name = "similarity_grouped" if grouped else "stratified_gene"
        return {"groups": self.groups, "paired_genes": sorted(self.labels)[2:], "settings": {
            name: p.nested_splits(self.labels, self.groups, sorted(self.labels)[2:], grouped)}}

    def test_nested_partitions_hold_every_gene_once_and_group_together(self):
        m = self.manifest()
        tests = []
        for outer in range(5):
            for inner in range(3):
                partitions = p.partition_ids(m, "similarity_grouped", outer, inner)
                p.assert_disjoint(**partitions, groups=self.groups)
                self.assertEqual(set().union(*map(set, partitions.values())), set(self.labels))
                for ids in partitions.values():
                    self.assertEqual({self.labels[g] for g in ids}, {"OG", "TSG"})
            tests.extend(p.partition_ids(m, "similarity_grouped", outer)["test"])
        self.assertEqual(sorted(tests), sorted(self.labels))

    def test_input_order_does_not_change_splits(self):
        a = p.nested_splits(self.labels, self.groups, self.labels, True)
        b = p.nested_splits(dict(reversed(list(self.labels.items()))), self.groups, self.labels, True)
        self.assertEqual(a, b)

    def test_paired_subset_inherits_outer_and_inner_assignments(self):
        m = self.manifest()
        full = p.partition_ids(m, "similarity_grouped", 0, 1)
        paired = p.partition_ids(m, "similarity_grouped", 0, 1, paired=True)
        for role in full:
            self.assertEqual(set(paired[role]), set(full[role]) & set(m["paired_genes"]))

    def test_fallback_depends_on_gene_and_group_counts(self):
        labels = {f"g{i}": "OG" if i % 2 else "TSG" for i in range(8)}
        groups = {g: g for g in labels}
        s = p.nested_splits(labels, groups, labels, True)
        self.assertEqual(s["outer_folds"], 4)
        self.assertTrue(s["feasibility_failures"])
        with self.assertRaises(ValueError):
            p.nested_splits(labels, dict.fromkeys(labels, "one-family"), labels, True)

    def test_derived_rows_cannot_split_or_omit_gene_identity(self):
        parts = {"train": ["a"], "validation": ["b"], "test": ["c"]}
        rows = [{"gene_id": "a", "kind": k} for k in ("transcript", "window", "augmentation")]
        self.assertEqual(p.assign_derived_rows(rows, parts)["train"], rows)
        for row in ({"gene_id": "unknown"}, {"seq": "ACGT"}):
            with self.assertRaises(ValueError):
                p.assign_derived_rows([row], parts)
        with self.assertRaises(ValueError):
            p.assert_disjoint(["a"], ["a"], [])
        with self.assertRaises(ValueError):
            p.assert_disjoint(["a"], ["b"], [], {"a": "family", "b": "family"})


class FitBoundaryTests(unittest.TestCase):
    def test_scaling_fits_exact_training_genes(self):
        scaler = p.fit_training_transformer(StandardScaler(), {"a": [0.], "b": [2.]}, ["a", "b"])
        self.assertEqual(scaler.mean_[0], 1.)
        self.assertEqual(scaler.transform([[100.]])[0, 0], 99.)
        with self.assertRaises(ValueError):
            p.fit_training_transformer(StandardScaler(), {"a": [0.], "heldout": [100.]}, ["a"])

    def test_class_weights_count_genes_from_training_only(self):
        labels = {"a": "OG", "b": "OG", "c": "TSG", "heldout": "TSG"}
        self.assertEqual(p.training_class_weights(labels, ["a", "b", "c"]), {"OG": .75, "TSG": 1.5})
        with self.assertRaises(ValueError):
            p.training_class_weights(labels, ["a", "a", "c"])

    def test_threshold_accepts_complete_inner_oof_only(self):
        rows = [{"gene_id": "a", "label": "OG", "probability_og": .8},
                {"gene_id": "b", "label": "TSG", "probability_og": .2}]
        self.assertEqual(p.select_threshold(rows, ["a", "b"]), .5)
        for expected in (["a"], ["a", "b", "outer-test"]):
            with self.assertRaises(ValueError):
                p.select_threshold(rows, expected)
        with self.assertRaises(ValueError):
            p.select_threshold(rows + [rows[0]], ["a", "b"])
        rows[0]["probability_og"] = float("nan")
        with self.assertRaises(ValueError):
            p.select_threshold(rows, ["a", "b"])

    def test_initialization_blocks_legacy_and_unpinned_models(self):
        good = {"kind": "pristine_pretrained", "task_finetuned": False, "revision": "a" * 40,
                "model_id": "upstream/model", "provenance_url": "https://example.org/model-card",
                "artifact_sha256": {k: "b" * 64 for k in ("config", "tokenizer", "weights")}}
        p.validate_initialization(good)
        for change in ({"model_id": "models/output/checkpoint-916"}, {"task_finetuned": True},
                       {"revision": "main"}, {"artifact_sha256": {}}):
            with self.assertRaises(ValueError):
                p.validate_initialization(dict(good, **change))


class FrozenArtifactTests(unittest.TestCase):
    def test_frozen_splits_and_edges_are_consistent(self):
        root = Path(__file__).resolve().parents[1] / "evaluation/v1"
        manifest = json.loads((root / "splits.json").read_text())
        report = json.loads((root / "report.json").read_text())
        for name, expected in report["artifacts"].items():
            self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), expected)
        edges = json.loads((root / "homology_edges.json").read_text())
        self.assertEqual(manifest["groups"], p.components(sorted(manifest["groups"]), edges))
        for name, setting in manifest["settings"].items():
            for outer in range(setting["outer_folds"]):
                for inner in range(setting["inner_folds"][str(outer)]):
                    p.partition_ids(manifest, name, outer, inner)
        self.assertTrue(edges)  # A degenerate zero-edge homology run must not pass unnoticed.

    def test_checkpoint_audit_replays_from_pinned_history(self):
        root = Path(__file__).resolve().parents[1]
        record = json.loads((root / "evaluation/v1/checkpoint_audit.json").read_text())
        self.assertEqual(audit(root, record["audited_revision"]), record)
        self.assertFalse(record["benchmark_allowed"])
        self.assertTrue(record["historical_dataset_blobs"])
        self.assertTrue(all(x["bytes"] == 0 for x in record["historical_dataset_blobs"]))


if __name__ == "__main__":
    unittest.main()
