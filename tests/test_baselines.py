"""Independent metric checks and real fold-local fitting on synthetic genes."""
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn import metrics as sk
from threadpoolctl import threadpool_limits

from scripts import benchmark_features as f, benchmark_metrics as m, evaluation_protocol as p, run_baselines as runner

ROOT = Path(__file__).resolve().parents[1]


class FeatureTests(unittest.TestCase):
    def test_kmer_counts_include_full_sequence_and_overlapping_positions(self):
        v = f.kmer_frequencies("AAAAAC", 3)
        # AAA at offsets 0,1,2 and AAC at offset 3; fixed ACGT base-4 order.
        self.assertEqual(v[0], .75)
        self.assertEqual(v[1], .25)
        self.assertAlmostEqual(v.sum(), 1)
        v = f.kmer_frequencies("AAAAAAC", 6)
        self.assertEqual((v[0], v[1]), (.5, .5))

    def test_invalid_sequence_does_not_silently_drop_kmers(self):
        for sequence, k in (("AANNN", 3), ("AC", 3), ("ACGT", 2)):
            with self.assertRaises(ValueError):
                f.kmer_frequencies(sequence, k)

    def test_length_gc_and_gene_order(self):
        x, names = f.make_features(["b", "a"], {"a": "ACGT", "b": "CCCCCCCC"}, {"features": "log_length_gc"})
        np.testing.assert_allclose(x, [[np.log(8), 1], [np.log(4), .5]])
        self.assertEqual(len(names), 2)

    def test_multiline_fasta_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sequences.fasta"
            path.write_text(">a\nAC\nGT\n>b\nAAA\n")
            self.assertEqual(f.read_fasta(path), {"a": "ACGT", "b": "AAA"})
            path.write_text(">a\nAAA\n>a\nCCC\n")
            with self.assertRaises(ValueError):
                f.read_fasta(path)


class MetricTests(unittest.TestCase):
    def test_weighted_metrics_match_explicit_duplicate_samples_with_ties(self):
        rng = np.random.default_rng(2026)
        y = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        probability = np.array([.2, .2, .2, .9, .6, .6, .9, .9])
        predicted = probability >= .5
        for _ in range(15):
            w = rng.integers(0, 5, len(y))
            if not all(w[y == label].sum() for label in (0, 1)):
                continue
            index = np.repeat(np.arange(len(y)), w)
            yy, pp, dd = y[index], probability[index], predicted[index]
            result = {k: v[0] for k, v in m.metric_arrays(y, probability, predicted, w).items()}
            expected = {"balanced_accuracy": sk.balanced_accuracy_score(yy, dd),
                        "accuracy": sk.accuracy_score(yy, dd), "auroc": sk.roc_auc_score(yy, pp),
                        "average_precision_og": sk.average_precision_score(yy, pp),
                        "average_precision_tsg": sk.average_precision_score(1 - yy, 1 - pp),
                        "mcc": sk.matthews_corrcoef(yy, dd), "macro_f1": sk.f1_score(yy, dd, average="macro"),
                        "brier_score": sk.brier_score_loss(yy, pp),
                        "precision_og": sk.precision_score(yy, dd, zero_division=0),
                        "precision_tsg": sk.precision_score(1 - yy, 1 - dd, zero_division=0),
                        "recall_og": sk.recall_score(yy, dd), "recall_tsg": sk.recall_score(1 - yy, 1 - dd)}
            for key, value in expected.items():
                self.assertAlmostEqual(result[key], value, places=12, msg=key)

    def test_constant_scores_and_predictions_have_defined_metrics(self):
        result = m.point_metrics([0, 0, 1], [.7, .7, .7], [1, 1, 1])
        self.assertEqual(result["balanced_accuracy"], .5)
        self.assertEqual(result["auroc"], .5)
        self.assertAlmostEqual(result["average_precision_og"], 1 / 3)
        self.assertEqual(result["mcc"], 0)
        self.assertEqual(result["precision_tsg"], 0)

    def test_bootstrap_repeats_whole_mixed_label_groups_and_is_deterministic(self):
        y = np.array([0, 1] * 6)
        groups = np.repeat(np.arange(6), 2)
        folds = np.repeat([0, 0, 1, 1, 2, 2], 2)
        w, info = m.bootstrap_weights(y, folds, groups, "similarity_grouped", 30)
        again, _ = m.bootstrap_weights(y, folds, groups, "similarity_grouped", 30)
        np.testing.assert_array_equal(w, again)
        for group in set(groups):
            np.testing.assert_array_equal(w[:, groups == group][:, 0], w[:, groups == group][:, 1])
        self.assertEqual(info["units"], 6)
        self.assertTrue(np.all(w.sum(axis=1) == len(y)))
        with self.assertRaises(ValueError):
            m.bootstrap_weights(y, np.arange(len(y)) % 3, groups, "similarity_grouped", 3)

    def test_identical_models_have_zero_paired_difference_interval(self):
        rows = []
        for name in ("a", "b"):
            for i in range(60):
                label = "OG" if i % 2 else "TSG"
                rows.append({"setting": "stratified_gene", "model": name, "gene_id": f"g{i:03}", "label": label,
                             "outer_fold": i // 20, "group": str(i), "probability_og": .6 if i % 2 else .4,
                             "prediction": label})
        with threadpool_limits(limits=1):
            summary = m.summarize_setting(rows, "stratified_gene", 30)
        for interval in summary["paired_comparisons"][0]["metrics"].values():
            self.assertEqual(interval, {"estimate": 0., "lower": 0., "upper": 0.})
        self.assertEqual(summary["models"]["a"]["mean_outer_balanced_accuracy"], 1)
        rows.pop()
        with self.assertRaises(ValueError):
            m.summarize_setting(rows, "stratified_gene", 5)

    def test_invalid_probabilities_and_missing_weighted_class_fail(self):
        with self.assertRaises(ValueError):
            m.metric_arrays([0, 1], [float("nan"), .5], [0, 1])
        with self.assertRaises(ValueError):
            m.metric_arrays([0, 1], [.2, .8], [0, 1], [0, 1])


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.plan = json.loads((ROOT / "evaluation/baseline_plan_v1.json").read_text())
        self.model = self.plan["models"][1]
        self.ids = [f"g{i:03}" for i in range(60)]
        self.labels = {g: "OG" if i % 2 else "TSG" for i, g in enumerate(self.ids)}
        self.x = np.random.default_rng(42).normal(size=(60, 2))

    def test_training_fit_and_class_weights_ignore_heldout_features_and_labels(self):
        train = self.ids[:40]
        with threadpool_limits(limits=1):
            model, record = runner.fit_model(self.x, self.ids, self.labels, train, self.model, .1, self.plan)
            changed = self.x.copy()
            changed[40:] = 100000
            labels = {g: ("OG" if i >= 40 else self.labels[g]) for i, g in enumerate(self.ids)}
            second, again = runner.fit_model(changed, self.ids, labels, train, self.model, .1, self.plan)
        self.assertEqual(record, again)
        self.assertEqual(record["fit_gene_ids_sha256"], f.object_sha(train))
        np.testing.assert_array_equal(model["scale"].mean_, self.x[:40].mean(axis=0))
        np.testing.assert_array_equal(model["classifier"].coef_, second["classifier"].coef_)

    def test_outer_labels_cannot_change_that_folds_selection_threshold_or_fit(self):
        groups = {g: str(i // 2) for i, g in enumerate(self.ids)}
        splits = {"groups": groups, "paired_genes": self.ids, "settings": {
            "similarity_grouped": p.nested_splits(self.labels, groups, self.ids, True)}}
        parts = p.partition_ids(splits, "similarity_grouped", 0)
        flipped = dict(self.labels)
        for g in parts["test"]:
            flipped[g] = "TSG" if flipped[g] == "OG" else "OG"
        def evaluate(labels):
            fits, candidates, inner = [], [], []
            with redirect_stdout(io.StringIO()), threadpool_limits(limits=1):
                rows = runner.evaluate_model(self.ids, self.x, labels, splits, self.model,
                                             "similarity_grouped", self.plan, {}, fits, candidates, inner)
            return rows, fits, candidates, inner
        a, b = evaluate(self.labels), evaluate(flipped)
        self.assertEqual([r for r in a[2] if r["outer_fold"] == 0], [r for r in b[2] if r["outer_fold"] == 0])
        self.assertEqual([r for r in a[1] if "/outer0/" in r["fit_id"]], [r for r in b[1] if "/outer0/" in r["fit_id"]])
        self.assertEqual([r for r in a[3] if r["outer_fold"] == 0], [r for r in b[3] if r["outer_fold"] == 0])
        for row in a[0]:
            matching = next(r for r in b[0] if r["gene_id"] == row["gene_id"])
            if row["outer_fold"] == 0:
                self.assertEqual(row["probability_og"], matching["probability_og"])
                self.assertNotEqual(row["label"], matching["label"])

    def test_plan_rejects_neural_initialization_and_changed_dependency_versions(self):
        runner.verify_plan(self.plan)
        for key, value in (("neural_training_authorized_by_this_plan", True), ("pretrained_checkpoint_required", True)):
            with self.assertRaises(ValueError):
                runner.verify_plan(dict(self.plan, **{key: value}))
        self.plan["software"]["scikit_learn"] = "latest"
        with self.assertRaises(ValueError):
            runner.verify_plan(self.plan)


if __name__ == "__main__":
    unittest.main()
