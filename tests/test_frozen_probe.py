"""End-to-end probe tests with explicitly synthetic embeddings and baseline rows."""
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import benchmark_features as f, evaluation_protocol as protocol, run_baselines as baseline
from scripts import extract_frozen_embeddings as encoder, prepare_pretrained_checkpoint as checkpoint
from scripts import run_frozen_encoder_benchmark as probe, verify_frozen_encoder_run as auditor

ROOT = Path(__file__).resolve().parents[1]


def fixture(root):
    root = Path(root)
    plan = json.loads((ROOT / "evaluation/frozen_encoder_plan_v1.json").read_text())
    base_plan = json.loads((ROOT / "evaluation/baseline_plan_v1.json").read_text())
    ids = [f"g{i:03}" for i in range(60)]
    labels = {g: "OG" if i % 2 else "TSG" for i, g in enumerate(ids)}
    groups = {g: str(i // 2) for i, g in enumerate(ids)}
    splits = {"policy": protocol.POLICY, "groups": groups, "paired_genes": ids,
              "label_vector_sha256": hashlib.sha256(json.dumps(labels, sort_keys=True).encode()).hexdigest(),
              "settings": {s: protocol.nested_splits(labels, groups, ids, s == "similarity_grouped") for s in plan["settings"]}}
    split_path = root / "splits.json"
    baseline.write_json(split_path, splits)
    for p in (plan, base_plan):
        p["split_manifest_sha256"] = f.sha(split_path)
        p["uncertainty"]["resamples"] = 20
    base_dir = root / "baselines"; base_dir.mkdir()
    baseline.write_json(base_dir / "plan.json", base_plan)
    rows = []
    for setting in plan["settings"]:
        for model in base_plan["models"]:
            for g in ids:
                probability = .5 if model["id"] == "class_prior" else (.65 if labels[g] == "OG" else .35)
                rows.append({"setting": setting, "model": model["id"], "gene_id": g, "label": labels[g],
                             "group": groups[g], "outer_fold": splits["settings"][setting]["outer"][g],
                             "selected_candidate": "synthetic", "threshold": .5, "probability_og": probability,
                             "prediction": "OG" if probability >= .5 else "TSG", "fit_id": "synthetic"})
    baseline.write_csv(base_dir / "outer_predictions.csv", rows,
                       ["setting", "model", "gene_id", "label", "group", "outer_fold", "selected_candidate", "probability_og", "threshold", "prediction", "fit_id"])
    baseline.write_json(base_dir / "run.json", {"status": "completed", "dataset_kind": "synthetic_test",
                        "artifacts": {name: f.sha(base_dir / name) for name in ["plan.json", "outer_predictions.csv"]}})
    plan["baseline_run_receipt_sha256"] = f.sha(base_dir / "run.json")
    plan_path = root / "plan.json"; baseline.write_json(plan_path, plan)
    embedding_dir = root / "embeddings"; embedding_dir.mkdir()
    x = np.random.default_rng(42).normal(size=(len(ids), 768)).astype(np.float32)
    np.savez_compressed(embedding_dir / "gene_embeddings.npz", gene_ids=np.array(ids), embeddings=x)
    token_audit = [{"window_id": g + "|cds|r0|w0", "gene_id": g, "bases": 57, "real_kmers": 52,
                    "model_tokens": 54, "unknown_tokens": 0, "pool_weight": 1.} for g in ids]
    baseline.write_json(embedding_dir / "tokenization_audit.json", token_audit)
    baseline.write_json(embedding_dir / "block_hashes.json", {})
    baseline.write_json(embedding_dir / "extraction_identity.json", {"dataset_kind": "synthetic_test"})
    record_path = ROOT / "evaluation/dnabert6_initialization_v1.json"
    receipt = {"status": "completed", "dataset_kind": "synthetic_test", "plan_sha256": f.sha(plan_path),
               "initialization_record_sha256": f.sha(record_path), "encoder": plan["encoder"],
               "parameters_unchanged": True, "parameter_sha256_before": "0" * 64, "parameter_sha256_after": "0" * 64,
               "loading": {"all_parameters_frozen": True, "strict_load": True}, "unknown_tokens": 0,
               "maximum_tokens_including_special": 54, "gene_matrix_sha256": f.array_sha(x), "windows": len(ids),
               "window_ids_sha256": f.object_sha([r["window_id"] for r in token_audit]),
               "source_sha256": {Path(m.__file__).name: f.sha(m.__file__) for m in (encoder, checkpoint, f, protocol)},
               "artifacts": {name: f.sha(embedding_dir / name) for name in ["gene_embeddings.npz", "tokenization_audit.json", "block_hashes.json", "extraction_identity.json"]}}
    baseline.write_json(embedding_dir / "extraction.json", receipt)
    return {"plan": plan, "plan_path": plan_path, "base_plan": base_plan, "baseline_dir": base_dir,
            "embedding_dir": embedding_dir, "splits": splits, "split_path": split_path,
            "initialization_path": record_path, "ids": ids, "labels": labels, "x": x}


class FrozenProbeTests(unittest.TestCase):
    def test_plan_rejects_posthoc_head_grid_or_baseline_policy_changes(self):
        plan = json.loads((ROOT / "evaluation/frozen_encoder_plan_v1.json").read_text())
        base = json.loads((ROOT / "evaluation/baseline_plan_v1.json").read_text())
        probe.verify_plan(plan, base)
        for key, value in (("encoder_finetuning", True), ("cohort", "paired")):
            with self.assertRaises(ValueError):
                probe.verify_plan(dict(plan, **{key: value}), base)
        plan = copy.deepcopy(plan)
        plan["models"][0]["C"].append(10.)
        with self.assertRaises(ValueError):
            probe.verify_plan(plan, base)

    def test_embeddings_reject_wrong_ids_changed_parameters_and_corrupt_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = fixture(tmp)
            x, _ = probe.load_embeddings(s["embedding_dir"], s["plan_path"], s["initialization_path"], s["ids"])
            np.testing.assert_array_equal(x, s["x"])
            with self.assertRaises(ValueError):
                probe.load_embeddings(s["embedding_dir"], s["plan_path"], s["initialization_path"], list(reversed(s["ids"])))
            path = s["embedding_dir"] / "extraction.json"
            receipt = json.loads(path.read_text())
            changed = dict(receipt, parameter_sha256_after="1" * 64)
            baseline.write_json(path, changed)
            with self.assertRaises(ValueError):
                probe.load_embeddings(s["embedding_dir"], s["plan_path"], s["initialization_path"], s["ids"])
            baseline.write_json(path, receipt)
            with (s["embedding_dir"] / "gene_embeddings.npz").open("ab") as stream:
                stream.write(b"corruption")
            with self.assertRaises(ValueError):
                probe.load_embeddings(s["embedding_dir"], s["plan_path"], s["initialization_path"], s["ids"])

    def test_complete_synthetic_probe_and_independent_audit_reuse_baselines(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = fixture(tmp)
            output = Path(tmp) / "output"
            original = (s["baseline_dir"] / "outer_predictions.csv").read_bytes()
            with patch.object(f, "load_cds", return_value=(s["splits"], s["labels"], {})), redirect_stdout(io.StringIO()):
                probe.run(Path(tmp) / "unused-sequences", s["split_path"], s["plan_path"], s["initialization_path"],
                          s["embedding_dir"], s["baseline_dir"], output)
            self.assertEqual(original, (s["baseline_dir"] / "outer_predictions.csv").read_bytes())
            result = auditor.verify(output, s["split_path"], s["plan_path"], s["initialization_path"], s["baseline_dir"], bootstrap=True)
            self.assertEqual(result["status"], "passed")
            self.assertEqual((result["fits"], result["outer_predictions"], result["combined_outer_predictions"]), (100, 120, 600))
            self.assertTrue(result["baseline_predictions_unchanged"])
            self.assertEqual(result["embedding_shape"], [60, 768])


if __name__ == "__main__":
    unittest.main()
