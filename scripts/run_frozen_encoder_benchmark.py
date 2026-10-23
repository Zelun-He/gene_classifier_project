"""Fit the predeclared linear probe to frozen gene vectors and compare baselines."""
import argparse
import json
from pathlib import Path
import platform
import shutil

import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_limits

if __package__:
    from . import benchmark_features as f, benchmark_metrics as metrics, evaluation_protocol as protocol
    from . import run_baselines as baseline, verify_baseline_run as baseline_audit
else:
    import benchmark_features as f
    import benchmark_metrics as metrics
    import evaluation_protocol as protocol
    import run_baselines as baseline
    import verify_baseline_run as baseline_audit


def verify_plan(plan, baseline_plan):
    if (plan["run_id"] != "dnabert6-frozen-cds-v1" or plan["protocol"] != protocol.POLICY
            or plan["cohort"] != "full_cds" or plan["pretrained_checkpoint_required"] is not True
            or plan["encoder_finetuning"] is not False or plan["encoder"]["frozen"] is not True):
        raise ValueError("Only the preregistered frozen-encoder comparison is supported")
    for key in ("schema_version", "protocol", "split_manifest_sha256", "sequence_summary_sha256", "cohort", "settings",
                "seed", "logistic", "selection", "uncertainty"):
        if plan[key] != baseline_plan[key]:
            raise ValueError(f"Comparison changed baseline evaluation policy: {key}")
    if plan["models"] != [{"id": "dnabert6_frozen", "features": "frozen_gene_embeddings", "estimator": "logistic", "C": [.01, .1, 1.]}]:
        raise ValueError("Encoder classifier candidates differ from preregistration")
    versions = {"numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}
    if any(plan["software"][key] != value for key, value in versions.items()):
        raise ValueError("Classical numerical runtime differs from preregistration")


def load_embeddings(embedding_dir, plan_path, initialization_path, expected_ids):
    embedding_dir, plan_path = Path(embedding_dir), Path(plan_path)
    plan = json.loads(plan_path.read_text())
    receipt = json.loads((embedding_dir / "extraction.json").read_text())
    if (receipt["status"] != "completed" or receipt["plan_sha256"] != f.sha(plan_path)
            or receipt["initialization_record_sha256"] != f.sha(initialization_path)
            or receipt["initialization_record_sha256"] != plan["initialization_record_sha256"]
            or receipt["encoder"] != plan["encoder"] or receipt["parameters_unchanged"] is not True
            or receipt["parameter_sha256_before"] != receipt["parameter_sha256_after"]
            or receipt["loading"]["all_parameters_frozen"] is not True or receipt["loading"]["strict_load"] is not True
            or receipt["unknown_tokens"] != 0 or receipt["maximum_tokens_including_special"] > 512):
        raise ValueError("Embeddings lack a complete, unchanged frozen-encoder lineage")
    protocol.validate_initialization(json.loads(Path(initialization_path).read_text()))
    for name, expected in receipt["artifacts"].items():
        if Path(name).name != name or f.sha(embedding_dir / name) != expected:
            raise ValueError(f"Embedding artifact changed: {name}")
    with np.load(embedding_dir / "gene_embeddings.npz", allow_pickle=False) as data:
        if set(data.files) != {"gene_ids", "embeddings"}:
            raise ValueError("Unexpected embedding archive schema")
        ids, x = data["gene_ids"].tolist(), data["embeddings"].copy()
    if (ids != list(expected_ids) or len(ids) != len(set(ids)) or x.shape != (len(ids), 768)
            or x.dtype != np.float32 or not np.isfinite(x).all() or receipt["gene_matrix_sha256"] != f.array_sha(x)):
        raise ValueError("Embedding matrix does not match gene cohort/dimensions/hash")
    return x, receipt


def run(sequence_dir, split_path, plan_path, initialization_path, embedding_dir, baseline_dir, output):
    plan_path, baseline_dir, output = Path(plan_path), Path(baseline_dir), Path(output)
    plan = json.loads(plan_path.read_text())
    base_plan = json.loads((baseline_dir / "plan.json").read_text())
    verify_plan(plan, base_plan)
    if f.sha(baseline_dir / "run.json") != plan["baseline_run_receipt_sha256"]:
        raise ValueError("Frozen baseline run receipt changed")
    base_receipt = json.loads((baseline_dir / "run.json").read_text())
    if base_receipt["status"] != "completed":
        raise ValueError("Baseline run incomplete")
    for name, expected in base_receipt["artifacts"].items():
        if f.sha(baseline_dir / name) != expected:
            raise ValueError(f"Frozen baseline artifact changed: {name}")
    splits, labels, _ = f.load_cds(sequence_dir, split_path, plan, Path(split_path).with_name("report.json"))
    ids = sorted(labels)
    x, extraction = load_embeddings(embedding_dir, plan_path, initialization_path, ids)
    old_rows = baseline_audit.load_rows(baseline_dir / "outer_predictions.csv")
    if sorted({r["model"] for r in old_rows}) != sorted(plan["comparisons"]["baselines"]):
        raise ValueError("Unexpected baseline models")
    output.mkdir(parents=True, exist_ok=False)
    baseline.write_json(output / "plan.json", plan)
    (output / "embeddings").mkdir()
    for name in ["extraction.json", *extraction["artifacts"]]:
        shutil.copyfile(Path(embedding_dir) / name, output / "embeddings" / name)
    receipt = {"status": "running", "plan_sha256": f.sha(plan_path), "split_manifest_sha256": f.sha(split_path),
               "sequence_summary_sha256": plan["sequence_summary_sha256"], "initialization_record_sha256": f.sha(initialization_path),
               "baseline_run_receipt_sha256": f.sha(baseline_dir / "run.json"),
               "extraction_receipt_sha256": f.sha(Path(embedding_dir) / "extraction.json"),
               "code_sha256": {p.name: f.sha(p) for p in [Path(__file__), Path(baseline.__file__), Path(f.__file__),
                                                          Path(metrics.__file__), Path(protocol.__file__)]},
               "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                            "scikit_learn": sklearn.__version__}, "genes": len(ids), "encoder_finetuned": False}
    baseline.write_json(output / "run.json", receipt)
    partitions, fits, candidates, inner_rows, outer_rows = {}, [], [], [], []
    model = plan["models"][0]
    try:
        with threadpool_limits(limits=plan["software"]["thread_limit"]):
            for setting in plan["settings"]:
                outer_rows.extend(baseline.evaluate_model(ids, x, labels, splits, model, setting, plan,
                                                          partitions, fits, candidates, inner_rows))
            combined = [*old_rows, *outer_rows]
            summary = {setting: metrics.summarize_setting([r for r in combined if r["setting"] == setting], setting,
                                                          plan["uncertainty"]["resamples"], plan["uncertainty"]["seed"])
                       for setting in plan["settings"]}
        baseline.write_json(output / "features.json", {model["id"]: {"shape": list(x.shape), "gene_order_sha256": f.object_sha(ids),
                              "matrix_sha256": f.array_sha(x), "source": "embeddings/gene_embeddings.npz",
                              "extraction_receipt_sha256": receipt["extraction_receipt_sha256"]}})
        for name, value in (("partitions", partitions), ("fits", fits), ("candidate_scores", candidates), ("summary", summary)):
            baseline.write_json(output / f"{name}.json", value)
        columns = ["setting", "model", "gene_id", "label", "group", "outer_fold", "selected_candidate", "probability_og", "threshold", "prediction", "fit_id"]
        baseline.write_csv(output / "outer_predictions.csv", outer_rows, columns)
        baseline.write_csv(output / "combined_outer_predictions.csv", combined, columns)
        baseline.write_csv(output / "selected_inner_oof.csv", inner_rows,
                            ["setting", "model", "outer_fold", "inner_fold", "gene_id", "label", "probability_og", "fit_id"])
        receipt.update({"status": "completed", "fits": len(fits), "outer_predictions": len(outer_rows),
                        "selected_inner_oof_predictions": len(inner_rows), "combined_outer_predictions": len(combined),
                        "artifacts": {str(p.relative_to(output)): f.sha(p) for p in sorted(output.rglob("*"))
                                      if p.is_file() and p != output / "run.json"}})
        baseline.write_json(output / "run.json", receipt)
    except Exception as error:
        receipt.update({"status": "failed", "error": str(error)})
        baseline.write_json(output / "run.json", receipt)
        raise
    print(json.dumps({"completed": True, "fits": len(fits), "outer_predictions": len(outer_rows)}), flush=True)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence-dir", required=True, type=Path)
    p.add_argument("--embedding-dir", required=True, type=Path)
    p.add_argument("--splits", default=Path("evaluation/v1/splits.json"), type=Path)
    p.add_argument("--plan", default=Path("evaluation/frozen_encoder_plan_v1.json"), type=Path)
    p.add_argument("--initialization-record", default=Path("evaluation/dnabert6_initialization_v1.json"), type=Path)
    p.add_argument("--baseline-dir", default=Path("evaluation/baselines-v1"), type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    run(a.sequence_dir, a.splits, a.plan, a.initialization_record, a.embedding_dir, a.baseline_dir, a.output)
