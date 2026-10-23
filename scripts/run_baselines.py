"""Run predeclared classical baselines on frozen nested gene folds.

No pretrained weights, vocabulary learning, or window-level labelled samples
are used. Every learned transform and class weight is local to a training fit.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import platform
import warnings

import numpy as np
import scipy
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

if __package__:
    from . import benchmark_features as features, benchmark_metrics as metrics, evaluation_protocol as protocol
else:
    import benchmark_features as features
    import benchmark_metrics as metrics
    import evaluation_protocol as protocol


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path, rows, columns):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def verify_plan(plan):
    """Support an intentionally narrow baseline scope; never unlock neural fits."""
    if (plan.get("schema_version") != 1 or plan.get("cohort") != "full_cds"
            or plan.get("protocol") != protocol.POLICY or plan.get("pretrained_checkpoint_required") is not False
            or plan.get("neural_training_authorized_by_this_plan") is not False
            or plan.get("selection", {}).get("metric") != "mean_inner_balanced_accuracy_at_0.5"):
        raise ValueError("Unsupported baseline plan")
    if set(plan["settings"]) != {"stratified_gene", "similarity_grouped"}:
        raise ValueError("Both evaluation settings are required")
    if len({m["id"] for m in plan["models"]}) != len(plan["models"]):
        raise ValueError("Duplicate model ID")
    for model in plan["models"]:
        if model["estimator"] == "prior":
            if model["features"] != "constant" or model["C"] != [None]:
                raise ValueError("Invalid class-prior specification")
        elif model["estimator"] == "logistic":
            if not model["C"] or any(not isinstance(c, (int, float)) or not np.isfinite(c) or c <= 0 for c in model["C"]):
                raise ValueError("Invalid C candidates")
            if model["C"] != sorted(set(model["C"])):
                raise ValueError("Candidate simplicity order must be ascending C")
            if model["features"] not in {"log_length_gc", "kmer"}:
                raise ValueError("Unsupported logistic features")
            if model["features"] == "kmer" and model.get("k") not in {3, 6}:
                raise ValueError("Unsupported k-mer size")
        else:
            raise ValueError("Only fresh prior/logistic baselines are supported")
    if plan["logistic"]["solver"] != "liblinear" or plan["logistic"]["l1_ratio"] != 0:
        raise ValueError("Only the predeclared L2 liblinear configuration is supported")
    versions = {"scikit_learn": sklearn.__version__, "numpy": np.__version__, "scipy": scipy.__version__}
    if any(plan["software"][key] != value for key, value in versions.items()):
        raise ValueError(f"Install requirements-evaluation.txt; actual versions: {versions}")


def fit_model(x, ids, labels, train_ids, model, c, plan):
    """Only training IDs reach fit(); callers predict held-out data separately."""
    if len(ids) != len(set(ids)) or x.shape[0] != len(ids) or not np.isfinite(x).all():
        raise ValueError("Invalid feature matrix / IDs")
    index = {gene: i for i, gene in enumerate(ids)}
    train_ids = sorted(train_ids)
    if len(train_ids) != len(set(train_ids)) or not set(train_ids) <= set(index):
        raise ValueError("Invalid training IDs")
    counts = Counter(labels[g] for g in train_ids)
    if set(counts) != {"OG", "TSG"}:
        raise ValueError("Training partition lacks a class")
    weights = None
    if model["estimator"] == "prior":
        estimator = DummyClassifier(strategy="prior", random_state=plan["seed"])
    else:
        weights = protocol.training_class_weights(labels, train_ids)
        options = plan["logistic"]
        classifier = LogisticRegression(C=c, solver=options["solver"], l1_ratio=options["l1_ratio"],
                                        fit_intercept=options["fit_intercept"], intercept_scaling=options["intercept_scaling"],
                                        class_weight={0: weights["TSG"], 1: weights["OG"]},
                                        max_iter=options["max_iter"], tol=options["tol"], random_state=plan["seed"])
        estimator = Pipeline([("scale", StandardScaler()), ("classifier", classifier)])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        estimator = protocol.fit_training_transformer(
            estimator, {g: x[index[g]] for g in train_ids}, train_ids,
            {g: int(labels[g] == "OG") for g in train_ids})
    record = {"training_genes": len(train_ids), "training_class_counts": dict(counts),
              "fit_gene_ids_sha256": features.object_sha(train_ids), "class_weights": weights,
              "initialization": "fresh_unfitted_classical_estimator", "C": c}
    if model["estimator"] == "logistic":
        scaler, classifier = estimator["scale"], estimator["classifier"]
        record.update({"preprocessor_fit_gene_ids_sha256": features.object_sha(estimator.fit_gene_ids_),
                       "scaler_mean_sha256": features.array_sha(scaler.mean_),
                       "scaler_scale_sha256": features.array_sha(scaler.scale_),
                       "coefficient_sha256": features.array_sha(classifier.coef_),
                       "intercept_sha256": features.array_sha(classifier.intercept_),
                       "iterations": int(classifier.n_iter_.max())})
    else:
        record.update({"preprocessor_fit_gene_ids_sha256": None, "class_prior_og": float(estimator.class_prior_[1])})
    return estimator, record


def probabilities(estimator, x):
    if list(estimator.classes_) != [0, 1]:
        raise ValueError("Unexpected class order")
    p = estimator.predict_proba(x)[:, 1]
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid model probabilities")
    return p


def evaluate_model(ids, x, labels, splits, model, setting, plan, partitions, fits, candidates, inner_rows):
    index = {g: i for i, g in enumerate(ids)}
    outer_rows = []
    setting_folds = splits["settings"][setting]
    for outer in range(setting_folds["outer_folds"]):
        outer_partition = protocol.partition_ids(splits, setting, outer)
        outer_key = f"{setting}/outer{outer}/refit"
        partitions.setdefault(outer_key, outer_partition)
        candidate_results = []
        for rank, c in enumerate(model["C"]):
            candidate = "prior" if c is None else f"C{c:g}"
            oof, scores = [], []
            for inner in range(setting_folds["inner_folds"][str(outer)]):
                parts = protocol.partition_ids(splits, setting, outer, inner)
                partition_key = f"{setting}/outer{outer}/inner{inner}"
                partitions.setdefault(partition_key, parts)
                fit_id = f"{model['id']}/{partition_key}/{candidate}"
                estimator, lineage = fit_model(x, ids, labels, parts["train"], model, c, plan)
                fits.append({"fit_id": fit_id, "partition": partition_key, "model": model["id"],
                             "candidate": candidate, **lineage})
                p = probabilities(estimator, x[[index[g] for g in parts["validation"]]])
                y = np.array([int(labels[g] == "OG") for g in parts["validation"]])
                scores.append(metrics.point_metrics(y, p, (p >= .5).astype(int))["balanced_accuracy"])
                oof.extend({"gene_id": g, "label": labels[g], "probability_og": float(prob),
                            "inner_fold": inner, "fit_id": fit_id} for g, prob in zip(parts["validation"], p))
            if sorted(r["gene_id"] for r in oof) != outer_partition["train"]:
                raise ValueError("Inner OOF predictions are not exactly outer-training genes")
            candidate_results.append({"candidate": candidate, "C": c, "rank": rank,
                                      "mean_inner_balanced_accuracy": float(np.mean(scores)),
                                      "inner_balanced_accuracy": scores, "oof": oof})
        selected = min(candidate_results, key=lambda r: (-r["mean_inner_balanced_accuracy"], r["rank"], r["candidate"]))
        threshold = protocol.select_threshold(selected["oof"], outer_partition["train"])
        candidates.extend({"setting": setting, "outer_fold": outer, "model": model["id"],
                           **{k: v for k, v in r.items() if k != "oof"}, "selected": r is selected,
                           "selected_threshold": threshold if r is selected else None} for r in candidate_results)
        inner_rows.extend({"setting": setting, "outer_fold": outer, "model": model["id"], **r}
                          for r in sorted(selected["oof"], key=lambda r: r["gene_id"]))
        estimator, lineage = fit_model(x, ids, labels, outer_partition["train"], model, selected["C"], plan)
        fit_id = f"{model['id']}/{outer_key}/{selected['candidate']}"
        fits.append({"fit_id": fit_id, "partition": outer_key, "model": model["id"],
                     "candidate": selected["candidate"], "threshold": threshold, **lineage})
        p = probabilities(estimator, x[[index[g] for g in outer_partition["test"]]])
        outer_rows.extend({"setting": setting, "model": model["id"], "gene_id": gene,
                           "label": labels[gene], "group": splits["groups"][gene], "outer_fold": outer,
                           "selected_candidate": selected["candidate"], "threshold": threshold,
                           "probability_og": float(prob), "prediction": "OG" if prob >= threshold else "TSG", "fit_id": fit_id}
                          for gene, prob in zip(outer_partition["test"], p))
        print(json.dumps({"setting": setting, "model": model["id"], "completed_outer_fold": outer,
                          "selected_candidate": selected["candidate"]}), flush=True)
    if sorted(r["gene_id"] for r in outer_rows) != ids:
        raise ValueError("Expected one outer OOF prediction per gene")
    return outer_rows


def run(sequence_dir, split_path, report_path, plan_path, output):
    plan_path, output = Path(plan_path), Path(output)
    plan = json.loads(plan_path.read_text())
    verify_plan(plan)
    splits, labels, sequences = features.load_cds(sequence_dir, split_path, plan, report_path)
    ids = sorted(labels)
    for setting in plan["settings"]:
        for outer, inner_count in splits["settings"][setting]["inner_folds"].items():
            for inner in range(inner_count):
                protocol.partition_ids(splits, setting, int(outer), inner)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    # The run receipt is written BEFORE the first fit; plans and source hashes
    # cannot be chosen after reading outer results. Use a new directory to rerun.
    receipt = {"status": "running", "plan_sha256": features.sha(plan_path), "split_manifest_sha256": features.sha(split_path),
               "sequence_summary_sha256": features.sha(Path(sequence_dir) / "summary.json"),
               "code_sha256": {p.name: features.sha(p) for p in
                               [Path(__file__), Path(features.__file__), Path(metrics.__file__), Path(protocol.__file__)]},
               "software": {"python": platform.python_version(), "numpy": np.__version__,
                            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__}, "genes": len(ids)}
    write_json(output / "run.json", receipt)
    fits, candidates, inner_rows, outer_rows, partitions, feature_records = [], [], [], [], {}, {}
    try:
        with threadpool_limits(limits=plan["software"]["thread_limit"]):
            for model in plan["models"]:
                x, names = features.make_features(ids, sequences, model)
                feature_records[model["id"]] = {"shape": list(x.shape), "gene_order_sha256": features.object_sha(ids),
                                                "feature_names_sha256": features.object_sha(names), "matrix_sha256": features.array_sha(x)}
                for setting in plan["settings"]:
                    outer_rows.extend(evaluate_model(ids, x, labels, splits, model, setting, plan,
                                                     partitions, fits, candidates, inner_rows))
            summaries = {}
            for setting in plan["settings"]:
                print(f"Bootstrapping {setting}", flush=True)
                summaries[setting] = metrics.summarize_setting(
                    [r for r in outer_rows if r["setting"] == setting], setting,
                    plan["uncertainty"]["resamples"], plan["uncertainty"]["seed"])
        write_json(output / "features.json", feature_records)
        write_json(output / "partitions.json", partitions)
        write_json(output / "fits.json", fits)
        write_json(output / "candidate_scores.json", candidates)
        write_json(output / "summary.json", summaries)
        write_csv(output / "outer_predictions.csv", outer_rows,
                  ["setting", "model", "gene_id", "label", "group", "outer_fold", "selected_candidate",
                   "probability_og", "threshold", "prediction", "fit_id"])
        write_csv(output / "selected_inner_oof.csv", inner_rows,
                  ["setting", "model", "outer_fold", "inner_fold", "gene_id", "label", "probability_og", "fit_id"])
        receipt.update({"status": "completed", "fits": len(fits), "outer_predictions": len(outer_rows),
                        "selected_inner_oof_predictions": len(inner_rows),
                        "artifacts": {p.name: features.sha(p) for p in sorted(output.iterdir()) if p.name != "run.json"}})
        write_json(output / "run.json", receipt)
    except Exception as error:
        receipt.update({"status": "failed", "error": str(error)})
        write_json(output / "run.json", receipt)
        raise
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-dir", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=Path("evaluation/v1/splits.json"))
    parser.add_argument("--evaluation-report", type=Path, default=Path("evaluation/v1/report.json"))
    parser.add_argument("--plan", type=Path, default=Path("evaluation/baseline_plan_v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.sequence_dir, args.splits, args.evaluation_report, args.plan, args.output)
