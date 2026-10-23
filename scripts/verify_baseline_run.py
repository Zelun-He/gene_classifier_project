"""Independently audit saved baseline metrics, fitting boundaries and thresholds."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
from sklearn import metrics as sk
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

if __package__:
    from . import benchmark_features as f, benchmark_metrics as m, evaluation_protocol as p
else:
    import benchmark_features as f
    import benchmark_metrics as m
    import evaluation_protocol as p


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(a, b, message):
    require(bool(np.allclose(a, b, atol=1e-12, rtol=1e-12)), message)


def load_rows(path):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("outer_fold", "inner_fold"):
            if key in row:
                row[key] = int(row[key])
        for key in ("probability_og", "threshold"):
            if key in row:
                row[key] = float(row[key])
    return rows


def independent_metrics(rows):
    """Use sklearn, independently of the runner's vectorized implementation."""
    y = np.array([int(r["label"] == "OG") for r in rows])
    probability = np.array([r["probability_og"] for r in rows])
    predicted = np.array([int(r["prediction"] == "OG") for r in rows])
    tn, fp, fn, tp = sk.confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {"balanced_accuracy": sk.balanced_accuracy_score(y, predicted),
            "accuracy": sk.accuracy_score(y, predicted), "auroc": sk.roc_auc_score(y, probability),
            "average_precision_og": sk.average_precision_score(y, probability),
            "average_precision_tsg": sk.average_precision_score(1 - y, 1 - probability),
            "mcc": sk.matthews_corrcoef(y, predicted), "macro_f1": sk.f1_score(y, predicted, average="macro"),
            "precision_og": sk.precision_score(y, predicted, zero_division=0),
            "precision_tsg": sk.precision_score(1 - y, 1 - predicted, zero_division=0),
            "recall_og": sk.recall_score(y, predicted), "recall_tsg": sk.recall_score(1 - y, 1 - predicted),
            "brier_score": sk.brier_score_loss(y, probability), "prevalence_og": y.mean(), "prevalence_tsg": 1 - y.mean(),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def audit_records(plan, splits, partitions, fits, candidates, inner_rows, outer_rows, summary):
    models = {model["id"]: model for model in plan["models"]}
    fit_lookup = {record["fit_id"]: record for record in fits}
    require(len(fit_lookup) == len(fits), "Duplicate fit IDs")
    genes = sorted(splits["groups"])
    labels = {}
    for row in outer_rows:
        require(row["label"] in {"OG", "TSG"}, "Invalid truth label")
        require(row["prediction"] in {"OG", "TSG"}, "Invalid predicted label")
        require(labels.setdefault(row["gene_id"], row["label"]) == row["label"], "Inconsistent gene labels")
        require(row["group"] == splits["groups"][row["gene_id"]], "Changed homology group")
        require(np.isfinite(row["probability_og"]) and 0 <= row["probability_og"] <= 1, "Invalid probability")
        require(np.isfinite(row["threshold"]), "Invalid threshold")
        require((row["probability_og"] >= row["threshold"]) == (row["prediction"] == "OG"), "Prediction/threshold mismatch")
    require(sorted(labels) == genes, "Incomplete prediction cohort")
    import hashlib
    require(hashlib.sha256(json.dumps(labels, sort_keys=True).encode()).hexdigest() == splits["label_vector_sha256"], "Changed truth labels")
    expected_partitions = {}
    for setting in plan["settings"]:
        s = splits["settings"][setting]
        for outer in range(s["outer_folds"]):
            expected_partitions[f"{setting}/outer{outer}/refit"] = p.partition_ids(splits, setting, outer)
            for inner in range(s["inner_folds"][str(outer)]):
                expected_partitions[f"{setting}/outer{outer}/inner{inner}"] = p.partition_ids(splits, setting, outer, inner)
    require(partitions == expected_partitions, "Saved partitions differ from frozen folds")
    for fit in fits:
        train_ids = partitions[fit["partition"]]["train"]
        require(fit["fit_gene_ids_sha256"] == f.object_sha(train_ids), "Fit training-gene hash mismatch")
        require(fit["training_genes"] == len(train_ids), "Fit gene count mismatch")
        require(fit["training_class_counts"] == dict(Counter(labels[g] for g in train_ids)), "Training class counts changed")
        require(fit["initialization"] == "fresh_unfitted_classical_estimator", "Unexpected initialization")
        model = models[fit["model"]]
        if model["estimator"] == "logistic":
            require(fit["class_weights"] == p.training_class_weights(labels, train_ids), "Weights not training-local")
            require(fit["preprocessor_fit_gene_ids_sha256"] == f.object_sha(train_ids), "Scaler fit-gene mismatch")
        else:
            require(fit["class_weights"] is None and fit["preprocessor_fit_gene_ids_sha256"] is None, "Unexpected prior fitting")
            close(fit["class_prior_og"], np.mean([labels[g] == "OG" for g in train_ids]), "Prior not training-local")
    expected_fits, expected_inner_rows, expected_fit_ids = 0, 0, set()
    for setting in plan["settings"]:
        s = splits["settings"][setting]
        for name, model in models.items():
            records = [r for r in outer_rows if r["setting"] == setting and r["model"] == name]
            require(sorted(r["gene_id"] for r in records) == genes, "Not one OOF prediction per gene/model/setting")
            pooled = independent_metrics(records)
            for key, value in pooled.items():
                close(summary[setting]["models"][name]["pooled_oof"][key], value, f"Independent metric mismatch: {key}")
            balanced = []
            for outer in range(s["outer_folds"]):
                part_key = f"{setting}/outer{outer}/refit"
                parts = partitions[part_key]
                out = [r for r in records if r["outer_fold"] == outer]
                require(sorted(r["gene_id"] for r in out) == parts["test"], "Outer fold membership changed")
                fold_stats = independent_metrics(out)
                balanced.append(fold_stats["balanced_accuracy"])
                saved_fold = next(r for r in summary[setting]["models"][name]["outer_folds"] if r["outer_fold"] == outer)
                for key, value in fold_stats.items():
                    close(saved_fold[key], value, f"Independent outer-fold metric mismatch: {key}")
                grid = [r for r in candidates if r["setting"] == setting and r["model"] == name and r["outer_fold"] == outer]
                require(len(grid) == len(model["C"]) and [r["C"] for r in sorted(grid, key=lambda r: r["rank"])] == model["C"], "Candidate grid changed")
                require(sum(r["selected"] for r in grid) == 1, "Invalid candidate selection count")
                chosen = min(grid, key=lambda r: (-r["mean_inner_balanced_accuracy"], r["rank"], r["candidate"]))
                require(chosen["selected"], "Candidate was not selected by declared rule")
                for candidate in grid:
                    require(len(candidate["inner_balanced_accuracy"]) == s["inner_folds"][str(outer)], "Candidate has wrong inner fold count")
                    close(np.mean(candidate["inner_balanced_accuracy"]), candidate["mean_inner_balanced_accuracy"], "Candidate mean changed")
                    for inner in range(s["inner_folds"][str(outer)]):
                        expected_fit_ids.add(f"{name}/{setting}/outer{outer}/inner{inner}/{candidate['candidate']}")
                inners = [r for r in inner_rows if r["setting"] == setting and r["model"] == name and r["outer_fold"] == outer]
                require(sorted(r["gene_id"] for r in inners) == parts["train"], "Threshold OOF IDs not exactly outer training")
                expected_inner_rows += len(parts["train"])
                for inner in range(s["inner_folds"][str(outer)]):
                    key = f"{setting}/outer{outer}/inner{inner}"
                    rows = [r for r in inners if r["inner_fold"] == inner]
                    require(sorted(r["gene_id"] for r in rows) == partitions[key]["validation"], "Inner OOF membership changed")
                    expected_id = f"{name}/{key}/{chosen['candidate']}"
                    require(all(r["fit_id"] == expected_id for r in rows), "Inner OOF model lineage mismatch")
                    require(expected_id in fit_lookup, "Missing inner fit")
                    for row in rows:
                        require(row["label"] == labels[row["gene_id"]], "Changed inner truth label")
                    yy = [int(r["label"] == "OG") for r in rows]
                    pp = [r["probability_og"] >= .5 for r in rows]
                    close(sk.balanced_accuracy_score(yy, pp), chosen["inner_balanced_accuracy"][inner], "Selected inner score changed")
                threshold = p.select_threshold(inners, parts["train"])
                close(threshold, chosen["selected_threshold"], "Threshold not chosen from inner OOF")
                expected_id = f"{name}/{part_key}/{chosen['candidate']}"
                expected_fit_ids.add(expected_id)
                require(expected_id in fit_lookup, "Missing outer refit")
                for row in out:
                    require(row["fit_id"] == expected_id and row["selected_candidate"] == chosen["candidate"], "Outer model lineage mismatch")
                    close(row["threshold"], threshold, "Outer threshold changed")
                expected_fits += len(model["C"]) * s["inner_folds"][str(outer)] + 1
            close(summary[setting]["models"][name]["mean_outer_balanced_accuracy"], np.mean(balanced), "Primary mean mismatch")
            close(summary[setting]["models"][name]["sd_outer_balanced_accuracy"], np.std(balanced, ddof=1), "Fold SD mismatch")
    require(len(fits) == expected_fits and len(inner_rows) == expected_inner_rows, "Unexpected fit/OOF count")
    require(set(fit_lookup) == expected_fit_ids, "Saved fit identities do not cover every declared candidate/fold")
    require(len(outer_rows) == len(genes) * len(models) * len(plan["settings"]), "Unexpected outer row count")
    return {"status": "passed", "genes": len(genes), "fits": len(fits), "partitions": len(partitions),
            "outer_predictions": len(outer_rows), "selected_inner_oof_predictions": len(inner_rows),
            "independent_metrics": "sklearn_pooled_and_per_outer_fold", "fitting_boundaries": "all_saved_fits_checked",
            "thresholds": "recomputed_from_complete_selected_inner_oof"}


def verify(run_dir, splits_path, source_plan, sequence_dir=None, bootstrap=False):
    run_dir, splits_path = Path(run_dir), Path(splits_path)
    read = lambda name: json.loads((run_dir / name).read_text())
    receipt, plan, splits = read("run.json"), read("plan.json"), json.loads(splits_path.read_text())
    require(receipt["status"] == "completed", "Run did not complete")
    for name, expected in receipt["artifacts"].items():
        require(f.sha(run_dir / name) == expected, f"Run artifact hash changed: {name}")
    for name, expected in receipt["code_sha256"].items():
        require(f.sha(Path(__file__).with_name(name)) == expected, f"Run source code changed: {name}")
    require(f.sha(splits_path) == receipt["split_manifest_sha256"] == plan["split_manifest_sha256"], "Split hash changed")
    require(f.sha(source_plan) == receipt["plan_sha256"], "Predeclared plan hash changed")
    require(json.loads(Path(source_plan).read_text()) == plan, "Plan values changed")
    partitions, fits, candidates = read("partitions.json"), read("fits.json"), read("candidate_scores.json")
    inner, outer = load_rows(run_dir / "selected_inner_oof.csv"), load_rows(run_dir / "outer_predictions.csv")
    summary = read("summary.json")
    result = audit_records(plan, splits, partitions, fits, candidates, inner, outer, summary)
    with threadpool_limits(limits=1):
        if sequence_dir is not None:
            _, labels, sequences = f.load_cds(sequence_dir, splits_path, plan, splits_path.with_name("report.json"))
            ids, stored_features = sorted(labels), read("features.json")
            for model in plan["models"]:
                x, names = f.make_features(ids, sequences, model)
                require(stored_features[model["id"]] == {"shape": list(x.shape), "gene_order_sha256": f.object_sha(ids),
                        "feature_names_sha256": f.object_sha(names), "matrix_sha256": f.array_sha(x)}, "Feature reconstruction mismatch")
                if model["estimator"] != "logistic":
                    continue
                index = {gene: i for i, gene in enumerate(ids)}
                for key, partition in partitions.items():
                    train = partition["train"]
                    scaler = StandardScaler().fit(x[[index[g] for g in train]])
                    for fit in fits:
                        if fit["model"] == model["id"] and fit["partition"] == key:
                            require(fit["scaler_mean_sha256"] == f.array_sha(scaler.mean_) and
                                    fit["scaler_scale_sha256"] == f.array_sha(scaler.scale_), "Scaler statistics are not training-local")
            result["features_and_scalers"] = "reconstructed_all_features_and_training_only_scaler_statistics"
        if bootstrap:
            for setting in plan["settings"]:
                actual = m.summarize_setting([r for r in outer if r["setting"] == setting], setting,
                                             plan["uncertainty"]["resamples"], plan["uncertainty"]["seed"])
                require(actual == summary[setting], "Bootstrap result replay differs")
            result["bootstrap"] = "all_2000_paired_resamples_reproduced_in_both_settings"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--splits", default=Path("evaluation/v1/splits.json"), type=Path)
    parser.add_argument("--plan", default=Path("evaluation/baseline_plan_v1.json"), type=Path)
    parser.add_argument("--sequence-dir", type=Path)
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.run_dir, args.splits, args.plan, args.sequence_dir, args.bootstrap)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
