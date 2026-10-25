"""Audit frozen vectors, fold-local probe fitting and paired baseline comparisons."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

if __package__:
    from . import benchmark_features as f, benchmark_metrics as metrics, extract_frozen_embeddings as encoder
    from . import verify_baseline_run as audit, run_frozen_encoder_benchmark as runner
else:
    import benchmark_features as f
    import benchmark_metrics as metrics
    import extract_frozen_embeddings as encoder
    import verify_baseline_run as audit
    import run_frozen_encoder_benchmark as runner


def verify(run_dir, splits_path, plan_path, initialization_path, baseline_dir, sequence_dir=None,
           extraction_cache=None, bootstrap=False, model_dir=None):
    run_dir, baseline_dir = Path(run_dir), Path(baseline_dir)
    read = lambda name: json.loads((run_dir / name).read_text())
    receipt, plan = read("run.json"), read("plan.json")
    audit.require(receipt["status"] == "completed" and receipt["encoder_finetuned"] is False, "Incomplete/unexpected encoder run")
    audit.require(receipt["plan_sha256"] == f.sha(plan_path), "Frozen plan changed")
    audit.require(json.loads(Path(plan_path).read_text()) == plan, "Plan values changed")
    audit.require(receipt["baseline_run_receipt_sha256"] == plan["baseline_run_receipt_sha256"] == f.sha(baseline_dir / "run.json"), "Baseline run changed")
    audit.require(receipt["split_manifest_sha256"] == plan["split_manifest_sha256"] == f.sha(splits_path), "Split manifest changed")
    for name, expected in receipt["artifacts"].items():
        audit.require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Unexpected artifact path")
        audit.require(f.sha(run_dir / name) == expected, f"Output artifact changed: {name}")
    for name, expected in receipt["code_sha256"].items():
        audit.require(f.sha(Path(__file__).with_name(name)) == expected, f"Recorded training source changed: {name}")
    baseline_receipt = json.loads((baseline_dir / "run.json").read_text())
    for name, expected in baseline_receipt["artifacts"].items():
        audit.require(f.sha(baseline_dir / name) == expected, "Original baseline evidence changed")
    runner.verify_plan(plan, json.loads((baseline_dir / "plan.json").read_text()))
    splits = json.loads(Path(splits_path).read_text())
    ids = sorted(splits["groups"])
    x, extraction = runner.load_embeddings(run_dir / "embeddings", plan_path, initialization_path, ids)
    audit.require(receipt["extraction_receipt_sha256"] == f.sha(run_dir / "embeddings/extraction.json"), "Extraction receipt changed")
    for name, expected in extraction["source_sha256"].items():
        audit.require(f.sha(Path(__file__).with_name(name)) == expected, f"Extraction source changed: {name}")
    token_audit = read("embeddings/tokenization_audit.json")
    audit.require(len(token_audit) == extraction["windows"] and len({w["window_id"] for w in token_audit}) == len(token_audit), "Incomplete/duplicate encoded windows")
    audit.require(f.object_sha([w["window_id"] for w in token_audit]) == extraction["window_ids_sha256"], "Window order changed")
    by_gene = defaultdict(list)
    for w in token_audit:
        audit.require(w["real_kmers"] == w["bases"] - 5 and w["model_tokens"] == w["real_kmers"] + 2 <= 512 and w["unknown_tokens"] == 0, "Tokenizer coverage mismatch")
        by_gene[w["gene_id"]].append(w)
    audit.require(sorted(by_gene) == ids, "Encoded gene coverage changed")
    for gene, members in by_gene.items():
        length = sum(w["bases"] for w in members)
        audit.close(sum(w["pool_weight"] for w in members), 1, "Incomplete gene pooling weights")
        for w in members:
            audit.close(w["pool_weight"], w["bases"] / length, "Incorrect length pooling weight")
    outer, inner = audit.load_rows(run_dir / "outer_predictions.csv"), audit.load_rows(run_dir / "selected_inner_oof.csv")
    partitions, fits, candidates, summary = read("partitions.json"), read("fits.json"), read("candidate_scores.json"), read("summary.json")
    result = audit.audit_records(plan, splits, partitions, fits, candidates, inner, outer, summary)
    original = audit.load_rows(baseline_dir / "outer_predictions.csv")
    combined = audit.load_rows(run_dir / "combined_outer_predictions.csv")
    audit.require(combined == [*original, *outer], "Baseline predictions were altered, reordered or refitted")
    for setting in plan["settings"]:
        for model in plan["comparisons"]["baselines"]:
            rows = [r for r in combined if r["setting"] == setting and r["model"] == model]
            independent = audit.independent_metrics(rows)
            for key, value in independent.items():
                audit.close(summary[setting]["models"][model]["pooled_oof"][key], value, "Baseline comparison metric mismatch")
    model_id = plan["models"][0]["id"]
    feature_info = read("features.json")[model_id]
    audit.require(feature_info["shape"] == list(x.shape) and feature_info["matrix_sha256"] == f.array_sha(x)
                  and feature_info["gene_order_sha256"] == f.object_sha(ids), "Probe features differ from immutable gene vectors")
    with threadpool_limits(limits=1):
        index = {gene: i for i, gene in enumerate(ids)}
        for key, partition in partitions.items():
            scaler = StandardScaler().fit(x[[index[g] for g in partition["train"]]])
            for fit in fits:
                if fit["partition"] == key:
                    audit.require(fit["scaler_mean_sha256"] == f.array_sha(scaler.mean_) and
                                  fit["scaler_scale_sha256"] == f.array_sha(scaler.scale_), "Embedding scaler was not training-local")
        if bootstrap:
            for setting in plan["settings"]:
                actual = metrics.summarize_setting([r for r in combined if r["setting"] == setting], setting,
                                                    plan["uncertainty"]["resamples"], plan["uncertainty"]["seed"])
                audit.require(actual == summary[setting], "Paired five-model bootstrap replay differs")
            result["bootstrap"] = "replayed_all_five_models_and_all_pairwise_comparisons"
    if sequence_dir is not None:
        windows = encoder.load_windows(sequence_dir, splits_path, plan)
        audit.require([w["window_id"] for w in windows] == [w["window_id"] for w in token_audit], "Encoded windows differ from source CDS")
        for w, metadata in zip(windows, token_audit):
            audit.require(w["gene_id"] == metadata["gene_id"] and w["length_bp"] == metadata["bases"], "Source window metadata differs")
        result["sequence_coverage"] = "all_source_windows_reconstruct_full_cds"
    if extraction_cache is not None:
        hashes = read("embeddings/block_hashes.json")
        window_vectors, window_ids = [], []
        for name, expected in sorted(hashes.items()):
            path = Path(extraction_cache) / "blocks" / name
            audit.require(f.sha(path) == expected, "Cached window-vector block changed")
            with np.load(path, allow_pickle=False) as block:
                window_vectors.append(block["embeddings"].copy())
                window_ids.extend(block["ids"].tolist())
        vectors = np.vstack(window_vectors)
        audit.require(window_ids == [w["window_id"] for w in token_audit], "Window-vector order changed")
        audit.require(f.array_sha(vectors) == extraction["window_matrix_sha256"], "Window-vector matrix changed")
        pooled_ids, pooled = encoder.pool_genes(token_audit, vectors)
        audit.require(pooled_ids == ids and np.array_equal(pooled, x), "Gene vectors do not reproduce length-weighted window pooling")
        result["pooling"] = "reconstructed_gene_vectors_exactly_from_all_cached_window_vectors"
        if model_dir is not None:
            audit.require(sequence_dir is not None, "Fresh encoder spot checks require source sequence windows")
            model, tokenizer, _ = encoder.load_encoder(model_dir, initialization_path, plan)
            audit.require(encoder.parameter_sha(model) == extraction["parameter_sha256_before"], "Freshly loaded encoder differs")
            source_windows = encoder.load_windows(sequence_dir, splits_path, plan)
            selected = sorted({0, len(source_windows) - 1, int(np.argmin([len(w["sequence"]) for w in source_windows])),
                               int(np.argmax([len(w["sequence"]) for w in source_windows]))})
            encoded, _ = encoder.encode_windows([source_windows[i] for i in selected], tokenizer)
            fresh = encoder.forward_batch(model, tokenizer, encoded)
            expected = vectors[selected]
            error = float(np.max(np.abs(fresh - expected)))
            audit.require(np.allclose(fresh, expected, rtol=1e-5, atol=2e-5), "Fresh encoder/cache spot checks differ")
            result["fresh_encoder_spot_checks"] = {"window_indices": selected, "max_absolute_error": error}
    elif model_dir is not None:
        raise ValueError("Fresh encoder spot checks require --extraction-cache")
    result.update({"encoder_parameters_unchanged": True, "embedding_shape": list(x.shape),
                   "encoded_windows": len(token_audit), "combined_outer_predictions": len(combined),
                   "baseline_predictions_unchanged": True, "scalers": "all_training_only_moments_reconstructed"})
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--splits", default=Path("evaluation/v1/splits.json"), type=Path)
    p.add_argument("--plan", default=Path("evaluation/frozen_encoder_plan_v1.json"), type=Path)
    p.add_argument("--initialization-record", default=Path("evaluation/dnabert6_initialization_v1.json"), type=Path)
    p.add_argument("--baseline-dir", default=Path("evaluation/baselines-v1"), type=Path)
    p.add_argument("--sequence-dir", type=Path)
    p.add_argument("--extraction-cache", type=Path)
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--bootstrap", action="store_true")
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    result = verify(a.run_dir, a.splits, a.plan, a.initialization_record, a.baseline_dir,
                    a.sequence_dir, a.extraction_cache, a.bootstrap, a.model_dir)
    if a.output:
        a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
