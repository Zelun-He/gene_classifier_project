"""One entry point for fixed-fold baseline/frozen evaluation and archive audits."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

if __package__:
    from . import benchmark_features as f, run_baselines as baseline, verify_baseline_run as baseline_audit
else:
    import benchmark_features as f
    import run_baselines as baseline
    import verify_baseline_run as baseline_audit

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"class_prior": "Class prior", "length_gc": "Length/GC logistic", "kmer3": "3-mer logistic",
         "kmer6": "6-mer logistic", "dnabert6_frozen": "Frozen DNABERT-6 + logistic"}


def read(path):
    return json.loads(Path(path).read_text())


def render_table(run_dir, audit):
    """Render saved numbers, never rounded/manual copies of a separate experiment."""
    root = Path(run_dir)
    receipt, summary = read(root / "run.json"), read(root / "summary.json")
    if receipt["status"] != "completed" or audit["status"] != "passed":
        raise ValueError("Only completed independently audited results can be rendered")
    for name, digest in receipt["artifacts"].items():
        if f.sha(root / name) != digest:
            raise ValueError(f"Saved result changed: {name}")
    settings = ("stratified_gene", "similarity_grouped")
    names = [name for name in NAMES if name in summary[settings[0]]["models"]]
    if any(set(summary[s]["models"]) != set(names) for s in settings):
        raise ValueError("Expected the same fixed models in both settings")
    genes = {summary[s]["models"][name]["genes"] for s in settings for name in names}
    if len(genes) != 1:
        raise ValueError("Model results are not from one common gene cohort")
    lines = ["# Fixed-fold gene classification results", "",
             f"{genes.pop()} genes; five outer folds and three inner folds. "
             "Primary metric: mean outer-fold balanced accuracy (%). Brackets are conditional 95% bootstrap intervals.", "",
             "| Model | Stratified BA %, 95% interval | Similarity-grouped BA %, 95% interval |",
             "| --- | ---: | ---: |"]
    for name in names:
        cells = []
        for setting in settings:
            row = summary[setting]["models"][name]
            ci = row["confidence_intervals_95"]["mean_outer_balanced_accuracy"]
            cells.append(f"{100 * row['mean_outer_balanced_accuracy']:.2f} [{100 * ci['lower']:.2f}, {100 * ci['upper']:.2f}]")
        lines.append("| " + " | ".join([NAMES[name], *cells]) + " |")
    lines += ["", "All transformations and class weights are fitted on training genes only. "
              "Candidates use inner validation; decision thresholds use complete selected-candidate inner OOF predictions. "
              "Every outer gene has one prediction per model, regardless of its number of sequence windows.", "",
              "Uncertainty uses 2,000 paired gene resamples (stratified) or complete detected-homology-component resamples (grouped). "
              "Intervals condition on the fitted models and thresholds; they do not include full retraining or source-label uncertainty. "
              "The analyses are exploratory. Inspect paired differences rather than treating overlapping marginal intervals as a test.", "",
              "Evidence in the run directory:", "",
              "- `outer_predictions.csv`: gene IDs, labels, probabilities, predictions, groups, folds, selected candidate and threshold.",
              "- `partitions.json`: exact training, inner-validation and outer-test gene IDs, including all 40 partition definitions.",
              "- `fits.json`: each fit's partition key, training-ID hash, training counts, weights and parameter/scaler hashes.",
              "- `selected_inner_oof.csv`, `candidate_scores.json`: threshold and configuration selection evidence.",
              "- `plan.json`, `run.json`, `features.json`, `summary.json`: fixed configurations, input/source/output identities and full metrics.", "",
              f"Run receipt SHA-256: `{f.sha(root / 'run.json')}`. "
              "`checkpoint-916` is excluded. The baseline sequence features use complete CDS directly; "
              "the frozen encoder pools real tokens within windows and length-weights all windows into one vector per gene."]
    return "\n".join(lines) + "\n"


def verify(args):
    if args.stage == "baselines":
        return baseline_audit.verify(args.run_dir, args.splits, args.plan or ROOT / "evaluation/baseline_plan_v1.json",
                                     args.sequence_dir, args.bootstrap)
    if __package__:
        from . import verify_frozen_encoder_run as auditor
    else:
        import verify_frozen_encoder_run as auditor
    return auditor.verify(args.run_dir, args.splits, args.plan or ROOT / "evaluation/frozen_encoder_plan_v1.json",
                          args.initialization_record, args.baseline_dir, args.sequence_dir,
                          args.embedding_dir, args.bootstrap, args.model_dir)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["baselines", "frozen", "verify"])
    parser.add_argument("--stage", choices=["baselines", "frozen"], default="baselines", help="Stage for verify")
    parser.add_argument("--sequence-dir", type=Path)
    parser.add_argument("--splits", type=Path, default=ROOT / "evaluation/v1/splits.json")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path, help="New run directory when fitting")
    parser.add_argument("--run-dir", type=Path, help="Existing run directory for verify")
    parser.add_argument("--embedding-dir", type=Path, help="Frozen extraction cache, resumable with identical inputs")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--initialization-record", type=Path, default=ROOT / "evaluation/dnabert6_initialization_v1.json")
    parser.add_argument("--baseline-dir", type=Path, default=ROOT / "evaluation/baselines-v1")
    parser.add_argument("--bootstrap", action="store_true", help="Replay uncertainty during verify; always done for new fits")
    parser.add_argument("--report", type=Path, help="Write an audited Markdown results table")
    args = parser.parse_args(argv)
    if args.action != "verify":
        if args.sequence_dir is None or args.output is None:
            parser.error("New fits require --sequence-dir and --output")
        if args.output.exists():
            parser.error("Use a new output directory; existing runs are never overwritten")
        args.stage, args.run_dir, args.bootstrap = args.action, args.output, True
        if args.stage == "baselines":
            plan = args.plan or ROOT / "evaluation/baseline_plan_v1.json"
            baseline.run(args.sequence_dir, args.splits, args.splits.with_name("report.json"), plan, args.output)
        else:
            if args.model_dir is None or args.embedding_dir is None:
                parser.error("Frozen evaluation requires --model-dir and --embedding-dir")
            if __package__:
                from . import extract_frozen_embeddings as encoder, run_frozen_encoder_benchmark as probe
            else:
                import extract_frozen_embeddings as encoder
                import run_frozen_encoder_benchmark as probe
            plan = args.plan or ROOT / "evaluation/frozen_encoder_plan_v1.json"
            encoder.extract(args.sequence_dir, args.splits, args.model_dir, args.initialization_record, plan, args.embedding_dir)
            probe.run(args.sequence_dir, args.splits, plan, args.initialization_record, args.embedding_dir,
                      args.baseline_dir, args.output)
    elif args.run_dir is None:
        parser.error("verify requires --run-dir")
    audit = verify(args)
    # Supplemental execution record leaves the original immutable run receipt intact.
    proof = {"status": "passed", "stage": args.stage, "run_receipt_sha256": f.sha(args.run_dir / "run.json"),
             "runner_sha256": f.sha(__file__), "checked_at_utc": datetime.now(timezone.utc).isoformat(),
             "bootstrap_replayed": args.bootstrap, "sequence_features_reconstructed": args.sequence_dir is not None,
             "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "audit": audit}
    baseline.write_json(args.run_dir / "evaluation_runner_receipt.json", proof)
    if args.report is not None:
        if not args.bootstrap:
            parser.error("Results reports require --bootstrap so intervals are verified")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_table(args.run_dir, audit))
    print(json.dumps({"status": "passed", "stage": args.stage, "run_dir": str(args.run_dir),
                      "report": None if args.report is None else str(args.report)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
