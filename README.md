# Gene Sequence Classification: Oncogenes vs Tumor Suppressors

Research with Dr. Hui Liu on whether DNA sequence representations can distinguish
oncogenes from tumor suppressor genes when labeled data are limited. The intended
comparison is pretrained DNA representations versus simple sequence features,
with evaluation on held-out genes and related-gene groups.

## Auditable labels

The label preparation pipeline preserves COSMIC and OncoKB annotations instead
of keeping whichever label appears first. It separates exclusive roles,
explicit dual roles, unresolved annotations, and genes without a target role.
Only exclusive roles with a source-provided stable identifier enter the binary
cohort. Original records, inclusion decisions, source release metadata, and file
hashes remain traceable.

**Start with [the label audit guide](docs/label_audit.md).** It includes the
real-data command, policy, output schemas, and a runnable synthetic demonstration.

The label audit requires only Python 3.10+. The complete test suite also needs
the evaluation dependencies (Python 3.11+):

```bash
python scripts/prepare_gene_labels.py --help
python -m pip install -r requirements-evaluation.txt
python -m unittest discover -s tests -v
```

The original exports were recovered from the initial Git commit and audited:
**816 eligible genes (447 OG, 369 TSG)**. See the
[real-data results and recovery command](docs/label_audit_results.md).
Raw exports remain untracked; sequence data and checkpoints are not bundled.
Synthetic fixtures test software behavior; historical transformer scores remain
unverified. Completed classical baseline results are documented below.

## Gene-level sequence inputs

The new sequence pipeline uses **GRCh38.p14 / Ensembl 114** and one transcript
per gene: MANE Select, then Ensembl Canonical, then longest translated CDS.
It extracts complete spliced coding sequences plus fixed promoter and sampled
genomic controls, with reference, transcript, strand, quality, and coverage
metadata. Model windows retain their gene ID and pool to one gene representation.
The completed extraction retains **809 CDS genes**, with **799 genes** available
for the paired CDS/promoter/genomic comparison.

See the [sequence policy](docs/sequence_policy.md) and
[real cohort coverage report](docs/sequence_coverage.md). The extraction and
pooling helpers prepare data; model training remains separate work.

## Evaluation before training

The [evaluation protocol](docs/evaluation_protocol.md) freezes **five outer folds
and three inner folds** for both stratified-gene and similarity-grouped settings.
The [feasibility report](docs/evaluation_report.md) documents 673 protein homology
groups, threshold sensitivity, per-fold class counts and checkpoint provenance.
All models consume the same [committed split IDs](evaluation/v1/splits.json).
Every transcript, window and augmentation follows its gene; scaling, selection,
weights and thresholds use training/inner-validation data only.

`checkpoint-916` has unverifiable exposure history and is excluded from the
benchmark. Future pretrained-DNA fits must start from documented pristine, revision-pinned
pretraining. Classical baselines have now completed; no DNA encoder has been
trained. Evaluation dependencies are separate:
`python -m pip install -r requirements-evaluation.txt`.

## First baseline results

The [completed nested benchmark](docs/baseline_results.md) compares class prior,
length/GC logistic regression, and 3-mer/6-mer logistic regression. Mean outer-fold
balanced accuracy:

| Baseline | Stratified BA %, 95% interval | Similarity-grouped BA %, 95% interval |
| --- | ---: | ---: |
| Class prior | 50.00 [50.00, 50.00] | 50.00 [50.00, 50.00] |
| Length/GC | 55.75 [52.60, 58.98] | 55.18 [51.51, 58.62] |
| 3-mer | 57.79 [54.41, 61.08] | 56.50 [53.01, 59.94] |
| 6-mer | 59.95 [56.54, 63.17] | 54.47 [50.87, 57.68] |

The report includes conditional uncertainty intervals and paired comparisons;
the grouped 3-mer/6-mer difference is inconclusive. The plan was published before
fitting. All 340 fits use fold-local preprocessing and gene-level labels; predictions,
fit-gene IDs and selection evidence are [committed](evaluation/baselines-v1).

The [generated results table](docs/evaluation_results.md) is rendered from the
saved summary after an independent audit. Use the unified runner to reproduce
fits or verify existing evidence, including without downloading sequence inputs:

```bash
# Recheck saved predictions, all fit boundaries and both uncertainty calculations.
python scripts/run_evaluation.py verify --stage baselines \
  --run-dir evaluation/baselines-v1 --bootstrap \
  --report docs/evaluation_results.md

# Fresh fits: sequence files must match the committed input/split hashes.
python scripts/run_evaluation.py baselines \
  --sequence-dir data/sequence_inputs/ensembl114-v1 \
  --output data/evaluation/baselines-reproduced --report docs/evaluation_results.md

# Pristine frozen encoder, complete window pooling, nested linear probe and audit.
python scripts/run_evaluation.py frozen \
  --sequence-dir data/sequence_inputs/ensembl114-v1 \
  --model-dir models/dnabert6-pretrained-2021 \
  --embedding-dir data/evaluation/dnabert6-embeddings-v1 \
  --output data/evaluation/dnabert6-frozen-v1 --report docs/frozen_comparison_table.md
```

New fits require a new output directory. Each fit refers to its exact training
gene IDs in `partitions.json`; its record saves the ID hash, class counts,
weights, configuration and preprocessing/parameter hashes. Frozen extraction
resumes only when all input, plan, source and runtime identities match.

## Pipeline and implementation status

| Stage | Entry point | Status |
| --- | --- | --- |
| Label preparation | `scripts/prepare_gene_labels.py` | Original exports recovered and audited; release/download metadata explicitly unknown |
| CDS and control sequence preparation | `scripts/prepare_sequence_inputs.py` | Pinned archive, deterministic transcript selection, gene-grouped windows and coverage audit |
| Nested evaluation design | `scripts/prepare_homology.py`, `scripts/prepare_evaluation.py` | Fixed stratified and similarity-grouped folds; feasibility and checkpoint audit completed |
| Classical benchmark | `scripts/run_baselines.py`, `scripts/verify_baseline_run.py` | Four predeclared baselines completed; nested predictions and independent audit saved |
| 6-mer vocabulary and dataset preparation | `scripts/build_vocab.py`, `scripts/make_datasets.py` | Legacy prototype; use committed gene-level folds for future training |
| Model training | Root `run_finetune.py` | Legacy prototype; checkpoint loading, tokenization, and loss integration need repair |
| Prediction | `predict.py` | Legacy prototype; preprocessing must be aligned with training |

Do not treat the legacy training scripts as a verified end-to-end benchmark.
`scripts/fetch_sequences.py` is also a legacy prototype; use the new sequence
pipeline for this research protocol.
The root training and prediction scripts currently expect an unbundled local
checkpoint at `models/output/checkpoint-916`.

## Labels and downstream compatibility

The audited export uses `gene,label`, with `TSG` for tumor suppressor genes
(class 0) and `OG` for oncogenes (class 1). Historical documentation and model
output also use `ONC` for the oncogene class.

Review a research audit before copying its filtered `gene_labels.csv` to the
legacy `data/gene_labels.csv` path. Existing processed datasets must be rebuilt
for the audited cohort. See the guide for the full handoff procedure.

## Acknowledgments

- Zelun He — Researcher
- Dr. Hui Liu — Research Supervisor
- Model inspiration: DNABERT
