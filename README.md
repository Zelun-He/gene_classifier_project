# Gene Sequence Classification: Oncogenes vs Tumor Suppressors

Research with Dr. Hui Liu on whether DNA sequence representations can distinguish
oncogenes from tumor suppressor genes when labeled data are limited. The intended
comparison is pretrained DNA representations versus simple sequence features,
with evaluation on held-out genes and related-gene groups.

## Current research milestone: auditable labels

The label preparation pipeline preserves COSMIC and OncoKB annotations instead
of keeping whichever label appears first. It separates exclusive roles,
explicit dual roles, unresolved annotations, and genes without a target role.
Only exclusive roles with a source-provided stable identifier enter the binary
cohort. Original records, inclusion decisions, source release metadata, and file
hashes remain traceable.

**Start with [the label audit guide](docs/label_audit.md).** It includes the
real-data command, policy, output schemas, and a runnable synthetic demonstration.

The audit and its tests require only Python 3.10+:

```bash
python scripts/prepare_gene_labels.py --help
python -m unittest discover -s tests -v
```

Original database exports, sequence data, and model checkpoints are not bundled.
The synthetic fixtures test software behavior; they are not biological evidence.
Historical dataset sizes and performance figures have not been reproduced.

## Pipeline and implementation status

| Stage | Entry point | Status |
| --- | --- | --- |
| Label preparation | `scripts/prepare_gene_labels.py` | Auditable offline pipeline; real cohort requires original exports and metadata |
| Genomic sequence retrieval | `scripts/fetch_sequences.py` | Legacy prototype; sequence/reference policy still needs verification |
| 6-mer vocabulary and dataset preparation | `scripts/build_vocab.py`, `scripts/make_datasets.py` | Legacy prototype; reproducible grouped splits are not implemented |
| Model training | Root `run_finetune.py` | Legacy prototype; checkpoint loading, tokenization, and loss integration need repair |
| Prediction | `predict.py` | Legacy prototype; preprocessing must be aligned with training |

Do not treat the legacy training scripts as a verified end-to-end benchmark.
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
