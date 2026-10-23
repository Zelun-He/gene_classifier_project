# Evaluation protocol: gene-nested-homology-v1

Status: partitions and feasibility checks completed **before training**. No new
model performance is claimed. The legacy training entry point does not implement
this protocol and must not be used to produce benchmark scores.

## Cohort and estimands

The biological unit is one audited gene. Use the fixed [sequence policy](sequence_policy.md):
GRCh38.p14, Ensembl 114, one consistently selected transcript, complete CDS.
There are 809 retained CDS genes (441 OG, 368 TSG) and 799 genes with all three
sequence inputs (435 OG, 364 TSG).

| Setting | Question | Partition unit |
| --- | --- | --- |
| Stratified gene | How well does the model classify additional genes from this labelled distribution? | Gene, stratified by class |
| Similarity grouped | How well does it classify genes outside training's detected near-full-length protein homology components? | Connected homology component, approximately stratified by class |

Both are closed-cohort OG/TSG tasks. Neither estimates cancer risk, variant effects,
tissue-specific function, or performance on arbitrary unlabelled human genes.

The committed [split manifest](../evaluation/v1/splits.json) is authoritative.
Its IDs, seed, input hashes and package versions are fixed. All models, baselines,
representations and hyperparameter candidates use the same outer **and inner**
assignments within a setting. The two settings intentionally have different
assignments. Do not regenerate splits to improve scores or balance.

For CDS/promoter/genomic comparisons, restrict **training, inner validation and
outer test** to the same 799 paired genes, inheriting these assignments. Report
the 809-gene CDS experiment separately. Do not compare a model trained on 809
genes with one trained on 799 as though only sequence type changed.

## Homology policy and threshold justification

[SpanSeq](https://doi.org/10.1093/nargab/lqae106) motivates similarity-aware
partitioning throughout development and assessment. This implementation uses
MMseqs2 alignments plus connected components; **the SpanSeq executable was not
run**, and its distance parameters are not interchangeable with alignment identity.

Retrieve the Ensembl 114 **protein translation belonging to each selected CDS
transcript**, checking translation ID, version and length. Protein sequences are
used only to define groups; the classification inputs remain DNA. This groups
coding homologs despite synonymous nucleotide substitutions. Protein grouping
also defines the same family-generalization challenge for the promoter/genomic
controls; it is not a direct promoter-sequence similarity guarantee.

Use MMseqs2 **18-8cc5c**, full revision
`8cc5ce367b5638c4306c2d7cfc652dd099a4643f`, with all-vs-all search,
`--prefilter-mode 2` (no prefilter), `--alignment-mode 3 -a 1`,
`--max-seqs 809`, unlimited accepted/rejected alignment counts, and E-value ≤1e-5.
The log records **654,481 = 809² alignments calculated**. Record the executable,
FASTA, raw alignment and log hashes. Default BLOSUM62 scoring, gap penalties,
composition correction and masking are recorded in the raw log.

An undirected edge requires, in **either** search direction:

- Exact aligned identity `nident/alnlen ≥ 0.30`, including internal gaps in alignment length.
- Aligned query span/query length **and** target span/target length each ≥0.80.
- E-value ≤1e-5.

Compute identity/coverage from integer counts and 1-based inclusive coordinates,
not rounded `pident`, `fident`, or coverage output. Pool edges transitively:
if A–B and B–C qualify, A/B/C stay together even without an A–C edge. A
representative-only cluster does not provide this guarantee. Mixed-label groups
remain intact, retaining the original gene labels.

The 30% identity rule is an explicitly chosen broad coding-homology challenge,
not a universal biological family boundary. Bidirectional 80% coverage focuses
on similarly sized, near-full-length relationships rather than isolated domains;
the E-value requirement limits chance matches, especially for short proteins.
Before examining model scores, diagnostics at 40%, 50% and 70% were compared:
30% yields 673 groups, 347 containing OG and 342 containing TSG, a largest group
of 13, and no collapse into an infeasible giant component. The largest component
contains small GTPases including RAS/RAC/RHO members; receptor and cadherin
components are also recovered. These sequence-only diagnostics support the
chosen operational challenge; they do not establish an optimal threshold or
prove uniform functional similarity.

Important limits: domain-only homology, remote homology, unequal-length family
members, unselected isoforms and promoter similarity can remain across partitions.
The search returned a partial self-alignment for MUC16 (7,248 of 14,507 aa),
explicitly flagged in the report; all other 808 self-alignments were full length.
All-pairs candidate enumeration does not make a local aligner an oracle for all
biological relationships. Claims must refer to the specified detected-homology
graph, not to complete absence of related genes. Threshold or family-definition
changes require a new version and a separately labelled sensitivity experiment.

## Nested cross-validation

Use seed 42, target **five outer folds and three inner folds**. StratifiedKFold
serves the gene setting; StratifiedGroupKFold serves the grouped setting.
Both classes must occur in every training/validation/test partition, including
the paired subset. Check distinct groups containing each class, gene counts,
complete ID coverage, outer/inner disjointness and homology-group isolation.
Try fewer folds only if these checks fail; record the failure and reduction.
The completed cohort supports **5 × 3** in both settings without reduction.
Group-stratification is approximate; publish actual per-fold class counts.

For each outer fold:

1. Lock outer-test IDs. Do not use their labels, predictions, statistics or loss
   to choose preprocessing, model settings, epochs or thresholds.
2. Within outer training, fit each predeclared candidate in the three inner
   folds. Initialize each fit independently. Select by mean inner balanced
   accuracy at threshold 0.5; deterministic ties prefer the simpler candidate,
   then its predeclared candidate ID. Use identical candidate budgets and folds
   across comparisons. Freeze candidate lists before running this process.
3. For the selected candidate, collect exactly one out-of-fold prediction for
   each outer-training gene from its inner-held-out fold. Set the decision
   threshold to maximize balanced accuracy on these predictions only. Ties
   prefer closest to 0.5, then the smaller threshold. No probability calibration
   is included in the primary protocol. Add it only as an inner-fitted candidate.
4. Refit from fresh pristine initialization on all outer-training genes. Refit
   preprocessing and weights there; retain the chosen threshold. For neural
   models, use the median inner-selected epoch count rounded upward, with no
   outer-test early stopping. Evaluate outer-test genes once.

Do not continue weights, optimizer/scheduler state, supervised embeddings or
fitted preprocessing from another fold. Unsupervised upstream pretraining may
include human reference sequences; document that separately from supervised
exposure to this task's labels.

## Fitting and gene boundaries

| Operation | Permitted fitting data |
| --- | --- |
| Deterministic reference extraction, homology graph and split construction | Whole cohort, before model development; no performance-based threshold choices |
| Frozen pretrained encoder inference | Any partition independently; encoder must be unchanged and auditable |
| Learned vocabulary, imputation, scaling, feature selection, dimensionality reduction | Current inner-training genes; final refit uses outer-training genes |
| Class weights, resampling or learned augmentation | Current training genes only, counting one label per gene |
| Hyperparameters, early stopping and decision threshold | Inner training/validation or complete inner OOF predictions |
| Final performance reporting | Outer test only after all choices above are fixed |

Every transcript, window, reverse complement and augmentation inherits its
audited gene ID. Missing/unknown IDs abort assignment. Augmentation is applied
after splitting and only to training examples. Pool windows as specified in the
sequence policy to **one prediction per gene**. If fine-tuning over windows,
aggregate into one gene loss or explicitly normalize each gene's total loss
weight; more windows must not increase its labelled sample weight.

`scripts/evaluation_protocol.py` supplies partition lookup, disjointness checks,
derived-row assignment, training-only transformer fitting, gene-count class
weights, inner-OOF ID/threshold checks and an initialization-record guard.
They are reusable guards, not a completed training runner. ID checks cannot
prove that a caller's claimed OOF model was actually fitted correctly: retain
fit-gene IDs, initialization hashes and preprocessing lineage for every fit.

## Metrics, comparisons and uncertainty

Predeclare balanced accuracy as primary, OG as positive. Report per-outer-fold
and mean/SD balanced accuracy; also report pooled outer OOF AUROC, average
precision for each class (including class prevalence), MCC, macro-F1,
per-class precision/recall, confusion counts, Brier score and the selected
thresholds. Scores and confusion matrices count genes, never windows. Keep the
two evaluation settings separate and include class-prior and simple composition/
length baselines using the same folds and fitting boundaries.

Archive one outer OOF record per gene/model/setting/cohort with true label,
probability, fold, group, selected candidate, threshold and binary prediction.
Compare models on identical genes with paired differences. Use 2,000 seeded
paired bootstrap resamples of genes in the stratified setting and entire
homology components in the grouped setting, recomputing pooled metrics and
differences. Report percentile 95% intervals, discard/resample draws lacking a
class, and document effective bootstrap unit counts. These intervals describe
OOF-sample uncertainty conditional on the fits; they do not capture all training
variability. Five fold scores are not five independent experiments, and fold SD
is not a confidence interval. Report all predeclared comparisons; avoid treating
outer-fold model ranking as a new unbiased tuning set.

## Checkpoint decision

The [checkpoint audit](../evaluation/v1/checkpoint_audit.json) classifies
`models/output/checkpoint-916` as **exposure history unverifiable**, with
`benchmark_allowed: false`. No weights, trainer state, split IDs or training logs
were recovered. Historical train/test files are empty, and the legacy loader
uses `test.tsv` as its evaluation dataset. This establishes uncertainty, not
proof of contamination or proof of a clean checkpoint.

Restart every benchmark fit from a documented **pristine upstream pretrained
checkpoint** with immutable revision, configuration/tokenizer/weight SHA-256
hashes, fresh task head and fresh optimizer. Do not use `checkpoint-916`, even
on the new split. No replacement model has been trained or downloaded here;
the [benchmark plan](../evaluation/benchmark_plan.json) leaves initialization and
candidate configuration unresolved, so training remains blocked until those
concrete records are completed. The initialization guard validates required
record structure; model-card and artifact provenance still require review.

## Reproduction

First reproduce the label and sequence artifacts using their documented commands.
Install the separate split environment: `python -m pip install -r requirements-evaluation.txt`.
Download the official [MMseqs2 18-8cc5c SSE2 archive](https://github.com/soedinglab/MMseqs2/releases/download/18-8cc5c/mmseqs-linux-sse2.tar.gz)
(archive SHA-256 `7f660ed8257adec1ddce885bb8a6acbe6acd703692146825e9fa9e9be06902af`).

```bash
python scripts/prepare_homology.py \
  --sequence-dir data/sequence_inputs/ensembl114-v1 \
  --cache data/sequence_cache/ensembl114 \
  --output data/evaluation/homology-trace-v1 \
  --mmseqs /absolute/path/to/mmseqs/bin/mmseqs
python scripts/prepare_evaluation.py \
  --sequence-dir data/sequence_inputs/ensembl114-v1 \
  --homology-dir data/evaluation/homology-trace-v1 \
  --output evaluation/reproduced-v1
python scripts/audit_checkpoint.py \
  --revision 161c6d7abc28f2680f0f5f385a96ca177d90c61f \
  --output evaluation/reproduced-v1/checkpoint_audit.json
python -m unittest discover -s tests -v
```

Use `--offline` for cached Ensembl replay. Compare split IDs/hashes against the
committed manifest. MMseqs2 multithreaded row ordering and runtime log paths/times
can vary; group construction sorts and deduplicates edges. Raw artifact hashes
and local paths document the actual run, rather than implying logs must replay
byte-for-byte. Existing split files reject any changed manifest; create a new
version directory for a changed protocol.
