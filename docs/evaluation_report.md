# Evaluation design and feasibility report

This report covers the design stage completed before model training. These
results describe the fixed cohort and partitions. Completed classical model
results are now in the [baseline report](baseline_results.md). See the [protocol](evaluation_protocol.md),
[machine-readable report](../evaluation/v1/report.json),
[frozen splits](../evaluation/v1/splits.json), and
[checkpoint audit](../evaluation/v1/checkpoint_audit.json).

## Cohort and homology

The 809 CDS genes comprise **441 OG and 368 TSG**. The paired three-input cohort
has **799 genes: 435 OG and 364 TSG**. Protein translations match the selected
Ensembl 114 transcripts. MMseqs2 18-8cc5c enumerated all **654,481 ordered pairs**;
10,739 alignments, including self-hits, passed its E-value filter. Identity was
recomputed from exact match counts and alignment columns, with bidirectional
coverage computed from inclusive aligned spans.

Sequence-only threshold sensitivity, with coverage ≥80% of each sequence and
E-value ≤1e-5 throughout:

| Minimum protein identity | Undirected edges | Components | Singletons | Largest component | Mixed-label components |
| --- | ---: | ---: | ---: | ---: | ---: |
| **30%: primary** | **205** | **673** | **586** | **13** | **16** |
| 40% | 143 | 699 | 616 | 7 | 13 |
| 50% | 101 | 735 | 681 | 7 | 6 |
| 70% | 42 | 773 | 744 | 3 | 3 |

At the primary threshold, 347 components contain OG genes and 342 contain TSG
genes. Mixed components count toward both totals. There is sufficient support
for five outer and three inner folds in both settings, including the paired
subset. No fold-count reduction was necessary.

The largest component contains RAB35, RRAS2, HRAS, KRAS, RHOH, NRAS, RAC1,
RAC2, RAP1B, RHEB, RIT1, RRAS and CDC42. Connected-component grouping intentionally
keeps transitively related genes together. A receptor component contains CSF1R,
FLT1, FLT3, FLT4, KDR, KIT, PDGFRA and PDGFRB.

The threshold defines a broad, near-full-length protein similarity challenge;
remote/domain-only relationships and direct promoter similarity are outside its
guarantee. One partial self-alignment is flagged: MUC16, 7,248 aligned residues
of 14,507. The next longest cohort protein is MUC4 at 5,412 aa. Candidate
enumeration is exhaustive, while the aligner's reported local alignments and
the specified graph remain the basis of the guarantee. The report retains this
limitation instead of treating all genes as proven unrelated across folds.

## Outer test folds

All indices are zero-based. Each gene appears in exactly one outer-test fold
per setting. Every inner/outer training and held-out partition contains both
classes, and the paired cohort inherits assignments rather than being resplit.

| Setting | Fold | OG | TSG | Total genes | Homology groups | Paired genes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stratified gene | 0 | 89 | 73 | 162 | 158 | 160 |
| Stratified gene | 1 | 88 | 74 | 162 | 153 | 161 |
| Stratified gene | 2 | 88 | 74 | 162 | 149 | 160 |
| Stratified gene | 3 | 88 | 74 | 162 | 153 | 160 |
| Stratified gene | 4 | 88 | 73 | 161 | 154 | 158 |
| Similarity grouped | 0 | 88 | 73 | 161 | 134 | 160 |
| Similarity grouped | 1 | 88 | 73 | 161 | 135 | 160 |
| Similarity grouped | 2 | 88 | 74 | 162 | 134 | 159 |
| Similarity grouped | 3 | 88 | 73 | 161 | 134 | 158 |
| Similarity grouped | 4 | 89 | 75 | 164 | 136 | 162 |

**165 of 205** qualifying homology edges cross outer-test fold assignments in
the stratified setting. **Zero** cross in the grouped setting; all group checks
also pass in its inner folds. This is a structural leakage opportunity in the
stratified design, not a measured amount of performance inflation.

Across inner-validation folds, individual class counts range from 98–118 in
the stratified design and 97–119 in the grouped design. Detailed counts are in
the JSON report. Fold assignments are reused across all models; threshold
sensitivity diagnostics have not been used to choose a threshold based on model
scores.

## Checkpoint audit and readiness

`checkpoint-916` is **ineligible**: its supervised exposure history is unknown.
The audited history at `161c6d7abc28f2680f0f5f385a96ca177d90c61f` contains
no checkpoint weights, trainer-state files, original split IDs or training logs.
Four historical train/test paths point to the same empty Git blob
`e69de29bb2d1d6434b8b29ae775ad8c2e48c5391`; they cannot establish test-gene
membership. The legacy generator discarded gene IDs and did not seed its
shuffle. The trainer loads the local checkpoint and uses `test.tsv` as its
evaluation dataset. The checkpoint directory is also absent locally.

This evidence cannot prove which test genes were encountered. The defensible
benchmark decision is to restart each fit from verified, revision-pinned pristine
pretraining. The fixed [benchmark plan](../evaluation/benchmark_plan.json) records
that requirement and the remaining training-runner work. No training has been
started and no replacement checkpoint has been selected implicitly.

Verification includes reproducible split generation, frozen artifact hashes,
transitive components, exact threshold boundaries, inner/outer gene and group
disjointness, paired-subset inheritance, derived-row identity checks,
training-only scaling/weights/threshold inputs, and replay of the historical
checkpoint audit. All **20,419 sequence windows** were also assigned by gene
without missing or crossing IDs in every outer/inner partition. Regenerating
the evaluation from the frozen search reproduces its output files byte-for-byte.
The full repository test suite passes **72 tests**.
