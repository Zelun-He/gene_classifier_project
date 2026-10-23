# First nested baseline benchmark: full CDS

**Simple sequence features show modest OG/TSG signal. The strongest stratified
result does not carry over unchanged to the grouped challenge, and the paired
intervals do not identify a clear winner among the feature baselines.**

These are completed model results on the fixed **809-gene CDS cohort**: 441 OG
and 368 TSG, GRCh38.p14 / Ensembl 114. Each gene supplies one complete-CDS feature
vector and one outer out-of-fold prediction per model and setting. No pretrained
DNA encoder or `checkpoint-916` was used.

The [plan and runner were published before the real-data fits](https://github.com/Zelun-He/gene_classifier_project/commit/460fb4031ec4b379ff19b8eff9e4a8b40f0705e5).
The [predeclared candidate plan](../evaluation/baseline_plan_v1.json),
[fixed split protocol](evaluation_protocol.md), and
[run receipt with code/input/output hashes](../evaluation/baselines-v1/run.json)
identify the actual experiment. The candidate grid was not expanded or retuned
after inspecting these results.

## Models and selection

| Model | Features | Candidates |
| --- | --- | --- |
| Class prior | Empirical OG prevalence in the current training genes; no sequence information | One |
| Length/GC | Natural log of full CDS length and GC fraction | L2 logistic regression, C = 0.01, 0.1, 1 |
| 3-mer | All 64 overlapping 3-mer frequencies across the full CDS | Same three C values |
| 6-mer | All 4,096 overlapping 6-mer frequencies across the full CDS | Same three C values |

K-mers use a fixed complete A/C/G/T vocabulary in transcript orientation; counts
are divided by `L−k+1`. They include positions crossing the earlier model-window
boundaries. Reverse complements are not collapsed. No features are selected,
no vocabulary is learned, and no augmentation or calibration is performed.

Every logistic fit uses a fresh StandardScaler fitted only on its training
genes and gene-count-balanced class weights derived from those same genes.
LIBLINEAR L2 regression uses tolerance 1e-6, at most 3,000 iterations, seed 42,
and one computational thread. Nonconvergence aborts the run. No fit failed.

All models reuse the frozen **five outer × three inner** assignments. Select C
by mean inner balanced accuracy at threshold 0.5, breaking ties toward lower C.
For that candidate, choose the threshold using complete inner OOF predictions,
then refit on outer-training genes and evaluate the outer test once. The class
prior also follows the inner threshold rule; it predicts a constant class within
each outer test fold and therefore has fold balanced accuracy 0.5.

## Results

Primary metric: unweighted **mean outer-fold balanced accuracy**. Brackets are
conditional 95% bootstrap intervals. AUROC is pooled across outer OOF genes.
All values are gene-level; windows do not increase sample size.

| Model | Stratified BA %, 95% interval | Grouped BA %, 95% interval | Stratified AUROC | Grouped AUROC |
| --- | ---: | ---: | ---: | ---: |
| Class prior | 50.00 [50.00, 50.00] | 50.00 [50.00, 50.00] | 0.4975 | 0.4982 |
| Length/GC | 55.75 [52.60, 58.98] | 55.18 [51.51, 58.62] | 0.5783 | 0.5691 |
| 3-mer | 57.79 [54.41, 61.08] | 56.50 [53.01, 59.94] | 0.5850 | 0.5954 |
| 6-mer | 59.95 [56.54, 63.17] | 54.47 [50.87, 57.68] | 0.6431 | 0.5901 |

Pooled class-prior AUROC can differ slightly from 0.5 because training prevalence,
and therefore its constant score, varies between outer folds. Its AUROC within
each outer fold is 0.5. The primary mean is distinct from pooled OOF balanced
accuracy; both are archived, along with fold SD, macro-F1, MCC, per-class
precision/recall, confusion counts, Brier score, average precision and prevalence.

The 6-mer result is **5.48 percentage points lower** in the grouped setting;
the 3-mer difference is 1.29 points. These are descriptive differences between
two evaluation challenges with different partitions and tuning outcomes. They
do not isolate a causal amount of similarity-driven inflation.

The following paired differences use identical bootstrap draws for each model
within a setting. Units are percentage points of mean outer balanced accuracy;
the direction is first model minus second. All predeclared model pairs are shown.

| Pair | Stratified difference, 95% interval | Grouped difference, 95% interval |
| --- | ---: | ---: |
| Prior − 3-mer | −7.79 [−11.08, −4.41] | −6.50 [−9.94, −3.01] |
| Prior − 6-mer | −9.95 [−13.17, −6.54] | −4.47 [−7.68, −0.87] |
| Prior − length/GC | −5.75 [−8.98, −2.60] | −5.18 [−8.62, −1.51] |
| 3-mer − 6-mer | −2.16 [−5.82, 1.46] | 2.03 [−2.19, 6.36] |
| 3-mer − length/GC | 2.04 [−2.04, 6.17] | 1.32 [−3.20, 5.81] |
| 6-mer − length/GC | 4.20 [−0.11, 8.41] | −0.71 [−5.46, 3.77] |

## Uncertainty and interpretation

Use 2,000 seeded percentile bootstrap resamples: 809 gene units in the stratified
setting and **673 complete homology-component units** in the grouped setting.
Sample units uniformly with replacement and retain all genes of each selected
component, including mixed-label groups. Recompute the mean fold balanced
accuracy and pooled metrics per draw; share draws across models. Discard draws
lacking a class in any fold; no draws needed rejection in this run.

The intervals condition on the fitted models, selected candidates and thresholds.
They do not include refitting variability, all uncertainty from the small cohort,
source-label uncertainty, or an independent external test. They are exploratory
and are not adjusted for multiple comparisons. Fold SD is not a confidence interval.

The grouped 3-mer/6-mer paired interval includes zero, as do all grouped
feature-versus-feature differences. The stronger stratified 6-mer result therefore
does not establish that it is a better approach to the grouped task. Even
length/GC carries signal, so future pretrained models must be compared against
composition/length baselines rather than only a class prior. These results do
not establish a cancer mechanism or a clinical predictor.

Grouping isolates the predeclared detected protein-homology graph; remote,
domain-only and promoter similarities can remain. The previously documented
partial MUC16 self-alignment remains a limitation. No grouping threshold,
transcript policy, class eligibility or outer assignment was changed for this run.

## Saved evidence and verification

The [complete result directory](../evaluation/baselines-v1) contains:

- `outer_predictions.csv`: **6,472** records = 809 genes × 4 models × 2 settings;
  includes true class, probability, prediction, group, outer fold, candidate,
  selected threshold and fitted-model ID.
- `selected_inner_oof.csv`: **25,888** records used for threshold selection,
  linked to their excluded-gene inner fits.
- `partitions.json` and `fits.json`: **40** exact training/validation/test ID
  sets and **340** fit records (300 inner fits, 40 final outer refits), with
  training class counts, weights, scaler/parameter hashes and convergence counts.
- `candidate_scores.json`: all candidates' inner scores and selection decisions.
- `features.json`, `plan.json`, `run.json`, `summary.json`: feature identities,
  immutable plan, source/artifact hashes, full metrics and all paired intervals.
- `audit.json`: independent verification outcome.

The audit reconstructs all features and each training partition's scaler
statistics, checks every fit's class weights and IDs, recomputes thresholds from
saved inner OOF records, and compares every pooled/per-fold metric against
scikit-learn. All 2,000 bootstrap resamples in both settings were replayed exactly.
Synthetic tests also change held-out features/labels and verify that the
corresponding training fit and selection remain unchanged. The complete
repository suite passes **87 tests**. Classifier weight
files are not archived; rerun the fixed fits to regenerate fitted estimators.

## Reproduce

Use Python 3.11+ with the pinned evaluation dependencies. First reproduce the
label/sequence artifacts as documented in their guides. A completed output
directory is never silently overwritten.

```bash
python -m pip install -r requirements-evaluation.txt
python scripts/run_baselines.py \
  --sequence-dir data/sequence_inputs/ensembl114-v1 \
  --output data/evaluation/cds-baselines-reproduced
python scripts/verify_baseline_run.py \
  --run-dir data/evaluation/cds-baselines-reproduced \
  --sequence-dir data/sequence_inputs/ensembl114-v1 --bootstrap
python -m unittest discover -s tests -v
```

Next: predeclare and evaluate a frozen pretrained DNA encoder plus a linear
head using these same folds. Pristine initialization, tokenizer compatibility,
window pooling and the neural training runner still require implementation.
Keep these baseline results fixed; do not use outer results to expand the C
grid or choose features and then call the resulting comparison confirmatory.
