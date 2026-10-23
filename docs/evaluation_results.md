# Fixed-fold gene classification results

809 genes; five outer folds and three inner folds. Primary metric: mean outer-fold balanced accuracy (%). Brackets are conditional 95% bootstrap intervals.

| Model | Stratified BA %, 95% interval | Similarity-grouped BA %, 95% interval |
| --- | ---: | ---: |
| Class prior | 50.00 [50.00, 50.00] | 50.00 [50.00, 50.00] |
| Length/GC logistic | 55.75 [52.60, 58.98] | 55.18 [51.51, 58.62] |
| 3-mer logistic | 57.79 [54.41, 61.08] | 56.50 [53.01, 59.94] |
| 6-mer logistic | 59.95 [56.54, 63.17] | 54.47 [50.87, 57.68] |

All transformations and class weights are fitted on training genes only. Candidates use inner validation; decision thresholds use complete selected-candidate inner OOF predictions. Every outer gene has one prediction per model, regardless of its number of sequence windows.

Uncertainty uses 2,000 paired gene resamples (stratified) or complete detected-homology-component resamples (grouped). Intervals condition on the fitted models and thresholds; they do not include full retraining or source-label uncertainty. The analyses are exploratory. Inspect paired differences rather than treating overlapping marginal intervals as a test.

Evidence in the run directory:

- `outer_predictions.csv`: gene IDs, labels, probabilities, predictions, groups, folds, selected candidate and threshold.
- `partitions.json`: exact training, inner-validation and outer-test gene IDs, including all 40 partition definitions.
- `fits.json`: each fit's partition key, training-ID hash, training counts, weights and parameter/scaler hashes.
- `selected_inner_oof.csv`, `candidate_scores.json`: threshold and configuration selection evidence.
- `plan.json`, `run.json`, `features.json`, `summary.json`: fixed configurations, input/source/output identities and full metrics.

Run receipt SHA-256: `f90fec9af29bbc7aa3f75bf8e1db49d7475fd368e5540ef6a759cc0371700fe3`. `checkpoint-916` is excluded. The baseline sequence features use complete CDS directly; the frozen encoder pools real tokens within windows and length-weights all windows into one vector per gene.
