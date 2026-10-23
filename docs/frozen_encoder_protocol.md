# Frozen DNABERT-6 comparison, version 1

This experiment compares a **frozen pretrained encoder plus a linear classifier**
with all four completed classical baselines. It does not fine-tune the DNA
encoder. The [candidate plan](../evaluation/frozen_encoder_plan_v1.json) fixes
the model, final layer, pooling, numerical runtime and classifier grid before
task fitting. No layer, pooling or checkpoint search is conducted on outer scores.

## Initialization evidence

The [DNABERT authors' repository](https://github.com/jerryji1993/DNABERT#update-20250708)
links `zhihan1996/DNA_bert_6` as the pretrained 6-mer distribution, separately
from their task-finetuned models. Pin the original complete 2021 revision:
**`a79a8fd96ad172f964a4dbef3f4d7545a5034baa`**.
Its configuration declares `BertForMaskedLM`, with no finetuning task. The
original weight LFS SHA-256 is
`e1688c3a3f881daff02a42b487e04612c4ec9ef33f1f02fa5f4e67d7b9865ca9`;
the same weight identity occurs in the author-linked repository's 2025-06-30
revision `c56e67ea5827e0ddc67ef059addcf71569b1216e`.

The [initialization record](../evaluation/dnabert6_initialization_v1.json) records
all configuration/tokenizer/weight hashes and original Git blob/LFS identities.
The download script checks sizes and those identities before publishing each
local artifact. The model weights remain untracked and are reproducible from
the immutable upstream revision.

Load locally with standard `BertConfig`, `BertModel` and `BertTokenizer` classes.
No upstream custom Python is executed. Use `torch.load(..., weights_only=True)`
and strict state loading: every encoder tensor must be present and shape-compatible.
Only the MLM prediction head is discarded. The original pooler is loaded if
present but its output is not used. Any historical position/token-type buffers
require explicit validation, and any other unexpected tensor is an error.

Pristine status follows author distribution and immutable artifact provenance;
it is not an independent reconstruction of the authors' training ledger.
DNABERT was pretrained on human reference sequence, so benchmark genes may have
appeared **without these task labels** during pretraining. The grouped evaluation
tests supervised transfer outside the detected homology groups; it is not a
claim of unseen sequence during upstream pretraining. `checkpoint-916` remains
ineligible and is never loaded.

## Tokenization and representation

Reuse the exact 4,389 CDS windows for the same 809 genes. Windows are balanced,
contiguous and nonoverlapping, at most 510 raw bases, with no first-prefix
truncation. Validate that they reconstruct every complete selected CDS and that
their saved raw-length pooling weights agree with the full region length.

Convert each window of length L to its **L−5 overlapping 6-mers**, separated by
spaces for the original tokenizer. Require each k-mer to map to one known
vocabulary ID. Verify all 4,096 A/C/G/T 6-mers; no vocabulary is learned from the
task. Compare the tokenizer's output against those exact IDs plus `[CLS]` and
`[SEP]`. Disable truncation; the maximum input has **505 real tokens plus two
special tokens**, within the 512-position model limit. Padding is applied only
within an inference batch. Count unknown tokens and abort if any occur.

Use the last encoder layer, mean pooling over real tokens only, excluding both
padding and special tokens. Pool window vectors into one gene vector using the
existing window-length weights. Accumulate in float64, then store the gene
matrix in float32. There is no L2 normalization or learned pooling. The result
is one **768-dimensional vector per gene**. All genes have equal labelled sample
weight regardless of window count.

This covers every base using bounded local contexts. It does not provide
attention across an entire long gene. The five 6-mers spanning each raw-window
boundary are not formed, whereas the whole-CDS classical k-mer baselines count
them. This is an explicit property of the existing nonoverlapping window policy.

## Inference and caching

CPU float32 inference uses PyTorch **2.9.1+cpu**, Transformers **4.57.1**, standard
BERT scaled-dot-product attention, evaluation mode, disabled gradients and seed
42. Use eight threads and batches of eight; these computational settings were
chosen using label-free timing probes, not classifier scores. Record every
parameter's combined hash before and after extraction, and require unchanged
parameters. A separate short-sequence-alone versus padded-batch check verifies
padding invariance to maximum absolute error 5e-5.

Save batch blocks with checksums. Resume only with identical inputs, plan,
extractor/integrity-helper source and numerical library versions; reject altered
blocks. Extraction does not fit a vocabulary, transform, encoder or label-dependent
operation, so the same immutable gene matrix can serve every evaluation fold.

## Task fitting and comparisons

Use the identical five outer × three inner folds and full CDS cohort from the
[evaluation protocol](evaluation_protocol.md). The head is fresh L2 logistic
regression with training-only StandardScaler and gene-count-balanced training
weights. Use **C = 0.01, 0.1, 1**, the same solver, tolerance, seed and candidate
selection/threshold rules as the classical baselines. No PCA or feature selection
is performed. No encoder state, head, scaler or optimizer is continued between fits.

Compare against **all four fixed baseline predictions**, without refitting or
selecting a baseline using the new outer results. Primary: mean outer-fold
balanced accuracy. Secondary: the same pooled gene metrics and per-class reports
already specified. Use 2,000 seeded paired gene/component bootstrap draws with
all five models sharing the same draws within each setting. Report every model
comparison; do not select a layer, pooling strategy or fine-tuning schedule from
these outer outcomes. Bootstrap intervals remain conditional on the fitted heads
and thresholds and do not capture upstream or full training variability.

## Reproduce extraction

Use a separate Python environment so the original baseline runtime remains
intact. Pinned versions are intentional reproducibility choices, not a claim to
be the newest available releases.

```bash
python -m pip install -r requirements-embeddings.txt
python scripts/prepare_pretrained_checkpoint.py \
  --output models/dnabert6-pretrained-2021 \
  --record evaluation/dnabert6_initialization_v1.json
python scripts/extract_frozen_embeddings.py \
  --sequence-dir data/sequence_inputs/ensembl114-v1 \
  --model-dir models/dnabert6-pretrained-2021 \
  --output data/evaluation/dnabert6-embeddings-v1
python -m unittest discover -s tests -v
```

`--offline` on checkpoint preparation checks existing artifacts without network
access. Reusing the same extraction directory resumes checked batches. Gene
embeddings and extraction metadata will be archived with the completed comparison;
the 359 MB upstream weights need not be committed to the project.
