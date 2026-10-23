# Gene sequence policy: gene-cds-mane-canonical-v1

This policy defines a protein-coding baseline for the audited OG/TSG cohort and
two reference-sequence controls. The biological sample is always an audited
**gene ID**. Transcripts, regions, reverse strands, and model windows are not
independent labelled samples.

## Fixed reference and gene identity

- Species: `homo_sapiens`.
- Assembly: **GRCh38.p14**, assembly accession `GCA_000001405.29`.
- Annotation: **Ensembl release 114**, May 2025 archive. This is an explicit
  experimental reference choice, not an inferred version of the label databases.
- REST server: `https://may2025.rest.ensembl.org`.
- BioMart: `https://may2025.archive.ensembl.org/biomart/martservice`.
- Reference release/assembly are checked before and after extraction. Changed
  metadata aborts the run. Every successful response is cached with its request
  and SHA-256 for offline replay. Network failures abort publication of the run;
  they are not converted into biological exclusions.

The input is the integrity-checked label audit's complete `gene_manifest.csv`,
filtered by `included=True`. Source-provided Entrez IDs are mapped through the
same archived Ensembl BioMart. Display names and historical aliases are not used
to guess an Ensembl ID. If an audited Ensembl ID is also present, it must agree
with that mapping. Ensembl-only audited IDs can be looked up directly.

Only a uniquely mapped protein-coding locus on nuclear chromosomes 1–22, X, or
Y qualifies. Alternative scaffolds, mitochondrial loci, noncoding genes, absent
mappings, and multiple primary coding mappings are excluded with distinct
reasons. If two audited IDs select the same Ensembl locus, both are quarantined;
they cannot become duplicate or oppositely labelled examples. These exclusions
define the scope of this CDS baseline, not the validity of the original labels.

## One transcript per gene

Candidates must be protein-coding transcripts with a positive-length translation
and matching parent gene, chromosome, assembly, and strand. Rank candidates by:

1. `MANE_Select` annotation.
2. Ensembl Canonical annotation or the gene's canonical transcript ID.
3. Longest translation length, equivalent to longest translated CDS before any
   terminal stop codon. UTR/transcript length is not a surrogate for CDS length.
4. Lexicographically smallest stable transcript ID as the final tie-break.

The first candidate is selected before fetching its CDS. No alternate transcript
is substituted after observing a sequence quality failure. All considered
transcripts, versions, MANE evidence, canonical status, translation lengths, and
the selection decision are retained in `transcript_candidates.jsonl`.

MANE Select is a biologically motivated first choice because it identifies
matching Ensembl/RefSeq transcripts. Canonical and longest-CDS fallbacks preserve
coverage where MANE is unavailable; their effects should be examined separately.
This is not tissue-specific isoform selection.

## Primary input: complete spliced CDS

Fetch `type=cds` for the selected transcript, in transcript 5′→3′ orientation.
Ensembl defines CDS here as the spliced transcript sequence without UTRs.
Introns and UTRs are absent. Negative-strand CDS is already oriented by the API
and must not be reverse-complemented a second time.

Require the returned transcript version to match the selection. Record the
entire sequence's length, ambiguous-base count/fraction, GC fraction, and hash
through the output artifact manifest. Require nonempty IUPAC DNA, length divisible
by three, and a length equal to three times the annotated protein length, with
or without one terminal stop codon. The default maximum ambiguous fraction is
**0.0**: sequences containing any non-ACGT base are reported and excluded, never
silently imputed. Threshold changes are explicit run parameters.

Non-ATG starts and absent terminal stops are reported as flags rather than
automatically interpreted as errors. Alternative initiation and annotation
conventions exist. Internal stop codons are not used as a blanket exclusion
without accounting for biological recoding. No protein-function or variant-effect
inference is claimed from these sequence checks.

## Follow-up inputs

These controls are extracted for genes that pass the CDS baseline, and use its
selected transcript to define the promoter. A failed control does not remove
that gene from the CDS baseline; it removes it from the relevant paired analysis.

### Promoter: 2,048 bp around the selected transcript's TSS

Use transcription-relative offsets **[-1536, +512)**, including the TSS base.
Coordinates sent to Ensembl are 1-based inclusive:

| Transcript strand | TSS | Genomic interval | Returned orientation |
| --- | --- | --- | --- |
| +1 | transcript start | `[TSS−1536, TSS+511]` | +1 |
| −1 | transcript end | `[TSS−511, TSS+1536]` | −1 |

The negative-strand request returns the reverse complement in transcriptional
orientation. Require the full 2,048 bp interval inside the chromosome; do not pad
or clip chromosome boundaries. This is a fixed experimental window, not a claim
to capture every gene's complete regulatory region. It can overlap neighbouring
genes or other regulatory elements.

### Genomic control: up to three 2,048 bp gene-body regions

Sample distinct start positions without replacement inside the annotated gene
span, allowing the sampled intervals themselves to overlap. Derive a per-gene
random seed from SHA-256 of `policy_version:seed:audited_gene_id`; the run seed
defaults to 42. Sampling is independent of labels and input ordering. Record the
Python version used by the deterministic sampler; the reported run uses Python
3.12.14. Selected coordinates, not a seed alone, are preserved in the manifest.

If the gene span is shorter than 2,048 bp, exclude this control. If fewer than
three distinct full-width starts exist, use all available starts. Each interval
is requested in gene-strand orientation and checked for length and ambiguity.
If any sampled region fails, exclude the complete genomic input for that gene
rather than silently changing its sampling budget. Report genomic span coverage
using the **union** of intervals, not their summed lengths.

## Windows and one prediction per gene

Use nonoverlapping balanced chunks with at most **510 raw bases**. For a sequence
of length L, create `ceil(L/510)` chunks with integer boundaries
`floor(i*L/n)` and `floor((i+1)*L/n)`. This avoids tiny trailing chunks and covers
every retained base exactly once. A long CDS is never reduced to its prefix.

The 510 bp budget conservatively fits a 512-token model with two special tokens
when DNA tokenization uses no more than one token per base; a proper 6-mer
tokenizer also fits. Use the actual pretrained model's tokenizer, request its
special-token mask, verify encoded length, and **raise an error instead of
truncating**. DNABERT-2 takes raw DNA; original DNABERT requires its matching
k-mer preparation. This pipeline does not reuse the legacy custom vocabulary.

The provided standard-library pooling helpers support frozen embeddings:

1. `mean_token_embedding`: mean real token vectors, excluding padding and special
   tokens using the attention and special-token masks.
2. Within each region, weight window vectors by their actual base lengths.
3. Within each gene/input type, average region vectors equally. CDS and promoter
   each have one region; genomic controls have one to three regions.
4. Apply a classifier to **one pooled vector per gene per experiment**.

`windows.jsonl` stores normalized `pool_weight` values implementing steps 2–3.
`pool_gene_embeddings` rejects missing/extra/duplicate window IDs, inconsistent
dimensions or labels, nonfinite vectors, and incomplete weight totals. Pass the
full manifest and plain Python lists of model embeddings/masks to the helpers.
For differentiable fine-tuning, the same pooling rule must be implemented inside
the model graph and the loss reduced per gene; the supplied helper is for frozen
features, not an autograd implementation.

This gives complete nucleotide coverage with local context. Mean pooling does
not introduce cross-window attention, preserve full transcript order, or recreate
one global long-context forward pass. Window count and sequence length remain
possible confounders and should be checked with simple baselines.

## Controlled comparisons and data splitting

`paired_genes.csv` contains the intersection with all three usable inputs. Use
that common cohort, identical gene-grouped outer/inner splits, the same classifier
and tuning budget, and the same evaluation metrics for region comparisons.
Every window, transcript, sequence type, and augmentation from a gene must follow
that gene into the same partition. Similarity/family grouping remains the next
evaluation milestone; this extraction does not create train/test splits.

Full CDS, a 2,048 bp promoter, and up to 6,144 sampled genomic bases have different
information and compute budgets. Their natural-length comparison alone cannot
attribute performance differences solely to region biology. Predeclare a
secondary 2,048 bp-per-gene comparison on paired genes with at least 2,048 CDS
bases, taking one label-independent sampled CDS interval, the fixed promoter,
and one sampled genomic interval. Use the same chunk budget and include length/
GC controls. That budget-matched ablation and model training are follow-up
experiments, not results of the present coverage report.

## Reproduce and outputs

First recover and audit the original labels as described in
[the label audit results](label_audit_results.md). Then run:

```bash
python scripts/prepare_sequence_inputs.py \
  --audit-dir data/label_audit/original-exports-2026-10-23 \
  --output-dir data/sequence_inputs/ensembl114-v1

# A new output directory; no API requests when replaying a complete cache.
python scripts/prepare_sequence_inputs.py \
  --audit-dir data/label_audit/original-exports-2026-10-23 \
  --output-dir data/sequence_inputs/ensembl114-offline \
  --offline

python -m unittest discover -s tests -v
```

| Output | Purpose |
| --- | --- |
| `gene_coverage.csv` | Every audited gene, mapping candidates, selection/exclusion, gene and transcript versions, assembly/release, strand, coordinates, CDS QC, control status and genomic coverage |
| `transcript_candidates.jsonl` | All transcript candidates for selected loci and selection evidence |
| `region_manifest.csv` | CDS and control region lengths, ambiguity/GC, reference coordinates, orientation, status and failure reason |
| `sequences.fasta` | Complete retained CDS and control-region sequences |
| `windows.jsonl` | Model windows with parent region, gene ID, label, offset, raw DNA and pooling weight |
| `paired_genes.csv` | Common gene cohort for the three-input comparison |
| `response_provenance.json` | Request URLs/bodies and hashes of the archived responses used |
| `summary.json` | Aggregate coverage, class-specific attrition, policy/parameters, source and code hashes, and output hashes |

Raw responses and generated data remain in ignored directories. Existing outputs
are never overwritten. The legacy `fetch_sequences.py`, `make_datasets.py`, and
training scripts do not consume these files automatically. No classifier or
biological interpretation is validated by sequence retrieval alone.

## Source documentation

- [Ensembl MANE](https://static.ensembl.org/info/genome/genebuild/mane.html)
- [Ensembl Canonical transcript selection](https://static.ensembl.org/info/genome/genebuild/canonical.html)
- [Archive releases](https://www.ensembl.org/help/articles/archives)
- [Expanded gene lookup](https://rest.ensembl.org/documentation/info/lookup)
- [CDS sequence endpoint](https://rest.ensembl.org/documentation/info/sequence_id_post)
- [Region sequence endpoint](https://rest.ensembl.org/documentation/info/sequence_region_post)
