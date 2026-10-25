# Sequence-input coverage: original audited cohort

Generated on 2026-10-23 using **GRCh38.p14, Ensembl 114** and policy
[`gene-cds-mane-canonical-v1`](sequence_policy.md). Input: the 816 eligible genes
from the [original-export label audit](label_audit_results.md). These are retrieved
reference sequences, not synthetic examples. No classifier was trained.

The [machine-readable report](sequence_coverage.json) records parameters, input
and code hashes, class-specific attrition, sequence statistics, and output hashes.

## Coverage and experimental units

| Input | Genes | OG | TSG | Regions | Model windows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Complete spliced CDS | **809** | 441 | 368 | 809 | 4,389 |
| Fixed promoter | 809 | 441 | 368 | 809 | 4,045 |
| Sampled genomic control | 799 | 435 | 364 | 2,397 | 11,985 |

The primary CDS baseline retains **809/816 genes (99.1%)**. There are **799
genes (435 OG, 364 TSG)** with all three usable inputs. Region comparisons must
use that paired intersection and identical gene-grouped folds.

The 20,419 windows are computational chunks. They do not increase the number of
labelled genes. Across three separate input experiments there are 2,417 pooled
gene/input representations: 809 CDS, 809 promoter, and 799 genomic. Each model
experiment produces one prediction per gene, not one per window.

## Transcript selection and sequence quality

Among retained CDS genes, **808 use MANE Select** and **one uses Ensembl
Canonical** (`MUC16`). No longest-translation fallback was needed. CDS sequences
are spliced, omit UTRs/introns, and are in transcript 5′→3′ orientation.

| CDS statistic | Result |
| --- | ---: |
| Minimum length | 135 bp |
| Median length | 1,764 bp |
| Mean length | 2,516.5 bp |
| Maximum length | 43,524 bp |
| Total retained CDS | 2,035,842 bp |
| Ambiguous-base fraction in every retained input | 0.0 |
| Retained CDS longer than 515 bp | 770/809 (95.2%) |

The default ambiguity threshold was fixed at zero before extraction. `MYC` and
`VEGFA` have CTG starts in the selected archived CDS; they are retained with
`non_ATG_start` flags rather than discarded by an oversimplified start-codon
rule. All retained CDS lengths are divisible by three and agree with the
annotated translation length, allowing a terminal stop codon.

## Exclusions

| Reason | Genes | Count |
| --- | --- | ---: |
| No Entrez-to-Ensembl gene mapping in the pinned archive | TRA, TRB, TRD, TRG | 4 |
| Multiple primary coding loci linked to the audited identity | CRLF2, SSX4 | 2 |
| Non-ACGT bases in the selected CDS | RYBP | 1 |

The four unmapped records are retained in the coverage manifest as unavailable
for this transcript-based representation; no replacement sequence was invented.
Ambiguous primary mappings were not resolved by choosing the first result.
`RYBP`'s selected Canonical CDS contains two ambiguous bases in 657 bp
(fraction 0.00304414), so it fails the declared threshold. No different isoform
was chosen after observing that failure.

CDS retention by class is 441/447 OG (98.7%) and 368/369 TSG (99.7%). Exclusion
status is traceable per gene and must accompany later performance reporting.

All 809 retained CDS genes have a usable full 2,048 bp promoter. Ten genes have
a genomic span shorter than 2,048 bp and are excluded from the genomic control:
H1-3, H1-4, H1-5, ID3, EZHIP, TACSTD2, TAL2, H2AC17, H3C2, and SOCS1. The 799
remaining genes each have three sampled full-width genomic regions. No additional
control-region ambiguity or chromosome-boundary exclusion occurred.

## Coverage instead of prefix truncation

Every base of each retained CDS and control region is reconstructed exactly from
its nonoverlapping balanced model windows. No retained CDS tail is dropped.
Windows contain at most 510 raw bases and carry normalized pooling weights.

For scale, keeping only the first 515 bases of these CDS inputs would retain
**36.4% of bases per gene on average**, with a **29.2% median retention fraction**.
This is an unweighted average of per-gene fractions, not a fraction of the pooled
base total. It is a hypothetical CDS truncation analysis: the legacy code used
genomic sequence, so these figures do not estimate its actual coding coverage.

The genomic controls deliberately sample only part of each gene body. After
merging overlapping sampled intervals, median gene-span coverage is **10.7%**
(mean 20.2%; range 0.27%–99.5%). Summing overlapping interval lengths would
overstate coverage. Promoter coverage is complete for the specified 2,048 bp
window, not for all possible regulatory sequence.

Complete nucleotide coverage through local windows is different from global
long-range attention. The supplied pooling removes padding/special tokens,
weights windows by bases within regions, averages regions within gene/input
type, and classifies one pooled representation. Tissue-dependent isoforms and
long-range interactions remain outside this baseline.

## Verification

- **57 automated tests pass**, covering selection priority, stable tie-breaks,
  mapping ambiguity, strand coordinates, boundary handling, reproducible sampling,
  overlap-aware coverage, sequence QC, transcript versions, token masks, complete
  window coverage, per-gene pooling, missing vectors, and cache corruption.
- Complete cache replay with `--offline` reproduced **all eight output files
  byte-for-byte** in two fresh run directories.
- Independently reconstructed all **4,015 retained regions** from their 20,419
  windows; checked contiguous offsets, maximum window length, ACGT content, and
  equality to the complete FASTA sequences.
- Checked every generated artifact hash and both pipeline source-code hashes.
- Independently checked the negative-strand `BRCA1` promoter against the reverse
  complement of a forward-strand reference request. All cached archive responses
  also retained their requested archive host without cross-host redirection.
- A constant-vector pooling check produced exactly 2,417 gene/input vectors,
  preserving the input vectors rather than multiplying samples by window count.
- No identical complete CDS sequences were found among the 809 retained genes.
  This does **not** rule out homologous-sequence leakage; similarity/family-grouped
  evaluation is still required.

Use the commands and output schemas in the [sequence policy](sequence_policy.md).
Full manifests, sequences, and hashed API responses remain in ignored data
directories and are reproducible using the pinned archive and the recovery
procedure. The committed JSON provides the aggregate result and integrity hashes.

## Next evaluation boundary

Natural-length CDS, promoter, and sampled genomic inputs have different total
base/compute budgets. Their initial comparison must use paired genes and equal
model-selection procedures. A predeclared 2,048 bp-per-input ablation on an
eligible paired subset should accompany it to test that confound. Model training,
tokenizer-specific checks, similarity grouping, and performance claims are not
completed by this extraction milestone.
