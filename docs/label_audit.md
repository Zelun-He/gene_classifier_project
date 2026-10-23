# Auditable gene labels

This is the first research milestone: construct an explicit, reproducible binary
label cohort before sequence selection or model comparison. The original COSMIC
and OncoKB exports are not included in this repository. Historical dataset sizes
and model scores have not been independently reproduced.

## Inputs and provenance

The audit runs offline with Python 3.10+ and the standard library. It reads a
COSMIC Cancer Gene Census CSV and an OncoKB Cancer Gene List TSV. Acquire and
retain the original exports under the databases' applicable access terms; do not
commit them or the resulting real-data audit to this public repository.

| Source | Required columns | Recognized optional stable ID columns |
| --- | --- | --- |
| COSMIC CSV | `Gene Symbol`, `Role in Cancer` | `Entrez GeneId`, `Entrez Gene ID`, `Entrez ID`, `Ensembl Gene ID`, `Ensembl GeneId`, `Ensembl ID` |
| OncoKB TSV | `Hugo Symbol`, `Is Oncogene`, `Is Tumor Suppressor Gene` | Same stable ID variants |

Stable ID header matching ignores case and repeated whitespace. Other required
headers retain their spelling. UTF-8 with or without a BOM is supported.

Supply the exact release or documented snapshot identifier and actual retrieval
date for each export. A filename date is not evidence of the database release.
If a release is unavailable, explicitly record that fact in a snapshot identifier
(for example, `release-unreported-local-snapshot-2025-05-22`); do not invent a
release. The SHA-256 of the original bytes identifies the precise input anyway.
The audit records metadata as supplied; it does not authenticate release claims.

```bash
python scripts/prepare_gene_labels.py \
  --cosmic 'data/Census_allThu May 22 21_50_23 2025.csv' \
  --oncokb data/cancerGeneList.tsv \
  --cosmic-release '<actual COSMIC release or documented snapshot>' \
  --cosmic-retrieved-on '<actual YYYY-MM-DD>' \
  --oncokb-release '<actual OncoKB release or documented snapshot>' \
  --oncokb-retrieved-on '<actual YYYY-MM-DD>' \
  --output-dir data/label_audit/run-001
```

Use a new output directory for every run. Existing directories are refused, and
validation errors do not publish partial audit files. Inputs are never changed.

## Classification policy: exclusive-cancer-roles-v1

Every data record is preserved, including duplicate rows, genes with no target
role, and additional source columns. No database wins merely by appearing first.

| Category | Rule | Binary benchmark |
| --- | --- | --- |
| `oncogene_only` | At least one exclusive OG annotation; no opposing, dual, or incomplete annotation | `OG` if identity is resolved |
| `tumor_suppressor_only` | At least one exclusive TSG annotation; no opposing, dual, or incomplete annotation | `TSG` if identity is resolved |
| `dual_role` | A source explicitly assigns both roles, without another unresolved issue | Excluded |
| `unresolved` | Opposing exclusive annotations, conflicting stable IDs, missing role values, or unknown role tokens | Excluded |
| `unannotated` | Only fusion annotations and/or explicit false/false OncoKB flags | Excluded |

The fifth category is an audit bucket for genes outside the binary task; it is
not a negative class. A gene present in only one source may be included if its
role is exclusive and it has a source-provided stable ID. An explicit no-target-
role annotation in the other source is not treated as an opposing role.
`included_source_presence` and `included_label_support` are reported separately;
appearing in both sources is not the same as both sources assigning the label.
Database agreement is not independent experimental validation.

Two opposing exclusive observations are a disagreement even if they occur
within one source or alongside a third, dual-role observation. A dual-role
observation plus a single-role observation is conservatively excluded as dual.
Missing or unknown annotations take precedence over dual and exclusive labels.
Every triggering issue is retained in `exclusion_reasons`; raw records remain
available when multiple conditions apply.

### Parsing and identity

- COSMIC roles are matched as complete, case-insensitive comma-, semicolon-, or
  pipe-separated tokens: `oncogene`/`OG`/`ONC`, `TSG`/`tumor suppressor`/`tumour
  suppressor` (with optional `gene`), and `fusion`. Unknown tokens are quarantined.
- OncoKB flags accept true/false, yes/no, and 1/0, case-insensitively. Blank,
  NA, N/A, nan, none, null, and `-` mean unknown, not false. Other values fail the
  run with a source record number, preventing silent label corruption.
- Symbols are stripped and uppercased for matching; original strings remain in
  the raw records. Unsupported or absent symbols, malformed IDs, malformed CSV
  rows, duplicate headers, missing required columns, and empty inputs fail the run.
- Records connect through shared normalized symbols or source-provided Entrez
  or Ensembl gene IDs. Alias symbols with a shared ID can therefore merge.
  No fuzzy symbol matching, external alias lookup, or automatic ID reassignment
  is performed. Distinct symbols without a shared identifier remain separate.
- Entrez IDs are positive integers; integer-valued exports such as `123.0` are
  accepted. Ensembl IDs must be human `ENSG` gene IDs; version suffixes are
  normalized away. Zero and empty-like ID values mean missing. Raw IDs survive.
- An entire connected group is excluded if it contains multiple IDs in the
  same namespace. Its `gene_id` starts with `ambiguous:`; all IDs remain visible.
  Such a group can represent several biological genes, so summary counts use
  **identity groups**, not an inflated assertion of unique biological genes.
- A source-provided Entrez or Ensembl ID is required for inclusion. Symbol-only
  records retain their role category but are excluded with `missing_stable_id`.
- The display symbol prefers the lexicographically first OncoKB symbol, then
  the first available symbol. This is a deterministic representative, not proof
  of the currently approved gene symbol. Later sequence mapping must verify it.

## Outputs

| File | Contents |
| --- | --- |
| `source_annotations.jsonl` | Every original record, parsed roles, issues, normalized identifiers, source, and record number |
| `gene_manifest.csv` | Every identity group, all symbols and IDs, source role sets, category, candidate label, inclusion flag, exclusion reasons, and links to original records |
| `gene_labels.csv` | Only eligible genes, using the legacy `gene,label` schema with `OG` and `TSG` |
| `excluded_genes.csv` | All exclusions, including symbol-only exclusive genes |
| `dual_role_genes.csv` | Groups classified as dual role |
| `unresolved_genes.csv` | Groups with conflicting identities/roles or incomplete annotations |
| `summary.json` | Release/snapshot metadata, retrieval dates, input hashes, input row counts, per-source quality counts, category and class totals, exclusion reasons, support counts, policy version, script hash, and output hashes |

CSV list/dictionary cells contain JSON. `record_ids` such as `cosmic:1` refer to
one-based **data record numbers**, not physical lines (quoted records can span
multiple lines). Each observation belongs to exactly one manifest group. Candidate
`label` values alone are not eligibility; consume only `included=True`, or use
the filtered `gene_labels.csv`.

Outputs are deterministically ordered. Identical input bytes, metadata, and
script produce identical output bytes. Reordering input rows changes their
provenance IDs and hashes, but not the classification decisions.

## Synthetic demonstration and verification

The fixtures use invented `SYNTH_*` symbols and IDs and are **not biological data**.
The dates below are demonstration metadata, not database retrieval claims.

```bash
python scripts/prepare_gene_labels.py \
  --cosmic tests/fixtures/cosmic_synthetic.csv \
  --oncokb tests/fixtures/oncokb_synthetic.tsv \
  --cosmic-release synthetic-v1 --cosmic-retrieved-on 2026-10-23 \
  --oncokb-release synthetic-v1 --oncokb-retrieved-on 2026-10-23 \
  --dataset-kind synthetic \
  --output-dir data/label_audit/synthetic-demo

python -m unittest discover -s tests -v
```

Expected fixture result: 20 source records, 11 identity groups, **5 included
genes (3 OG, 2 TSG)**, and 6 excluded groups. These counts verify the software's
rules only. The tests cover conflicting source labels, explicit dual roles,
missingness, duplicate observations, alias merging, transitive ID collisions,
source presence versus label support, invalid input rejection, output integrity,
and deterministic reruns. They also run the CLI from another directory with
third-party site packages disabled.

## Before using real labels downstream

Review the real-data summary, manifest, and exclusions. Verify source-provided
identities against a fixed nomenclature/reference release before sequence work.
Resolve disputed cases using documented evidence in a future versioned policy;
do not edit generated CSV labels by hand. There is no manual override mechanism
in v1.

After review, copy the **research** audit's filtered `gene_labels.csv` to the
legacy `data/gene_labels.csv` location used by `fetch_sequences.py` and
`make_datasets.py`. Retain its associated audit directory. The script deliberately
does not promote labels or overwrite the old cohort automatically. Regenerate
downstream datasets: existing train/test files are not valid evidence for the
new cohort. Sequence policy, reproducible splitting, and model repairs remain
separate research milestones.
