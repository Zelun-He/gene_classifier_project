# Audit of the recovered original exports

Audit performed on 2026-10-23 using `exclusive-cancer-roles-v1`. These are real
records recovered from the user's original repository, not synthetic fixtures
or newly invented annotations. They are a historical snapshot, not a claim about
the current contents of either database.

## Recovery and provenance

Both original files were present in the repository's
[initial commit](https://github.com/Zelun-He/gene_classifier_project/commit/0b3b43e04fa7b613e6eec9634669cb5f05c3563f),
whose recorded author date is 2025-05-22. They were absent from the latest tree
but remain recoverable from Git objects. Recovery preserves exact bytes and
verifies these SHA-256 hashes:

| Source | Original path | Data records | SHA-256 |
| --- | --- | ---: | --- |
| COSMIC | `gene_classfifier_project/Census_allThu May 22 21_50_23 2025.csv` | 758 | `5f5fbe6e68533e9f771680eb1473f6242913993f2429891c9bf48af363b7d9de` |
| OncoKB | `gene_classfifier_project/cancerGeneList.tsv` | 1,192 | `802e957942bbfbe73127824e9b1841a4973bfa978741f81b99bd8ec4353f3994` |

The database release versions and original download dates are not established
by the files or their commit metadata. The audit records
`release=unreported-original-export` and `retrieved_on=unknown` for both, with
immutable source URLs. The COSMIC filename is not promoted into a verified
download date. OncoKB's `COSMIC CGC (v99)` column describes membership in a source
resource; it does not establish the release of either recovered export.

## Cohort results

The 1,950 source records resolve into 1,292 identity groups under the documented
symbol/identifier matching policy.

| Classification | Groups | Included in binary cohort |
| --- | ---: | ---: |
| Oncogene only | 447 | 447 |
| Tumor suppressor only | 369 | 369 |
| Explicit dual role, without another unresolved issue | 125 | 0 |
| Unresolved annotations or identities | 64 | 0 |
| No oncogene/TSG annotation | 287 | 0 |
| **Total** | **1,292** | **816** |

The resulting binary cohort contains **447 OG and 369 TSG genes**, with source-
provided stable IDs. The 476 excluded groups remain in the full manifest and
exclusion tables. These are eligibility decisions under policy v1, not
independently validated biological ground truth.

Among the 64 unresolved groups, 21 have opposing exclusive role annotations and
43 have missing COSMIC role annotations. One of those 43 also has conflicting
Entrez IDs. Exclusion reasons overlap and must not be summed as distinct genes.
Two groups lack a stable ID; they are already excluded in the categories above.

Source support for the eligible labels is:

| Label support | Eligible genes |
| --- | ---: |
| COSMIC and OncoKB both assign the exclusive label | 303 |
| COSMIC supplies the label without matching OncoKB label support | 167 |
| OncoKB supplies the label without matching COSMIC label support | 346 |

The 816-gene cohort is an exclusive-role **union**, not a consensus-only cohort.
The 303 genes with matching label support from both sources form a useful later
sensitivity analysis. Source presence alone is not counted as agreement.

## Data issues exposed

- The original OncoKB file uses `Yes`/`No` strings. These need explicit parsing;
  the previous script compared the values directly with the boolean `True`.
- Nine identity groups merge different source symbols through shared Entrez
  IDs, including `CARS`/`CARS1` and `H3F3A`/`H3-3A`. The manifest preserves all
  symbols; it does not assert that the selected display symbol is approved.
- `EIF1AX` has conflicting source-provided Entrez IDs (`107984923` and `1964`).
  The group is quarantined rather than silently choosing one.
- The recovered source sizes are 758 and 1,192 records; historical documentation
  listed 760 and 1,194. The old claim of 579 combined labels is not the result of
  this new audit and must not be carried forward as the audited cohort size.

No historical labels were silently promoted into the training pipeline. Existing
processed splits and model scores remain unverified.

## Reproduce

Use a full Git clone with the initial commit available. From the repository root:

```bash
python scripts/recover_label_sources.py \
  --output-dir data/label_sources/initial-commit

python scripts/prepare_gene_labels.py \
  --cosmic 'data/label_sources/initial-commit/Census_allThu May 22 21_50_23 2025.csv' \
  --oncokb data/label_sources/initial-commit/cancerGeneList.tsv \
  --cosmic-release unreported-original-export \
  --cosmic-retrieved-on unknown \
  --cosmic-provenance 'https://github.com/Zelun-He/gene_classifier_project/blob/0b3b43e04fa7b613e6eec9634669cb5f05c3563f/gene_classfifier_project/Census_allThu%20May%2022%2021_50_23%202025.csv' \
  --oncokb-release unreported-original-export \
  --oncokb-retrieved-on unknown \
  --oncokb-provenance 'https://github.com/Zelun-He/gene_classifier_project/blob/0b3b43e04fa7b613e6eec9634669cb5f05c3563f/gene_classfifier_project/cancerGeneList.tsv' \
  --output-dir data/label_audit/original-exports-2026-10-23

python -m unittest discover -s tests -v
```

Use new directories if a previous run already exists. Generated raw exports and
full audit files stay untracked. The committed
[machine-readable summary](label_audit_results.json) records source hashes,
policy, counts, metadata gaps, script hash, and hashes of every output table.

Verification completed: 33 automated tests pass. The real-data result was also
reconciled independently by Entrez ID against the original source role fields
for all 816 eligible genes; alias symbols were matched by identifier. Output
hashes and the audit script hash were checked against the generated summary.
