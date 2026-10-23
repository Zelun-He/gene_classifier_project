"""Retrieve and audit gene-level CDS and matched reference-sequence controls."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import tempfile

if __package__:
    from .ensembl_snapshot import ASSEMBLY, BIOMART, RELEASE, REST, SnapshotClient, digest
else:
    from ensembl_snapshot import ASSEMBLY, BIOMART, RELEASE, REST, SnapshotClient, digest


POLICY = "gene-cds-mane-canonical-v1"
CHROMOSOMES = {str(number) for number in range(1, 23)} | {"X", "Y"}
SEQUENCE_TYPES = ("cds", "promoter", "genomic")
UPSTREAM, DOWNSTREAM, REGION_BP, GENOMIC_SAMPLES = 1536, 512, 2048, 3
IUPAC = set("ACGTRYSWKMBDHVN")


def load_cohort(audit_dir):
    audit_dir = Path(audit_dir)
    summary = json.loads((audit_dir / "summary.json").read_text())
    if summary.get("dataset_kind") != "research":
        raise ValueError("A real research label audit is required")
    for filename, metadata in summary["artifacts"].items():
        if Path(filename).name != filename or digest((audit_dir / filename).read_bytes()) != metadata["sha256"]:
            raise ValueError(f"Label audit artifact failed integrity check: {filename}")
    if "gene_manifest.csv" not in summary["artifacts"]:
        raise ValueError("Label audit does not hash its gene manifest")
    with (audit_dir / "gene_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["included"] == "True"]
    genes, seen = [], set()
    for row in rows:
        if row["gene_id"] in seen or row["label"] not in {"OG", "TSG"}:
            raise ValueError("Duplicate gene identity or unexpected label in cohort")
        seen.add(row["gene_id"])
        entrez, ensembl = json.loads(row["entrez_ids"]), json.loads(row["ensembl_ids"])
        if len(entrez) > 1 or len(ensembl) > 1 or not (entrez or ensembl):
            raise ValueError("Cohort contains an unresolved identity")
        genes.append({"gene_id": row["gene_id"], "gene": row["gene"], "label": row["label"],
                      "entrez_id": entrez[0] if entrez else "", "provided_ensembl_id": ensembl[0] if ensembl else ""})
    if len(genes) != summary["counts"]["included_genes"] or not genes:
        raise ValueError("Cohort size disagrees with label audit")
    return sorted(genes, key=lambda row: row["gene_id"])


def choose_gene(candidate_ids, lookup):
    objects = [lookup[identifier] for identifier in sorted(candidate_ids) if lookup.get(identifier)]
    primary = [gene for gene in objects if gene.get("species") == "homo_sapiens"
               and gene.get("assembly_name") == "GRCh38" and gene.get("seq_region_name") in CHROMOSOMES]
    coding = [gene for gene in primary if gene.get("biotype") == "protein_coding"]
    if len(coding) == 1:
        return coding[0], ""
    if len(coding) > 1:
        return None, "ambiguous_primary_coding_gene_mapping"
    if primary:
        return None, "no_primary_protein_coding_gene"
    return None, "no_primary_gene_mapping" if objects else "no_ensembl_gene_mapping"


def choose_transcript(gene):
    transcripts = [transcript for transcript in gene.get("Transcript", [])
                   if transcript.get("biotype") == "protein_coding"
                   and transcript.get("Translation", {}).get("length", 0) > 0
                   and transcript.get("Parent") == gene["id"]
                   and transcript.get("strand") == gene["strand"]
                   and transcript.get("seq_region_name") == gene["seq_region_name"]
                   and transcript.get("assembly_name") == "GRCh38"]
    if not transcripts:
        return None, "no_coding_transcript"

    def mane(transcript):
        return any(item.get("type") == "MANE_Select" for item in transcript.get("MANE", []))

    def canonical(transcript):
        return bool(transcript.get("is_canonical")) or transcript["id"] == gene.get("canonical_transcript", "").split(".")[0]

    chosen = sorted(transcripts, key=lambda transcript: (
        not mane(transcript), not canonical(transcript), -transcript["Translation"]["length"], transcript["id"]
    ))[0]
    return chosen, "MANE_Select" if mane(chosen) else ("Ensembl_Canonical" if canonical(chosen) else "longest_translation")


def promoter_coordinates(transcript, chromosome_length):
    strand = transcript["strand"]
    if strand not in {-1, 1}:
        raise ValueError("Invalid transcript strand")
    tss = transcript["start"] if strand == 1 else transcript["end"]
    start, end = (tss - UPSTREAM, tss + DOWNSTREAM - 1) if strand == 1 else (tss - DOWNSTREAM + 1, tss + UPSTREAM)
    if start < 1 or end > chromosome_length:
        return None
    return start, end, strand


def genomic_coordinates(gene, gene_id, seed):
    choices = gene["end"] - gene["start"] + 1 - REGION_BP + 1
    if choices < 1:
        return []
    generator = random.Random(int(hashlib.sha256(f"{POLICY}:{seed}:{gene_id}".encode()).hexdigest(), 16))
    offsets = sorted(generator.sample(range(choices), min(GENOMIC_SAMPLES, choices)))
    return [(gene["start"] + offset, gene["start"] + offset + REGION_BP - 1, gene["strand"]) for offset in offsets]


def union_length(intervals):
    end, total = -1, 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end + 1) + 1)
        end = max(end, stop)
    return total


def sequence_quality(sequence, max_ambiguous):
    sequence = sequence.upper()
    length = len(sequence)
    ambiguous = sum(base not in "ACGT" for base in sequence)
    stats = {"length_bp": length, "ambiguous_bp": ambiguous,
             "ambiguous_fraction": ambiguous / length if length else None,
             "gc_fraction": sum(base in "GC" for base in sequence) / length if length else None}
    if not length:
        return stats, "empty_sequence"
    if set(sequence) - IUPAC:
        return stats, "non_iupac_sequence"
    if stats["ambiguous_fraction"] > max_ambiguous:
        return stats, "ambiguous_sequence"
    return stats, ""


def cds_quality(sequence, protein_length, max_ambiguous):
    stats, reason = sequence_quality(sequence, max_ambiguous)
    flags = []
    if not sequence.startswith("ATG"):
        flags.append("non_ATG_start")
    if sequence[-3:] not in {"TAA", "TAG", "TGA"}:
        flags.append("no_terminal_stop")
    stats["qc_flags"] = flags
    if reason:
        return stats, reason
    if len(sequence) % 3:
        return stats, "cds_length_not_multiple_of_three"
    if len(sequence) not in {3 * protein_length, 3 * protein_length + 3}:
        return stats, "cds_translation_length_mismatch"
    return stats, ""


def make_windows(regions, window_bp):
    if not 1 <= window_bp <= 510:
        raise ValueError("Window length must be 1..510 bp to fit a conservative 512-token budget")
    grouped = defaultdict(list)
    for region in regions:
        grouped[(region["gene_id"], region["sequence_type"])].append(region)
    windows = []
    for group, members in sorted(grouped.items()):
        for region in sorted(members, key=lambda item: item["region_id"]):
            sequence = region["sequence"]
            if not sequence:
                raise ValueError("Cannot window an empty sequence")
            count = math.ceil(len(sequence) / window_bp)
            for index in range(count):
                # Balance chunks so a long sequence does not end in a tiny,
                # mostly padding window; every base is still covered once.
                start = index * len(sequence) // count
                end = (index + 1) * len(sequence) // count
                windows.append({
                    "window_id": f"{region['region_id']}|w{index}", "region_id": region["region_id"],
                    "gene_id": group[0], "sequence_type": group[1], "label": region["label"],
                    "start_0based": start, "end_0based_exclusive": end, "length_bp": end - start,
                    "pool_weight": (end - start) / (len(sequence) * len(members)),
                    "sequence": sequence[start:end],
                })
    return windows


def mean_token_embedding(token_vectors, attention_mask, special_tokens_mask):
    """Mean real token vectors, excluding padding and special tokens."""
    if not (len(token_vectors) == len(attention_mask) == len(special_tokens_mask)):
        raise ValueError("Token vectors and masks must align")
    if any(value not in {0, 1} for value in attention_mask + special_tokens_mask):
        raise ValueError("Masks must contain only 0/1")
    real = [vector for vector, attention, special in zip(token_vectors, attention_mask, special_tokens_mask) if attention and not special]
    if not real or not real[0] or any(len(vector) != len(real[0]) for vector in real):
        raise ValueError("No real tokens or inconsistent vector dimensions")
    if any(not math.isfinite(value) for vector in real for value in vector):
        raise ValueError("Nonfinite real token embedding")
    return [sum(vector[index] for vector in real) / len(real) for index in range(len(real[0]))]


def pool_gene_embeddings(windows, embeddings):
    """Pool frozen window embeddings BEFORE classification; one vector per gene/type.

    All expected windows are required. A gene with many windows has the same one
    sample weight as every other gene; these rows are not training examples.
    """
    identifiers = [row["window_id"] for row in windows]
    if len(set(identifiers)) != len(identifiers) or set(identifiers) != set(embeddings):
        raise ValueError("Embeddings must match unique window IDs exactly")
    groups = defaultdict(list)
    dimension = None
    for row in windows:
        vector = embeddings[row["window_id"]]
        if not vector or any(not math.isfinite(value) for value in vector):
            raise ValueError("Embeddings must be nonempty finite vectors")
        dimension = len(vector) if dimension is None else dimension
        if len(vector) != dimension or not math.isfinite(row["pool_weight"]) or row["pool_weight"] <= 0:
            raise ValueError("Invalid embedding dimension or pooling weight")
        groups[(row["gene_id"], row["sequence_type"])].append(row)
    result = []
    for (gene_id, sequence_type), rows in sorted(groups.items()):
        if len({row["label"] for row in rows}) != 1 or not math.isclose(sum(row["pool_weight"] for row in rows), 1, abs_tol=1e-10):
            raise ValueError("Inconsistent gene labels or incomplete pooling weights")
        vector = [sum(row["pool_weight"] * embeddings[row["window_id"]][index] for row in rows) for index in range(dimension)]
        result.append({"gene_id": gene_id, "sequence_type": sequence_type, "label": rows[0]["label"], "embedding": vector})
    return result


def distribution(values):
    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {"min": min(values), "median": statistics.median(values), "mean": statistics.mean(values), "max": max(values)}


def write_csv(path, rows, fields=None):
    fields = fields or sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value for key, value in row.items()} for row in rows)


def prepare(audit_dir, output_dir, client, window_bp=510, seed=42, max_ambiguous=0.0):
    if not 1 <= window_bp <= 510 or not 0 <= max_ambiguous <= 1:
        raise ValueError("Invalid window size or ambiguous-base threshold")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError(f"Output directory already exists: {output_dir}")
    cohort = load_cohort(audit_dir)
    reference, assembly = client.reference(refresh=True)
    chromosome_lengths = {row["name"]: row["length"] for row in assembly["top_level_region"] if row["name"] in CHROMOSOMES}
    if set(chromosome_lengths) != CHROMOSOMES:
        raise ValueError("Reference is missing nuclear chromosome lengths")
    print(f"Mapping {len(cohort)} audited genes against Ensembl {RELEASE}, {ASSEMBLY}...", flush=True)
    mapping = client.map_entrez(sorted({row["entrez_id"] for row in cohort if row["entrez_id"]}))
    candidates = {}
    for row in cohort:
        ids = set(mapping.get(row["entrez_id"], set())) if row["entrez_id"] else {row["provided_ensembl_id"]}
        if row["provided_ensembl_id"]:
            ids &= {row["provided_ensembl_id"]}
        candidates[row["gene_id"]] = sorted(ids)
    lookup = client.lookup_genes(sorted({identifier for ids in candidates.values() for identifier in ids}))
    selections, objects, candidate_transcripts = [], {}, []
    for row in cohort:
        result = dict(row, assembly=ASSEMBLY, ensembl_release=RELEASE, status="excluded", exclusion_reason="",
                      candidate_ensembl_ids=candidates[row["gene_id"]])
        gene, reason = choose_gene(candidates[row["gene_id"]], lookup)
        if gene is None:
            result["exclusion_reason"] = reason
        else:
            transcript, rule = choose_transcript(gene)
            result.update({"ensembl_gene_id": gene["id"], "ensembl_gene_version": gene["version"],
                           "chromosome": gene["seq_region_name"], "strand": gene["strand"],
                           "gene_start_1based": gene["start"], "gene_end_1based": gene["end"]})
            if transcript is None:
                result["exclusion_reason"] = rule
            else:
                result.update({"transcript_id": transcript["id"], "transcript_version": transcript["version"],
                               "selection_rule": rule, "protein_length_aa": transcript["Translation"]["length"],
                               "transcript_start_1based": transcript["start"], "transcript_end_1based": transcript["end"],
                               "status": "selected"})
                objects[row["gene_id"]] = (gene, transcript)
                for candidate in gene.get("Transcript", []):
                    candidate_transcripts.append({"gene_id": row["gene_id"], "transcript_id": candidate["id"],
                        "version": candidate.get("version"), "biotype": candidate.get("biotype"),
                        "is_canonical": candidate.get("is_canonical", 0), "MANE": candidate.get("MANE", []),
                        "protein_length_aa": candidate.get("Translation", {}).get("length"),
                        "selected": candidate["id"] == transcript["id"]})
        selections.append(result)
    # A single Ensembl locus must not become two independently labelled genes.
    by_locus = defaultdict(list)
    for row in selections:
        if row["status"] == "selected":
            by_locus[row["ensembl_gene_id"]].append(row)
    for rows in by_locus.values():
        if len(rows) > 1:
            for row in rows:
                row.update(status="excluded", exclusion_reason="ensembl_locus_shared_by_audited_ids")
    chosen = [row for row in selections if row["status"] == "selected"]
    print(f"Retrieving CDS for {len(chosen)} selected transcripts...", flush=True)
    cds_records = client.sequences(sorted({row["transcript_id"] for row in chosen}), "cds")
    regions, region_audit, control_specs = [], [], []
    for row in chosen:
        record = cds_records.get(row["transcript_id"])
        if record is None:
            row.update(status="excluded", exclusion_reason="cds_not_returned")
            continue
        if record.get("version") != row["transcript_version"] or record.get("molecule") != "dna":
            raise ValueError(f"CDS version/type mismatch: {row['transcript_id']}")
        sequence = record["seq"].upper()
        stats, reason = cds_quality(sequence, row["protein_length_aa"], max_ambiguous)
        row.update({"cds_" + key: value for key, value in stats.items()})
        if reason:
            row.update(status="excluded", exclusion_reason=reason)
            continue
        row["status"] = "retained"
        gene, transcript = objects[row["gene_id"]]
        base = {key: row[key] for key in ("gene_id", "gene", "label", "assembly", "ensembl_release", "ensembl_gene_id",
                                         "ensembl_gene_version", "transcript_id", "transcript_version", "chromosome", "strand")}
        cds = dict(base, region_id=f"{row['gene_id']}|cds|r0", sequence_type="cds", sequence=sequence, **stats)
        regions.append(cds)
        region_audit.append({key: value for key, value in dict(cds, status="retained").items() if key != "sequence"})
        promoter = promoter_coordinates(transcript, chromosome_lengths[gene["seq_region_name"]])
        row["promoter_status"] = "pending" if promoter else "excluded_chromosome_boundary"
        genomic = genomic_coordinates(gene, row["gene_id"], seed)
        row["genomic_status"] = "pending" if genomic else "excluded_gene_shorter_than_2048bp"
        for kind, intervals in (("promoter", [promoter] if promoter else []), ("genomic", genomic)):
            for index, (start, end, strand) in enumerate(intervals):
                control_specs.append(dict(base, sequence_type=kind, region_id=f"{row['gene_id']}|{kind}|r{index}",
                    start_1based=start, end_1based=end, query=f"{gene['seq_region_name']}:{start}..{end}:{strand}"))
    print(f"Retrieving {len(control_specs)} promoter/genomic control regions...", flush=True)
    controls = client.sequences(sorted({row["query"] for row in control_specs}), "genomic")
    grouped_controls = defaultdict(list)
    for spec in control_specs:
        record = controls.get(spec["query"])
        stats, reason = sequence_quality(record["seq"].upper() if record else "", max_ambiguous)
        if record is None:
            reason = "region_not_returned"
        elif record.get("molecule") != "dna":
            raise ValueError("Non-DNA genomic response")
        elif stats["length_bp"] != REGION_BP:
            reason = "region_length_mismatch"
        item = dict(spec, **stats, status="excluded" if reason else "retained", exclusion_reason=reason)
        if not reason:
            item["sequence"] = record["seq"].upper()
        grouped_controls[(item["gene_id"], item["sequence_type"])].append(item)
    selection_by_id = {row["gene_id"]: row for row in selections}
    for (gene_id, kind), members in grouped_controls.items():
        failed = any(item["status"] != "retained" for item in members)
        selection_by_id[gene_id][kind + "_status"] = "excluded_region_qc" if failed else "retained"
        for item in members:
            if failed and item["status"] == "retained":
                item.update(status="excluded", exclusion_reason="another_region_in_gene_failed_qc")
            if not failed:
                regions.append(item)
            region_audit.append({key: value for key, value in item.items() if key != "sequence"})
        if kind == "genomic" and not failed:
            gene, _ = objects[gene_id]
            coverage = union_length([(item["start_1based"], item["end_1based"]) for item in members])
            selection_by_id[gene_id]["genomic_union_coverage_bp"] = coverage
            selection_by_id[gene_id]["genomic_gene_span_fraction"] = coverage / (gene["end"] - gene["start"] + 1)
    windows = make_windows(regions, window_bp)
    # Reference checks bracket the entire run; network failure aborts publication.
    if client.reference(refresh=True) != (reference, assembly):
        raise ValueError("Reference metadata changed during retrieval")
    retained = [row for row in selections if row["status"] == "retained"]
    summary = {
        "schema_version": 1, "policy_version": POLICY, "ensembl_release": RELEASE, "assembly": ASSEMBLY,
        "rest_archive": REST, "biomart_archive": BIOMART,
        "assembly_accession": assembly.get("assembly_accession"),
        "label_audit_summary_sha256": digest((Path(audit_dir) / "summary.json").read_bytes()),
        "script_sha256": {path.name: digest(path.read_bytes()) for path in (Path(__file__), Path(__file__).with_name("ensembl_snapshot.py"))},
        "parameters": {"window_bp": window_bp, "seed": seed, "max_ambiguous_fraction": max_ambiguous,
                       "python_version": ".".join(map(str, sys.version_info[:3])),
                       "promoter_upstream_bp": UPSTREAM, "promoter_downstream_bp": DOWNSTREAM,
                       "genomic_region_bp": REGION_BP, "genomic_regions_per_gene_max": GENOMIC_SAMPLES},
        "input_genes": len(cohort), "retained_cds_genes": len(retained),
        "excluded_genes": len(selections) - len(retained),
        "excluded_reasons": dict(sorted(Counter(row["exclusion_reason"] for row in selections if row["status"] == "excluded").items())),
        "selection_rules": dict(sorted(Counter(row["selection_rule"] for row in retained).items())),
        "per_class": {label: {"input": sum(row["label"] == label for row in selections), "retained_cds": sum(row["label"] == label for row in retained)} for label in ("OG", "TSG")},
        "cds_qc_flags": dict(sorted(Counter(flag for row in retained for flag in row["cds_qc_flags"]).items())),
        "cds_longer_than_515bp": sum(row["cds_length_bp"] > 515 for row in retained),
        "hypothetical_cds_prefix_515bp_retention_fraction": distribution([min(515, row["cds_length_bp"]) / row["cds_length_bp"] for row in retained]),
        "genomic_sample_span_coverage_fraction": distribution([row["genomic_gene_span_fraction"] for row in retained if "genomic_gene_span_fraction" in row]),
        "inputs": {},
    }
    gene_sets = {}
    for kind in SEQUENCE_TYPES:
        subset = [region for region in regions if region["sequence_type"] == kind]
        subset_windows = [window for window in windows if window["sequence_type"] == kind]
        gene_sets[kind] = {region["gene_id"] for region in subset}
        summary["inputs"][kind] = {
            "genes": len(gene_sets[kind]), "regions": len(subset), "windows": len(subset_windows),
            "per_class": {label: len({region["gene_id"] for region in subset if region["label"] == label}) for label in ("OG", "TSG")},
            "region_length_bp": distribution([region["length_bp"] for region in subset]),
            "ambiguous_fraction": distribution([region["ambiguous_fraction"] for region in subset]),
            "total_region_bp": sum(region["length_bp"] for region in subset),
        }
        if kind != "cds":
            summary["inputs"][kind]["excluded_relative_to_cds"] = dict(sorted(Counter(
                row[kind + "_status"] for row in retained if row[kind + "_status"] != "retained"
            ).items()))
    paired = set.intersection(*gene_sets.values())
    summary["paired_all_inputs_genes"] = len(paired)
    summary["paired_all_inputs_per_class"] = {label: sum(row["gene_id"] in paired and row["label"] == label for row in retained) for label in ("OG", "TSG")}
    summary["biological_sample_unit"] = "gene_id; never individual windows"
    summary["pooling"] = "mean non-padding/non-special token embeddings within window; base-length-weighted windows within region; equal regions within gene/type; classify one pooled vector"
    summary["limitations"] = ["No classifier is trained by this pipeline.", "Noncoding genes and non-nuclear/alternative loci are outside this CDS baseline.",
        "Canonical/longest fallbacks and CDS start/stop flags require sensitivity analyses.", "Promoter and genomic comparisons require paired genes and identical gene-grouped splits.",
        "Genomic windows may overlap and sample only part of a gene span.", "515bp retention is hypothetical for CDS; the legacy pipeline truncated genomic rather than CDS sequence."]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sequence-inputs-", dir=output_dir.parent) as temp:
        staged = Path(temp) / "inputs"
        staged.mkdir()
        write_csv(staged / "gene_coverage.csv", selections)
        write_csv(staged / "region_manifest.csv", region_audit, None if region_audit else ["gene_id", "sequence_type", "region_id", "status", "exclusion_reason"])
        write_csv(staged / "paired_genes.csv", [{"gene_id": row["gene_id"], "gene": row["gene"], "label": row["label"]} for row in retained if row["gene_id"] in paired], ["gene_id", "gene", "label"])
        for filename, items in (("transcript_candidates.jsonl", candidate_transcripts), ("windows.jsonl", windows)):
            with (staged / filename).open("w", encoding="utf-8", newline="\n") as handle:
                for item in items:
                    handle.write(json.dumps(item, sort_keys=True) + "\n")
        with (staged / "sequences.fasta").open("w", encoding="utf-8", newline="\n") as handle:
            for region in regions:
                handle.write(f">{region['region_id']}\n{region['sequence']}\n")
        (staged / "response_provenance.json").write_text(json.dumps(client.used, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary["artifacts"] = {path.name: {"sha256": digest(path.read_bytes()), "size_bytes": path.stat().st_size} for path in sorted(staged.iterdir())}
        (staged / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        staged.rename(output_dir)
    print(f"Saved {len(retained)} CDS genes, {len(windows)} grouped windows; {len(paired)} genes have all three inputs.", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/sequence_cache/ensembl114"))
    parser.add_argument("--offline", action="store_true", help="Replay hashed cached responses only")
    parser.add_argument("--window-bp", type=int, default=510)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-ambiguous-fraction", type=float, default=0.0)
    args = parser.parse_args(argv)
    try:
        prepare(args.audit_dir, args.output_dir, SnapshotClient(args.cache_dir, args.offline), args.window_bp, args.seed, args.max_ambiguous_fraction)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"Sequence preparation failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
