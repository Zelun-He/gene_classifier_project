"""Freeze nested evaluation partitions and report feasibility; never train models."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import platform

import numpy
import scipy
import sklearn

from evaluation_protocol import POLICY, components, nested_splits, partition_ids, similarity_edges


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def group_stats(groups, labels):
    members = defaultdict(list)
    for gene, group in groups.items():
        members[group].append(gene)
    return {"groups": len(members), "singleton_groups": sum(len(g) == 1 for g in members.values()),
            "largest_group": max(map(len, members.values())),
            "mixed_label_groups": sum(len({labels[g] for g in genes}) > 1 for genes in members.values()),
            "groups_containing_class": {c: len({groups[g] for g in labels if labels[g] == c}) for c in ("OG", "TSG")},
            "size_histogram": dict(sorted(Counter(map(len, members.values())).items()))}


def prepare(sequence_dir, homology_dir, output):
    sequence_dir, homology_dir, output = map(Path, (sequence_dir, homology_dir, output))
    output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((sequence_dir / "summary.json").read_text())
    provenance = json.loads((homology_dir / "homology_provenance.json").read_text())
    if provenance["sequence_summary_sha256"] != sha(sequence_dir / "summary.json"):
        raise ValueError("Homology was computed from a different sequence cohort")
    for directory, hashes in ((sequence_dir, summary["artifacts"]), (homology_dir, provenance["artifacts"])):
        for name, expected in hashes.items():
            expected = expected["sha256"] if isinstance(expected, dict) else expected
            if sha(directory / name) != expected:
                raise ValueError(f"Artifact hash mismatch: {name}")
    with (sequence_dir / "gene_coverage.csv").open() as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "retained"]
    if len({r["gene_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate genes in coverage report")
    labels = {r["gene_id"]: r["label"] for r in rows}
    paired = sorted(r["gene_id"] for r in rows if r["promoter_status"] == r["genomic_status"] == "retained")
    protein_manifest = json.loads((homology_dir / "protein_manifest.json").read_text())
    search_to_gene = {r["search_id"]: r["gene_id"] for r in protein_manifest}
    if len(search_to_gene) != len(rows) or set(search_to_gene.values()) != set(labels):
        raise ValueError("Protein and gene cohorts differ")
    lines = (homology_dir / "alignments.tsv").read_text().splitlines()
    self_hits, partial_self_hits = set(), []
    for line in lines:
        c = line.split("\t")
        if c[0] == c[1] and int(c[2]) == int(c[3]) and int(c[2]) > 0:
            self_hits.add(c[0])
            if int(c[3]) != int(c[6]):
                partial_self_hits.append({"gene_id": search_to_gene[c[0]], "aligned_aa": int(c[3]),
                                          "sequence_length_aa": int(c[6])})
    if self_hits != set(search_to_gene):
        raise ValueError("Search lacks identity-verified self-alignments; check -a 1 identity output")
    sensitivity, primary_edges, groups = [], None, None
    for identity in (0.30, 0.40, 0.50, 0.70):
        edges = similarity_edges(lines, search_to_gene, identity=identity)
        mapping = components(sorted(labels), edges)
        sensitivity.append({"identity": identity, "bidirectional_coverage": 0.8, "edges": len(edges),
                            **group_stats(mapping, labels)})
        if identity == 0.30:
            primary_edges, groups = edges, mapping
    settings = {name: nested_splits(labels, groups, paired, grouped=grouped) for name, grouped in
                (("stratified_gene", False), ("similarity_grouped", True))}
    manifest = {"policy": POLICY, "seed": 42, "sequence_summary_sha256": sha(sequence_dir / "summary.json"),
                "alignments_sha256": sha(homology_dir / "alignments.tsv"),
                "protein_manifest_sha256": sha(homology_dir / "protein_manifest.json"),
                "label_vector_sha256": hashlib.sha256(json.dumps(labels, sort_keys=True).encode()).hexdigest(),
                "software": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                             "numpy": numpy.__version__, "scipy": scipy.__version__},
                "groups": groups, "paired_genes": paired, "settings": settings}
    diagnostics = {}
    for name, split in settings.items():
        folds = []
        for outer in range(split["outer_folds"]):
            partitions = partition_ids(manifest, name, outer)
            fold_stats = {role: {"genes": len(ids), "classes": dict(Counter(labels[g] for g in ids)),
                                "groups": len({groups[g] for g in ids}),
                                "paired_genes": len(set(ids) & set(paired))} for role, ids in
                          partitions.items() if role != "validation"}
            inner_stats = []
            for inner in range(split["inner_folds"][str(outer)]):
                parts = partition_ids(manifest, name, outer, inner)
                inner_stats.append({role: dict(Counter(labels[g] for g in ids)) for role, ids in parts.items()
                                    if role != "test"})
            fold_stats["inner_class_counts"] = inner_stats
            fold_stats["outer_fold"] = outer
            folds.append(fold_stats)
        crossings = sum(split["outer"][a] != split["outer"][b] for a, b in primary_edges)
        diagnostics[name] = {"outer_folds": split["outer_folds"], "inner_folds": split["inner_folds"],
                             "cross_outer_similarity_edges": crossings, "folds": folds,
                             "feasibility_failures": split["feasibility_failures"]}
    # Existing partition identities cannot silently change. Use a new version directory.
    if (output / "splits.json").exists() and json.loads((output / "splits.json").read_text()) != manifest:
        raise ValueError("Frozen splits differ; use a new version directory and document the change")
    # Freeze the actual IDs; future model code consumes these instead of resampling.
    write_json(output / "splits.json", manifest)
    write_json(output / "homology_edges.json", primary_edges)
    write_json(output / "protein_manifest.json", protein_manifest)
    report = {"policy": POLICY, "genes": len(labels), "classes": dict(Counter(labels.values())),
              "partial_self_alignments": partial_self_hits,
              "paired_genes": len(paired), "paired_classes": dict(Counter(labels[g] for g in paired)),
              "homology_threshold_sensitivity": sensitivity, "settings": diagnostics,
              "artifacts": {name: sha(output / name) for name in
                            ("splits.json", "homology_edges.json", "protein_manifest.json")},
              "sequence_summary_sha256": manifest["sequence_summary_sha256"],
              "alignment_search": {k: provenance[k] for k in
                                   ("mmseqs_version", "mmseqs_binary_sha256", "command", "artifacts")},
              "code_sha256": {p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name("evaluation_protocol.py"),
                                                      Path(__file__).with_name("prepare_homology.py")]}}
    write_json(output / "report.json", report)
    print(json.dumps({"genes": len(labels), "sensitivity": sensitivity, "crossings": {
        k: v["cross_outer_similarity_edges"] for k, v in diagnostics.items()}}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence-dir", required=True, type=Path)
    p.add_argument("--homology-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    prepare(a.sequence_dir, a.homology_dir, a.output)
