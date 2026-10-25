"""Integrity-checked, label-independent whole-CDS features; one row per gene."""
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def array_sha(value):
    a = np.asarray(value, dtype="<f8")
    return hashlib.sha256(a.tobytes()).hexdigest()


def read_fasta(path):
    records, key, pieces = {}, None, []
    for line in Path(path).read_text().splitlines():
        if not line:
            continue
        if line.startswith(">"):
            if key is not None:
                records[key] = "".join(pieces)
            key, pieces = line[1:], []
            if not key or key in records:
                raise ValueError("Empty or duplicate FASTA identifier")
        elif key is None:
            raise ValueError("FASTA sequence before header")
        else:
            pieces.append(line.strip())
    if key is not None:
        records[key] = "".join(pieces)
    return records


def load_cds(sequence_dir, split_path, plan, report_path):
    """Fail before fitting if any frozen input/label identity has changed."""
    sequence_dir, split_path = Path(sequence_dir), Path(split_path)
    if sha(split_path) != plan["split_manifest_sha256"]:
        raise ValueError("Split manifest does not match predeclared plan")
    if sha(sequence_dir / "summary.json") != plan["sequence_summary_sha256"]:
        raise ValueError("Sequence summary does not match predeclared plan")
    report = json.loads(Path(report_path).read_text())
    for name, expected in report["artifacts"].items():
        if sha(split_path.parent / name) != expected:
            raise ValueError(f"Frozen evaluation artifact changed: {name}")
    summary = json.loads((sequence_dir / "summary.json").read_text())
    for name, record in summary["artifacts"].items():
        if sha(sequence_dir / name) != record["sha256"]:
            raise ValueError(f"Sequence artifact changed: {name}")
    splits = json.loads(split_path.read_text())
    if splits["policy"] != plan["protocol"] or splits["sequence_summary_sha256"] != sha(sequence_dir / "summary.json"):
        raise ValueError("Inconsistent sequence and evaluation policy")
    with (sequence_dir / "gene_coverage.csv").open() as f:
        genes = [r for r in csv.DictReader(f) if r["status"] == "retained"]
    labels = {r["gene_id"]: r["label"] for r in genes}
    if len(labels) != len(genes) or set(labels) != set(splits["groups"]):
        raise ValueError("Retained gene cohort differs from frozen splits")
    if hashlib.sha256(json.dumps(labels, sort_keys=True).encode()).hexdigest() != splits["label_vector_sha256"]:
        raise ValueError("Label vector differs from frozen splits")
    records = read_fasta(sequence_dir / "sequences.fasta")
    with (sequence_dir / "region_manifest.csv").open() as f:
        regions = [r for r in csv.DictReader(f) if r["sequence_type"] == "cds" and r["status"] == "retained"]
    sequences = {}
    for row in regions:
        gene, sequence = row["gene_id"], records[row["region_id"]]
        if gene in sequences or not sequence or set(sequence) - set("ACGT") or len(sequence) != int(row["length_bp"]):
            raise ValueError("Invalid/duplicate gene CDS")
        if labels.get(gene) != row["label"]:
            raise ValueError("Region and gene labels disagree")
        sequences[gene] = sequence
    if set(sequences) != set(labels):
        raise ValueError("Missing CDS genes")
    return splits, labels, sequences


def kmer_frequencies(sequence, k):
    if k not in (3, 6) or len(sequence) < k or set(sequence) - set("ACGT"):
        raise ValueError("Require ACGT sequence of length >= k, k=3 or 6")
    lookup = np.full(256, -1, dtype=np.int64)
    lookup[[ord(c) for c in "ACGT"]] = np.arange(4)
    codes = lookup[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]
    windows = np.lib.stride_tricks.sliding_window_view(codes, k)
    indices = windows @ (4 ** np.arange(k - 1, -1, -1))
    return np.bincount(indices, minlength=4 ** k).astype(np.float64) / len(indices)


def make_features(ids, sequences, model):
    kind = model["features"]
    if kind == "constant":
        return np.zeros((len(ids), 1)), ["constant_unused"]
    if kind == "log_length_gc":
        rows = []
        for gene in ids:
            seq = sequences[gene]
            if not seq or set(seq) - set("ACGT"):
                raise ValueError("Invalid CDS")
            rows.append([np.log(len(seq)), (seq.count("G") + seq.count("C")) / len(seq)])
        return np.asarray(rows), ["log_cds_length", "gc_fraction"]
    if kind == "kmer":
        k = model["k"]
        return np.vstack([kmer_frequencies(sequences[g], k) for g in ids]), [
            "".join(kmer) for kmer in itertools.product("ACGT", repeat=k)]
    raise ValueError(f"Unsupported baseline feature family: {kind}")
