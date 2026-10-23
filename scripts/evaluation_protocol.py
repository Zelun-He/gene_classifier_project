"""Gene-level nested CV, homology components, and benchmark input guards.

No estimator is trained here. Inner assignments refer only to an outer training
partition. Consumers must aggregate windows before scoring a gene.
"""
from collections import Counter, defaultdict
import hashlib
import math
import re

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

POLICY = "gene-nested-homology-v1"
IDENTITY = 0.30
COVERAGE = 0.80
EVALUE = 1e-5


def similarity_edges(alignments, search_to_gene, identity=IDENTITY, coverage=COVERAGE):
    """Either search direction is sufficient; both sequence coverages must pass."""
    edges = set()
    for line in alignments:
        columns = line.rstrip().split("\t")
        if len(columns) != 12:
            raise ValueError("Expected twelve MMseqs2 alignment columns")
        query, target = columns[:2]
        if query not in search_to_gene or target not in search_to_gene:
            raise ValueError("Unknown protein search ID")
        nident, alnlen, qs, qe, ql, ts, te, tl = map(int, columns[2:10])
        evalue = float(columns[10])
        if not (0 <= nident <= alnlen and alnlen > 0 and 1 <= qs <= qe <= ql and 1 <= ts <= te <= tl):
            raise ValueError("Invalid alignment coordinates/counts")
        fid, qcov, tcov = nident / alnlen, (qe - qs + 1) / ql, (te - ts + 1) / tl
        if not all(math.isfinite(x) for x in (fid, qcov, tcov, evalue)):
            raise ValueError("Nonfinite alignment")
        if fid >= identity and min(qcov, tcov) >= coverage and evalue <= EVALUE:
            a, b = search_to_gene[query], search_to_gene[target]
            if a != b:
                edges.add(tuple(sorted((a, b))))
    return sorted(edges)


def components(gene_ids, edges):
    parent = {g: g for g in sorted(gene_ids)}
    if len(parent) != len(gene_ids):
        raise ValueError("Duplicate gene IDs")

    def root(g):
        if g not in parent:
            raise ValueError("Unknown gene in similarity edge")
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    for a, b in edges:
        ra, rb = root(a), root(b)
        parent[max(ra, rb)] = min(ra, rb)
    members = defaultdict(list)
    for g in parent:
        members[root(g)].append(g)
    result = {}
    for genes in members.values():
        group = "homology:" + hashlib.sha256("\n".join(sorted(genes)).encode()).hexdigest()[:20]
        result.update({g: group for g in genes})
    return result


def _assign(ids, labels, groups, n, grouped, seed):
    if n < 2:
        raise ValueError("At least two folds required")
    if min(Counter(labels[g] for g in ids).values()) < n:
        raise ValueError("Too few genes per class")
    if grouped and any(len({groups[g] for g in ids if labels[g] == c}) < n for c in set(labels.values())):
        raise ValueError("Too few groups containing each class")
    splitter = (StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed) if grouped else
                StratifiedKFold(n_splits=n, shuffle=True, random_state=seed))
    assignment = {}
    split_args = (np.zeros(len(ids)), [labels[g] for g in ids])
    iterator = splitter.split(*split_args, groups=[groups[g] for g in ids]) if grouped else splitter.split(*split_args)
    for fold, (_, validation) in enumerate(iterator):
        assignment.update({ids[i]: fold for i in validation})
    return assignment


def _validate_assignment(ids, assignment, labels, groups, n, grouped, paired):
    if set(assignment) != set(ids):
        raise ValueError("Fold assignment missing genes")
    for fold in range(n):
        test = {g for g in ids if assignment[g] == fold}
        train = set(ids) - test
        for cohort in (set(ids), set(ids) & set(paired)):
            for partition in (train & cohort, test & cohort):
                if {labels[g] for g in partition} != {"OG", "TSG"}:
                    raise ValueError("Fold lacks a class in full or paired cohort")
        if grouped and {groups[g] for g in train} & {groups[g] for g in test}:
            raise ValueError("Homology group crosses folds")


def nested_splits(labels, groups, paired, grouped, seed=42, target_outer=5, target_inner=3):
    """Deterministically reduce fold counts only if feasibility validation fails."""
    ids = sorted(labels)
    if not ids or set(labels.values()) != {"OG", "TSG"} or set(groups) != set(ids):
        raise ValueError("Expected unique labelled genes and complete groups")
    if not set(paired) <= set(ids):
        raise ValueError("Unknown paired gene")
    failures = []
    for outer_n in range(target_outer, 1, -1):
        try:
            outer = _assign(ids, labels, groups, outer_n, grouped, seed)
            _validate_assignment(ids, outer, labels, groups, outer_n, grouped, paired)
            inners, inner_ns = {}, {}
            for fold in range(outer_n):
                train = [g for g in ids if outer[g] != fold]
                for inner_n in range(target_inner, 1, -1):
                    try:
                        inner = _assign(train, labels, groups, inner_n, grouped, seed + 100 + fold)
                        _validate_assignment(train, inner, labels, groups, inner_n, grouped, paired)
                        break
                    except ValueError:
                        if inner_n == 2:
                            raise
                inners[str(fold)], inner_ns[str(fold)] = inner, inner_n
            return {"outer_folds": outer_n, "inner_folds": inner_ns, "outer": outer,
                    "inner": inners, "feasibility_failures": failures}
        except ValueError as error:
            failures.append({"outer_folds": outer_n, "reason": str(error)})
    raise ValueError(f"No feasible nested design: {failures}")


def partition_ids(manifest, setting, outer_fold, inner_fold=None, paired=False):
    """The only supported way for consumers to derive training/evaluation IDs."""
    split = manifest["settings"][setting]
    if str(outer_fold) not in split["inner"]:
        raise ValueError("Unknown outer fold")
    cohort = set(manifest["paired_genes"] if paired else manifest["groups"])
    test = {g for g in cohort if split["outer"][g] == outer_fold}
    train = cohort - test
    validation = set()
    if inner_fold is not None:
        if not 0 <= inner_fold < split["inner_folds"][str(outer_fold)]:
            raise ValueError("Unknown inner fold")
        validation = {g for g in train if split["inner"][str(outer_fold)][g] == inner_fold}
        train -= validation
    assert_disjoint(train, validation, test,
                    manifest["groups"] if setting == "similarity_grouped" else None)
    return {"train": sorted(train), "validation": sorted(validation), "test": sorted(test)}


def assert_disjoint(train, validation, test, groups=None):
    partitions = [set(train), set(validation), set(test)]
    for i, first in enumerate(partitions):
        for second in partitions[i + 1:]:
            if first & second:
                raise ValueError("Gene leakage across partitions")
            if groups is not None and {groups[g] for g in first} & {groups[g] for g in second}:
                raise ValueError("Homology leakage across partitions")


def assign_derived_rows(rows, partitions):
    """Assign every transcript/window/augmentation by audited gene_id, never row."""
    assert_disjoint(**partitions)
    lookup = {g: role for role, ids in partitions.items() for g in ids}
    result = {role: [] for role in partitions}
    for row in rows:
        if row.get("gene_id") not in lookup:
            raise ValueError("Derived row has absent/unknown gene_id")
        result[lookup[row["gene_id"]]].append(row)
    return result


def training_class_weights(labels, permitted_train_ids):
    """Compute weights from exactly one label per training gene."""
    ids = list(permitted_train_ids)
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("Expected unique training gene IDs")
    counts = Counter(labels[g] for g in ids)
    if set(counts) != {"OG", "TSG"}:
        raise ValueError("Both classes required")
    return {c: len(ids) / (2 * counts[c]) for c in sorted(counts)}


def fit_training_transformer(transformer, features, permitted_train_ids, labels=None):
    """Fit a scaler/selector/Pipeline using exactly the training gene features.

    Features are gene embeddings or covariates, already pooled to one row/gene.
    Fit a fresh transformer on every inner fit; fit a fresh final transformer on
    outer training only after choosing hyperparameters. No global fit is reused.
    """
    ids = sorted(permitted_train_ids)
    if len(ids) != len(set(ids)) or not ids or set(features) != set(ids):
        raise ValueError("Transformer fit features must contain exactly the training genes")
    x = [features[g] for g in ids]
    transformer.fit(x, None if labels is None else [labels[g] for g in ids])
    transformer.fit_gene_ids_ = ids
    return transformer


def select_threshold(predictions, expected_inner_oof_ids):
    """Select balanced-accuracy threshold from complete inner OOF predictions.

    Each row is one gene's prediction from an inner model that excluded that
    gene/group. This checks IDs; callers must retain the model's fit-ID lineage.
    OG is positive. Ties prefer the threshold closest to 0.5, then smaller.
    """
    ids = [r["gene_id"] for r in predictions]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_inner_oof_ids):
        raise ValueError("Require exactly one inner OOF prediction per training gene")
    if {r["label"] for r in predictions} != {"OG", "TSG"}:
        raise ValueError("Both classes required")
    scores = [float(r["probability_og"]) for r in predictions]
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in scores):
        raise ValueError("Invalid probability")
    candidates = sorted({0.0, 0.5, 1.0, math.nextafter(1.0, math.inf), *scores})
    def objective(t):
        recalls = [sum((p >= t) == (c == "OG") for p, c in
                       zip(scores, [r["label"] for r in predictions]) if c == label) /
                   sum(r["label"] == label for r in predictions) for label in ("OG", "TSG")]
        return (sum(recalls) / 2, -abs(t - 0.5), -t)
    return max(candidates, key=objective)


def validate_initialization(record):
    """Fail closed on legacy task checkpoints or unpinned initialization."""
    if (record.get("kind") != "pristine_pretrained" or record.get("task_finetuned") is not False
            or not re.fullmatch(r"[0-9a-f]{40}", record.get("revision", ""))
            or not record.get("model_id") or "checkpoint-916" in str(record)
            or not record.get("provenance_url") or
            not {"config", "tokenizer", "weights"} <= set(record.get("artifact_sha256", {}))):
        raise ValueError("Benchmark requires documented pristine, revision-pinned pretrained initialization")
    if any(not re.fullmatch(r"[0-9a-f]{64}", h) for h in record["artifact_sha256"].values()):
        raise ValueError("Invalid initialization artifact hashes")
