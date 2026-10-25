"""Exact tie-aware weighted metrics and paired gene/component bootstrap.

Vectorization makes the predeclared 2,000 resamples practical. Formulae are
verified against sklearn for explicit duplicate samples, ties and zero weights.
"""
from itertools import combinations

import numpy as np


def _divide(a, b):
    a, b = np.broadcast_arrays(a, b)
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)


def _ranking(y, probability, weights):
    order = np.argsort(probability, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(probability[order])) + 1]
    pos = np.add.reduceat(weights[:, order] * y[order], starts, axis=1)
    neg = np.add.reduceat(weights[:, order] * (1 - y[order]), starts, axis=1)
    auc = np.sum(pos * (np.cumsum(neg, axis=1) - 0.5 * neg), axis=1) / (pos.sum(axis=1) * neg.sum(axis=1))
    tp = np.cumsum(pos[:, ::-1], axis=1)
    fp = np.cumsum(neg[:, ::-1], axis=1)
    ap = np.sum(_divide(tp, tp + fp) * pos[:, ::-1], axis=1) / pos.sum(axis=1)
    return auc, ap


def metric_arrays(y, probability, predicted, weights=None):
    y, probability, predicted = np.asarray(y), np.asarray(probability, dtype=float), np.asarray(predicted)
    if y.ndim != 1 or probability.shape != y.shape or predicted.shape != y.shape:
        raise ValueError("Metrics require one aligned record per gene")
    if set(np.unique(y)) != {0, 1} or not set(np.unique(predicted)) <= {0, 1}:
        raise ValueError("Require binary labels and predictions, both truth classes")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Invalid probabilities")
    weights = np.ones((1, len(y))) if weights is None else np.atleast_2d(np.asarray(weights, dtype=float))
    if weights.shape[1] != len(y) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Invalid sample weights")
    tp = weights @ ((y == 1) & (predicted == 1))
    tn = weights @ ((y == 0) & (predicted == 0))
    fp = weights @ ((y == 0) & (predicted == 1))
    fn = weights @ ((y == 1) & (predicted == 0))
    if np.any(tp + fn == 0) or np.any(tn + fp == 0):
        raise ValueError("Weighted sample lacks a truth class")
    auc, ap_og = _ranking(y, probability, weights)
    _, ap_tsg = _ranking(1 - y, 1 - probability, weights)
    return {"balanced_accuracy": .5 * (tp / (tp + fn) + tn / (tn + fp)),
            "accuracy": (tp + tn) / weights.sum(axis=1), "auroc": auc,
            "average_precision_og": ap_og, "average_precision_tsg": ap_tsg,
            "mcc": _divide(tp * tn - fp * fn, np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))),
            "macro_f1": .5 * (_divide(2 * tp, 2 * tp + fp + fn) + _divide(2 * tn, 2 * tn + fp + fn)),
            "precision_og": _divide(tp, tp + fp), "recall_og": tp / (tp + fn),
            "precision_tsg": _divide(tn, tn + fn), "recall_tsg": tn / (tn + fp),
            "brier_score": weights @ ((probability - y) ** 2) / weights.sum(axis=1),
            "prevalence_og": (tp + fn) / weights.sum(axis=1), "prevalence_tsg": (tn + fp) / weights.sum(axis=1),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def point_metrics(y, probability, predicted):
    return {k: float(v[0]) for k, v in metric_arrays(y, probability, predicted).items()}


def bootstrap_weights(y, folds, groups, setting, resamples=2000, seed=42):
    """Draw complete components with replacement, retaining all their genes.

    Unit multiplicities are multinomial counts. All models use the exact same
    matrix, including when a group has multiple genes and mixed class labels.
    """
    y, folds, groups = np.asarray(y), np.asarray(folds), np.asarray(groups)
    if y.ndim != 1 or folds.shape != y.shape or groups.shape != y.shape or resamples < 1:
        raise ValueError("Invalid bootstrap dimensions/count")
    if setting == "stratified_gene":
        unit_index = np.arange(len(y))
    elif setting == "similarity_grouped":
        _, unit_index = np.unique(groups, return_inverse=True)
        for group in np.unique(groups):
            if len(set(folds[groups == group])) != 1:
                raise ValueError("A homology component crosses outer folds")
    else:
        raise ValueError("Unknown setting")
    units = len(set(unit_index))
    rng, accepted, rejected = np.random.default_rng(seed), [], 0
    while sum(len(w) for w in accepted) < resamples:
        remaining = resamples - sum(len(w) for w in accepted)
        multiplicities = rng.multinomial(units, np.full(units, 1 / units), size=remaining)
        w = multiplicities[:, unit_index].astype(float)
        valid = np.ones(len(w), dtype=bool)
        for fold in np.unique(folds):
            for label in (0, 1):
                valid &= w[:, (folds == fold) & (y == label)].sum(axis=1) > 0
        accepted.append(w[valid])
        rejected += int((~valid).sum())
        if rejected > max(10000, resamples * 100):
            raise ValueError("Too many bootstrap draws lack a class in an outer fold")
    return np.vstack(accepted), {"units": units, "resamples": resamples, "seed": seed,
                                "rejected_draws": rejected, "unit": "gene" if setting == "stratified_gene" else "homology_component",
                                "conditioning": "fixed_outer_fits_and_thresholds"}


def summarize_setting(rows, setting, resamples=2000, seed=42):
    names = sorted({r["model"] for r in rows})
    by_model = {name: sorted((r for r in rows if r["model"] == name), key=lambda r: r["gene_id"]) for name in names}
    reference = by_model[names[0]]
    ids = [r["gene_id"] for r in reference]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate outer OOF gene")
    y = np.array([int(r["label"] == "OG") for r in reference])
    folds = np.array([r["outer_fold"] for r in reference])
    groups = np.array([r["group"] for r in reference])
    weights, bootstrap = bootstrap_weights(y, folds, groups, setting, resamples, seed)
    summaries, distributions, points = {}, {}, {}
    for name, records in by_model.items():
        if [(r["gene_id"], r["label"], r["outer_fold"], r["group"]) for r in records] != [
                (r["gene_id"], r["label"], r["outer_fold"], r["group"]) for r in reference]:
            raise ValueError("Model predictions are not paired on identical genes/folds/groups")
        probability = np.array([r["probability_og"] for r in records])
        predicted = np.array([int(r["prediction"] == "OG") for r in records])
        point = point_metrics(y, probability, predicted)
        dist = metric_arrays(y, probability, predicted, weights)
        fold_stats, fold_distributions = [], []
        for fold in sorted(set(folds)):
            mask = folds == fold
            fold_stats.append({"outer_fold": int(fold), **point_metrics(y[mask], probability[mask], predicted[mask])})
            # The primary statistic is the unweighted mean over outer folds.
            fold_distributions.append(metric_arrays(y[mask], probability[mask], predicted[mask], weights[:, mask])["balanced_accuracy"])
        point["mean_outer_balanced_accuracy"] = float(np.mean([r["balanced_accuracy"] for r in fold_stats]))
        dist["mean_outer_balanced_accuracy"] = np.mean(fold_distributions, axis=0)
        ci = {key: dict(zip(("lower", "upper"), map(float, np.quantile(values, [.025, .975]))))
              for key, values in dist.items() if key not in {"tn", "fp", "fn", "tp", "prevalence_og", "prevalence_tsg"}}
        summaries[name] = {"genes": len(ids), "pooled_oof": {k: v for k, v in point.items() if k != "mean_outer_balanced_accuracy"},
                           "mean_outer_balanced_accuracy": point["mean_outer_balanced_accuracy"],
                           "sd_outer_balanced_accuracy": float(np.std([r["balanced_accuracy"] for r in fold_stats], ddof=1)),
                           "confidence_intervals_95": ci, "outer_folds": fold_stats}
        distributions[name], points[name] = dist, point
    comparisons = []
    for a, b in combinations(names, 2):
        differences = {}
        for key in ("mean_outer_balanced_accuracy", "balanced_accuracy", "auroc", "average_precision_og", "average_precision_tsg", "mcc", "macro_f1", "brier_score"):
            lower, upper = np.quantile(distributions[a][key] - distributions[b][key], [.025, .975])
            differences[key] = {"estimate": points[a][key] - points[b][key], "lower": float(lower), "upper": float(upper)}
        comparisons.append({"model_a": a, "model_b": b, "direction": "a_minus_b", "metrics": differences})
    return {"bootstrap": bootstrap, "models": summaries, "paired_comparisons": comparisons}
