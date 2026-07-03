from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


REQUIRED_PAIR_COLUMNS = ("protein_A", "protein_B", "label")


def read_table(path: str) -> List[Dict[str, str]]:
    fp = Path(path)
    delimiter = "\t" if fp.suffix.lower() in (".tsv", ".txt") else ","
    with fp.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_tsv(path: str, rows: Sequence[Dict], fieldnames: Sequence[str] | None = None) -> None:
    fp = Path(path)
    fp.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with fp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def write_json(path: str, payload: Dict) -> None:
    fp = Path(path)
    fp.parent.mkdir(parents=True, exist_ok=True)
    with fp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def as_binary_label(value) -> int:
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "positive", "pos"):
        return 1
    if text in ("0", "false", "no", "negative", "neg"):
        return 0
    raise ValueError(f"Invalid binary label: {value!r}")


def canonicalize_pairs(rows: Iterable[Dict]) -> Tuple[List[Dict], List[Dict]]:
    rows = list(rows)
    if not rows:
        return [], []
    missing = [name for name in REQUIRED_PAIR_COLUMNS if name not in rows[0]]
    if missing:
        raise ValueError(f"Pair manifest is missing required columns: {missing}")

    accepted: Dict[Tuple[str, str], Dict] = {}
    conflicted = set()
    rejected: List[Dict] = []
    for source_row in rows:
        row = dict(source_row)
        a = str(row.get("protein_A", "")).strip()
        b = str(row.get("protein_B", "")).strip()
        if not a or not b:
            rejected.append({**row, "exclusion_reason": "missing_protein"})
            continue
        if a == b:
            rejected.append({**row, "exclusion_reason": "self_pair"})
            continue
        label = as_binary_label(row.get("label"))
        first, second = sorted((a, b))
        key = (first, second)
        row["protein_A"] = first
        row["protein_B"] = second
        row["label"] = str(label)
        row["pair_id"] = str(row.get("pair_id") or f"{first}__{second}")
        if key in conflicted:
            rejected.append({**row, "exclusion_reason": "label_conflict"})
            continue
        if key in accepted:
            previous = accepted[key]
            if int(previous["label"]) != label:
                rejected.append({**row, "exclusion_reason": "label_conflict"})
                rejected.append({**previous, "exclusion_reason": "label_conflict"})
                del accepted[key]
                conflicted.add(key)
            else:
                rejected.append({**row, "exclusion_reason": "duplicate_orientation"})
            continue
        accepted[key] = row
    return list(accepted.values()), rejected


def expected_calibration_error(prob, labels, n_bins: int = 15) -> float:
    prob = np.asarray(prob, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if prob.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, max(2, int(n_bins)) + 1)
    result = 0.0
    for idx in range(len(edges) - 1):
        upper_closed = idx == len(edges) - 2
        mask = (prob >= edges[idx]) & (
            (prob <= edges[idx + 1]) if upper_closed else (prob < edges[idx + 1])
        )
        if np.any(mask):
            result += float(mask.mean()) * abs(
                float(prob[mask].mean()) - float(labels[mask].mean())
            )
    return float(result)


def binary_metrics(prob, labels, threshold: float, ece_bins: int = 15) -> Dict[str, float]:
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    prob = np.asarray(prob, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    valid = np.isfinite(prob) & np.isfinite(labels)
    prob = prob[valid]
    labels = labels[valid]
    pred = (prob >= float(threshold)).astype(np.int32)
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    two_classes = len(np.unique(labels)) > 1
    return {
        "n": int(labels.size),
        "threshold": float(threshold),
        "prevalence": float(labels.mean()) if labels.size else 0.0,
        "auprc": float(average_precision_score(labels, prob)) if two_classes else 0.0,
        "auroc": float(roc_auc_score(labels, prob)) if two_classes else 0.0,
        "mcc": float(matthews_corrcoef(labels, pred)) if two_classes else 0.0,
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "precision": float(precision_score(labels, pred, zero_division=0)),
        "sensitivity": float(recall_score(labels, pred, zero_division=0)),
        "specificity": float(tn / max(1, tn + fp)),
        "accuracy": float((pred == labels).mean()) if labels.size else 0.0,
        "brier": float(np.mean((prob - labels) ** 2)) if labels.size else 0.0,
        "ece": expected_calibration_error(prob, labels, n_bins=ece_bins),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def select_threshold(prob, labels, objective: str = "mcc") -> float:
    prob = np.asarray(prob, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int32)
    best_threshold = 0.5
    best_value = -math.inf
    for threshold in np.linspace(0.01, 0.99, 197):
        metrics = binary_metrics(prob, labels, float(threshold))
        key = "mcc" if objective == "mcc" else "f1"
        if metrics[key] > best_value:
            best_threshold = float(threshold)
            best_value = float(metrics[key])
    return best_threshold


def paired_bootstrap_difference(
    labels,
    prob_full,
    prob_control,
    threshold: float,
    metric: str,
    repeats: int = 2000,
    seed: int = 1337,
) -> Dict[str, float]:
    labels = np.asarray(labels, dtype=np.int32)
    prob_full = np.asarray(prob_full, dtype=np.float64)
    prob_control = np.asarray(prob_control, dtype=np.float64)
    if not (labels.size == prob_full.size == prob_control.size):
        raise ValueError("Paired bootstrap inputs must have equal lengths")
    rng = np.random.default_rng(int(seed))
    observed = (
        binary_metrics(prob_full, labels, threshold)[metric]
        - binary_metrics(prob_control, labels, threshold)[metric]
    )
    values = []
    for _ in range(int(repeats)):
        index = rng.integers(0, labels.size, size=labels.size)
        sampled_labels = labels[index]
        if len(np.unique(sampled_labels)) < 2 and metric in ("auprc", "auroc", "mcc"):
            continue
        values.append(
            binary_metrics(prob_full[index], sampled_labels, threshold)[metric]
            - binary_metrics(prob_control[index], sampled_labels, threshold)[metric]
        )
    low, high = np.quantile(values, [0.025, 0.975]) if values else (0.0, 0.0)
    return {
        "metric": metric,
        "difference": float(observed),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_repeats": int(len(values)),
    }


def paired_wilcoxon(left, right) -> Dict[str, float]:
    from scipy.stats import rankdata, wilcoxon

    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    valid = np.isfinite(left) & np.isfinite(right)
    difference = left[valid] - right[valid]
    if difference.size == 0 or np.allclose(difference, 0.0):
        return {
            "n": int(difference.size),
            "statistic": 0.0,
            "p_value": 1.0,
            "rank_biserial": 0.0,
        }
    statistic, p_value = wilcoxon(difference, zero_method="wilcox", alternative="two-sided")
    ranks = rankdata(np.abs(difference), method="average")
    positive = float(ranks[difference > 0].sum())
    negative = float(ranks[difference < 0].sum())
    denominator = positive + negative
    return {
        "n": int(difference.size),
        "statistic": float(statistic),
        "p_value": float(p_value),
        "rank_biserial": float((positive - negative) / denominator) if denominator else 0.0,
    }


def empirical_null_probability(observed: float, null_values, two_sided: bool = False) -> float:
    null = np.asarray(null_values, dtype=np.float64).reshape(-1)
    null = null[np.isfinite(null)]
    if null.size == 0:
        return 1.0
    if two_sided:
        exceed = np.abs(null) >= abs(float(observed))
    else:
        exceed = null >= float(observed)
    return float((int(exceed.sum()) + 1) / (int(null.size) + 1))


def benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    values = np.asarray(p_values, dtype=np.float64)
    count = len(values)
    order = np.argsort(values)
    adjusted = np.ones(count, dtype=np.float64)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = count - reverse_rank + 1
        running = min(running, float(values[index]) * count / max(1, rank))
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def seeded_random(seed: int) -> random.Random:
    return random.Random(int(seed))
