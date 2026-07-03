from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments_l131.common import (
    binary_metrics,
    select_threshold,
    write_json,
    write_tsv,
)
from experiments_l131.pair_runtime import (
    build_runtime,
    forward_item,
    item_for_row,
    load_pair_manifest,
    sample_to_pair,
)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    n = min(len(left), len(right))
    if n < 2:
        return 0.0
    x, y = rankdata(left[:n]), rankdata(right[:n])
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def top_set(values: np.ndarray, k: int) -> set:
    k = max(1, min(int(k), len(values)))
    return set(np.argsort(-values)[:k].astype(int).tolist())


def jaccard(left: set, right: set) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 1.0


def support_set(output, swapped: bool = False) -> set:
    idx_a = output["pair_idxA"][0].detach().cpu().numpy().astype(int)
    idx_b = output["pair_idxB"][0].detach().cpu().numpy().astype(int)
    if swapped:
        return {(int(b), int(a)) for a, b in zip(idx_a, idx_b)}
    return {(int(a), int(b)) for a, b in zip(idx_a, idx_b)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate L131 chain-order symmetry.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--esm-local-dir", default="")
    parser.add_argument("--sequence-mode", default="esm", choices=("esm", "light", "hybrid"))
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=-1.0)
    parser.add_argument("--require-labels", action="store_true")
    args = parser.parse_args()

    rows = load_pair_manifest(args.manifest, require_labels=args.require_labels)
    proteins = [p for row in rows for p in (row["protein_A"], row["protein_B"])]
    runtime = build_runtime(
        args.root,
        args.checkpoint,
        proteins,
        esm_local_dir=args.esm_local_dir,
        sequence_mode=args.sequence_mode,
        pair_head_type="eb",
        require_native_eb=True,
    )
    records = []
    for row in rows:
        item_ab = item_for_row(runtime, row)
        item_ba = sample_to_pair(
            runtime.samples[row["protein_B"]],
            runtime.samples[row["protein_A"]],
            label=max(0, int(row["label"])),
            pair_id=row["pair_id"],
        )
        out_ab = forward_item(runtime, item_ab)
        out_ba = forward_item(runtime, item_ba)
        prob_ab = float(out_ab["pair_prob"][0].detach().cpu())
        prob_ba = float(out_ba["pair_prob"][0].detach().cpu())
        score_a_ab = out_ab["p_res_A"][0].detach().float().cpu().numpy()
        score_b_ab = out_ab["p_res_B"][0].detach().float().cpu().numpy()
        score_a_ba = out_ba["p_res_A"][0].detach().float().cpu().numpy()
        score_b_ba = out_ba["p_res_B"][0].detach().float().cpu().numpy()
        records.append({
            "pair_id": row["pair_id"],
            "protein_A": row["protein_A"],
            "protein_B": row["protein_B"],
            "label": int(row["label"]),
            "prob_AB": prob_ab,
            "prob_BA": prob_ba,
            "prob_swap_average": 0.5 * (prob_ab + prob_ba),
            "abs_probability_difference": abs(prob_ab - prob_ba),
            "residue_spearman_A": spearman(score_a_ab, score_b_ba),
            "residue_spearman_B": spearman(score_b_ab, score_a_ba),
            "topk_jaccard_A": jaccard(
                top_set(score_a_ab, args.topk), top_set(score_b_ba, args.topk)
            ),
            "topk_jaccard_B": jaccard(
                top_set(score_b_ab, args.topk), top_set(score_a_ba, args.topk)
            ),
            "support_jaccard": jaccard(
                support_set(out_ab), support_set(out_ba, swapped=True)
            ),
        })

    labels = np.asarray([row["label"] for row in records], dtype=np.int32)
    prob_ab = np.asarray([row["prob_AB"] for row in records], dtype=np.float64)
    prob_avg = np.asarray([row["prob_swap_average"] for row in records], dtype=np.float64)
    known = labels >= 0
    checkpoint_threshold = float(
        (runtime.checkpoint.get("val_metrics", {}) or {}).get("pair_thr", 0.5)
    )
    if args.threshold >= 0:
        threshold = float(args.threshold)
    elif known.sum() and len(np.unique(labels[known])) > 1:
        threshold = select_threshold(prob_ab[known], labels[known], objective="mcc")
    else:
        threshold = checkpoint_threshold
    differences = np.asarray(
        [row["abs_probability_difference"] for row in records], dtype=np.float64
    )
    summary = {
        "pairs": len(records),
        "threshold": threshold,
        "probability_difference_mean": float(np.mean(differences)),
        "probability_difference_median": float(np.median(differences)),
        "probability_difference_p95": float(np.quantile(differences, 0.95)),
        "probability_difference_p99": float(np.quantile(differences, 0.99)),
        "labeled_pairs": int(known.sum()),
    }
    if known.sum() and len(np.unique(labels[known])) > 1:
        summary["AB_metrics"] = binary_metrics(prob_ab[known], labels[known], threshold)
        summary["swap_average_metrics"] = binary_metrics(
            prob_avg[known], labels[known], threshold
        )
    out_dir = Path(args.out_dir)
    write_tsv(str(out_dir / "swap_symmetry_per_pair.tsv"), records)
    write_json(str(out_dir / "swap_symmetry_summary.json"), summary)


if __name__ == "__main__":
    main()
