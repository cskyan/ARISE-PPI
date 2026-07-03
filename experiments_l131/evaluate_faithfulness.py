from __future__ import annotations

import argparse
import collections
import math
import random
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments_l131.common import write_json, write_tsv
from experiments_l131.pair_runtime import (
    build_runtime,
    clone_pair_item,
    forward_item,
    item_for_row,
    load_pair_manifest,
)


def parse_budgets(text: str) -> List[float]:
    values = [float(value) for value in text.split(",") if value.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("Budgets must be positive")
    return values


def budget_count(value: float, total: int) -> int:
    count = int(round(value * total)) if value < 1.0 else int(round(value))
    return max(1, min(total, count))


def choose_indices(scores: np.ndarray, count: int, method: str, rng: random.Random) -> List[int]:
    valid = list(range(len(scores)))
    if method == "top":
        return np.argsort(-scores)[:count].astype(int).tolist()
    if method == "low":
        return np.argsort(scores)[:count].astype(int).tolist()
    if method == "random":
        return rng.sample(valid, k=min(count, len(valid)))
    raise ValueError(method)


def replace_features(
    item: Dict,
    side: str,
    selected: Sequence[int],
    operator: str,
    retain_only: bool,
) -> Dict:
    output = clone_pair_item(item)
    key = "resA" if side == "A" else "resB"
    mask_key = "maskA" if side == "A" else "maskB"
    features = output[key]
    valid = output[mask_key].bool()
    selected_mask = torch.zeros(features.shape[0], dtype=torch.bool)
    selected_mask[list(selected)] = True
    target = valid & (~selected_mask if retain_only else selected_mask)
    if operator == "zero":
        replacement = torch.zeros(features.shape[-1], dtype=features.dtype)
    elif operator == "background":
        source = valid & ~target
        if not bool(source.any()):
            source = valid
        replacement = features[source].mean(dim=0)
    else:
        raise ValueError(operator)
    features[target] = replacement
    output[key] = features
    return output


def scalar_output(output: Dict) -> Dict[str, float]:
    return {
        "pair_prob": float(output["pair_prob"][0].detach().cpu()),
        "pair_logit": float(output["pair_logit"][0].detach().cpu()),
    }


def fixed_selection(output: Dict) -> Dict:
    return {
        key: output[key].detach()
        for key in (
            "topk_idxA",
            "topk_idxB",
            "pair_idxA_local",
            "pair_idxB_local",
            "retained_pair_flat",
        )
        if key in output
    }


def support_intervention(
    runtime,
    item,
    original,
    selected: Sequence[int],
    retain_only: bool,
) -> Dict:
    support_count = int(original["pair_score"].shape[1])
    keep = torch.ones((1, support_count), dtype=torch.bool)
    if retain_only:
        keep[:] = False
        keep[0, list(selected)] = True
    else:
        keep[0, list(selected)] = False
    return forward_item(
        runtime,
        item,
        intervention={
            "pair_head_type": "eb",
            "fixed_selection": fixed_selection(original),
            "support_keep_mask": keep,
        },
    )


def append_measurement(
    rows: List[Dict],
    base: Dict,
    original: Dict[str, float],
    perturbed: Dict[str, float],
    repeats: int = 1,
) -> None:
    rows.append({
        **base,
        "original_probability": original["pair_prob"],
        "perturbed_probability": perturbed["pair_prob"],
        "original_logit": original["pair_logit"],
        "perturbed_logit": perturbed["pair_logit"],
        "comprehensiveness_probability": original["pair_prob"] - perturbed["pair_prob"],
        "comprehensiveness_logit": original["pair_logit"] - perturbed["pair_logit"],
        "sufficiency_error_probability": "",
        "sufficiency_error_logit": "",
        "random_repetitions": int(repeats),
    })


def aggregate_random(outputs: Sequence[Dict]) -> Dict[str, float]:
    return {
        "pair_prob": float(np.mean([value["pair_prob"] for value in outputs])),
        "pair_logit": float(np.mean([value["pair_logit"] for value in outputs])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run L131 EB faithfulness interventions.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--esm-local-dir", default="")
    parser.add_argument("--sequence-mode", default="esm", choices=("esm", "light", "hybrid"))
    parser.add_argument("--threshold", type=float, default=-1.0)
    parser.add_argument(
        "--cohort",
        default="predicted_positive",
        choices=("predicted_positive", "all_positive", "all"),
    )
    parser.add_argument("--residue-budgets", default="0.01,0.02,0.05,0.10,0.20")
    parser.add_argument("--support-budgets", default="1,5,0.10,0.20,0.50")
    parser.add_argument("--random-repetitions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
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
    checkpoint_threshold = (
        runtime.checkpoint.get("val_metrics", {}).get("pair_thr", 0.5)
        if isinstance(runtime.checkpoint, dict)
        else 0.5
    )
    threshold = float(args.threshold if args.threshold >= 0 else checkpoint_threshold)
    residue_budgets = parse_budgets(args.residue_budgets)
    support_budgets = parse_budgets(args.support_budgets)
    rng = random.Random(args.seed)
    output_rows: List[Dict] = []
    cohort_before = len(rows)
    cohort_after = 0

    for row in rows:
        item = item_for_row(runtime, row)
        original_output = forward_item(runtime, item)
        original = scalar_output(original_output)
        include = (
            args.cohort == "all"
            or (args.cohort == "all_positive" and int(row["label"]) == 1)
            or (
                args.cohort == "predicted_positive"
                and original["pair_prob"] >= threshold
            )
        )
        if not include:
            continue
        cohort_after += 1

        for side, score_key, feature_key in (
            ("A", "p_res_A", "resA"),
            ("B", "p_res_B", "resB"),
        ):
            scores = (
                original_output[score_key][0].detach().float().cpu().numpy()
            )
            total = int(item[feature_key].shape[0])
            scores = scores[:total]
            for operator in ("zero", "background"):
                for budget in residue_budgets:
                    count = budget_count(budget, total)
                    for method in ("top", "low"):
                        selected = choose_indices(scores, count, method, rng)
                        removed = scalar_output(forward_item(
                            runtime,
                            replace_features(item, side, selected, operator, retain_only=False),
                        ))
                        append_measurement(output_rows, {
                            "pair_id": row["pair_id"],
                            "label": int(row["label"]),
                            "level": "residue",
                            "side": side,
                            "operator": operator,
                            "method": method,
                            "budget": budget,
                            "count": count,
                            "analysis": "comprehensiveness",
                        }, original, removed)
                        retained = scalar_output(forward_item(
                            runtime,
                            replace_features(item, side, selected, operator, retain_only=True),
                        ))
                        output_rows.append({
                            "pair_id": row["pair_id"],
                            "label": int(row["label"]),
                            "level": "residue",
                            "side": side,
                            "operator": operator,
                            "method": method,
                            "budget": budget,
                            "count": count,
                            "analysis": "sufficiency",
                            "original_probability": original["pair_prob"],
                            "perturbed_probability": retained["pair_prob"],
                            "original_logit": original["pair_logit"],
                            "perturbed_logit": retained["pair_logit"],
                            "comprehensiveness_probability": "",
                            "comprehensiveness_logit": "",
                            "sufficiency_error_probability": original["pair_prob"] - retained["pair_prob"],
                            "sufficiency_error_logit": original["pair_logit"] - retained["pair_logit"],
                            "random_repetitions": 1,
                        })

                    random_removed, random_retained = [], []
                    for _ in range(max(1, args.random_repetitions)):
                        selected = choose_indices(scores, count, "random", rng)
                        random_removed.append(scalar_output(forward_item(
                            runtime,
                            replace_features(item, side, selected, operator, retain_only=False),
                        )))
                        random_retained.append(scalar_output(forward_item(
                            runtime,
                            replace_features(item, side, selected, operator, retain_only=True),
                        )))
                    append_measurement(output_rows, {
                        "pair_id": row["pair_id"],
                        "label": int(row["label"]),
                        "level": "residue",
                        "side": side,
                        "operator": operator,
                        "method": "random",
                        "budget": budget,
                        "count": count,
                        "analysis": "comprehensiveness",
                    }, original, aggregate_random(random_removed), args.random_repetitions)
                    retained = aggregate_random(random_retained)
                    output_rows.append({
                        "pair_id": row["pair_id"],
                        "label": int(row["label"]),
                        "level": "residue",
                        "side": side,
                        "operator": operator,
                        "method": "random",
                        "budget": budget,
                        "count": count,
                        "analysis": "sufficiency",
                        "original_probability": original["pair_prob"],
                        "perturbed_probability": retained["pair_prob"],
                        "original_logit": original["pair_logit"],
                        "perturbed_logit": retained["pair_logit"],
                        "comprehensiveness_probability": "",
                        "comprehensiveness_logit": "",
                        "sufficiency_error_probability": original["pair_prob"] - retained["pair_prob"],
                        "sufficiency_error_logit": original["pair_logit"] - retained["pair_logit"],
                        "random_repetitions": args.random_repetitions,
                    })

        support_scores = (
            original_output["pair_score"][0].detach().float().cpu().numpy()
        )
        support_total = len(support_scores)
        for budget in support_budgets:
            count = budget_count(budget, support_total)
            for method in ("top", "low"):
                selected = choose_indices(support_scores, count, method, rng)
                removed = scalar_output(support_intervention(
                    runtime, item, original_output, selected, retain_only=False
                ))
                append_measurement(output_rows, {
                    "pair_id": row["pair_id"], "label": int(row["label"]),
                    "level": "support", "side": "AB", "operator": "renormalize",
                    "method": method, "budget": budget, "count": count,
                    "analysis": "comprehensiveness",
                }, original, removed)
                retained = scalar_output(support_intervention(
                    runtime, item, original_output, selected, retain_only=True
                ))
                output_rows.append({
                    "pair_id": row["pair_id"], "label": int(row["label"]),
                    "level": "support", "side": "AB", "operator": "renormalize",
                    "method": method, "budget": budget, "count": count,
                    "analysis": "sufficiency",
                    "original_probability": original["pair_prob"],
                    "perturbed_probability": retained["pair_prob"],
                    "original_logit": original["pair_logit"],
                    "perturbed_logit": retained["pair_logit"],
                    "comprehensiveness_probability": "",
                    "comprehensiveness_logit": "",
                    "sufficiency_error_probability": original["pair_prob"] - retained["pair_prob"],
                    "sufficiency_error_logit": original["pair_logit"] - retained["pair_logit"],
                    "random_repetitions": 1,
                })

            random_removed, random_retained = [], []
            for _ in range(max(1, args.random_repetitions)):
                selected = choose_indices(support_scores, count, "random", rng)
                random_removed.append(scalar_output(support_intervention(
                    runtime, item, original_output, selected, retain_only=False
                )))
                random_retained.append(scalar_output(support_intervention(
                    runtime, item, original_output, selected, retain_only=True
                )))
            append_measurement(output_rows, {
                "pair_id": row["pair_id"], "label": int(row["label"]),
                "level": "support", "side": "AB", "operator": "renormalize",
                "method": "random", "budget": budget, "count": count,
                "analysis": "comprehensiveness",
            }, original, aggregate_random(random_removed), args.random_repetitions)
            retained = aggregate_random(random_retained)
            output_rows.append({
                "pair_id": row["pair_id"], "label": int(row["label"]),
                "level": "support", "side": "AB", "operator": "renormalize",
                "method": "random", "budget": budget, "count": count,
                "analysis": "sufficiency",
                "original_probability": original["pair_prob"],
                "perturbed_probability": retained["pair_prob"],
                "original_logit": original["pair_logit"],
                "perturbed_logit": retained["pair_logit"],
                "comprehensiveness_probability": "",
                "comprehensiveness_logit": "",
                "sufficiency_error_probability": original["pair_prob"] - retained["pair_prob"],
                "sufficiency_error_logit": original["pair_logit"] - retained["pair_logit"],
                "random_repetitions": args.random_repetitions,
            })

    grouped = collections.defaultdict(list)
    for result in output_rows:
        if result["analysis"] == "comprehensiveness":
            grouped[(
                result["pair_id"], result["level"], result["side"],
                result["operator"], result["method"],
            )].append(result)
    aopc_rows = []
    for key, values in grouped.items():
        values = sorted(values, key=lambda row: float(row["budget"]))
        aopc_rows.append({
            "pair_id": key[0], "level": key[1], "side": key[2],
            "operator": key[3], "method": key[4],
            "aopc_probability": float(np.mean([
                float(row["comprehensiveness_probability"]) for row in values
            ])),
            "aopc_logit": float(np.mean([
                float(row["comprehensiveness_logit"]) for row in values
            ])),
            "budgets": ",".join(str(row["budget"]) for row in values),
        })

    out_dir = Path(args.out_dir)
    write_tsv(str(out_dir / "faithfulness_per_pair.tsv"), output_rows)
    write_tsv(str(out_dir / "faithfulness_aopc_per_pair.tsv"), aopc_rows)
    write_json(str(out_dir / "faithfulness_cohort.json"), {
        "cohort_rule": args.cohort,
        "threshold": threshold,
        "pairs_before_filter": cohort_before,
        "pairs_after_filter": cohort_after,
        "random_repetitions": args.random_repetitions,
        "residue_budgets": residue_budgets,
        "support_budgets": support_budgets,
    })


if __name__ == "__main__":
    main()
