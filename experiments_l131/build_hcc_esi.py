from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np

from experiments_l131.common import read_table, write_json, write_tsv


def numeric(row, key, default=np.nan):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a frozen HCC evidence-sufficient interaction ledger.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--faithfulness", required=True)
    parser.add_argument("--symmetry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--validation-pairs", default="")
    parser.add_argument("--probability-quantile", type=float, default=0.90)
    parser.add_argument("--evidence-quantile", type=float, default=0.75)
    parser.add_argument("--sufficiency-quantile", type=float, default=0.25)
    parser.add_argument("--symmetry-quantile", type=float, default=0.90)
    args = parser.parse_args()

    predictions = read_table(args.predictions)
    faithfulness = read_table(args.faithfulness)
    symmetry = {row["pair_id"]: row for row in read_table(args.symmetry)}
    validation_ids = set()
    if args.validation_pairs:
        validation_ids = {
            row["pair_id"] for row in read_table(args.validation_pairs)
        }
    if not validation_ids:
        validation_ids = {
            row["pair_id"]
            for row in predictions
            if str(row.get("split", "")).lower() == "val"
        }
    if not validation_ids:
        raise ValueError(
            "Validation pairs are required through --validation-pairs or a split=val column"
        )

    faith_by_pair = collections.defaultdict(list)
    for row in faithfulness:
        faith_by_pair[row["pair_id"]].append(row)

    validation_predictions = [
        row for row in predictions if row["pair_id"] in validation_ids
    ]
    if not validation_predictions:
        raise ValueError("No prediction rows matched the validation pair set")
    val_prob = np.asarray(
        [numeric(row, "pair_prob") for row in validation_predictions], dtype=np.float64
    )
    val_evi = np.asarray(
        [numeric(row, "evi_score") for row in validation_predictions], dtype=np.float64
    )
    val_swap = np.asarray([
        numeric(symmetry.get(row["pair_id"], {}), "abs_probability_difference")
        for row in validation_predictions
    ], dtype=np.float64)
    val_suff = []
    for row in validation_predictions:
        values = [
            numeric(value, "sufficiency_error_probability")
            for value in faith_by_pair[row["pair_id"]]
            if value.get("analysis") == "sufficiency"
            and value.get("method") == "top"
            and value.get("level") == "support"
        ]
        if values:
            val_suff.append(float(np.nanmean(values)))
    evi_mean = float(np.nanmean(val_evi))
    evi_sd = float(np.nanstd(val_evi)) or 1.0
    thresholds = {
        "probability": float(np.nanquantile(val_prob, args.probability_quantile)),
        "evidence_z": float(np.nanquantile((val_evi - evi_mean) / evi_sd, args.evidence_quantile)),
        "sufficiency_error": float(np.nanquantile(val_suff, args.sufficiency_quantile)) if val_suff else 0.0,
        "swap_difference": float(np.nanquantile(val_swap, args.symmetry_quantile)),
        "evidence_validation_mean": evi_mean,
        "evidence_validation_sd": evi_sd,
    }

    ledger = []
    for row in predictions:
        pair_id = row["pair_id"]
        faith_rows = faith_by_pair[pair_id]
        top_comp = [
            numeric(value, "comprehensiveness_probability")
            for value in faith_rows
            if value.get("analysis") == "comprehensiveness"
            and value.get("method") == "top"
        ]
        random_comp = [
            numeric(value, "comprehensiveness_probability")
            for value in faith_rows
            if value.get("analysis") == "comprehensiveness"
            and value.get("method") == "random"
        ]
        low_comp = [
            numeric(value, "comprehensiveness_probability")
            for value in faith_rows
            if value.get("analysis") == "comprehensiveness"
            and value.get("method") == "low"
        ]
        top_suff = [
            numeric(value, "sufficiency_error_probability")
            for value in faith_rows
            if value.get("analysis") == "sufficiency"
            and value.get("method") == "top"
            and value.get("level") == "support"
        ]
        top_effect = float(np.nanmean(top_comp)) if top_comp else np.nan
        random_effect = float(np.nanmean(random_comp)) if random_comp else np.nan
        low_effect = float(np.nanmean(low_comp)) if low_comp else np.nan
        sufficiency = float(np.nanmean(top_suff)) if top_suff else np.nan
        probability = numeric(row, "pair_prob")
        evi_score = numeric(row, "evi_score")
        evidence_z = (evi_score - evi_mean) / evi_sd
        swap_difference = numeric(
            symmetry.get(pair_id, {}), "abs_probability_difference"
        )
        pass_probability = probability >= thresholds["probability"]
        pass_evidence = evidence_z >= thresholds["evidence_z"]
        pass_deletion = (
            np.isfinite(top_effect)
            and np.isfinite(random_effect)
            and np.isfinite(low_effect)
            and top_effect > random_effect
            and top_effect > low_effect
        )
        pass_sufficiency = (
            np.isfinite(sufficiency)
            and sufficiency <= thresholds["sufficiency_error"]
        )
        pass_symmetry = (
            np.isfinite(swap_difference)
            and swap_difference <= thresholds["swap_difference"]
        )
        ledger.append({
            **row,
            "evidence_z": evidence_z,
            "top_deletion_effect": top_effect,
            "random_deletion_effect": random_effect,
            "low_deletion_effect": low_effect,
            "support_sufficiency_error": sufficiency,
            "swap_probability_difference": swap_difference,
            "pass_probability": int(pass_probability),
            "pass_evidence": int(pass_evidence),
            "pass_deletion": int(pass_deletion),
            "pass_sufficiency": int(pass_sufficiency),
            "pass_symmetry": int(pass_symmetry),
            "HCC_ESI_member": int(
                pass_probability
                and pass_evidence
                and pass_deletion
                and pass_sufficiency
                and pass_symmetry
            ),
        })

    write_tsv(args.output, ledger)
    write_json(args.summary, {
        "thresholds": thresholds,
        "validation_pairs": len(validation_ids),
        "candidate_pairs": len(ledger),
        "HCC_ESI_pairs": sum(int(row["HCC_ESI_member"]) for row in ledger),
        "quantiles": {
            "probability": args.probability_quantile,
            "evidence": args.evidence_quantile,
            "sufficiency": args.sufficiency_quantile,
            "symmetry": args.symmetry_quantile,
        },
    })


if __name__ == "__main__":
    main()
