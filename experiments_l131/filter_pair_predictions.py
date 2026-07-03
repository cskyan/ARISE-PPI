from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments_l131.common import read_table, write_json, write_tsv


def probability(row):
    try:
        return float(row.get("pair_prob", "nan"))
    except (TypeError, ValueError):
        return float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter candidate pairs with a validation-frozen probability threshold."
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--validation-predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--quantile", type=float, default=0.90)
    args = parser.parse_args()

    validation = read_table(args.validation_predictions)
    values = np.asarray([probability(row) for row in validation], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Validation predictions contain no finite pair_prob values")
    threshold = float(np.quantile(values, args.quantile))
    rows = read_table(args.predictions)
    selected = [
        row for row in rows
        if np.isfinite(probability(row)) and probability(row) >= threshold
    ]
    write_tsv(args.output, selected)
    write_json(args.summary, {
        "validation_pairs": len(validation),
        "candidate_pairs": len(rows),
        "selected_pairs": len(selected),
        "probability_quantile": args.quantile,
        "probability_threshold": threshold,
    })
    print(
        f"[candidate-filter] threshold={threshold:.6f} "
        f"selected={len(selected)}/{len(rows)}"
    )


if __name__ == "__main__":
    main()
