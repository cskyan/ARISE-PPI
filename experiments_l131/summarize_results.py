from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments_l131.common import read_table, write_json, write_tsv


METRICS = (
    "pair_auprc",
    "pair_auroc",
    "pair_mcc",
    "pair_f1",
    "pair_precision",
    "pair_recall",
    "pair_specificity",
    "pair_brier",
    "pair_ece",
)


def numeric(row, key):
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def load_json(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate L131 revision outputs into manuscript-ready tables."
    )
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    root = Path(args.results_root)
    out = Path(args.output_dir)
    control_rows = []
    controls = root / "controls"
    if controls.exists():
        for variant_dir in sorted(path for path in controls.iterdir() if path.is_dir()):
            for seed_dir in sorted(path for path in variant_dir.iterdir() if path.is_dir()):
                metric_files = sorted(
                    (seed_dir / "checkpoints").glob(
                        "test_PairAUPRC_*_metrics.tsv"
                    )
                )
                if not metric_files:
                    continue
                rows = read_table(str(metric_files[-1]))
                if not rows:
                    continue
                control_rows.append({
                    "variant": variant_dir.name,
                    "seed": seed_dir.name.replace("seed_", ""),
                    **rows[-1],
                    "metrics_file": str(metric_files[-1]),
                })
    write_tsv(str(out / "control_metrics_by_seed.tsv"), control_rows)

    aggregate_rows = []
    by_variant = collections.defaultdict(list)
    for row in control_rows:
        by_variant[row["variant"]].append(row)
    for variant, rows in sorted(by_variant.items()):
        summary = {"variant": variant, "seeds": len(rows)}
        for metric in METRICS:
            values = np.asarray([numeric(row, metric) for row in rows], dtype=np.float64)
            values = values[np.isfinite(values)]
            summary[f"{metric}_mean"] = float(values.mean()) if values.size else ""
            summary[f"{metric}_sd"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        aggregate_rows.append(summary)
    write_tsv(str(out / "control_metrics_aggregate.tsv"), aggregate_rows)

    faithfulness_path = root / "test" / "faithfulness" / "faithfulness_per_pair.tsv"
    faithfulness_rows = read_table(str(faithfulness_path)) if faithfulness_path.exists() else []
    faith_groups = collections.defaultdict(list)
    for row in faithfulness_rows:
        key = (
            row.get("level", ""),
            row.get("side", ""),
            row.get("operator", ""),
            row.get("method", ""),
            row.get("analysis", ""),
            row.get("budget", ""),
        )
        faith_groups[key].append(row)
    faith_summary = []
    for key, rows in sorted(faith_groups.items()):
        comp = np.asarray(
            [numeric(row, "comprehensiveness_probability") for row in rows],
            dtype=np.float64,
        )
        suff = np.asarray(
            [numeric(row, "sufficiency_error_probability") for row in rows],
            dtype=np.float64,
        )
        comp = comp[np.isfinite(comp)]
        suff = suff[np.isfinite(suff)]
        faith_summary.append({
            "level": key[0],
            "side": key[1],
            "operator": key[2],
            "method": key[3],
            "analysis": key[4],
            "budget": key[5],
            "pairs": len(rows),
            "comprehensiveness_mean": float(comp.mean()) if comp.size else "",
            "comprehensiveness_sd": float(comp.std(ddof=1)) if comp.size > 1 else 0.0,
            "sufficiency_error_mean": float(suff.mean()) if suff.size else "",
            "sufficiency_error_sd": float(suff.std(ddof=1)) if suff.size > 1 else 0.0,
        })
    write_tsv(str(out / "faithfulness_summary.tsv"), faith_summary)

    payload = {
        "control_variants": len(by_variant),
        "control_runs": len(control_rows),
        "control_metrics": aggregate_rows,
        "test_symmetry": load_json(
            root / "test" / "symmetry" / "swap_symmetry_summary.json"
        ),
        "contact_enrichment": load_json(
            root / "structure" / "contact_enrichment" /
            "contact_enrichment_summary.json"
        ),
        "hcc_esi": load_json(root / "hcc" / "HCC_ESI_summary.json"),
        "hcc_module": load_json(root / "hcc" / "module_summary.json"),
        "degree_matched_null": load_json(
            root / "hcc" / "degree_null" /
            "degree_matched_module_summary.json"
        ),
        "faithfulness_groups": len(faith_summary),
    }
    write_json(str(out / "revision_results_summary.json"), payload)
    print(
        f"[summary] control_runs={len(control_rows)} "
        f"faithfulness_groups={len(faith_summary)} output={out}"
    )


if __name__ == "__main__":
    main()
