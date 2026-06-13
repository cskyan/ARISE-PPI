#!/usr/bin/env python
"""Run ARISE-PPI prediction with selectable binary or top-k checkpoint mode."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arise_ppi.cli_utils import apply_env, load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with ARISE-PPI.")
    parser.add_argument("--config", help="JSON config file. CLI values override it.")
    parser.add_argument("--task", choices=["topk", "binary"], default=None, help="Prediction objective/checkpoint family.")
    parser.add_argument("--root", help="Dataset root.")
    parser.add_argument("--id-list", dest="id_list", help="All protein IDs.")
    parser.add_argument("--train-list", dest="train_list", help="Training ID list.")
    parser.add_argument("--val-list", dest="val_list", help="Validation ID list.")
    parser.add_argument("--test-list", dest="test_list", help="Test ID list.")
    parser.add_argument("--save-dir", dest="save_dir", help="Training run directory containing checkpoints.")
    parser.add_argument("--esm-local-dir", dest="esm_local_dir", help="Directory containing esm2_t33_650M_UR50D.pt.")
    parser.add_argument("--structure-dir", dest="structure_dir", help="Structure subdirectory under dataset root.")
    parser.add_argument("--sequence-mode", dest="sequence_mode", choices=["esm", "light", "hybrid"], default=None)
    parser.add_argument("--structure-source", dest="structure_source", choices=["auto", "pdb", "coords"], default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int)
    parser.add_argument("--num-workers", dest="num_workers", type=int)
    parser.add_argument("--allow-zero-esm-fallback", dest="allow_zero_esm_fallback", choices=["0", "1"], default=None)
    parser.add_argument("--checkpoint", help="Explicit checkpoint path.")
    parser.add_argument("--checkpoint-tag", choices=["auto", "topk", "binary", "auprc", "auc", "auroc"], default=None)
    parser.add_argument("--split", default=None, help="train|val|test|all|/path/to/id_list.txt")
    parser.add_argument("--ids", help="Explicit ID list for prediction.")
    parser.add_argument("--out-dir", dest="out_dir", help="Prediction output directory.")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--threshold-mode", dest="threshold_mode")
    parser.add_argument("--topk", type=int)
    parser.add_argument("--ager-scan", action="store_true")
    parser.add_argument("--ager-scan-apply-best", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    options = load_config(args.config)
    options.update({k: v for k, v in vars(args).items() if k != "config" and v is not None})
    options.setdefault("task", "topk")
    apply_env(options)

    forwarded = []
    for key in (
        "root", "id_list", "train_list", "val_list", "test_list", "save_dir",
        "structure_dir", "esm_local_dir", "checkpoint", "checkpoint_tag", "split",
        "ids", "out_dir", "threshold", "threshold_mode", "topk",
    ):
        value = options.get(key)
        if value is not None:
            forwarded.extend(["--" + key.replace("_", "-"), str(value)])
    if options.get("ager_scan"):
        forwarded.append("--ager-scan")
    if options.get("ager_scan_apply_best"):
        forwarded.append("--ager-scan-apply-best")
    sys.argv = [sys.argv[0]] + forwarded

    from arise_ppi.predict_core import main as predict_main

    predict_main()


if __name__ == "__main__":
    main()
