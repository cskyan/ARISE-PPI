#!/usr/bin/env python
"""Train ARISE-PPI with selectable binary or top-k objective."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arise_ppi.cli_utils import apply_env, load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ARISE-PPI.")
    parser.add_argument("--config", help="JSON config file. CLI values override it.")
    parser.add_argument("--task", choices=["topk", "binary"], default=None, help="Training objective.")
    parser.add_argument("--root", help="Dataset root.")
    parser.add_argument("--id-list", dest="id_list", help="All protein IDs.")
    parser.add_argument("--train-list", dest="train_list", help="Training ID list.")
    parser.add_argument("--val-list", dest="val_list", help="Validation ID list.")
    parser.add_argument("--test-list", dest="test_list", help="Test ID list.")
    parser.add_argument("--save-dir", dest="save_dir", help="Run/checkpoint output directory.")
    parser.add_argument("--esm-local-dir", dest="esm_local_dir", help="Directory containing esm2_t33_650M_UR50D.pt.")
    parser.add_argument("--structure-dir", dest="structure_dir", help="Structure subdirectory under dataset root.")
    parser.add_argument("--sequence-mode", dest="sequence_mode", choices=["esm", "light", "hybrid"], default=None)
    parser.add_argument("--structure-source", dest="structure_source", choices=["auto", "pdb", "coords"], default=None)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", dest="batch_size", type=int)
    parser.add_argument("--num-workers", dest="num_workers", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--use-pssm", dest="use_pssm", choices=["0", "1"], default=None)
    parser.add_argument("--use-dssp", dest="use_dssp", choices=["0", "1"], default=None)
    parser.add_argument("--allow-zero-esm-fallback", dest="allow_zero_esm_fallback", choices=["0", "1"], default=None)
    parser.add_argument("--report-topk-metrics", dest="report_topk_metrics", choices=["0", "1"], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    options = load_config(args.config)
    options.update({k: v for k, v in vars(args).items() if k != "config" and v is not None})
    options.setdefault("task", "topk")
    apply_env(options)
    os.environ.setdefault("PYTHONPATH", str(ROOT / "src"))

    from arise_ppi.train_core import main as train_main

    train_main()


if __name__ == "__main__":
    main()
