"""Command-line helpers for ARISE-PPI wrappers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


ENV_KEYS = {
    "task": "PRIMARY_OBJECTIVE",
    "root": "RBP400_ROOT",
    "id_list": "ID_LIST",
    "train_list": "TRAIN_LIST",
    "val_list": "VAL_LIST",
    "test_list": "TEST_LIST",
    "save_dir": "SAVE_DIR",
    "esm_local_dir": "ESM_LOCAL_DIR",
    "structure_dir": "RBP400_STRUCTURE_DIR",
    "sequence_mode": "SEQUENCE_MODE",
    "structure_source": "STRUCTURE_SOURCE",
    "epochs": "EPOCHS",
    "batch_size": "BATCH_SITE",
    "num_workers": "NUM_WORKERS",
    "lr": "LR",
    "seed": "SEED",
    "use_pssm": "USE_PSSM",
    "use_dssp": "USE_DSSP",
    "allow_zero_esm_fallback": "ALLOW_ZERO_ESM_FALLBACK",
    "report_topk_metrics": "REPORT_TOPK_METRICS",
}


def load_config(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    fp = Path(path)
    with fp.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def apply_env(options: Dict[str, Any]) -> None:
    for key, value in options.items():
        if value is None:
            continue
        env_key = ENV_KEYS.get(key)
        if not env_key:
            continue
        os.environ[env_key] = str(value)
        os.environ[env_key.lower()] = str(value)
