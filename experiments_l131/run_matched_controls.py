from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from experiments_l131.common import write_json


VARIANTS = {
    "full_eb": {
        "PAIR_HEAD_TYPE": "eb",
        "EB_PROPOSAL_MODE": "top",
        "EB_SUPPORT_MODE": "top",
        "MANUSCRIPT_PRIMARY": "1",
    },
    "no_eb": {
        "PAIR_HEAD_TYPE": "no_eb",
        "EB_PROPOSAL_MODE": "top",
        "EB_SUPPORT_MODE": "top",
        "MANUSCRIPT_PRIMARY": "0",
    },
    "attention_pool": {
        "PAIR_HEAD_TYPE": "attention",
        "EB_PROPOSAL_MODE": "top",
        "EB_SUPPORT_MODE": "top",
        "MANUSCRIPT_PRIMARY": "0",
    },
    "random_proposals": {
        "PAIR_HEAD_TYPE": "eb",
        "EB_PROPOSAL_MODE": "random",
        "EB_SUPPORT_MODE": "top",
        "MANUSCRIPT_PRIMARY": "0",
    },
    "low_proposals": {
        "PAIR_HEAD_TYPE": "eb",
        "EB_PROPOSAL_MODE": "low",
        "EB_SUPPORT_MODE": "top",
        "MANUSCRIPT_PRIMARY": "0",
    },
    "all_candidates": {
        "PAIR_HEAD_TYPE": "eb",
        "EB_PROPOSAL_MODE": "top",
        "EB_SUPPORT_MODE": "all_candidates",
        "MANUSCRIPT_PRIMARY": "0",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run matched L131 EvidenceBridge controls with fixed splits and seeds."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--seeds", default="1337,2027,3407,4517,5651")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    variants = [value.strip() for value in args.variants.split(",") if value.strip()]
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")

    project_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root).resolve()
    runs = []
    for variant in variants:
        for seed in seeds:
            run_dir = output_root / variant / f"seed_{seed}"
            env_updates = {
                "DATA_ROOT": str(Path(args.data_root).resolve()),
                "PAIR_TRAIN_MANIFEST": str(Path(args.train_manifest).resolve()),
                "PAIR_VAL_MANIFEST": str(Path(args.val_manifest).resolve()),
                "PAIR_TEST_MANIFEST": str(Path(args.test_manifest).resolve()),
                "SAVE_DIR": str(run_dir),
                "DATASET_MODE": "pair",
                "PRIMARY_OBJECTIVE": "pair",
                "TASK_LOSS_PROFILE": "pair",
                "PAIR_EVAL_MAKE_NEGATIVES": "0",
                "TRAIN_NEG_RATIO": "0",
                "SEED": str(seed),
                "SPLIT_SEED": str(seed),
                **VARIANTS[variant],
            }
            command = [args.python, str(project_root / "train_L131.py")]
            record = {
                "variant": variant,
                "seed": seed,
                "output_dir": str(run_dir),
                "command": command,
                "environment": env_updates,
                "status": "planned" if args.dry_run else "running",
            }
            runs.append(record)
            write_json(str(output_root / "matched_control_runs.json"), {"runs": runs})
            if args.dry_run:
                print(f"[dry-run] {variant} seed={seed} -> {run_dir}")
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment.update(env_updates)
            completed = subprocess.run(
                command,
                cwd=str(project_root),
                env=environment,
                check=False,
            )
            record["return_code"] = int(completed.returncode)
            record["status"] = "completed" if completed.returncode == 0 else "failed"
            write_json(str(output_root / "matched_control_runs.json"), {"runs": runs})
            if completed.returncode != 0:
                raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
