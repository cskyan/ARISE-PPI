from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments_l131.common import read_table, write_json, write_tsv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the complete resumable L131 revision experiment pipeline."
    )
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument("--pp-root", default="")
    parser.add_argument("--rbp-root", default="")
    parser.add_argument("--hcc-manifest", default="")
    parser.add_argument("--string-network", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--sequence-mode", default="hybrid", choices=("esm", "light", "hybrid"))
    parser.add_argument("--esm-local-dir", default="")
    parser.add_argument("--seeds", default="1337,2027,3407,4517,5651")
    parser.add_argument(
        "--variants",
        default="full_eb,no_eb,attention_pool,random_proposals,low_proposals,all_candidates",
    )
    parser.add_argument("--split-seed", type=int, default=1337)
    parser.add_argument("--random-repetitions", type=int, default=100)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-hcc", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    project = Path(args.project_root).resolve()
    pp_root = Path(args.pp_root or project / "pp_prepared").resolve()
    rbp_root = Path(args.rbp_root or project / "RBP400").resolve()
    hcc_manifest = Path(
        args.hcc_manifest or rbp_root / "pairs_strict" / "hcc_all_pairs.tsv"
    ).resolve()
    string_network = Path(
        args.string_network or rbp_root / "annotations" / "string_network.tsv"
    ).resolve()
    output = Path(args.output_root or project / "results" / "plos_revision").resolve()
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    os.environ["ARISE_PPI_ROOT"] = str(project)
    os.environ["SEQUENCE_MODE"] = args.sequence_mode
    os.environ["STRUCTURE_SOURCE"] = "auto"
    os.environ["ESM_CACHE_DIR"] = str(pp_root / "esm_cache")
    if args.esm_local_dir:
        os.environ["ESM_LOCAL_DIR"] = args.esm_local_dir
    records = []
    state_path = output / "pipeline_status.json"

    def save_state():
        write_json(str(state_path), {
            "project_root": str(project),
            "output_root": str(output),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stages": records,
        })

    def run_stage(name, command, expected=None, environment=None):
        expected_paths = [Path(value) for value in (expected or [])]
        record = {
            "stage": name,
            "command": [str(value) for value in command],
            "expected_outputs": [str(value) for value in expected_paths],
        }
        records.append(record)
        if expected_paths and all(path.exists() for path in expected_paths) and not args.force:
            record["status"] = "skipped_existing_outputs"
            save_state()
            print(f"[pipeline][skip] {name}")
            return
        log_path = logs / f"{name}.log"
        env = os.environ.copy()
        if environment:
            env.update({key: str(value) for key, value in environment.items()})
        print(f"[pipeline][run] {name}")
        started = time.time()
        with log_path.open("w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                [str(value) for value in command],
                cwd=str(project),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record.update({
            "status": "completed" if completed.returncode == 0 else "failed",
            "return_code": int(completed.returncode),
            "elapsed_seconds": float(time.time() - started),
            "log": str(log_path),
        })
        save_state()
        if completed.returncode != 0:
            raise SystemExit(
                f"Stage {name!r} failed. Inspect {log_path}"
            )

    manifest = project / "manifests" / "pp_prepared_labeled_pairs.tsv"
    manifest_summary = manifest.with_suffix(".summary.json")
    split_dir = project / "results" / "splits" / f"pp_component_seed{args.split_seed}"
    run_stage(
        "01_build_pair_manifest",
        [
            python, "-m", "experiments_l131.build_pair_manifest",
            "--root", pp_root,
            "--output", manifest,
            "--summary", manifest_summary,
            "--negative-ratio", "1",
            "--seed", str(args.split_seed),
        ],
        expected=[manifest, manifest_summary],
    )
    train_manifest = split_dir / "train.tsv"
    val_manifest = split_dir / "val.tsv"
    test_manifest = split_dir / "test.tsv"
    feature_audit = project / "manifests" / "pp_prepared_feature_audit.tsv"
    run_stage(
        "01b_audit_pair_features",
        [
            python, "-m", "experiments_l131.audit_pair_features",
            "--root", pp_root,
            "--manifest", manifest,
            "--output", feature_audit,
            "--summary", project / "manifests" / "pp_prepared_feature_audit.summary.json",
            "--strict",
        ],
        expected=[
            feature_audit,
            project / "manifests" / "pp_prepared_feature_audit.summary.json",
        ],
    )
    run_stage(
        "02_build_component_splits",
        [
            python, "-m", "experiments_l131.build_splits",
            "--manifest", manifest,
            "--strategy", "component",
            "--seed", str(args.split_seed),
            "--ratios", "0.7,0.15,0.15",
            "--out-dir", split_dir,
        ],
        expected=[
            train_manifest,
            val_manifest,
            test_manifest,
            split_dir / "split_audit.json",
        ],
    )
    if args.prepare_only:
        save_state()
        print(f"[pipeline] preparation complete: {state_path}")
        return

    controls = output / "controls"
    control_variants = [
        value.strip() for value in args.variants.split(",") if value.strip()
    ]
    control_seeds = [
        value.strip() for value in args.seeds.split(",") if value.strip()
    ]
    control_command = [
        python, "-m", "experiments_l131.run_matched_controls",
        "--data-root", pp_root,
        "--train-manifest", train_manifest,
        "--val-manifest", val_manifest,
        "--test-manifest", test_manifest,
        "--output-root", controls,
        "--seeds", args.seeds,
        "--variants", args.variants,
        "--python", python,
    ]
    if args.force:
        control_command.append("--force")
    run_stage(
        "03_train_matched_controls",
        control_command,
        expected=[
            controls / variant / f"seed_{seed}" /
            "checkpoints" / "best_PairAUPRC.pt"
            for variant in control_variants
            for seed in control_seeds
        ],
        environment={
            "SEQUENCE_MODE": args.sequence_mode,
            "ESM_LOCAL_DIR": args.esm_local_dir,
            "ARISE_PPI_ROOT": project,
        },
    )

    analysis_seed = control_seeds[0]
    checkpoint = (
        controls / "full_eb" / f"seed_{analysis_seed}" /
        "checkpoints" / "best_PairAUPRC.pt"
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"Primary checkpoint was not produced: {checkpoint}")

    def model_command(module, manifest_path, out_dir, extra=None, root=None):
        command = [
            python, "-m", module,
            "--root", root or pp_root,
            "--checkpoint", checkpoint,
            "--manifest", manifest_path,
            "--out-dir", out_dir,
            "--sequence-mode", args.sequence_mode,
        ]
        if args.esm_local_dir:
            command.extend(["--esm-local-dir", args.esm_local_dir])
        command.extend(extra or [])
        return command

    validation_native = output / "validation" / "native_eb"
    validation_symmetry = output / "validation" / "symmetry"
    validation_faithfulness = output / "validation" / "faithfulness"
    run_stage(
        "04_validation_native_eb",
        model_command(
            "experiments_l131.export_native_eb",
            val_manifest,
            validation_native,
            ["--require-labels", "--require-native-eb", "--export-candidates"],
        ),
        expected=[validation_native / "pair_mode_native_EB_outputs.tsv"],
    )
    run_stage(
        "05_validation_symmetry",
        model_command(
            "experiments_l131.evaluate_symmetry",
            val_manifest,
            validation_symmetry,
            ["--require-labels"],
        ),
        expected=[validation_symmetry / "swap_symmetry_per_pair.tsv"],
    )
    run_stage(
        "06_validation_faithfulness",
        model_command(
            "experiments_l131.evaluate_faithfulness",
            val_manifest,
            validation_faithfulness,
            [
                "--require-labels", "--cohort", "all",
                "--random-repetitions", str(args.random_repetitions),
            ],
        ),
        expected=[validation_faithfulness / "faithfulness_per_pair.tsv"],
    )

    test_native = output / "test" / "native_eb"
    test_symmetry = output / "test" / "symmetry"
    test_faithfulness = output / "test" / "faithfulness"
    run_stage(
        "07_test_native_eb",
        model_command(
            "experiments_l131.export_native_eb",
            test_manifest,
            test_native,
            ["--require-labels", "--require-native-eb", "--export-candidates"],
        ),
        expected=[test_native / "native_EB_supports.tsv"],
    )
    run_stage(
        "08_test_symmetry",
        model_command(
            "experiments_l131.evaluate_symmetry",
            test_manifest,
            test_symmetry,
            ["--require-labels"],
        ),
        expected=[test_symmetry / "swap_symmetry_summary.json"],
    )
    run_stage(
        "09_test_faithfulness",
        model_command(
            "experiments_l131.evaluate_faithfulness",
            test_manifest,
            test_faithfulness,
            [
                "--require-labels", "--cohort", "predicted_positive",
                "--random-repetitions", str(args.random_repetitions),
            ],
        ),
        expected=[test_faithfulness / "faithfulness_aopc_per_pair.tsv"],
    )

    structure_dir = output / "structure"
    structure_manifest = structure_dir / "pp_structure_manifest.tsv"
    run_stage(
        "10_build_structure_manifest",
        [
            python, "-m", "experiments_l131.build_structure_manifest",
            "--dataset", "pp",
            "--root", pp_root,
            "--output", structure_manifest,
        ],
        expected=[structure_manifest],
    )
    run_stage(
        "11_contact_enrichment",
        [
            python, "-m", "experiments_l131.evaluate_contacts",
            "--structure-manifest", structure_manifest,
            "--supports", test_native / "native_EB_supports.tsv",
            "--candidates", test_native / "native_EB_candidates.tsv",
            "--out-dir", structure_dir / "contact_enrichment",
            "--null-repetitions", "1000",
            "--seed", str(args.split_seed),
        ],
        expected=[structure_dir / "contact_enrichment" / "contact_enrichment_summary.json"],
    )

    if args.skip_hcc or not hcc_manifest.exists():
        records.append({
            "stage": "12_hcc_pipeline",
            "status": "skipped",
            "reason": "disabled or HCC manifest missing",
        })
        run_stage(
            "19_summarize_revision_results",
            [
                python, "-m", "experiments_l131.summarize_results",
                "--results-root", output,
                "--output-dir", output / "summary",
            ],
            expected=[output / "summary" / "revision_results_summary.json"],
        )
        save_state()
        return

    hcc_native = output / "hcc" / "native_eb"
    run_stage(
        "12_hcc_native_eb",
        model_command(
            "experiments_l131.export_native_eb",
            hcc_manifest,
            hcc_native,
            ["--require-native-eb"],
            root=rbp_root,
        ),
        expected=[hcc_native / "pair_mode_native_EB_outputs.tsv"],
    )
    hcc_selected = output / "hcc" / "selected_candidates.tsv"
    run_stage(
        "13_filter_hcc_candidates",
        [
            python, "-m", "experiments_l131.filter_pair_predictions",
            "--predictions", hcc_native / "pair_mode_native_EB_outputs.tsv",
            "--validation-predictions",
            validation_native / "pair_mode_native_EB_outputs.tsv",
            "--output", hcc_selected,
            "--summary", output / "hcc" / "candidate_filter_summary.json",
            "--quantile", "0.90",
        ],
        expected=[hcc_selected],
    )
    selected_count = len(read_table(str(hcc_selected))) if hcc_selected.exists() else 0
    hcc_symmetry = output / "hcc" / "symmetry"
    hcc_faithfulness = output / "hcc" / "faithfulness"
    if selected_count:
        run_stage(
            "14_hcc_symmetry",
            model_command(
                "experiments_l131.evaluate_symmetry",
                hcc_selected,
                hcc_symmetry,
                [],
                root=rbp_root,
            ),
            expected=[hcc_symmetry / "swap_symmetry_per_pair.tsv"],
        )
        run_stage(
            "15_hcc_faithfulness",
            model_command(
                "experiments_l131.evaluate_faithfulness",
                hcc_selected,
                hcc_faithfulness,
                [
                    "--cohort", "all",
                    "--random-repetitions", str(args.random_repetitions),
                ],
                root=rbp_root,
            ),
            expected=[hcc_faithfulness / "faithfulness_per_pair.tsv"],
        )
    else:
        write_tsv(
            str(hcc_symmetry / "swap_symmetry_per_pair.tsv"),
            [],
            fieldnames=("pair_id", "abs_probability_difference"),
        )
        write_tsv(
            str(hcc_faithfulness / "faithfulness_per_pair.tsv"),
            [],
            fieldnames=(
                "pair_id", "analysis", "method", "level",
                "comprehensiveness_probability",
                "sufficiency_error_probability",
            ),
        )
        records.append({
            "stage": "14_15_hcc_mechanism_analysis",
            "status": "completed_empty_cohort",
            "selected_pairs": 0,
        })
        save_state()

    hcc_ledger = output / "hcc" / "HCC_ESI_ledger.tsv"
    run_stage(
        "16_build_hcc_esi",
        [
            python, "-m", "experiments_l131.build_hcc_esi",
            "--predictions", hcc_native / "pair_mode_native_EB_outputs.tsv",
            "--faithfulness", hcc_faithfulness / "faithfulness_per_pair.tsv",
            "--symmetry", hcc_symmetry / "swap_symmetry_per_pair.tsv",
            "--validation-predictions",
            validation_native / "pair_mode_native_EB_outputs.tsv",
            "--validation-faithfulness",
            validation_faithfulness / "faithfulness_per_pair.tsv",
            "--validation-symmetry",
            validation_symmetry / "swap_symmetry_per_pair.tsv",
            "--output", hcc_ledger,
            "--summary", output / "hcc" / "HCC_ESI_summary.json",
        ],
        expected=[hcc_ledger, output / "hcc" / "HCC_ESI_summary.json"],
    )
    selected_module = output / "hcc" / "selected_module.tsv"
    universe = output / "hcc" / "candidate_universe.tsv"
    run_stage(
        "17_extract_hcc_module",
        [
            python, "-m", "experiments_l131.extract_hcc_module",
            "--ledger", hcc_ledger,
            "--module-output", selected_module,
            "--universe-output", universe,
            "--summary", output / "hcc" / "module_summary.json",
        ],
        expected=[selected_module, universe],
    )
    if len(read_table(str(selected_module))) and string_network.exists():
        run_stage(
            "18_degree_matched_null",
            [
                python, "-m", "experiments_l131.degree_matched_null",
                "--network", string_network,
                "--module", selected_module,
                "--universe", universe,
                "--out-dir", output / "hcc" / "degree_null",
                "--repetitions", "1000",
                "--seed", str(args.split_seed),
            ],
            expected=[
                output / "hcc" / "degree_null" /
                "degree_matched_module_summary.json"
            ],
        )
    run_stage(
        "19_summarize_revision_results",
        [
            python, "-m", "experiments_l131.summarize_results",
            "--results-root", output,
            "--output-dir", output / "summary",
        ],
        expected=[output / "summary" / "revision_results_summary.json"],
    )
    save_state()
    print(f"[pipeline] complete: {state_path}")


if __name__ == "__main__":
    main()
