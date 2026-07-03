from __future__ import annotations

import argparse
from pathlib import Path

from experiments_l131.common import write_json, write_tsv
from experiments_l131.pair_runtime import (
    build_runtime,
    forward_item,
    item_for_row,
    load_pair_manifest,
    native_evidence_record,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export native L131 EvidenceBridge outputs.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--esm-local-dir", default="")
    parser.add_argument("--sequence-mode", default="esm", choices=("esm", "light", "hybrid"))
    parser.add_argument(
        "--pair-head-type",
        default="eb",
        choices=("eb", "no_eb", "attention", "mean", "eb_l2", "gpeh"),
    )
    parser.add_argument(
        "--proposal-mode",
        default="top",
        choices=("top", "random", "low", "shuffled"),
    )
    parser.add_argument(
        "--support-mode",
        default="top",
        choices=("top", "all_candidates"),
    )
    parser.add_argument("--export-candidates", action="store_true")
    parser.add_argument("--require-labels", action="store_true")
    parser.add_argument("--require-native-eb", action="store_true")
    args = parser.parse_args()

    rows = load_pair_manifest(args.manifest, require_labels=args.require_labels)
    proteins = [
        protein
        for row in rows
        for protein in (row["protein_A"], row["protein_B"])
    ]
    runtime = build_runtime(
        args.root,
        args.checkpoint,
        proteins,
        esm_local_dir=args.esm_local_dir,
        sequence_mode=args.sequence_mode,
        pair_head_type=args.pair_head_type,
        require_native_eb=args.require_native_eb,
    )
    predictions, supports, candidates = [], [], []
    for row in rows:
        output = forward_item(
            runtime,
            item_for_row(runtime, row),
            intervention={
                "pair_head_type": args.pair_head_type,
                "proposal_mode": args.proposal_mode,
                "support_mode": args.support_mode,
                "export_candidates": args.export_candidates,
            },
        )
        prediction, pair_supports = native_evidence_record(row, output)
        predictions.append(prediction)
        supports.extend(pair_supports)
        if args.export_candidates and "candidate_score" in output:
            idx_a = output["candidate_idxA"][0].detach().cpu().tolist()
            idx_b = output["candidate_idxB"][0].detach().cpu().tolist()
            values = output["candidate_score"][0].detach().float().cpu().tolist()
            candidates.extend({
                "pair_id": row["pair_id"],
                "residue_index_A": int(a),
                "residue_index_B": int(b),
                "candidate_score": float(value),
            } for a, b, value in zip(idx_a, idx_b, values))

    out_dir = Path(args.out_dir)
    write_tsv(str(out_dir / "pair_mode_native_EB_outputs.tsv"), predictions)
    write_tsv(str(out_dir / "native_EB_supports.tsv"), supports)
    if candidates:
        write_tsv(str(out_dir / "native_EB_candidates.tsv"), candidates)
    write_json(str(out_dir / "checkpoint_provenance.json"), {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_provenance": runtime.provenance,
        "pair_head_type": args.pair_head_type,
        "proposal_mode": args.proposal_mode,
        "support_mode": args.support_mode,
        "pairs": len(rows),
    })


if __name__ == "__main__":
    main()
