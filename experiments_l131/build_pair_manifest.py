from __future__ import annotations

import argparse
import collections
import random
import sys
from pathlib import Path
from typing import Dict, List

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments_l131.common import write_json, write_tsv


def complex_id(protein_id: str) -> str:
    return str(protein_id).split("_", 1)[0].upper()


def load_proteins(root: Path) -> Dict[str, Dict]:
    ids_path = root / "all_ids.txt"
    if not ids_path.exists():
        raise FileNotFoundError(f"Missing protein list: {ids_path}")
    metadata = {}
    manifest_path = root / "pp_manifest.tsv"
    if manifest_path.exists():
        import csv

        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                metadata[str(row.get("pid", "")).strip()] = row

    proteins = {}
    for raw in ids_path.read_text(encoding="utf-8").splitlines():
        protein_id = raw.strip()
        if not protein_id:
            continue
        side = protein_id.rsplit("_", 1)[-1].lower()
        if side not in ("l", "r"):
            continue
        proteins[protein_id] = {
            "protein_id": protein_id,
            "complex_id": complex_id(protein_id),
            "side": side,
            "length": metadata.get(protein_id, {}).get("length", ""),
            "split_source": metadata.get(protein_id, {}).get("split_source", ""),
        }
    return proteins


def build_manifest(root: Path, negative_ratio: int, seed: int):
    proteins = load_proteins(root)
    groups = collections.defaultdict(lambda: {"l": [], "r": []})
    for protein in proteins.values():
        groups[protein["complex_id"]][protein["side"]].append(protein)
    complete_groups = {
        group_id: sides
        for group_id, sides in groups.items()
        if sides["l"] and sides["r"]
    }
    positives: List[Dict] = []
    for group_id, sides in sorted(complete_groups.items()):
        for left in sorted(sides["l"], key=lambda row: row["protein_id"]):
            for right in sorted(sides["r"], key=lambda row: row["protein_id"]):
                positives.append({
                    "pair_id": f"PP_POS__{left['protein_id']}__{right['protein_id']}",
                    "protein_A": left["protein_id"],
                    "protein_B": right["protein_id"],
                    "label": 1,
                    "data_source": "pp_prepared",
                    "evidence_type": "same_complex_structure_positive",
                    "label_scope": "structural_positive",
                    "complex_id_A": group_id,
                    "complex_id_B": group_id,
                    "length_A": left["length"],
                    "length_B": right["length"],
                })

    group_ids = sorted(complete_groups)
    rng = random.Random(int(seed))
    rng.shuffle(group_ids)
    partner = {}
    for index in range(0, len(group_ids) - 1, 2):
        left_group = group_ids[index]
        right_group = group_ids[index + 1]
        partner[left_group] = right_group
        partner[right_group] = left_group
    if len(group_ids) % 2:
        partner[group_ids[-1]] = group_ids[0]

    negatives: List[Dict] = []
    seen = set()
    for repetition in range(max(0, int(negative_ratio))):
        for positive in positives:
            group_a = positive["complex_id_A"]
            group_b = partner[group_a]
            candidates = sorted(
                complete_groups[group_b]["r"],
                key=lambda row: row["protein_id"],
            )
            if not candidates:
                continue
            offset = (
                repetition
                + sum(ord(char) for char in positive["protein_A"])
            ) % len(candidates)
            right = candidates[offset]
            key = tuple(sorted((positive["protein_A"], right["protein_id"])))
            if key in seen:
                continue
            seen.add(key)
            negatives.append({
                "pair_id": f"PP_NEG__{positive['protein_A']}__{right['protein_id']}",
                "protein_A": positive["protein_A"],
                "protein_B": right["protein_id"],
                "label": 0,
                "data_source": "pp_prepared",
                "evidence_type": "cross_complex_frozen_synthetic_negative",
                "label_scope": "synthetic_negative_not_experimentally_verified",
                "complex_id_A": group_a,
                "complex_id_B": group_b,
                "length_A": positive["length_A"],
                "length_B": right["length"],
            })
    return positives + negatives, {
        "root": root.name,
        "seed": int(seed),
        "negative_ratio_requested": int(negative_ratio),
        "proteins": len(proteins),
        "complete_complexes": len(complete_groups),
        "positive_pairs": len(positives),
        "synthetic_negative_pairs": len(negatives),
        "negative_label_warning": (
            "Cross-complex negatives are frozen synthetic controls and are not "
            "experimentally verified non-interactions."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a fixed labeled pair manifest from pp_prepared complexes."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--negative-ratio", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    rows, summary = build_manifest(
        Path(args.root),
        negative_ratio=args.negative_ratio,
        seed=args.seed,
    )
    write_tsv(args.output, rows)
    summary_path = args.summary or str(Path(args.output).with_suffix(".summary.json"))
    write_json(summary_path, summary)
    print(
        f"[pair-manifest] positives={summary['positive_pairs']} "
        f"synthetic_negatives={summary['synthetic_negative_pairs']} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
