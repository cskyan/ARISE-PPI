from __future__ import annotations

import argparse
import collections
from pathlib import Path

from experiments_l131.common import read_table, write_tsv


def build_pp(root: Path):
    ids = [
        line.strip()
        for line in (root / "all_ids.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    groups = collections.defaultdict(lambda: {"l": [], "r": []})
    for protein_id in ids:
        side = protein_id.rsplit("_", 1)[-1].lower()
        if side not in ("l", "r"):
            continue
        pdb_id = protein_id.split("_", 1)[0].lower()
        groups[pdb_id][side].append(protein_id)
    rows = []
    for pdb_id, sides in sorted(groups.items()):
        for left in sides["l"]:
            for right in sides["r"]:
                coords_a = root / "coords" / f"{left}.npz"
                coords_b = root / "coords" / f"{right}.npz"
                if coords_a.exists() and coords_b.exists():
                    rows.append({
                        "pair_id": f"PP__{pdb_id}__{left}__{right}",
                        "protein_A": left,
                        "protein_B": right,
                        "pdb_id": pdb_id,
                        "coords_A": str(coords_a.resolve()),
                        "coords_B": str(coords_b.resolve()),
                        "pdb_path": "",
                        "chains_A": "",
                        "chains_B": "",
                        "contact_definition_available": "ca",
                        "source": "pp_prepared",
                    })
    return rows


def build_dest(root: Path):
    rows = read_table(str(root / "dest_manifest.tsv"))
    groups = collections.defaultdict(list)
    for row in rows:
        coords = root / "coords" / f"{row['pid']}.npz"
        if row.get("coord_status") == "ok" and coords.exists():
            groups[row["pdb_id"].lower()].append((row, coords))
    output = []
    for pdb_id, members in sorted(groups.items()):
        for index, (left, coords_a) in enumerate(members):
            for right, coords_b in members[index + 1:]:
                pdb_path = root / "pdb_raw" / f"{pdb_id}.pdb"
                output.append({
                    "pair_id": f"DEST__{pdb_id}__{left['pid']}__{right['pid']}",
                    "protein_A": left["pid"],
                    "protein_B": right["pid"],
                    "pdb_id": pdb_id,
                    "coords_A": str(coords_a.resolve()),
                    "coords_B": str(coords_b.resolve()),
                    "pdb_path": str(pdb_path.resolve()) if pdb_path.exists() else "",
                    "chains_A": left.get("chains", ""),
                    "chains_B": right.get("chains", ""),
                    "contact_definition_available": "heavy_atom,ca" if pdb_path.exists() else "ca",
                    "source": "Dest_prepared",
                })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build structural complex manifests.")
    parser.add_argument("--dataset", required=True, choices=("pp", "dest"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    rows = build_pp(root) if args.dataset == "pp" else build_dest(root)
    write_tsv(args.output, rows)
    print(f"[structure-manifest] dataset={args.dataset} pairs={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
