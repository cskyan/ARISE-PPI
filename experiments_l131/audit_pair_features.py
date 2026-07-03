from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments_l131.common import read_table, write_json, write_tsv


def find_first(root: Path, directories, protein_id: str, extensions):
    for directory in directories:
        base = root / directory
        for extension in extensions:
            path = base / f"{protein_id}{extension}"
            if path.exists():
                return str(path)
    return ""


def find_structure(root: Path, protein_id: str):
    direct = find_first(
        root,
        ("coords", "structures", "structure", "structures_af", "pdb"),
        protein_id,
        (".npz", ".npz.gz", ".npy", ".pdb", ".ent"),
    )
    if direct:
        return direct
    patterns = (
        root / "structures" / f"AF-{protein_id}-F1-*.pdb",
        root / "structures_af" / f"AF-{protein_id}-F1-*.pdb",
    )
    for pattern in patterns:
        matches = sorted(glob.glob(str(pattern)))
        if matches:
            return matches[0]
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit feature coverage for every protein in a pair manifest."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    pairs = read_table(args.manifest)
    proteins = sorted({
        str(row[key]).strip()
        for row in pairs
        for key in ("protein_A", "protein_B")
        if str(row.get(key, "")).strip()
    })
    rows = []
    def relative_path(value):
        if not value:
            return ""
        try:
            return Path(value).relative_to(root).as_posix()
        except ValueError:
            return Path(value).as_posix()

    for protein_id in proteins:
        sequence = find_first(
            root, ("seq", "SEQ", "fasta"), protein_id,
            (".fa", ".fasta", ".faa", ".txt"),
        )
        pssm = find_first(
            root, ("pssm", "PSSM"), protein_id,
            (".npy", ".npz", ".npz.gz"),
        )
        dssp = find_first(
            root, ("dssp_rsa_asa", "dssp_asa_rsa", "dssp", "DSSP"),
            protein_id, (".npy", ".npz", ".npz.gz"),
        )
        structure = find_structure(root, protein_id)
        rows.append({
            "protein_id": protein_id,
            "sequence_found": int(bool(sequence)),
            "pssm_found": int(bool(pssm)),
            "dssp_found": int(bool(dssp)),
            "structure_found": int(bool(structure)),
            "sequence_path": relative_path(sequence),
            "pssm_path": relative_path(pssm),
            "dssp_path": relative_path(dssp),
            "structure_path": relative_path(structure),
        })
    summary = {
        "root": root.name,
        "pairs": len(pairs),
        "proteins": len(rows),
        "sequence_found": sum(row["sequence_found"] for row in rows),
        "pssm_found": sum(row["pssm_found"] for row in rows),
        "dssp_found": sum(row["dssp_found"] for row in rows),
        "structure_found": sum(row["structure_found"] for row in rows),
        "complete_proteins": sum(
            int(all(row[key] for key in (
                "sequence_found", "pssm_found", "dssp_found", "structure_found"
            )))
            for row in rows
        ),
    }
    write_tsv(args.output, rows)
    write_json(args.summary, summary)
    print(
        f"[feature-audit] complete={summary['complete_proteins']}/"
        f"{summary['proteins']} output={args.output}"
    )
    if args.strict and summary["complete_proteins"] != summary["proteins"]:
        raise SystemExit("Feature audit failed: one or more proteins are incomplete")


if __name__ == "__main__":
    main()
