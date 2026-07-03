from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments_l131.common import read_table, write_json, write_tsv


def first_value(row, names):
    for name in names:
        value = str(row.get(name, "")).strip()
        if value:
            return value
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract selected HCC genes and the tested gene universe."
    )
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--module-output", required=True)
    parser.add_argument("--universe-output", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    rows = read_table(args.ledger)
    selected = set()
    universe = set()
    selected_pairs = 0
    for row in rows:
        gene_a = first_value(row, ("gene_A", "protein_A", "uniprot_A"))
        gene_b = first_value(row, ("gene_B", "protein_B", "uniprot_B"))
        universe.update(gene for gene in (gene_a, gene_b) if gene)
        if str(row.get("HCC_ESI_member", "0")).strip() in ("1", "true", "True"):
            selected_pairs += 1
            selected.update(gene for gene in (gene_a, gene_b) if gene)

    write_tsv(args.module_output, [{"gene": gene} for gene in sorted(selected)])
    write_tsv(args.universe_output, [{"gene": gene} for gene in sorted(universe)])
    write_json(args.summary, {
        "ledger_pairs": len(rows),
        "selected_pairs": selected_pairs,
        "selected_genes": len(selected),
        "universe_genes": len(universe),
    })
    print(
        f"[hcc-module] selected_pairs={selected_pairs} "
        f"selected_genes={len(selected)} universe_genes={len(universe)}"
    )


if __name__ == "__main__":
    main()
