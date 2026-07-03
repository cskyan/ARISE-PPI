from __future__ import annotations

import argparse
import collections
import random
from pathlib import Path

import numpy as np

from experiments_l131.common import read_table, write_json, write_tsv


def choose_column(row, candidates):
    for name in candidates:
        if name in row:
            return name
    raise ValueError(f"None of the expected columns were found: {candidates}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate degree-matched random modules.")
    parser.add_argument("--network", required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--universe", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument("--degree-bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    network = read_table(args.network)
    if not network:
        raise ValueError("Network is empty")
    left_key = choose_column(
        network[0], ("protein_A", "gene_A", "preferredName_A", "stringId_A")
    )
    right_key = choose_column(
        network[0], ("protein_B", "gene_B", "preferredName_B", "stringId_B")
    )
    adjacency = collections.defaultdict(set)
    for row in network:
        left = str(row[left_key]).strip()
        right = str(row[right_key]).strip()
        if left and right and left != right:
            adjacency[left].add(right)
            adjacency[right].add(left)

    module_rows = read_table(args.module)
    universe_rows = read_table(args.universe)
    module_key = choose_column(
        module_rows[0], ("gene", "gene_name", "protein", "protein_id")
    )
    universe_key = choose_column(
        universe_rows[0], ("gene", "gene_name", "protein", "protein_id")
    )
    module = list(dict.fromkeys(str(row[module_key]).strip() for row in module_rows))
    universe = list(dict.fromkeys(str(row[universe_key]).strip() for row in universe_rows))
    universe = [gene for gene in universe if gene]
    degrees = {gene: len(adjacency.get(gene, set())) for gene in universe}
    degree_values = np.asarray(list(degrees.values()), dtype=np.float64)
    edges = np.unique(np.quantile(
        degree_values,
        np.linspace(0.0, 1.0, max(2, args.degree_bins) + 1),
    ))
    if len(edges) < 2:
        edges = np.asarray([degree_values.min(), degree_values.max() + 1.0])

    def degree_bin(gene):
        return int(np.searchsorted(edges[1:-1], degrees.get(gene, 0), side="right"))

    pools = collections.defaultdict(list)
    for gene in universe:
        pools[degree_bin(gene)].append(gene)
    rng = random.Random(args.seed)
    output, module_stats = [], []
    observed_edges = sum(
        1
        for i, left in enumerate(module)
        for right in module[i + 1:]
        if right in adjacency.get(left, set())
    )
    for repetition in range(args.repetitions):
        selected, used = [], set()
        for observed in module:
            target_bin = degree_bin(observed)
            candidates = [
                gene for gene in pools[target_bin]
                if gene not in used and gene not in module
            ]
            if not candidates:
                candidates = [
                    gene for gene in universe
                    if gene not in used and gene not in module
                ]
            replacement = rng.choice(candidates)
            selected.append(replacement)
            used.add(replacement)
            output.append({
                "module_id": repetition,
                "observed_gene": observed,
                "replacement_gene": replacement,
                "degree_bin": target_bin,
                "observed_degree": degrees.get(observed, 0),
                "replacement_degree": degrees.get(replacement, 0),
            })
        sampled_edges = sum(
            1
            for i, left in enumerate(selected)
            for right in selected[i + 1:]
            if right in adjacency.get(left, set())
        )
        module_stats.append({
            "module_id": repetition,
            "node_count": len(selected),
            "edge_count": sampled_edges,
            "density": (
                2.0 * sampled_edges / max(1, len(selected) * (len(selected) - 1))
            ),
        })

    out_dir = Path(args.out_dir)
    write_tsv(str(out_dir / "degree_matched_module_null.tsv"), output)
    write_tsv(str(out_dir / "degree_matched_module_stats.tsv"), module_stats)
    write_json(str(out_dir / "degree_matched_module_summary.json"), {
        "repetitions": args.repetitions,
        "degree_bins": args.degree_bins,
        "module_nodes": len(module),
        "observed_edges": observed_edges,
        "observed_density": (
            2.0 * observed_edges / max(1, len(module) * (len(module) - 1))
        ),
        "network_nodes": len(adjacency),
        "network_edges": sum(len(value) for value in adjacency.values()) // 2,
    })


if __name__ == "__main__":
    main()
