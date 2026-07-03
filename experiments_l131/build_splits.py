from __future__ import annotations

import argparse
import collections
import hashlib
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments_l131.common import (
    canonicalize_pairs,
    read_table,
    write_json,
    write_tsv,
)


SPLITS = ("train", "val", "test")


def stable_int(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def stratified_pair_random(rows: Sequence[Dict], seed: int, ratios) -> Tuple[List[Dict], List[Dict]]:
    groups = collections.defaultdict(list)
    for row in rows:
        groups[int(row["label"])].append(row)
    output = []
    for label_rows in groups.values():
        ordered = sorted(label_rows, key=lambda r: stable_int(r["pair_id"], seed))
        n = len(ordered)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        for index, row in enumerate(ordered):
            split = "train" if index < n_train else ("val" if index < n_train + n_val else "test")
            output.append({**row, "split": split})
    return output, []


def partition_units(units: Iterable[str], seed: int, ratios) -> Dict[str, str]:
    ordered = sorted(set(units), key=lambda value: stable_int(value, seed))
    n = len(ordered)
    boundaries = (
        int(round(n * ratios[0])),
        int(round(n * (ratios[0] + ratios[1]))),
    )
    result = {}
    for index, unit in enumerate(ordered):
        result[unit] = "train" if index < boundaries[0] else ("val" if index < boundaries[1] else "test")
    return result


def one_unseen(rows: Sequence[Dict], seed: int, ratios) -> Tuple[List[Dict], List[Dict]]:
    proteins = {row["protein_A"] for row in rows} | {row["protein_B"] for row in rows}
    assignment = partition_units(proteins, seed, ratios)
    output = []
    for row in rows:
        sa = assignment[row["protein_A"]]
        sb = assignment[row["protein_B"]]
        if "test" in (sa, sb):
            split = "test"
        elif "val" in (sa, sb):
            split = "val"
        else:
            split = "train"
        output.append({**row, "split": split})
    return output, []


def load_clusters(path: str | None) -> Dict[str, str]:
    if not path:
        return {}
    rows = read_table(path)
    if not rows:
        return {}
    protein_key = "protein_id" if "protein_id" in rows[0] else "protein"
    cluster_key = "cluster_id" if "cluster_id" in rows[0] else "cluster"
    return {
        str(row[protein_key]).strip(): str(row[cluster_key]).strip()
        for row in rows
    }


def both_unseen(rows: Sequence[Dict], seed: int, ratios, clusters=None) -> Tuple[List[Dict], List[Dict]]:
    clusters = clusters or {}
    proteins = {row["protein_A"] for row in rows} | {row["protein_B"] for row in rows}
    unit = {protein: clusters.get(protein, protein) for protein in proteins}
    assignment = partition_units(unit.values(), seed, ratios)
    output, removed = [], []
    for row in rows:
        sa = assignment[unit[row["protein_A"]]]
        sb = assignment[unit[row["protein_B"]]]
        if sa != sb:
            removed.append({**row, "exclusion_reason": "cross_partition_pair"})
        else:
            output.append({**row, "split": sa})
    return output, removed


def connected_components(rows: Sequence[Dict]) -> List[Set[str]]:
    adjacency = collections.defaultdict(set)
    for row in rows:
        a, b = row["protein_A"], row["protein_B"]
        adjacency[a].add(b)
        adjacency[b].add(a)
    seen, components = set(), []
    for node in adjacency:
        if node in seen:
            continue
        stack, component = [node], set()
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.add(current)
            stack.extend(adjacency[current] - seen)
        components.append(component)
    return components


def component_disjoint(rows: Sequence[Dict], seed: int, ratios) -> Tuple[List[Dict], List[Dict]]:
    components = connected_components(rows)
    node_to_component = {}
    for index, component in enumerate(components):
        for node in component:
            node_to_component[node] = str(index)
    assignment = partition_units(node_to_component.values(), seed, ratios)
    output = [
        {**row, "split": assignment[node_to_component[row["protein_A"]]]}
        for row in rows
    ]
    return output, []


def anchor_disjoint(rows: Sequence[Dict], seed: int, ratios) -> Tuple[List[Dict], List[Dict]]:
    if not rows or "anchor_id" not in rows[0]:
        raise ValueError("anchor_disjoint requires an anchor_id column")
    anchors = {str(row.get("anchor_id", "")).strip() for row in rows}
    if "" in anchors:
        raise ValueError("anchor_disjoint requires a non-empty anchor_id for every row")
    assignment = partition_units(anchors, seed, ratios)
    return [
        {**row, "split": assignment[str(row["anchor_id"]).strip()]}
        for row in rows
    ], []


def split_audit(rows: Sequence[Dict]) -> Dict:
    by_split = {split: [row for row in rows if row["split"] == split] for split in SPLITS}
    proteins = {
        split: {
            protein
            for row in split_rows
            for protein in (row["protein_A"], row["protein_B"])
        }
        for split, split_rows in by_split.items()
    }
    anchors = {
        split: {
            str(row.get("anchor_id", "")).strip()
            for row in split_rows
            if str(row.get("anchor_id", "")).strip()
        }
        for split, split_rows in by_split.items()
    }
    summary = {}
    for split, split_rows in by_split.items():
        positives = sum(int(row["label"]) for row in split_rows)
        degrees = collections.Counter(
            protein
            for row in split_rows
            for protein in (row["protein_A"], row["protein_B"])
        )
        summary[split] = {
            "pairs": len(split_rows),
            "positives": positives,
            "prevalence": positives / max(1, len(split_rows)),
            "unique_proteins": len(proteins[split]),
            "unique_anchors": len(anchors[split]),
            "degree_min": min(degrees.values()) if degrees else 0,
            "degree_median": sorted(degrees.values())[len(degrees) // 2] if degrees else 0,
            "degree_max": max(degrees.values()) if degrees else 0,
        }
    overlap = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap[f"protein_{left}_{right}"] = sorted(proteins[left] & proteins[right])
        overlap[f"anchor_{left}_{right}"] = sorted(anchors[left] & anchors[right])
    return {
        "splits": summary,
        "overlap": overlap,
        "connected_components": len(connected_components(rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-aware pair splits.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--strategy",
        required=True,
        choices=("pair_random", "one_unseen", "both_unseen", "homology", "component", "anchor"),
    )
    parser.add_argument("--clusters", help="TSV with protein_id and cluster_id.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--ratios", default="0.8,0.1,0.1")
    args = parser.parse_args()

    ratios = tuple(float(value) for value in args.ratios.split(","))
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("--ratios must contain three values summing to 1")
    rows, rejected = canonicalize_pairs(read_table(args.manifest))
    if args.strategy == "pair_random":
        split_rows, removed = stratified_pair_random(rows, args.seed, ratios)
    elif args.strategy == "one_unseen":
        split_rows, removed = one_unseen(rows, args.seed, ratios)
    elif args.strategy == "both_unseen":
        split_rows, removed = both_unseen(rows, args.seed, ratios)
    elif args.strategy == "homology":
        clusters = load_clusters(args.clusters)
        if not clusters:
            raise ValueError("--clusters is required for homology strategy")
        split_rows, removed = both_unseen(rows, args.seed, ratios, clusters=clusters)
    elif args.strategy == "component":
        split_rows, removed = component_disjoint(rows, args.seed, ratios)
    else:
        split_rows, removed = anchor_disjoint(rows, args.seed, ratios)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(str(out_dir / "split_manifest.tsv"), split_rows)
    for split in SPLITS:
        write_tsv(
            str(out_dir / f"{split}.tsv"),
            [
                {key: value for key, value in row.items() if key != "split"}
                for row in split_rows
                if row["split"] == split
            ],
        )
    write_tsv(str(out_dir / "excluded_pairs.tsv"), rejected + removed)
    audit = split_audit(split_rows)
    audit["strategy"] = args.strategy
    audit["seed"] = args.seed
    audit["ratios"] = ratios
    audit["input_pairs"] = len(rows)
    audit["excluded_pairs"] = len(rejected) + len(removed)
    write_json(str(out_dir / "split_audit.json"), audit)

    if args.strategy in ("both_unseen", "homology", "component"):
        violations = (
            audit["overlap"]["protein_train_val"]
            + audit["overlap"]["protein_train_test"]
            + audit["overlap"]["protein_val_test"]
        )
        if violations:
            raise RuntimeError(f"Protein leakage detected: {violations[:10]}")
    if args.strategy == "anchor":
        anchor_violations = (
            audit["overlap"]["anchor_train_val"]
            + audit["overlap"]["anchor_train_test"]
            + audit["overlap"]["anchor_val_test"]
        )
        if anchor_violations:
            raise RuntimeError("Anchor leakage detected")
    counts = audit["splits"]
    print(
        f"[split] strategy={args.strategy} "
        f"train={counts['train']['pairs']} "
        f"val={counts['val']['pairs']} "
        f"test={counts['test']['pairs']} "
        f"excluded={audit['excluded_pairs']} out={out_dir}"
    )


if __name__ == "__main__":
    main()
