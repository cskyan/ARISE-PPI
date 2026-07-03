from __future__ import annotations

import argparse
import collections
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments_l131.common import read_table, write_json, write_tsv


def load_coords(path: str) -> np.ndarray:
    payload = np.load(path)
    if isinstance(payload, np.ndarray):
        array = payload
    else:
        key = "coords" if "coords" in payload.files else payload.files[0]
        array = payload[key]
    return np.asarray(array, dtype=np.float32).reshape(-1, 3)


def parse_heavy_atom_residues(path: str, chains: str) -> List[np.ndarray]:
    wanted = set(str(chains).strip())
    residue_order = []
    atom_map = collections.OrderedDict()
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
                continue
            chain = line[21].strip()
            if wanted and chain not in wanted:
                continue
            element = line[76:78].strip().upper() if len(line) >= 78 else line[12:14].strip().upper()
            if element.startswith("H"):
                continue
            altloc = line[16].strip()
            if altloc not in ("", "A"):
                continue
            key = (chain, line[22:26].strip(), line[26].strip())
            if key not in atom_map:
                atom_map[key] = []
                residue_order.append(key)
            try:
                atom_map[key].append([
                    float(line[30:38]), float(line[38:46]), float(line[46:54])
                ])
            except ValueError:
                continue
    return [
        np.asarray(atom_map[key], dtype=np.float32)
        for key in residue_order
        if atom_map[key]
    ]


def heavy_distance_matrix(residues_a: Sequence[np.ndarray], residues_b: Sequence[np.ndarray]) -> np.ndarray:
    matrix = np.full((len(residues_a), len(residues_b)), np.nan, dtype=np.float32)
    for i, atoms_a in enumerate(residues_a):
        for j, atoms_b in enumerate(residues_b):
            delta = atoms_a[:, None, :] - atoms_b[None, :, :]
            matrix[i, j] = float(np.sqrt(np.sum(delta * delta, axis=-1)).min())
    return matrix


def ca_distance_matrix(coords_a: np.ndarray, coords_b: np.ndarray) -> np.ndarray:
    delta = coords_a[:, None, :] - coords_b[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=-1))


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 2:
        return 0.0
    x, y = rankdata(left[valid]), rankdata(right[valid])
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) and np.std(y) else 0.0


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score
    labels = labels.astype(np.int32)
    return float(average_precision_score(labels, scores)) if len(np.unique(labels)) > 1 else 0.0


def sampled_precision(
    contacts: np.ndarray,
    population: np.ndarray,
    count: int,
    repeats: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    values = []
    population = np.asarray(population, dtype=np.int64)
    for _ in range(repeats):
        selected = rng.choice(
            population, size=min(count, len(population)), replace=False
        )
        values.append(float(contacts.reshape(-1)[selected].mean()))
    return float(np.mean(values)), float(np.std(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate native EB structural contact enrichment.")
    parser.add_argument("--structure-manifest", required=True)
    parser.add_argument("--supports", required=True)
    parser.add_argument("--candidates", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--null-repetitions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--heavy-cutoff", type=float, default=5.0)
    parser.add_argument("--ca-cutoff", type=float, default=8.0)
    args = parser.parse_args()

    structures = {row["pair_id"]: row for row in read_table(args.structure_manifest)}
    supports_by_pair = collections.defaultdict(list)
    for row in read_table(args.supports):
        supports_by_pair[row["pair_id"]].append(row)
    candidates_by_pair = collections.defaultdict(list)
    if args.candidates:
        for row in read_table(args.candidates):
            candidates_by_pair[row["pair_id"]].append(row)

    rng = np.random.default_rng(args.seed)
    results, exclusions = [], []
    for pair_id, structure in structures.items():
        supports = supports_by_pair.get(pair_id, [])
        if not supports:
            exclusions.append({"pair_id": pair_id, "reason": "missing_native_supports"})
            continue
        coords_a = load_coords(structure["coords_A"])
        coords_b = load_coords(structure["coords_B"])
        ca_distance = ca_distance_matrix(coords_a, coords_b)
        heavy_distance = None
        if structure.get("pdb_path") and Path(structure["pdb_path"]).exists():
            residues_a = parse_heavy_atom_residues(
                structure["pdb_path"], structure.get("chains_A", "")
            )
            residues_b = parse_heavy_atom_residues(
                structure["pdb_path"], structure.get("chains_B", "")
            )
            if residues_a and residues_b:
                heavy_distance = heavy_distance_matrix(residues_a, residues_b)

        definitions = [("ca", ca_distance, args.ca_cutoff)]
        if heavy_distance is not None:
            definitions.insert(0, ("heavy_atom", heavy_distance, args.heavy_cutoff))

        for definition, distance, cutoff in definitions:
            contact = distance < cutoff
            valid_flat = np.flatnonzero(np.isfinite(distance.reshape(-1)))
            selected_indices, selected_scores, selected_distances = [], [], []
            for support in supports:
                i = int(support["residue_index_A"])
                j = int(support["residue_index_B"])
                if i >= distance.shape[0] or j >= distance.shape[1]:
                    continue
                selected_indices.append(i * distance.shape[1] + j)
                selected_scores.append(float(support["support_score"]))
                selected_distances.append(float(distance[i, j]))
            if not selected_indices:
                exclusions.append({
                    "pair_id": pair_id,
                    "contact_definition": definition,
                    "reason": "support_mapping_failed",
                })
                continue
            selected_indices = np.asarray(selected_indices, dtype=np.int64)
            selected_scores = np.asarray(selected_scores, dtype=np.float64)
            selected_distances = np.asarray(selected_distances, dtype=np.float64)
            selected_contacts = contact.reshape(-1)[selected_indices]
            precision = float(selected_contacts.mean())
            recall = float(selected_contacts.sum() / max(1, int(contact.sum())))
            null_mean, null_std = sampled_precision(
                contact, valid_flat, len(selected_indices), args.null_repetitions, rng
            )

            candidate_rows = candidates_by_pair.get(pair_id, [])
            candidate_indices, candidate_scores = [], []
            for candidate in candidate_rows:
                i = int(candidate["residue_index_A"])
                j = int(candidate["residue_index_B"])
                if i < distance.shape[0] and j < distance.shape[1]:
                    candidate_indices.append(i * distance.shape[1] + j)
                    candidate_scores.append(float(candidate["candidate_score"]))
            candidate_null_mean = 0.0
            candidate_null_std = 0.0
            candidate_auprc = 0.0
            has_candidates = bool(candidate_indices)
            if candidate_indices:
                candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
                candidate_scores = np.asarray(candidate_scores, dtype=np.float64)
                candidate_null_mean, candidate_null_std = sampled_precision(
                    contact, candidate_indices, len(selected_indices),
                    args.null_repetitions, rng,
                )
                candidate_auprc = average_precision(
                    contact.reshape(-1)[candidate_indices], candidate_scores
                )
            results.append({
                "pair_id": pair_id,
                "source": structure.get("source", ""),
                "contact_definition": definition,
                "contact_cutoff": cutoff,
                "supports_mapped": len(selected_indices),
                "precision_at_M": precision,
                "recall_at_M": recall,
                "candidate_auprc": candidate_auprc,
                "median_min_distance": float(np.median(selected_distances)),
                "fraction_within_5A": float((selected_distances < 5.0).mean()),
                "fraction_within_8A": float((selected_distances < 8.0).mean()),
                "fraction_within_10A": float((selected_distances < 10.0).mean()),
                "all_valid_null_precision": null_mean,
                "all_valid_null_sd": null_std,
                "all_valid_fold_enrichment": precision / max(1e-12, null_mean),
                "candidate_null_precision": candidate_null_mean,
                "candidate_null_sd": candidate_null_std,
                "candidate_fold_enrichment": (
                    precision / max(1e-12, candidate_null_mean)
                    if has_candidates
                    else 0.0
                ),
                "support_score_vs_negative_distance_spearman": spearman(
                    selected_scores, -selected_distances
                ),
            })

    # Avoid truth-value ambiguity after candidate_indices becomes an ndarray.
    for row in results:
        if not np.isfinite(float(row["candidate_fold_enrichment"])):
            row["candidate_fold_enrichment"] = 0.0
    summary = {
        "complexes_evaluated": len(results),
        "unique_complexes_evaluated": len({row["pair_id"] for row in results}),
        "complexes_excluded": len(exclusions),
        "macro_precision_at_M": float(np.mean([float(row["precision_at_M"]) for row in results])) if results else 0.0,
        "macro_all_valid_fold_enrichment": float(np.mean([float(row["all_valid_fold_enrichment"]) for row in results])) if results else 0.0,
        "macro_candidate_fold_enrichment": float(np.mean([float(row["candidate_fold_enrichment"]) for row in results])) if results else 0.0,
        "null_repetitions": args.null_repetitions,
    }
    out_dir = Path(args.out_dir)
    write_tsv(str(out_dir / "contact_enrichment_per_complex.tsv"), results)
    write_tsv(str(out_dir / "contact_mapping_exclusions.tsv"), exclusions)
    write_json(str(out_dir / "contact_enrichment_summary.json"), summary)


if __name__ == "__main__":
    main()
