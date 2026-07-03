from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from experiments_l131.common import as_binary_label, read_table


@dataclass
class RuntimeBundle:
    model: torch.nn.Module
    trainlib: object
    dataset: object
    samples: Dict[str, Dict]
    device: str
    checkpoint: Dict
    provenance: Dict


def sample_to_pair(sample_a: Dict, sample_b: Dict, label: int = 0, pair_id: str = "") -> Dict:
    return {
        "complex": pair_id or f"{sample_a['complex']}__{sample_b['complex']}",
        "resA": sample_a["resA"],
        "resB": sample_b["resA"],
        "coordsA": sample_a["coordsA"],
        "coordsB": sample_b["coordsA"],
        "maskA": sample_a["maskA"],
        "maskB": sample_b["maskA"],
        "chainA": sample_a.get("chainA"),
        "chainB": sample_b.get("chainA"),
        "y_res_A": sample_a.get("y_res_A", torch.zeros_like(sample_a["maskA"])),
        "y_res_B": sample_b.get("y_res_A", torch.zeros_like(sample_b["maskA"])),
        "y_pair": torch.tensor(float(label), dtype=torch.float32),
        "has_contact": torch.tensor(float(label), dtype=torch.float32),
        "site_mode": torch.tensor(0.0, dtype=torch.float32),
    }


def clone_pair_item(item: Dict) -> Dict:
    output = {}
    for key, value in item.items():
        output[key] = value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
    return output


def batch_to_device(batch: Dict, device: str) -> Tuple:
    res_a = torch.nan_to_num(batch["resA"], nan=0.0).to(device).float()
    res_b = torch.nan_to_num(batch["resB"], nan=0.0).to(device).float()
    coords_a = torch.nan_to_num(batch["coordsA"], nan=0.0).to(device).float()
    coords_b = torch.nan_to_num(batch["coordsB"], nan=0.0).to(device).float()
    mask_a = (batch["maskA"] > 0.5).to(device)
    mask_b = (batch["maskB"] > 0.5).to(device)
    chain_a = batch.get("chainA")
    chain_b = batch.get("chainB")
    if chain_a is not None:
        chain_a = torch.nan_to_num(chain_a).to(device).float()
    if chain_b is not None:
        chain_b = torch.nan_to_num(chain_b).to(device).float()
    return res_a, mask_a, chain_a, coords_a, res_b, mask_b, chain_b, coords_b


def forward_item(
    runtime: RuntimeBundle,
    item: Dict,
    intervention: Optional[Dict] = None,
) -> Dict:
    batch = runtime.trainlib.dips_collate([item])
    args = batch_to_device(batch, runtime.device)
    with torch.no_grad():
        return runtime.model(
            *args,
            site_mode=False,
            intervention=intervention or {},
        )


def load_pair_manifest(path: str, require_labels: bool = True) -> List[Dict]:
    rows = read_table(path)
    output = []
    for index, source in enumerate(rows):
        row = dict(source)
        a = str(row.get("protein_A", "")).strip()
        b = str(row.get("protein_B", "")).strip()
        if not a or not b:
            raise ValueError(f"Row {index + 2} is missing protein_A/protein_B")
        if require_labels:
            row["label"] = as_binary_label(row.get("label"))
        elif str(row.get("label", "")).strip():
            row["label"] = as_binary_label(row["label"])
        else:
            row["label"] = -1
        row["pair_id"] = str(row.get("pair_id") or f"{a}__{b}")
        row["protein_A"] = a
        row["protein_B"] = b
        output.append(row)
    return output


def build_runtime(
    root: str,
    checkpoint: str,
    protein_ids: Sequence[str],
    esm_local_dir: str = "",
    sequence_mode: str = "esm",
    pair_head_type: Optional[str] = None,
    require_native_eb: bool = False,
    device: Optional[str] = None,
) -> RuntimeBundle:
    import train_L131 as trainlib
    from model_L131 import L13PDBGVPModel

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    trainlib.P.rbp_root = os.path.abspath(root)
    trainlib.P.dips_root = os.path.abspath(root)
    trainlib.P.sequence_mode = str(sequence_mode).lower()
    if esm_local_dir:
        trainlib.P.esm_local_dir = esm_local_dir

    embedder = None
    if trainlib.P.sequence_mode in ("esm", "hybrid"):
        try:
            embedder = trainlib.SiteEmbedder(
                device=device,
                esm_local_dir=trainlib.P.esm_local_dir,
            )
        except Exception:
            if not bool(getattr(trainlib.P, "allow_zero_esm_fallback", False)):
                raise

    unique_ids = list(dict.fromkeys(str(value) for value in protein_ids))
    dataset = trainlib.RBP296Dataset(
        root,
        unique_ids,
        embedder=embedder,
        use_pssm=bool(getattr(trainlib.P, "use_pssm", True)),
        use_dssp=bool(getattr(trainlib.P, "use_dssp", True)),
        esm_cache_dir=os.path.join(os.path.dirname(os.path.abspath(checkpoint)), "esm_cache"),
        verbose=False,
    )
    samples = {}
    for index in range(len(dataset)):
        sample = dataset[index]
        samples[str(sample["complex"])] = sample
    missing_ids = sorted(set(unique_ids) - set(samples))
    if missing_ids:
        raise RuntimeError(f"Failed to load {len(missing_ids)} proteins: {missing_ids[:10]}")

    first = samples[unique_ids[0]]
    d_res = int(first["resA"].shape[-1])
    chain = first.get("chainA")
    d_chain = int(chain.shape[-1]) if torch.is_tensor(chain) and chain.ndim >= 1 else 0
    checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    cfg = checkpoint_payload.get("cfg")
    if cfg is None:
        cfg = trainlib.build_model_config(trainlib.P, d_res, d_chain)
    if pair_head_type:
        cfg.pair_head_type = str(pair_head_type).lower()
    model = L13PDBGVPModel(cfg).to(device)
    state = checkpoint_payload.get("model_state", checkpoint_payload)
    model.load_state_dict(state, strict=False)
    model.eval()

    provenance = dict(checkpoint_payload.get("model_provenance", {}))
    effective_head = str(getattr(cfg, "pair_head_type", "")).lower()
    if require_native_eb:
        if effective_head != "eb":
            raise RuntimeError(
                f"Native EB evaluation requires pair_head_type='eb', received {effective_head!r}"
            )
        if provenance and not bool(provenance.get("native_eb_export", False)):
            raise RuntimeError("Checkpoint provenance does not confirm native EB")

    return RuntimeBundle(
        model=model,
        trainlib=trainlib,
        dataset=dataset,
        samples=samples,
        device=device,
        checkpoint=checkpoint_payload,
        provenance=provenance,
    )


def item_for_row(runtime: RuntimeBundle, row: Dict) -> Dict:
    return sample_to_pair(
        runtime.samples[row["protein_A"]],
        runtime.samples[row["protein_B"]],
        label=max(0, int(row.get("label", 0))),
        pair_id=row["pair_id"],
    )


def tensor_row(tensor: torch.Tensor, batch_index: int = 0) -> np.ndarray:
    return tensor[batch_index].detach().float().cpu().numpy()


def native_evidence_record(row: Dict, output: Dict) -> Tuple[Dict, List[Dict]]:
    prediction = {
        **row,
        "pair_prob": float(output["pair_prob"][0].detach().cpu()),
        "pair_logit": float(output["pair_logit"][0].detach().cpu()),
        "evi_score": float(output["evi_score"][0].detach().cpu()),
        "pair_head_type": output.get("model_metadata", {}).get("pair_head_type", ""),
        "proposal_mode": output.get("model_metadata", {}).get("proposal_mode", ""),
        "support_mode": output.get("model_metadata", {}).get("support_mode", ""),
    }
    supports = []
    idx_a = tensor_row(output["pair_idxA"]).astype(int)
    idx_b = tensor_row(output["pair_idxB"]).astype(int)
    score = tensor_row(output["pair_score"])
    weight = tensor_row(output["pair_weight"])
    for rank, (a, b, value, support_weight) in enumerate(
        zip(idx_a, idx_b, score, weight), start=1
    ):
        supports.append({
            "pair_id": row["pair_id"],
            "rank": rank,
            "residue_index_A": int(a),
            "residue_index_B": int(b),
            "support_score": float(value),
            "support_weight": float(support_weight),
        })
    return prediction, supports
