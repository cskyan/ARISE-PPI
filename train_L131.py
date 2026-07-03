# -*- coding: utf-8 -*-
"""Train the L131 residue-to-pair model from sequence and structure features."""

import os, math, time, random, re, uuid, gzip, hashlib, glob, json, shutil, csv
import numpy as np
from typing import Optional, Dict, Tuple, List
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from contextlib import nullcontext

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

AA_ORDER = "ACDEFGHIKLMNPQRSTVWYX"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}
AA_PHYS = {
    "A": [1.8,  0.0, 0.0, 0.0, 0.0],
    "C": [2.5,  0.0, 0.0, 0.5, 0.0],
    "D": [-3.5, -1.0, 0.0, 0.0, 1.0],
    "E": [-3.5, -1.0, 0.0, 0.0, 1.0],
    "F": [2.8,  0.0, 1.0, 0.0, 0.0],
    "G": [-0.4, 0.0, 0.0, 0.0, 0.0],
    "H": [-3.2, 0.1, 1.0, 0.5, 0.0],
    "I": [4.5,  0.0, 1.0, 0.0, 0.0],
    "K": [-3.9, 1.0, 0.0, 0.0, 0.0],
    "L": [3.8,  0.0, 1.0, 0.0, 0.0],
    "M": [1.9,  0.0, 0.0, 0.5, 0.0],
    "N": [-3.5, 0.0, 0.0, 0.0, 1.0],
    "P": [-1.6, 0.0, 0.0, 0.0, 0.0],
    "Q": [-3.5, 0.0, 0.0, 0.0, 1.0],
    "R": [-4.5, 1.0, 0.0, 0.0, 0.0],
    "S": [-0.8, 0.0, 0.0, 0.0, 1.0],
    "T": [-0.7, 0.0, 0.0, 0.0, 1.0],
    "V": [4.2,  0.0, 1.0, 0.0, 0.0],
    "W": [-0.9, 0.0, 1.0, 0.5, 0.0],
    "Y": [-1.3, 0.0, 1.0, 0.5, 1.0],
    "X": [0.0,  0.0, 0.0, 0.0, 0.0],
}

try:
    from torch.cuda.amp import autocast, GradScaler
except Exception:
    autocast = None
    GradScaler = None

from config_L131 import Params, build_model_config
from model_L131 import L13PDBGVPModel

# ============================================================
# Global configuration
# ============================================================
P = Params.from_env()


def _force_dest_binary_params(p):
    """Optional compatibility lock for Dest_prepared site training.

    This branch no longer defaults to DIPS.  The configured/current data root
    is kept unless FORCE_DEST=1 is explicitly provided.  Use DATA_ROOT,
    TRAIN_LIST, VAL_LIST and TEST_LIST to point the script to a new prepared
    dataset without changing the task/loss profile.
    """
    force_dest = os.environ.get("FORCE_DEST", "0").strip().lower() not in ("0", "false", "no", "off", "")
    if not force_dest:
        return p
    dest_root = os.environ.get(
        "DEST_PREPARED_ROOT",
        os.path.join("data", "Dest_prepared"),
    )
    p.dataset_mode = "rbp"
    p.primary_objective = "binary"
    p.dips_root = dest_root
    p.rbp_root = dest_root
    p.rbp_id_list = os.path.join(dest_root, "all_ids.txt")
    p.rbp_train_list = os.path.join(dest_root, "train.txt")
    p.rbp_val_list = os.path.join(dest_root, "val.txt")
    p.rbp_test_list = os.path.join(dest_root, "test.txt")
    p.save_dir = os.environ.get(
        "DEST_SAVE_DIR",
        os.path.join("runs", "L131_Dest"),
    )
    p.rbp_structure_dir = "coords"
    p.structure_source = "coords"
    p.use_pssm = True
    p.use_dssp = True

    seq_mode = (
        os.environ.get("DEST_SEQUENCE_MODE")
        or os.environ.get("sequence_mode")
        or os.environ.get("SEQUENCE_MODE")
        or getattr(p, "sequence_mode", "hybrid")
        or "hybrid"
    )
    p.sequence_mode = str(seq_mode).lower()

    # DEST is an auxiliary L1 benchmark; use the best stable linear residue head unless explicitly overridden.
    p.site_head_type = os.environ.get("site_head_type", os.environ.get("SITE_HEAD_TYPE", "linear")).lower()
    p.site_self_cross = bool(getattr(p, "site_self_cross", True))
    p.site_graph_layers = int(getattr(p, "site_graph_layers", 0))
    p.binary_ager_eval = bool(getattr(p, "binary_ager_eval", False))
    p.ager_enable = bool(getattr(p, "ager_enable", False))

    if p.sequence_mode in ("esm", "hybrid"):
        p.num_workers = int(os.environ.get("num_workers", os.environ.get("NUM_WORKERS", 0)))
        p.allow_cuda_workers = False
        p.prefetch_factor = int(os.environ.get("prefetch_factor", os.environ.get("PREFETCH_FACTOR", 2)))

    p.dips_use_embedder_on_miss = False
    _restore_dest_prepared_splits(dest_root)
    return p


def _restore_dest_prepared_splits(dest_root: str):
    """Restore Dest_prepared train/val/test from the original fused split pkl files."""
    try:
        import csv as _csv
        import pickle as _pickle
        root = os.path.abspath(str(dest_root))
        manifest_path = os.path.join(root, "dest_manifest.tsv")
        src_root = os.path.join(os.path.dirname(root), "Dest")
        split_map = {
            "train.txt": "fused_training_list.pkl",
            "val.txt": "fused_validing_list.pkl",
            "test.txt": "fused_test_list.pkl",
        }
        if not os.path.exists(manifest_path):
            return
        rows = []
        with open(manifest_path, "r", encoding="utf-8", errors="ignore") as f:
            for row in _csv.DictReader(f, delimiter="\t"):
                rows.append(row)
        rows = sorted(rows, key=lambda r: int(r.get("fused_idx", len(rows))))
        pids = [r["pid"] for r in rows]
        if len(pids) != 422:
            return
        expected = {"train.txt": 269, "val.txt": 68, "test.txt": 85}
        current_ok = True
        for name, n in expected.items():
            path = os.path.join(root, name)
            if not os.path.exists(path):
                current_ok = False
                break
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                ids = [x.strip().split()[0] for x in f if x.strip() and not x.startswith("#")]
            if len(ids) != n:
                current_ok = False
                break
        if current_ok:
            try:
                same_all = True
                for out_name, pkl_name in split_map.items():
                    pkl_path = os.path.join(src_root, pkl_name)
                    if not os.path.exists(pkl_path):
                        same_all = False
                        break
                    with open(pkl_path, "rb") as f:
                        idxs = _pickle.load(f)
                    expected_ids = [pids[int(i)] for i in idxs]
                    txt_path = os.path.join(root, out_name)
                    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                        current_ids = [x.strip().split()[0] for x in f if x.strip() and not x.startswith("#")]
                    if expected_ids != current_ids:
                        same_all = False
                        break
                if same_all:
                    return
                print("[Dest] split count is correct but IDs differ; restoring original fused splits.", flush=True)
            except Exception:
                pass
        for out_name, pkl_name in split_map.items():
            pkl_path = os.path.join(src_root, pkl_name)
            if not os.path.exists(pkl_path):
                return
            with open(pkl_path, "rb") as f:
                idxs = _pickle.load(f)
            with open(os.path.join(root, out_name), "w", encoding="utf-8") as f:
                for i in idxs:
                    f.write(pids[int(i)] + "\n")
        print(f"[Dest] restored fixed fused splits under {root}: train=269 val=68 test=85", flush=True)
    except Exception as e:
        print(f"[Dest][warn] failed to restore fixed splits: {e}", flush=True)


P = _force_dest_binary_params(P)

# Generic current/custom data override.  This is intentionally independent from
# primary_objective and task_loss_profile.  It prevents accidental fallback to
# old DIPS paths while still allowing explicit pair-mode experiments.
def _apply_current_data_root(p):
    root = (
        os.environ.get("DATA_ROOT")
        or os.environ.get("CUSTOM_DATA_ROOT")
        or os.environ.get("DEST_PREPARED_ROOT")
        or ""
    )
    if str(root).strip():
        root = os.path.abspath(str(root).strip())
        p.rbp_root = root
        p.dips_root = root
        p.rbp_id_list = os.environ.get("ID_LIST", os.path.join(root, "all_ids.txt"))
        p.rbp_train_list = os.environ.get("TRAIN_LIST", os.path.join(root, "train.txt"))
        p.rbp_val_list = os.environ.get("VAL_LIST", os.path.join(root, "val.txt"))
        p.rbp_test_list = os.environ.get("TEST_LIST", os.path.join(root, "test.txt"))
        p.dips_train_list = p.rbp_train_list
        p.dips_val_list = p.rbp_val_list
        p.dips_test_list = p.rbp_test_list
    return p

P = _apply_current_data_root(P)

DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
DIPS_ROOT        = getattr(P, 'dips_root', P.rbp_root)
DIPS_TRAIN_LIST  = getattr(P, 'dips_train_list', P.rbp_id_list)
DIPS_VAL_LIST    = getattr(P, 'dips_val_list',   P.rbp_id_list)
ESM_LOCAL_DIR    = P.esm_local_dir
SAVE_DIR         = P.save_dir
CKPT_DIR         = os.path.join(SAVE_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

ESM_LOCAL_CKPT   = os.path.join(ESM_LOCAL_DIR, "esm2_t33_650M_UR50D.pt")
ESM_HUB_CACHE    = os.path.join(ESM_LOCAL_DIR, "hub")
os.makedirs(ESM_HUB_CACHE, exist_ok=True)
os.environ.setdefault("TORCH_HOME", ESM_HUB_CACHE)
os.environ.setdefault("TRANSFORMERS_CACHE", ESM_HUB_CACHE)
os.environ.setdefault("HF_HOME", ESM_HUB_CACHE)
os.environ.setdefault("XDG_CACHE_HOME", ESM_HUB_CACHE)

SEED             = P.seed
BATCH_SITE       = P.batch_site
NUM_WORKERS_SITE = P.num_workers
CONTACT_CUTOFF   = P.contact_cutoff

PRIMARY_OBJ = P.primary_objective.lower()
DATASET_MODE = str(getattr(P, "dataset_mode", "auto")).lower()
if PRIMARY_OBJ == "binary":
    BEST_TAG = "AUPRC"
elif PRIMARY_OBJ == "topk":
    BEST_TAG = "TOPK"
else:
    BEST_TAG = "PairAUPRC" if str(getattr(P, "pair_primary_metric", "pair_auprc")).lower() == "pair_auprc" else "MedAUC"
RUN_STAMP = time.strftime("%Y%m%d_%H%M%S")
GLOBAL_BEST_CKPT = os.path.join(CKPT_DIR, f"best_{BEST_TAG}.pt")
RUN_BEST_CKPT = os.path.join(CKPT_DIR, f"best_{BEST_TAG}_{RUN_STAMP}.pt")
GLOBAL_BEST_META = os.path.join(CKPT_DIR, f"best_{BEST_TAG}_meta.json")
BEST_CKPT = RUN_BEST_CKPT
METRIC_FIELDS = [
    "acc", "precision", "recall", "f1", "auprc", "mcc",
    "auprc_macro", "surface_auprc", "surface_macro_auprc",
    "ca10_auprc", "ca10_macro_auprc", "ca12_auprc", "ca12_macro_auprc",
    "ca14_auprc", "ca14_macro_auprc", "ca16_auprc", "ca16_macro_auprc",
]
TOPK_FIELDS = [
    "recall_L5", "recall_L10", "precision_K10", "hit_K20",
    "enrichment_K10", "topk_score",
]
PAIR_FIELDS = [
    "pair_acc", "pair_precision", "pair_recall", "pair_specificity",
    "pair_f1", "pair_auroc", "pair_auprc", "pair_mcc",
    "pair_brier", "pair_ece",
    "medauc", "medauc_geom", "focus_pos_recall", "focus_frac", "n_auc",
    "prob_mean_l2", "pos_rate_l2",
]
METRICS_HISTORY = os.path.join(SAVE_DIR, f"metrics_history_{RUN_STAMP}.tsv")
RUN_BEST_METRICS_TSV = os.path.join(CKPT_DIR, f"best_{BEST_TAG}_{RUN_STAMP}_metrics.tsv")
GLOBAL_BEST_METRICS_TSV = os.path.join(CKPT_DIR, f"best_{BEST_TAG}_metrics.tsv")
RUN_TEST_METRICS_TSV = os.path.join(CKPT_DIR, f"test_{BEST_TAG}_{RUN_STAMP}_metrics.tsv")
RUN_TEST_PREDICTIONS_TSV = os.path.join(CKPT_DIR, f"test_{BEST_TAG}_{RUN_STAMP}_predictions.tsv")

USE_AMP = (os.environ.get('AMP', '1') not in ('0', 'false', '')) \
          and torch.cuda.is_available() \
          and autocast is not None and GradScaler is not None
AMP_DTYPE = (torch.bfloat16 if torch.cuda.is_available()
             and torch.cuda.is_bf16_supported() else torch.float16)


def _deranged_perm(B: int, device=None) -> torch.Tensor:
    if B <= 1:
        return torch.empty(0, dtype=torch.long, device=device)
    perm = torch.randperm(B, device=device)
    for i in range(B):
        if perm[i].item() == i:
            j = (i + 1) % B
            tmp = perm[i].clone()
            perm[i] = perm[j]
            perm[j] = tmp
    if torch.any(perm == torch.arange(B, device=device)):
        perm = torch.roll(torch.arange(B, device=device), shifts=1, dims=0)
    return perm


# ============================================================
# Data utilities
# ============================================================

_AA_ALLOWED = set("ACDEFGHIKLMNPQRSTVWYBXZUO")


def _clean_protein_seq(seq):
    if not seq:
        return ""
    out = []
    for ch in str(seq).upper():
        if ch.isalpha():
            out.append(ch if ch in _AA_ALLOWED else "X")
    return "".join(out)


def _read_fasta_one(path):
    if path is None or not os.path.exists(str(path)):
        return ""
    try:
        opener = gzip.open if str(path).endswith('.gz') else open
        with opener(path, 'rt', errors='ignore') as f:
            seqs = []
            for line in f:
                line = line.strip()
                if not line or line.startswith('>') or line.startswith('#'):
                    continue
                seqs.append(line.split('\t')[-1] if '\t' in line else line)
        return _clean_protein_seq(''.join(seqs))
    except Exception:
        return ""


def _read_fasta_two(path):
    if path is None or not os.path.exists(str(path)):
        return "", ""
    try:
        opener = gzip.open if str(path).endswith('.gz') else open
        seqs, cur = [], []
        with opener(path, 'rt', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if cur: seqs.append(''.join(cur)); cur = []
                elif line:
                    cur.append(line)
        if cur: seqs.append(''.join(cur))
        return (seqs[0], seqs[1]) if len(seqs) >= 2 else (seqs[0] if seqs else "", "")
    except Exception:
        return "", ""


def _seq_to_light_feats(seq: str, length: int) -> torch.Tensor:
    seq = (seq or "")[:length]
    out = torch.zeros((length, len(AA_ORDER) + 6), dtype=torch.float32)
    if length <= 0:
        return out
    for i in range(length):
        aa = seq[i] if i < len(seq) else "X"
        aa = aa if aa in AA_TO_IDX else "X"
        out[i, AA_TO_IDX[aa]] = 1.0
        phys = AA_PHYS.get(aa, AA_PHYS["X"])
        out[i, len(AA_ORDER):len(AA_ORDER)+5] = torch.tensor(phys, dtype=torch.float32)
        out[i, -1] = float(i) / float(max(1, length - 1))
    return out


def _load_pdb_ca(path: str):
    if path is None or not os.path.exists(path):
        return None, None
    coords = []
    seen = set()
    try:
        opener = gzip.open if str(path).lower().endswith(".gz") else open
        with opener(path, 'rt', errors='ignore') as f:
            for line in f:
                if not line.startswith('ATOM'):
                    continue
                atom = line[12:16].strip()
                if atom != 'CA':
                    continue
                resseq = line[22:27].strip()
                icode = line[26:27].strip()
                key = (resseq, icode)
                if key in seen:
                    continue
                try:
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                except Exception:
                    continue
                coords.append((x, y, z))
                seen.add(key)
    except Exception:
        return None, None
    if not coords:
        return None, None
    arr = torch.tensor(coords, dtype=torch.float32)
    return arr, torch.ones(arr.shape[0], dtype=torch.bool)


def _geom_features_from_ca(coords: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if coords is None or coords.numel() == 0:
        return None
    L = coords.shape[0]
    feat = torch.zeros((L, 6), dtype=torch.float32)
    if L == 0:
        return feat
    if L > 1:
        d = (coords[1:] - coords[:-1]).pow(2).sum(-1).sqrt()
        feat[1:, 0] = d
        feat[:-1, 1] = d
    if L > 2:
        v1 = coords[1:-1] - coords[:-2]
        v2 = coords[2:] - coords[1:-1]
        n1 = v1 / v1.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        n2 = v2 / v2.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        feat[1:-1, 2] = (n1 * n2).sum(-1)
    center = coords.mean(dim=0, keepdim=True)
    feat[:, 3] = (coords - center).pow(2).sum(-1).sqrt()
    dist = torch.cdist(coords, coords)
    feat[:, 4] = (dist < 8.0).float().sum(dim=1) - 1.0
    feat[:, 5] = torch.linspace(0.0, 1.0, steps=L)
    # scale to moderate range
    feat[:, 0:2] = feat[:, 0:2] / 4.0
    feat[:, 3] = feat[:, 3] / feat[:, 3].max().clamp_min(1.0)
    feat[:, 4] = feat[:, 4] / feat[:, 4].max().clamp_min(1.0)
    return feat


def _chain_geom_summary(geom: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if geom is None or geom.numel() == 0:
        return None
    mu = geom.mean(dim=0)
    sd = geom.std(dim=0, unbiased=False)
    return torch.cat([mu, sd], dim=0)

def _first_existing_dir(root: str, names: List[str]) -> str:
    for name in names:
        p = os.path.join(root, name)
        if os.path.isdir(p):
            return p
    return os.path.join(root, names[0])

def _first_existing_file(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _resolve_dssp_dir(root: str) -> str:
    """Resolve DSSP/RSA feature directory.

    Priority:
      1) DSSP_DIR / rbp_dssp_dir if provided;
      2) root/dssp_rsa_asa or root/dssp_asa_rsa;
      3) legacy root/dssp or root/DSSP.
    """
    explicit = os.environ.get("DSSP_DIR", "") or str(getattr(P, "rbp_dssp_dir", "") or "")
    if explicit.strip():
        d = explicit.strip()
        if not os.path.isabs(d):
            d = os.path.join(root, d)
        return d
    return _first_existing_dir(root, ["dssp_rsa_asa", "dssp_asa_rsa", "dssp", "DSSP"])


def _configured_dssp_dim(default: int = 9) -> int:
    try:
        return max(1, int(getattr(P, "dssp_dim", default)))
    except Exception:
        return int(default)


def _load_coords(coords_npz: str):
    """
    Read C-alpha coordinates from an NPZ file.
    Return `(coords [L,3], mask [L])` or `(None, None)`.
    """
    if coords_npz is None or not os.path.exists(coords_npz):
        return None, None
    try:
        opener = gzip.open if coords_npz.endswith('.gz') else open
        with opener(coords_npz, 'rb') as f:
            z = np.load(f, allow_pickle=True)
            arr = None
            for k in ("ca", "CA", "coords", "xyz"):
                if k in z.files:
                    arr = z[k]; break
            if arr is None:
                for k in z.files:
                    try:
                        a = z[k].astype(np.float32)
                        if a.ndim >= 2 and a.shape[-1] == 3:
                            arr = a.reshape(-1, 3); break
                    except Exception:
                        pass
            if arr is None:
                return None, None
            coords = torch.from_numpy(arr.reshape(-1, 3).astype(np.float32))
            mask   = torch.ones(len(coords), dtype=torch.bool)
            return coords, mask
    except Exception:
        return None, None


def _pairwise_contact(coordsA, coordsB, maskA, maskB, cutoff=8.0):
    """
    Compute a binary residue contact map from C-alpha coordinates.
    """
    La, Lb = coordsA.shape[0], coordsB.shape[0]
    vA = maskA.bool() if maskA is not None else torch.ones(La, dtype=torch.bool)
    vB = maskB.bool() if maskB is not None else torch.ones(Lb, dtype=torch.bool)
    contact = torch.zeros(La, Lb, dtype=torch.float32)
    chunk = 64
    for i0 in range(0, La, chunk):
        i1 = min(i0 + chunk, La)
        if not vA[i0:i1].any(): continue
        diff = coordsA[i0:i1, None, :] - coordsB[None, :, :]  # [chunk, Lb, 3]
        dist = diff.pow(2).sum(-1).sqrt()
        c    = (dist < cutoff).float()
        c    = c * vA[i0:i1].float().unsqueeze(1) * vB.float().unsqueeze(0)
        contact[i0:i1] = c
    return contact


def _load_npz_tensor(path, keys, dtype=np.float32):
    if path is None or not os.path.exists(str(path)):
        return None
    try:
        opener = gzip.open if str(path).endswith('.gz') else open
        with opener(path, 'rb') as f:
            z = np.load(f, allow_pickle=True)
            if isinstance(z, np.lib.npyio.NpzFile):
                try:
                    for k in keys:
                        if k in z.files:
                            return torch.from_numpy(z[k].astype(dtype))
                    return None
                finally:
                    z.close()
            arr = np.asarray(z)
            if arr.dtype == object:
                arr = np.asarray(arr.tolist())
            return torch.from_numpy(arr.astype(dtype))
    except Exception:
        pass
    return None


def _load_pssm(path, length):
    arr = _load_npz_tensor(path, ("pssm", "PSSM", "data"))
    if arr is None: return None
    if arr.ndim == 1: arr = arr.unsqueeze(-1)
    return arr[:length, :20] if arr.shape[-1] >= 20 else None


def _load_dssp(path, length):
    """Load DSSP-derived residue features.

    Legacy files may contain 9-dim secondary-structure one-hot features.
    Updated files can contain 2-dim [ASA, RSA] or other numeric DSSP/RSA
    descriptors. Do not force-truncate to 9 columns; the model dimension is
    inferred from the first sample.
    """
    arr = _load_npz_tensor(path, ("dssp", "DSSP", "ss", "rsa_asa", "asa_rsa", "data"))
    if arr is None:
        return None
    if arr.ndim == 1:
        arr = arr.unsqueeze(-1)
    if arr.ndim != 2 or arr.shape[-1] < 1:
        return None
    return arr[:length, :].float()


def _fit_len_feat(x: Optional[torch.Tensor], L: int, D: int) -> torch.Tensor:
    L = int(max(1, L))
    D = int(max(1, D))
    if x is None:
        return torch.zeros((L, D), dtype=torch.float32)
    x = torch.as_tensor(x).float()
    if x.ndim == 1:
        x = x.unsqueeze(-1)
    if x.ndim != 2:
        x = x.reshape(-1, x.shape[-1] if x.numel() else D)
    if x.shape[1] < D:
        x = F.pad(x, (0, D - x.shape[1]))
    elif x.shape[1] > D:
        x = x[:, :D]
    if x.shape[0] >= L:
        return x[:L].contiguous()
    pad = torch.zeros((L - x.shape[0], D), dtype=x.dtype)
    return torch.cat([x, pad], dim=0).contiguous()


def _fit_len_coords(coords: Optional[torch.Tensor], L: int) -> torch.Tensor:
    L = int(max(1, L))
    if coords is None:
        return _pseudo_coords(L)
    coords = torch.as_tensor(coords).float().reshape(-1, 3)
    if coords.numel() == 0:
        return _pseudo_coords(L)
    if coords.shape[0] >= L:
        return coords[:L].contiguous()
    extra = _pseudo_coords(L - coords.shape[0])
    if coords.shape[0] > 0:
        extra = extra + coords[-1:].detach()
    return torch.cat([coords, extra], dim=0).contiguous()


def _run_embedder(embedder, cid, tag, seq_str):
    if embedder is None: return None
    try:
        return embedder.embed(seq_str, cid=str(cid), tag=str(tag))
    except Exception as e:
        print(f"[ESM][warn] cid={cid} tag={tag} err={e}", flush=True)
        return None


class SiteEmbedder:
    """Embedded ESM2 loader adapted from train_sitepairs.py.
    Exposes .embed(seq, cid=None, tag=None) to match this training script.
    """
    def __init__(self, backbone="esm2_t33_650M_UR50D", cache_dir=None, device="cpu", esm_local_dir=None):
        self.backbone = backbone
        self.cache = cache_dir or os.path.join((esm_local_dir or ESM_LOCAL_DIR), "cache_embed")
        os.makedirs(self.cache, exist_ok=True)
        self.device = device
        self.mem = {}

        import argparse
        import esm

        ckpt = os.path.join((esm_local_dir or ESM_LOCAL_DIR), "esm2_t33_650M_UR50D.pt")
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Local ESM checkpoint not found: {ckpt}")

        try:
            import torch.serialization as _ts
            _ts.add_safe_globals([argparse.Namespace])
        except Exception:
            pass

        try:
            self.esm_model, self.esm_alphabet = esm.pretrained.load_model_and_alphabet_local(ckpt)
        except Exception:
            from esm.pretrained import load_model_and_alphabet_core
            model_data = torch.load(ckpt, map_location="cpu", weights_only=False)
            try:
                self.esm_model, self.esm_alphabet = load_model_and_alphabet_core(self.backbone, model_data)
            except TypeError:
                self.esm_model, self.esm_alphabet = load_model_and_alphabet_core(model_data)

        self.esm_model.to(device).eval()
        self.batch_converter = self.esm_alphabet.get_batch_converter()
        with torch.no_grad():
            self.d_out = int(getattr(self.esm_model, 'embed_dim', 1280))

    def _safe_filename(self, uid: str) -> str:
        safe = re.sub(r'[^A-Za-z0-9._-]+', '_', uid)[:80]
        hid = hashlib.md5(uid.encode('utf-8')).hexdigest()[:8]
        return f"{safe}__{hid}.npy"

    def embed(self, seq: str, cid=None, tag=None):
        uid = f"{cid}_{tag}" if (cid is not None or tag is not None) else "anon"
        seq = _clean_protein_seq(seq)
        fp = os.path.join(self.cache, self._safe_filename(uid))
        if uid in self.mem:
            return self.mem[uid]
        if os.path.isfile(fp):
            try:
                arr = np.load(fp)
                self.mem[uid] = arr
                return arr
            except Exception:
                pass
        if not seq:
            arr = np.zeros((1, self.d_out), np.float32)
            self.mem[uid] = arr
            return arr
        with torch.no_grad():
            batch_labels, batch_strs, batch_tokens = self.batch_converter([(uid, seq)])
            batch_tokens = batch_tokens.to(self.device)
            out = self.esm_model(batch_tokens, repr_layers=[33] if "t33" in self.backbone else [12])
            reps = out["representations"][max(out["representations"].keys())]
            rep = reps[0, 1:1+len(seq), :].detach().cpu().float().numpy()
        try:
            np.save(fp, rep)
        except Exception:
            pass
        self.mem[uid] = rep
        return rep


# ============================================================
# DIPSIndexedPairs Dataset
# Indexed DIPS pair dataset
# ============================================================

class DIPSIndexedPairs(torch.utils.data.Dataset):
    """
    Indexed DIPS-Plus dataset with ESM caching and residue features.

    Each item contains residue features, masks, coordinates, contact labels,
    chain-level interface labels, the pair label, and the complex identifier.
    """

    def __init__(self, dips_root, complex_ids, contact_cutoff=8.0, embedder=None,
                 use_pssm=False, use_dssp=False, esm_cache_dir=None,
                 skip_filter=True, verbose=False):
        self.root     = str(dips_root)
        self.cutoff   = float(contact_cutoff)
        self.use_pssm = bool(use_pssm)
        self.use_dssp = bool(use_dssp)
        self.use_geom = bool(getattr(P, "use_geom", True))
        self.use_chain_geom = bool(getattr(P, "use_chain_geom", True))
        self.sequence_mode = str(getattr(P, "sequence_mode", "light")).lower()
        self.structure_source = str(getattr(P, "structure_source", "auto")).lower()
        self.embedder = embedder
        self.verbose  = verbose

        self.dir_coords = _first_existing_dir(self.root, ["coords", "struct", "processed"])
        self.dir_pdb    = _first_existing_dir(self.root, ["pdb", "sch_pdb", "struct"])
        self.dir_seq    = _first_existing_dir(self.root, ["seq", "SEQ", "fasta"])
        self.dir_pssm   = _first_existing_dir(self.root, ["pssm", "PSSM"])
        self.dir_dssp   = _resolve_dssp_dir(self.root)
        self.dir_esm    = str(esm_cache_dir) if esm_cache_dir \
                          else os.path.join(self.root, "esm_cache")
        os.makedirs(self.dir_esm, exist_ok=True)
        self.filter_cache_dir = os.path.join(SAVE_DIR, "dips_index_cache")

        self._index_built = False
        self.ids = (list(complex_ids) if skip_filter
                    else self._filter_valid_ids_cached(list(complex_ids), verbose))
        if not self.ids:
            raise RuntimeError(f"[DIPS] empty complex list (root={self.root})")
        self._esm_calls = self._esm_hit = self._esm_miss = 0

    def __len__(self):
        return len(self.ids)

    def _atomic_save(self, obj, path):
        tmp = path + f".tmp.{os.getpid()}.{uuid.uuid4().hex}"
        torch.save(obj, tmp)
        os.replace(tmp, path)

    def _load_esm_feature_file(self, path, length):
        try:
            if str(path).endswith(".npy"):
                feat = torch.from_numpy(np.load(path))
            else:
                feat = torch.load(path, map_location="cpu")
            if torch.is_tensor(feat) and feat.ndim == 2 and feat.shape[1] >= 1280:
                self._esm_hit += 1
                return feat[:length, :1280].float()
        except Exception:
            return None
        return None

    def _esm_cache_candidates(self, cid, tag):
        uid = f"{cid}_{tag}"
        safe_cid = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(cid))
        safe_uid = re.sub(r"[^A-Za-z0-9._-]+", "_", uid)[:80]
        hid = hashlib.md5(uid.encode("utf-8")).hexdigest()[:8]
        names = [
            f"{safe_cid}_{tag}_esm.pt",
            f"{safe_cid}_{tag}.pt",
            f"{safe_cid}_{tag}_esm.npy",
            f"{safe_cid}_{tag}.npy",
            f"{safe_uid}__{hid}.npy",
        ]
        return [os.path.join(self.dir_esm, n) for n in names]

    def _get_esm_cached(self, seq_path, cid, tag, length):
        cache = self._esm_cache_candidates(cid, tag)[0]
        self._esm_calls += 1
        for cand in self._esm_cache_candidates(cid, tag):
            if os.path.exists(cand):
                feat = self._load_esm_feature_file(cand, length)
                if feat is not None:
                    return feat

        if isinstance(seq_path, tuple) and seq_path[0] == "SEQ":
            seq_str = _clean_protein_seq(seq_path[1] or "")
        else:
            seq_str = _read_fasta_one(seq_path) if seq_path else ""

        if not seq_str or self.embedder is None or not bool(getattr(P, "dips_use_embedder_on_miss", False)):
            self._esm_miss += 1
            miss_log_limit = int(getattr(P, "dips_esm_miss_log_limit", 0) or 0)
            if self.verbose and miss_log_limit > 0 and self._esm_miss <= miss_log_limit and self.embedder is None:
                print(f"[ESM][cache-miss] cid={cid} tag={tag}; using zero fallback", flush=True)
            return torch.zeros((length, 1280))

        feat = _run_embedder(self.embedder, cid, tag, seq_str)
        if feat is None:
            self._esm_miss += 1
            return torch.zeros((length, 1280))
        if isinstance(feat, np.ndarray):
            feat = torch.from_numpy(feat)
        feat = feat.detach().float().cpu()
        try: self._atomic_save(feat, cache)
        except Exception: pass
        self._esm_miss += 1
        return feat[:length, :1280]

    def _build_file_index(self, verbose=False):
        if self._index_built: return
        tag_re = re.compile(r"^(.*)_([0-9A-Za-z]+)$")

        def scan(d):
            if not os.path.isdir(d):
                return []
            out = []
            try:
                for root, _, files in os.walk(d):
                    for fn in files:
                        out.append((root, fn))
            except Exception:
                return []
            return out

        def _idx(scan_dir, exts, out_map, out_pair):
            exts = tuple(sorted(exts, key=len, reverse=True))
            for root, fn in scan(scan_dir):
                low = fn.lower()
                for ext in exts:
                    if low.endswith(ext):
                        stem = fn[:-len(ext)]
                        path = os.path.join(root, fn)
                        m = tag_re.match(stem)
                        if m:
                            out_map.setdefault(m.group(1), {})[m.group(2)] = path
                        else:
                            out_pair[stem] = path
                        break

        self._cm, self._cp = {}, {}   # coords map / pair
        self._xm, self._xp = {}, {}   # pdb map / pair
        self._sm, self._sp = {}, {}   # seq
        self._pm, self._pp = {}, {}   # pssm
        self._dm, self._dp = {}, {}   # dssp

        _idx(self.dir_coords, (".npz", ".npz.gz"), self._cm, self._cp)
        _idx(self.dir_pdb, (".pdb", ".ent", ".pdb.gz", ".ent.gz"), self._xm, self._xp)
        _idx(self.dir_seq, (".fa", ".fasta", ".faa", ".tsv", ".fa.gz", ".fasta.gz"),
             self._sm, self._sp)
        if self.use_pssm:
            _idx(self.dir_pssm, (".npz", ".npz.gz", ".pt", ".tsv"), self._pm, self._pp)
        if self.use_dssp:
            _idx(self.dir_dssp, (".npy", ".npz", ".npz.gz", ".pt", ".tsv"), self._dm, self._dp)

        self._index_built = True
        if verbose:
            print(f"[DIPS-index] coords: split={sum(len(v) for v in self._cm.values())} pair={len(self._cp)} | "
                  f"pdb: split={sum(len(v) for v in self._xm.values())} pair={len(self._xp)}", flush=True)

    def _resolve(self, cid):
        self._build_file_index()
        raw = str(cid)
        raw = re.sub(r"\.(npz|npy|pt|pth|pkl|pickle|fa|fasta|faa|tsv|txt|pdb|ent)(\.gz)?$",
                     "", raw, flags=re.I)
        bases = [raw]
        m = re.match(r"^(.*)_([0-9A-Za-z]+)$", cid)
        if m:
            bases.append(m.group(1))
        # DIPS list files sometimes store a single-chain id. Keep order but remove duplicates.
        bases = list(dict.fromkeys([b for b in bases if b]))

        source = self.structure_source
        if source not in ("pdb", "coords", "auto"):
            source = "auto"

        for base in bases:
            if source in ("coords", "auto") and base in self._cp:
                return ("pair", self._cp[base],
                        self._sp.get(base), self._pp.get(base), self._dp.get(base))

            split_map = None
            if source == "pdb":
                split_map = self._xm.get(base)
            elif source == "coords":
                split_map = self._cm.get(base)
            else:
                # DIPS-Plus was originally prepared from coords; use PDB only as a fallback.
                split_map = self._cm.get(base) or self._xm.get(base)

            if not split_map:
                continue

            tags = sorted(split_map.keys())
            if "0" in tags and "1" in tags:   tA, tB = "0", "1"
            elif "A" in tags and "B" in tags: tA, tB = "A", "B"
            elif len(tags) >= 2:                tA, tB = tags[0], tags[1]
            else:                               continue

            smap = self._sm.get(base, {})
            pmap = self._pm.get(base, {})
            dmap = self._dm.get(base, {})
            return ("split",
                    split_map.get(tA), split_map.get(tB),
                    smap.get(tA), smap.get(tB),
                    pmap.get(tA), pmap.get(tB),
                    dmap.get(tA), dmap.get(tB),
                    tA, tB)
        return None

    def _filter_cache_path(self, ids_in):
        key_src = {
            "root": os.path.abspath(self.root),
            "n": len(ids_in),
            "ids_md5": hashlib.md5("\n".join(map(str, ids_in)).encode("utf-8")).hexdigest(),
            "structure_source": self.structure_source,
            "coords": os.path.abspath(self.dir_coords) if self.dir_coords else "",
            "pdb": os.path.abspath(self.dir_pdb) if self.dir_pdb else "",
        }
        key = hashlib.md5(json.dumps(key_src, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return os.path.join(self.filter_cache_dir, f"filtered_{key}.txt")

    def _filter_valid_ids_cached(self, ids_in, verbose=False):
        if not bool(getattr(P, "dips_filter_cache", True)):
            return self._filter_valid_ids(ids_in, verbose)
        os.makedirs(self.filter_cache_dir, exist_ok=True)
        cache_path = self._filter_cache_path(ids_in)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    kept = [ln.strip() for ln in f if ln.strip()]
                if kept:
                    if verbose:
                        print(f"[DIPS] filter cache hit: {len(ids_in)} -> {len(kept)} {cache_path}", flush=True)
                    return kept
            except Exception:
                pass
        kept = self._filter_valid_ids(ids_in, verbose)
        try:
            tmp = cache_path + f".tmp.{os.getpid()}.{uuid.uuid4().hex}"
            with open(tmp, "w", encoding="utf-8") as f:
                for cid in kept:
                    f.write(str(cid) + "\n")
            os.replace(tmp, cache_path)
            if verbose:
                print(f"[DIPS] filter cache saved: {cache_path}", flush=True)
        except Exception:
            pass
        return kept

    def _filter_valid_ids(self, ids_in, verbose=False):
        self._build_file_index()
        kept = []
        dropped = 0
        total = len(ids_in)
        t0 = time.time()
        for i, cid in enumerate(ids_in, 1):
            if self._resolve(cid) is not None:
                kept.append(cid)
            else:
                dropped += 1
            if verbose and (i % 5000 == 0 or i == total):
                print(f"[DIPS] filtering {i}/{total} kept={len(kept)} dropped={dropped} dt={time.time()-t0:.1f}s", flush=True)
        if verbose:
            print(f"[DIPS] filter: {len(ids_in)} -> {len(kept)} dropped={dropped}", flush=True)
        return kept

    def __getitem__(self, idx: int):
        cid = self.ids[int(idx)]
        r   = self._resolve(cid)
        if r is None:
            raise FileNotFoundError(f"[DIPS] resolve failed: {cid}")

        mode = r[0]
        if mode == "split":
            # Split mode: one coordinate file per chain.
            _, cA_p, cB_p, sA_p, sB_p, pA_p, pB_p, dA_p, dB_p, tagA, tagB = r
            load_fn_A = _load_pdb_ca if str(cA_p).lower().endswith((".pdb", ".ent", ".pdb.gz", ".ent.gz")) else _load_coords
            load_fn_B = _load_pdb_ca if str(cB_p).lower().endswith((".pdb", ".ent", ".pdb.gz", ".ent.gz")) else _load_coords
            cA, mA = load_fn_A(cA_p)
            cB, mB = load_fn_B(cB_p)
            if cA is None or cB is None:
                raise FileNotFoundError(f"[DIPS] missing structure: {cid}")
            L, M = cA.shape[0], cB.shape[0]
        else:
            # Pair mode prefers a paired NPZ file.
            _, cP_p, sP_p, pP_p, dP_p = r
            if str(cP_p).lower().endswith((".pdb", ".ent", ".pdb.gz", ".ent.gz")):
                raise RuntimeError(f"[DIPS] pair mode with single pdb is unsupported: {cid}")
            try:
                opener = gzip.open if cP_p.endswith('.gz') else open
                with opener(cP_p, 'rb') as f:
                    z = np.load(f, allow_pickle=True)
                    files = list(z.files)
                    caL = caR = None
                    for k in ("caL", "ca0", "coordsA", "ca_A", "coords_A"):
                        if k in files: caL = z[k]; break
                    for k in ("caR", "ca1", "coordsB", "ca_B", "coords_B"):
                        if k in files: caR = z[k]; break
                    if caL is None and "coords" in files:
                        arr = z["coords"]
                        if arr.ndim == 3 and arr.shape[0] == 2:
                            caL, caR = arr[0], arr[1]
                    seqL = str(z["seqL"]) if "seqL" in files else ""
                    seqR = str(z["seqR"]) if "seqR" in files else ""
            except Exception as e:
                raise RuntimeError(f"[DIPS] bad pair npz {cid}: {e}")
            if caL is None or caR is None:
                raise RuntimeError(f"[DIPS] cannot extract two chains from: {cid}")
            cA = torch.from_numpy(caL.reshape(-1, 3).astype(np.float32))
            cB = torch.from_numpy(caR.reshape(-1, 3).astype(np.float32))
            mA = torch.ones(len(cA), dtype=torch.bool)
            mB = torch.ones(len(cB), dtype=torch.bool)
            L, M = len(cA), len(cB)
            if sP_p and os.path.exists(sP_p):
                seqL, seqR = _read_fasta_two(sP_p)
            sA_p = ("SEQ", seqL); sB_p = ("SEQ", seqR)
            pA_p = pB_p = dA_p = dB_p = None
            tagA, tagB = "0", "1"

        def _assemble(s_p, p_p, d_p, length, tag, coords):
            if isinstance(s_p, tuple) and s_p[0] == "SEQ":
                seq_str = s_p[1] or ""
            else:
                seq_str = _read_fasta_one(s_p) if s_p else ""
            feats = []
            if self.sequence_mode in ("esm", "hybrid"):
                esm = self._get_esm_cached(("SEQ", seq_str), cid, tag, length)
                if esm is None or not torch.is_tensor(esm):
                    esm = torch.zeros((length, 1280))
                feats.append(esm[:length, :1280])
            if self.sequence_mode in ("light", "hybrid"):
                feats.append(_seq_to_light_feats(seq_str, length))
            if self.use_pssm:
                ps = _load_pssm(p_p, length)
                feats.append(ps[:length, :20] if ps is not None else torch.zeros((length, 20)))
            if self.use_dssp:
                ds = _load_dssp(d_p, length)
                dssp_dim = int(ds.shape[-1]) if ds is not None else _configured_dssp_dim(2)
                feats.append(ds[:length, :] if ds is not None else torch.zeros((length, dssp_dim)))
            if self.use_geom:
                gf = _geom_features_from_ca(coords)
                feats.append(gf[:length, :6] if gf is not None else torch.zeros((length, 6)))
            Lmin = min(x.shape[0] for x in feats) if feats else length
            out = torch.cat([x[:Lmin] for x in feats], dim=-1) if feats else torch.zeros((length, 1))
            return out[:length], (_chain_geom_summary(_geom_features_from_ca(coords)) if self.use_chain_geom else None)

        resA, chainA = _assemble(sA_p, pA_p, dA_p, L, tagA, cA)
        resB, chainB = _assemble(sB_p, pB_p, dB_p, M, tagB, cB)

        # Build the binary residue-pair contact labels.
        y2d = _pairwise_contact(cA, cB, mA, mB, cutoff=self.cutoff)

        LA, LB = resA.shape[0], resB.shape[0]
        mA, mB = mA[:LA], mB[:LB]
        if y2d.shape[0] != LA or y2d.shape[1] != LB:
            y2d = y2d[:LA, :LB]

        y_res_A = y2d.max(dim=1).values
        y_res_B = y2d.max(dim=0).values   # [LB]

        # Protein-pair label derived from the presence of an interface contact.
        has_contact = float(y2d.sum() > 0)

        return {
            "complex":     cid,
            "resA":        resA,                       # [LA, D]
            "resB":        resB,                       # [LB, D]
            "coordsA":     cA[:LA],                    # [LA, 3]
            "coordsB":     cB[:LB],                    # [LB, 3]
            "maskA":       mA.float(),                 # [LA]
            "maskB":       mB.float(),                 # [LB]
            "chainA":      chainA if chainA is not None else torch.zeros(12),
            "chainB":      chainB if chainB is not None else torch.zeros(12),
            "y2d":         y2d,
            "y_res_A":     y_res_A,
            "y_res_B":     y_res_B,
            "has_contact": torch.tensor(has_contact),
        }



def _read_id_list_optional(path: str) -> List[str]:
    if not path or not str(path).strip():
        return []
    return _read_rbp_id_list(path)

def _rbp_label_ids(root: str, ids: List[str]) -> List[str]:
    label_dir = _first_existing_dir(str(root), ['labels', 'label'])
    seq_dir = _first_existing_dir(str(root), ['seq', 'SEQ', 'fasta'])
    out = []
    for pid in ids:
        label_path = os.path.join(label_dir, f'{pid}.npy')
        if not os.path.exists(label_path):
            continue
        try:
            label_len = int(np.load(label_path).reshape(-1).shape[0])
        except Exception:
            continue
        ok = False
        for ext in ('.fa', '.fasta', '.faa', '.txt'):
            seq = _read_fasta_one(os.path.join(seq_dir, f'{pid}{ext}'))
            if seq and len(seq) == label_len:
                ok = True
                break
        if ok:
            out.append(pid)
    return out

def _write_id_list(path: str, ids: List[str]) -> None:
    d = os.path.dirname(str(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for x in ids:
            f.write(str(x).strip() + '\n')

def _rbp_label_stats(root: str, ids: List[str]) -> Dict[str, Dict[str, float]]:
    label_dir = _first_existing_dir(str(root), ['labels', 'label'])
    out = {}
    for pid in ids:
        fp = os.path.join(label_dir, f'{pid}.npy')
        if not os.path.exists(fp):
            continue
        try:
            y = np.load(fp).astype(np.float32).reshape(-1)
        except Exception:
            continue
        if y.size <= 0:
            continue
        pos = float((y > 0.5).sum())
        total = float(y.size)
        frac = pos / max(1.0, total)
        if pos <= 0:
            bucket = 'zero'
        elif pos >= total:
            bucket = 'full'
        elif frac < 0.10:
            bucket = 'low'
        elif frac < 0.50:
            bucket = 'mid'
        elif frac < 0.90:
            bucket = 'high'
        else:
            bucket = 'very_high'
        out[pid] = {'pos': pos, 'total': total, 'frac': frac, 'bucket': bucket}
    return out

def _stratified_split_once(ids: List[str], stats: Dict[str, Dict[str, float]], seed: int):
    rng = random.Random(seed)
    groups = {}
    for pid in ids:
        b = str(stats.get(pid, {}).get('bucket', 'unknown'))
        groups.setdefault(b, []).append(pid)
    tr, va, te = [], [], []
    for b in sorted(groups.keys()):
        g = list(groups[b])
        rng.shuffle(g)
        n = len(g)
        n_tr = int(round(n * float(getattr(P, 'split_train', 0.80))))
        n_va = int(round(n * float(getattr(P, 'split_val', 0.10))))
        n_tr = max(0, min(n_tr, n))
        n_va = max(0, min(n_va, n - n_tr))
        if n >= 3 and n_va == 0:
            n_va = 1
            if n_tr + n_va > n:
                n_tr = n - n_va
        if n >= 3 and n - n_tr - n_va == 0:
            if n_tr > 1:
                n_tr -= 1
            else:
                n_va = max(0, n_va - 1)
        tr.extend(g[:n_tr])
        va.extend(g[n_tr:n_tr+n_va])
        te.extend(g[n_tr+n_va:])
    rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
    return tr, va, te

def _split_pos_rate(ids: List[str], stats: Dict[str, Dict[str, float]]) -> float:
    pos = sum(float(stats.get(x, {}).get('pos', 0.0)) for x in ids)
    total = sum(float(stats.get(x, {}).get('total', 0.0)) for x in ids)
    return pos / max(1.0, total)

def _bucket_counts(ids: List[str], stats: Dict[str, Dict[str, float]]) -> Dict[str, int]:
    out = {}
    for pid in ids:
        b = str(stats.get(pid, {}).get('bucket', 'unknown'))
        out[b] = out.get(b, 0) + 1
    return out

def _split_balance_score(tr, va, te, stats):
    all_ids = list(tr) + list(va) + list(te)
    all_rate = _split_pos_rate(all_ids, stats)
    rates = [_split_pos_rate(x, stats) for x in (tr, va, te)]
    score = sum(abs(r - all_rate) for r in rates)
    all_b = _bucket_counts(all_ids, stats)
    n_all = max(1, len(all_ids))
    for part in (tr, va, te):
        bc = _bucket_counts(part, stats)
        n = max(1, len(part))
        for b, c in all_b.items():
            score += 0.15 * abs((bc.get(b, 0) / n) - (c / n_all))
    return score

def _stratified_split_ids(ids: List[str], stats: Dict[str, Dict[str, float]], seed: int):
    trials = max(1, int(getattr(P, 'split_search_trials', 1)))
    best = None
    best_score = 1e9
    for i in range(trials):
        cand = _stratified_split_once(ids, stats, seed + i)
        sc = _split_balance_score(cand[0], cand[1], cand[2], stats)
        if sc < best_score:
            best_score = sc
            best = cand
    return best

def _ensure_rbp_split_files() -> None:
    """
    Create deterministic RBP train/val/test split files if configured paths do not exist.
    Once files exist, training reads them directly, so runs remain fixed across launches.
    """
    trp = str(getattr(P, 'rbp_train_list', '') or '').strip()
    vap = str(getattr(P, 'rbp_val_list', '') or '').strip()
    tep = str(getattr(P, 'rbp_test_list', '') or '').strip()
    if not (trp and vap and tep):
        return
    all_ids = _read_rbp_id_list(P.rbp_id_list)
    labeled_ids = _rbp_label_ids(P.rbp_root, all_ids)
    if not labeled_ids:
        raise RuntimeError(f"[split][RBP] no labeled IDs found under {P.rbp_root}")
    split_strategy = str(getattr(P, 'split_strategy', 'random')).strip().lower()
    meta_path = os.path.join(os.path.dirname(trp) or '.', '.rbp_split_meta.tsv')
    if os.path.exists(trp) and os.path.exists(vap) and os.path.exists(tep):
        old_ids = _read_rbp_id_list(trp) + _read_rbp_id_list(vap) + _read_rbp_id_list(tep)
        old_set = set(old_ids)
        labeled_set = set(labeled_ids)
        meta_ok = False
        if os.path.exists(meta_path):
            try:
                txt = open(meta_path, 'r', encoding='utf-8', errors='ignore').read()
                meta_ok = (f"strategy\t{split_strategy}\n" in txt)
            except Exception:
                meta_ok = False
        if old_set == labeled_set and len(old_ids) == len(labeled_ids) and (split_strategy == 'random' or meta_ok):
            return
        print(f"[split][RBP] existing split does not match labeled IDs "
              f"(split={len(old_set)} labeled={len(labeled_set)} strategy={split_strategy}); regenerating.", flush=True)
    all_ids = labeled_ids
    rng = random.Random(getattr(P, 'split_seed', 42))
    stats = _rbp_label_stats(P.rbp_root, all_ids)
    if split_strategy.startswith('strat'):
        tr_ids, va_ids, te_ids = _stratified_split_ids(all_ids, stats, getattr(P, 'split_seed', 42))
    else:
        rng.shuffle(all_ids)
        n_total = len(all_ids)
        n_tr = int(round(n_total * float(getattr(P, 'split_train', 0.80))))
        n_va = int(round(n_total * float(getattr(P, 'split_val', 0.10))))
        n_tr = max(1, min(n_tr, n_total))
        n_va = max(1, min(n_va, max(0, n_total - n_tr)))
        if n_tr + n_va >= n_total and n_total >= 3:
            n_va = max(1, n_total - n_tr - 1)
        tr_ids = all_ids[:n_tr]
        va_ids = all_ids[n_tr:n_tr+n_va]
        te_ids = all_ids[n_tr+n_va:]
    _write_id_list(trp, tr_ids)
    _write_id_list(vap, va_ids)
    _write_id_list(tep, te_ids)
    with open(meta_path, 'w', encoding='utf-8') as f:
        f.write(f"strategy\t{split_strategy}\n")
        f.write(f"seed\t{getattr(P, 'split_seed', 42)}\n")
        f.write(f"train_pos\t{_split_pos_rate(tr_ids, stats):.6f}\n")
        f.write(f"val_pos\t{_split_pos_rate(va_ids, stats):.6f}\n")
        f.write(f"test_pos\t{_split_pos_rate(te_ids, stats):.6f}\n")
    print(f"[split][RBP] generated fixed split strategy={split_strategy} seed={getattr(P, 'split_seed', 42)} "
          f"Train={len(tr_ids)} Val={len(va_ids)} Test={len(te_ids)} "
          f"pos(train/val/test)={_split_pos_rate(tr_ids, stats):.3f}/"
          f"{_split_pos_rate(va_ids, stats):.3f}/{_split_pos_rate(te_ids, stats):.3f}", flush=True)

def _estimate_l1_balance_from_ids(root: str, ids: List[str], max_weight: float = 6.0):
    label_dir = _first_existing_dir(str(root), ['labels', 'label'])
    pos = 0.0
    total = 0.0
    n_zero = 0
    n_full = 0
    for pid in ids:
        fp = os.path.join(label_dir, f'{pid}.npy')
        if not os.path.exists(fp):
            continue
        try:
            y = np.load(fp).astype(np.float32).reshape(-1)
        except Exception:
            continue
        if y.size <= 0:
            continue
        yp = float((y > 0.5).sum())
        pos += yp
        total += float(y.size)
        if yp <= 0:
            n_zero += 1
        elif yp >= float(y.size):
            n_full += 1
    if total <= 0:
        return 1.0, 0.0, n_zero, n_full
    pos_rate = pos / max(1.0, total)
    neg = max(1.0, total - pos)
    pw = neg / max(1.0, pos)
    pw = max(0.25, min(float(max_weight), float(pw)))
    return pw, pos_rate, n_zero, n_full

def _feature_coverage(root: str, ids: List[str]) -> Dict[str, int]:
    pssm_dir = _first_existing_dir(str(root), ['pssm', 'PSSM'])
    dssp_dir = _resolve_dssp_dir(str(root))
    out = {'n': len(ids), 'pssm': 0, 'dssp': 0}
    for pid in ids:
        if _first_existing_file([
            os.path.join(pssm_dir, f'{pid}.npy'),
            os.path.join(pssm_dir, f'{pid}.npz'),
            os.path.join(pssm_dir, f'{pid}.npz.gz'),
        ]) is not None:
            out['pssm'] += 1
        if _first_existing_file([
            os.path.join(dssp_dir, f'{pid}.npy'),
            os.path.join(dssp_dir, f'{pid}.npz'),
            os.path.join(dssp_dir, f'{pid}.npz.gz'),
        ]) is not None:
            out['dssp'] += 1
    return out

def _rbp_structure_candidates(root: str, pid: str) -> List[str]:
    struct_dir = str(getattr(P, 'rbp_structure_dir', 'structures_af')).strip() or 'structures_af'
    explicit = [
        os.path.join(root, struct_dir, f'{pid}.pdb'),
        os.path.join(root, struct_dir, f'{pid}.ent'),
        os.path.join(root, 'structures', f'{pid}.pdb'),
        os.path.join(root, 'structures', f'{pid}.ent'),
        os.path.join(root, 'structure', f'{pid}.pdb'),
        os.path.join(root, 'structure', f'{pid}.ent'),
        os.path.join(root, 'structures_af', f'{pid}.pdb'),
        os.path.join(root, 'structures_af', f'{pid}.ent'),
        os.path.join(root, 'pdb', f'{pid}.pdb'),
        os.path.join(root, 'pdb', f'{pid}.ent'),
        os.path.join(root, 'coords', f'{pid}.npz'),
        os.path.join(root, 'coords', f'{pid}.npz.gz'),
        os.path.join(root, 'coords', f'{pid}.npy'),
    ]
    af_globs = [
        os.path.join(root, struct_dir, f'AF-{pid}-F1-*.pdb'),
        os.path.join(root, struct_dir, f'AF-{pid}-F1-*.ent'),
        os.path.join(root, struct_dir, f'{pid}*.pdb'),
        os.path.join(root, struct_dir, f'{pid}*.ent'),
    ]
    cands = list(explicit)
    for pat in af_globs:
        cands.extend(sorted(glob.glob(pat)))
    seen = set()
    out = []
    for p in cands:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out

def _structure_file_exists(root: str, pid: str) -> bool:
    return any(os.path.exists(fp) for fp in _rbp_structure_candidates(root, pid))

def _read_rbp_id_list(path: str) -> List[str]:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"RBP ID list not found: {path}")
    out, seen = [], set()
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            tok = s.split()[0].strip()
            tok = re.sub(r'\.(npy|npz|pt|fa|fasta|txt)(\.(gz))?$', '', tok, flags=re.IGNORECASE)
            if tok and tok not in seen:
                out.append(tok); seen.add(tok)
    return out


def _pseudo_coords(length: int) -> torch.Tensor:
    if length <= 0:
        return torch.zeros((1, 3), dtype=torch.float32)
    x = torch.arange(length, dtype=torch.float32).unsqueeze(-1)
    yz = torch.zeros((length, 2), dtype=torch.float32)
    return torch.cat([x, yz], dim=-1)


def _load_single_coords_any(root: str, pid: str):
    source = str(getattr(P, 'structure_source', 'pdb')).lower()
    all_cands = _rbp_structure_candidates(root, pid)
    pdb_cands = [fp for fp in all_cands if fp.endswith(('.pdb', '.ent'))]
    coord_cands = [fp for fp in all_cands if fp.endswith(('.npz', '.npz.gz', '.npy'))]
    if source == 'pdb':
        cands = pdb_cands
    elif source == 'coords':
        cands = coord_cands
    else:
        cands = pdb_cands + coord_cands
    for fp in cands:
        if not os.path.exists(fp):
            continue
        if fp.endswith('.npy'):
            try:
                arr = np.load(fp).astype(np.float32).reshape(-1, 3)
                coords = torch.from_numpy(arr)
                return coords, torch.ones(coords.shape[0], dtype=torch.bool)
            except Exception:
                continue
        if fp.endswith(('.pdb', '.ent')):
            coords, mask = _load_pdb_ca(fp)
        else:
            coords, mask = _load_coords(fp)
        if coords is not None:
            return coords, mask
    return None, None


class RBP296Dataset(torch.utils.data.Dataset):
    """Single-protein RNA-binding site dataset adapted to the original two-chain L13 interface.
    Chain A is real; chain B is a dummy placeholder so the innovation modules stay intact.
    """
    DUMMY_LEN = 2

    def __init__(self, rbp_root, pid_list, embedder=None, use_pssm=True, use_dssp=True, esm_cache_dir=None, verbose=False):
        self.root = str(rbp_root)
        self.pids = list(pid_list)
        self.embedder = embedder
        self.use_pssm = bool(use_pssm)
        self.use_dssp = bool(use_dssp)
        self.verbose = bool(verbose)
        self.sequence_mode = str(getattr(P, 'sequence_mode', 'esm')).lower()
        self.use_geom = bool(getattr(P, 'use_geom', True))
        self.use_chain_geom = bool(getattr(P, 'use_chain_geom', True))
        self.dir_seq = _first_existing_dir(self.root, ['seq', 'SEQ', 'fasta'])
        self.dir_labels = _first_existing_dir(self.root, ['labels', 'label'])
        self.dir_pssm = _first_existing_dir(self.root, ['pssm', 'PSSM'])
        self.dir_dssp = _resolve_dssp_dir(self.root)
        self.dir_esm = str(esm_cache_dir) if esm_cache_dir else os.path.join(self.root, 'esm_cache')
        os.makedirs(self.dir_esm, exist_ok=True)

    def __len__(self):
        return len(self.pids)

    def _get_esm_cached(self, pid: str, seq_str: str, length: int) -> torch.Tensor:
        safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(pid))
        cache = os.path.join(self.dir_esm, f'{safe}_esm.pt')
        if os.path.exists(cache):
            try:
                feat = torch.load(cache, map_location='cpu')
                if torch.is_tensor(feat) and feat.ndim == 2 and feat.shape[1] >= 1280:
                    feat = feat[:length, :1280]
                    if feat.shape[0] < length:
                        feat = F.pad(feat, (0, 0, 0, length - feat.shape[0]))
                    return feat
            except Exception:
                pass
        if not seq_str:
            return torch.zeros((length, 1280), dtype=torch.float32)
        feat = _run_embedder(self.embedder, pid, 'A', seq_str)
        if isinstance(feat, np.ndarray):
            feat = torch.from_numpy(feat)
        feat = torch.as_tensor(feat).detach().float().cpu()
        try:
            tmp = cache + f'.tmp.{os.getpid()}.{uuid.uuid4().hex}'
            torch.save(feat, tmp); os.replace(tmp, cache)
        except Exception:
            pass
        feat = feat[:length, :1280]
        if feat.shape[0] < length:
            feat = F.pad(feat, (0, 0, 0, length - feat.shape[0]))
        return feat

    def _read_seq_for_label(self, pid: str, target_len: Optional[int] = None) -> str:
        seqs = []
        for ext in ('.fa', '.fasta', '.faa', '.txt'):
            fp = os.path.join(self.dir_seq, f'{pid}{ext}')
            s = _read_fasta_one(fp)
            if s:
                seqs.append((fp, s))
        if not seqs:
            return ""
        if target_len is not None and target_len > 0:
            for _, s in seqs:
                if len(s) == int(target_len):
                    return s
        return seqs[0][1]

    def __getitem__(self, idx: int):
        pid = self.pids[int(idx)]
        label_path = os.path.join(self.dir_labels, f'{pid}.npy')
        arr = None
        label_len = 0
        if os.path.exists(label_path):
            try:
                arr = np.load(label_path).astype(np.float32).reshape(-1)
                label_len = int(arr.shape[0])
            except Exception:
                arr = None
                label_len = 0
        seq_str = self._read_seq_for_label(pid, label_len)
        L = label_len if label_len > 0 else (len(seq_str) if seq_str else 0)
        if L == 0:
            L = 1
        y_res_A = torch.zeros(L, dtype=torch.float32)
        if arr is not None:
            if len(arr) != L:
                arr = arr[:L] if len(arr) > L else np.pad(arr, (0, L - len(arr)))
            y_res_A = torch.from_numpy(arr)

        feats = []
        if self.sequence_mode in ('esm', 'hybrid'):
            if self.embedder is None and not getattr(P, 'allow_zero_esm_fallback', False):
                raise RuntimeError('ESM requested but unavailable and ALLOW_ZERO_ESM_FALLBACK=0')
            esm = self._get_esm_cached(pid, seq_str, L) if self.embedder is not None else torch.zeros((L, 1280), dtype=torch.float32)
            feats.append(_fit_len_feat(esm, L, 1280))
        if self.sequence_mode in ('light', 'hybrid'):
            feats.append(_fit_len_feat(_seq_to_light_feats(seq_str, L), L, len(AA_ORDER) + 6))
        if self.use_pssm:
            p = _load_pssm(_first_existing_file([
                os.path.join(self.dir_pssm, f'{pid}.npy'),
                os.path.join(self.dir_pssm, f'{pid}.npz'),
                os.path.join(self.dir_pssm, f'{pid}.npz.gz'),
            ]), L)
            feats.append(_fit_len_feat(p, L, 20))
        if self.use_dssp:
            d = _load_dssp(_first_existing_file([
                os.path.join(self.dir_dssp, f'{pid}.npy'),
                os.path.join(self.dir_dssp, f'{pid}.npz'),
                os.path.join(self.dir_dssp, f'{pid}.npz.gz'),
            ]), L)
            dssp_dim = int(d.shape[-1]) if d is not None else _configured_dssp_dim(2)
            feats.append(_fit_len_feat(d, L, dssp_dim))

        coordsA, _ = _load_single_coords_any(self.root, pid)
        if coordsA is None:
            # Graceful fallback: use pseudo-coordinates (equally-spaced along x-axis).
            # Structural features will be uninformative for this protein, but training
            # continues. A warning is printed once per missing protein.
            if self.verbose:
                print(f"[RBP][warn] no structure found for {pid} "
                      f"(searched {getattr(P, 'rbp_structure_dir', 'structures_af')} / pdb / coords); "
                      f"using pseudo-coords; GVP geometry features will be zero.", flush=True)
        coordsA = _fit_len_coords(coordsA, L)
        if self.use_geom:
            gf = _geom_features_from_ca(coordsA)
            feats.append(_fit_len_feat(gf, L, 6))
        resA = torch.cat(feats, dim=-1) if feats else torch.zeros((L, 1), dtype=torch.float32)
        y_res_A = y_res_A[:L]
        maskA = torch.ones(L, dtype=torch.float32)
        chainA = _chain_geom_summary(_geom_features_from_ca(coordsA)) if self.use_chain_geom else None
        if chainA is None:
            chainA = torch.zeros(12, dtype=torch.float32)

        Ld = max(self.DUMMY_LEN, 2)
        D = int(resA.shape[-1])
        resB = torch.zeros((Ld, D), dtype=torch.float32)
        coordsB = _pseudo_coords(Ld)
        maskB = torch.ones(Ld, dtype=torch.float32)
        y_res_B = torch.zeros(Ld, dtype=torch.float32)
        y2d = torch.zeros((L, Ld), dtype=torch.float32)
        chainB = torch.zeros(12, dtype=torch.float32)
        return {
            'complex': pid, 'resA': resA, 'resB': resB, 'coordsA': coordsA, 'coordsB': coordsB,
            'maskA': maskA, 'maskB': maskB, 'chainA': chainA, 'chainB': chainB,
            'y2d': y2d, 'y_res_A': y_res_A, 'y_res_B': y_res_B,
            'has_contact': torch.tensor(float((y_res_A > 0.5).any()), dtype=torch.float32),
            'site_mode': torch.tensor(1.0, dtype=torch.float32),
        }


class ExplicitPairDataset(torch.utils.data.Dataset):
    """Build fixed labeled protein pairs from an explicit split manifest."""

    REQUIRED_COLUMNS = ("protein_A", "protein_B", "label")

    def __init__(
        self,
        rbp_root,
        manifest_path,
        embedder=None,
        use_pssm=True,
        use_dssp=True,
        esm_cache_dir=None,
        verbose=False,
    ):
        self.manifest_path = os.path.abspath(str(manifest_path))
        with open(self.manifest_path, "r", encoding="utf-8-sig", newline="") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            delimiter = "\t" if self.manifest_path.lower().endswith((".tsv", ".txt")) else ","
            try:
                delimiter = csv.Sniffer().sniff(sample, delimiters="\t,").delimiter
            except csv.Error:
                pass
            reader = csv.DictReader(handle, delimiter=delimiter)
            fields = tuple(reader.fieldnames or ())
            missing = [name for name in self.REQUIRED_COLUMNS if name not in fields]
            if missing:
                raise ValueError(
                    f"Pair manifest is missing required columns {missing}: {self.manifest_path}"
                )
            self.rows = []
            for line_number, source in enumerate(reader, start=2):
                protein_a = str(source.get("protein_A", "")).strip()
                protein_b = str(source.get("protein_B", "")).strip()
                if not protein_a or not protein_b:
                    raise ValueError(f"Missing protein ID at line {line_number}: {self.manifest_path}")
                raw_label = str(source.get("label", "")).strip().lower()
                if raw_label in ("1", "1.0", "true", "yes", "positive"):
                    label = 1
                elif raw_label in ("0", "0.0", "false", "no", "negative"):
                    label = 0
                else:
                    raise ValueError(
                        f"Invalid binary label {source.get('label')!r} at line {line_number}"
                    )
                row = dict(source)
                row.update(
                    protein_A=protein_a,
                    protein_B=protein_b,
                    label=label,
                    pair_id=str(source.get("pair_id") or f"{protein_a}__{protein_b}"),
                )
                self.rows.append(row)
        if not self.rows:
            raise ValueError(f"Pair manifest is empty: {self.manifest_path}")

        self.ids = list(
            dict.fromkeys(
                protein_id
                for row in self.rows
                for protein_id in (row["protein_A"], row["protein_B"])
            )
        )
        self.proteins = RBP296Dataset(
            rbp_root,
            self.ids,
            embedder=embedder,
            use_pssm=use_pssm,
            use_dssp=use_dssp,
            esm_cache_dir=esm_cache_dir,
            verbose=verbose,
        )
        self._protein_index = {protein_id: index for index, protein_id in enumerate(self.ids)}
        self._sample_cache = {}

    def __len__(self):
        return len(self.rows)

    def _protein(self, protein_id):
        if protein_id not in self._sample_cache:
            self._sample_cache[protein_id] = self.proteins[self._protein_index[protein_id]]
        return self._sample_cache[protein_id]

    def __getitem__(self, idx):
        row = self.rows[int(idx)]
        sample_a = self._protein(row["protein_A"])
        sample_b = self._protein(row["protein_B"])
        length_a = int(sample_a["resA"].shape[0])
        length_b = int(sample_b["resA"].shape[0])
        label = float(row["label"])
        return {
            "complex": row["pair_id"],
            "resA": sample_a["resA"],
            "resB": sample_b["resA"],
            "coordsA": sample_a["coordsA"],
            "coordsB": sample_b["coordsA"],
            "maskA": sample_a["maskA"],
            "maskB": sample_b["maskA"],
            "chainA": sample_a["chainA"],
            "chainB": sample_b["chainA"],
            "y2d": torch.zeros((length_a, length_b), dtype=torch.float32),
            "y_res_A": sample_a["y_res_A"],
            "y_res_B": sample_b["y_res_A"],
            "has_contact": torch.tensor(label, dtype=torch.float32),
            "site_mode": torch.tensor(0.0, dtype=torch.float32),
        }


# ============================================================
# Collate
# ============================================================

def dips_collate(batch: List[Dict]) -> Dict:
    """
    Collate variable-length pairs and validate residue/contact alignment.
    """
    for i, d in enumerate(batch):
        y2d = d.get("y2d"); rA = d.get("resA"); rB = d.get("resB")
        if y2d is None or rA is None or rB is None: continue
        if y2d.shape[0] != rA.shape[0] or y2d.shape[1] != rB.shape[0]:
            raise RuntimeError(
                f"[collate] shape mismatch idx={i}: "
                f"A={tuple(rA.shape)} B={tuple(rB.shape)} y2d={tuple(y2d.shape)}"
            )

    out = {}
    def _pad1d(x, L): return F.pad(x, (0, L - x.shape[0]))
    def _pad2d(x, L): return F.pad(x, (0, 0, 0, L - x.shape[0]))

    for k in batch[0].keys():
        if k == "complex":
            out[k] = [d[k] for d in batch];
            continue
        if k == "y2d":
            mL = max(d[k].shape[0] for d in batch)
            mM = max(d[k].shape[1] for d in batch)
            out[k] = torch.stack([
                F.pad(d[k], (0, mM - d[k].shape[1], 0, mL - d[k].shape[0]))
                for d in batch]);
            continue
        if k in ("resA", "resB"):
            mL = max(d[k].shape[0] for d in batch)
            out[k] = torch.stack([_pad2d(d[k], mL) for d in batch]);
            continue
        if k in ("chainA", "chainB"):
            if batch[0][k].ndim == 1:
                out[k] = torch.stack([d[k] for d in batch]);
                continue
            mL = max(d[k].shape[0] for d in batch)
            out[k] = torch.stack([_pad2d(d[k], mL) for d in batch]);
            continue
        if k in ("coordsA", "coordsB"):
            mL = max(d[k].shape[0] for d in batch)
            out[k] = torch.stack([_pad2d(d[k], mL) for d in batch]);
            continue
        if k in ("maskA", "maskB", "y_res_A", "y_res_B"):
            mL = max(d[k].shape[0] for d in batch)
            out[k] = torch.stack([_pad1d(d[k], mL) for d in batch]);
            continue
        if torch.is_tensor(batch[0][k]):
            out[k] = torch.stack([d[k] for d in batch])
        else:
            out[k] = [d[k] for d in batch]

    # Preserve the explicit pair label.
    if "has_contact" in out:
        out["y_pair"] = out["has_contact"]   # [B] float 0/1
    return out


# ============================================================
# In-batch negative generation
# ============================================================

def make_inbatch_negatives(batch: Dict, device: str, neg_ratio: float = 0.5
                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                      torch.Tensor, torch.Tensor, torch.Tensor,
                                      Optional[torch.Tensor], Optional[torch.Tensor],
                                      torch.Tensor]:
    """
    Generate negative pairs by shuffling chain partners within a batch.
    """
    B = batch["resA"].size(0)
    if B <= 1 or neg_ratio <= 0:
        resA = batch["resA"]
        resB = batch["resB"]
        maskA = batch["maskA"]
        maskB = batch["maskB"]
        coordsA = batch["coordsA"]; coordsB = batch["coordsB"]
        emptyA = resA[:0].to(device)
        emptyB = resB[:0].to(device)
        emptyCA = coordsA[:0].to(device)
        emptyCB = coordsB[:0].to(device)
        emptyMA = (maskA[:0] > 0.5).to(device)
        emptyMB = (maskB[:0] > 0.5).to(device)
        chainA = batch.get("chainA", None)
        chainB = batch.get("chainB", None)
        chainA_empty = chainA[:0].to(device).float() if chainA is not None else None
        chainB_empty = chainB[:0].to(device).float() if chainB is not None else None
        return emptyA, emptyMA, emptyCA, emptyB, emptyMB, emptyCB, chainA_empty, chainB_empty, torch.zeros(0, device=device)

    n_neg = max(1, int(round(B * neg_ratio)))
    n_neg = min(n_neg, B)
    perm = _deranged_perm(B, device=None)
    neg_b_idx = perm[:n_neg]
    pos_a_idx = torch.arange(n_neg)

    resA_neg  = batch["resA"][pos_a_idx].to(device)
    coordsA_neg = batch["coordsA"][pos_a_idx].to(device)
    maskA_neg = (batch["maskA"][pos_a_idx] > 0.5).to(device)
    resB_neg  = batch["resB"][neg_b_idx].to(device)
    coordsB_neg = batch["coordsB"][neg_b_idx].to(device)
    maskB_neg = (batch["maskB"][neg_b_idx] > 0.5).to(device)

    coordsA = torch.nan_to_num(batch["coordsA"]).to(device).float()
    coordsB = torch.nan_to_num(batch["coordsB"]).to(device).float()
    chainA = batch.get("chainA", None)
    chainB = batch.get("chainB", None)
    chainA_neg = chainA[pos_a_idx].to(device).float() if chainA is not None else None
    chainB_neg = chainB[neg_b_idx].to(device).float() if chainB is not None else None

    y_pair_neg = torch.zeros(n_neg, device=device)
    return resA_neg, maskA_neg, coordsA_neg, resB_neg, maskB_neg, coordsB_neg, chainA_neg, chainB_neg, y_pair_neg


# ============================================================
# EMA
# ============================================================

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad: self.shadow[n] = p.detach().clone()

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            if n not in self.shadow: self.shadow[n] = p.detach().clone()
            else: self.shadow[n].mul_(d).add_(p.detach(), alpha=1 - d)

    @torch.no_grad()
    def apply_shadow(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n])

    @torch.no_grad()
    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self.backup: p.data.copy_(self.backup[n])
        self.backup = {}


# ============================================================
# LR Scheduler
# ============================================================

def build_warmup_cosine_scheduler(opt, total_updates, base_lr, P):
    if not P.lr_sched: return None, {"enabled": False}
    total_updates = max(1, int(total_updates))
    wu = int(round(float(P.lr_warmup_frac) * total_updates))
    wu = max(int(P.lr_warmup_min_updates), min(int(P.lr_warmup_max_updates), wu))
    wu = min(wu, total_updates)
    sr = max(0.0, min(1.0, float(P.lr_warmup_start_ratio)))
    mr = float(P.lr_min_ratio)
    def lr_lambda(step):
        if step < wu:
            return sr + (1 - sr) * (step / max(1, wu))
        prog = (step - wu) / max(1, total_updates - wu)
        return max(mr, 0.5 * (1 + math.cos(math.pi * prog)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return sched, {"enabled": True, "warmup": wu, "total": total_updates}


# ============================================================
# Loss helpers
# ============================================================

def _focal_bce(logits, targets, gamma=2.0, alpha=0.25, pw=None):
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw, reduction="none")
    p  = torch.sigmoid(logits)
    pt = targets * p + (1 - targets) * (1 - p)
    at = targets * alpha + (1 - targets) * (1 - alpha)
    return at * (1 - pt).pow(gamma) * bce


def _mixed_loss(logits, targets, bce_w, focal_w, fg, fa, pw=None, smooth=0.0):
    if smooth > 0:
        targets = targets * (1.0 - smooth) + 0.5 * smooth
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw, reduction="mean")
    if focal_w > 0:
        fl = _focal_bce(logits, targets, fg, fa, pw=pw).mean()
        return bce_w * bce + focal_w * fl
    return bce


def _sampled_l2_map_loss(
        slog: torch.Tensor,
        y2d: torch.Tensor,
        valid2d: torch.Tensor,
        pos_weight: float,
        neg_per_pos: int,
        neg_min: int,
        neg_cap: int,
        hardneg_frac: float,
        rank_alpha: float = 0.0,
        rank_margin: float = 0.2,
        rank_pairs: int = 256,
):
    if slog.ndim == 2:
        slog = slog.unsqueeze(0)
        y2d = y2d.unsqueeze(0)
        valid2d = valid2d.unsqueeze(0)
    losses = []
    pw = None if pos_weight <= 0 else slog.new_tensor(float(pos_weight))
    for b in range(slog.size(0)):
        valid = valid2d[b].bool()
        if not bool(valid.any()):
            continue
        s = slog[b][valid]
        y = y2d[b][valid].float()
        pos_idx = torch.nonzero(y > 0.5, as_tuple=False).flatten()
        neg_idx = torch.nonzero(y <= 0.5, as_tuple=False).flatten()
        if pos_idx.numel() == 0:
            continue
        if neg_idx.numel() == 0:
            losses.append(F.binary_cross_entropy_with_logits(s, y, pos_weight=pw, reduction="mean"))
            continue
        n_pos = int(pos_idx.numel())
        k_neg = max(int(neg_min), int(n_pos * max(1, int(neg_per_pos))))
        k_neg = min(int(neg_cap), min(k_neg, int(neg_idx.numel())))
        hard_frac = float(max(0.0, min(1.0, hardneg_frac)))
        k_hard = min(k_neg, int(round(k_neg * hard_frac)))
        k_rand = max(0, k_neg - k_hard)
        sel_neg = []
        if k_hard > 0:
            hard_scores = s[neg_idx]
            top_hard = torch.topk(hard_scores, k=min(k_hard, hard_scores.numel()), largest=True).indices
            sel_neg.append(neg_idx[top_hard])
        if k_rand > 0:
            neg_pool = neg_idx
            if sel_neg:
                used = torch.unique(torch.cat(sel_neg, dim=0))
                keep_mask = ~torch.isin(neg_pool, used)
                neg_pool = neg_pool[keep_mask]
            if neg_pool.numel() > 0:
                perm = torch.randperm(neg_pool.numel(), device=neg_pool.device)[:min(k_rand, neg_pool.numel())]
                sel_neg.append(neg_pool[perm])
        neg_sel = torch.cat(sel_neg, dim=0) if sel_neg else neg_idx[:0]
        sel = torch.cat([pos_idx, neg_sel], dim=0)
        loss_bce = F.binary_cross_entropy_with_logits(
            s[sel], y[sel], pos_weight=pw, reduction="mean",
        )
        if rank_alpha > 0.0 and neg_sel.numel() > 0:
            k_rank = max(1, min(int(rank_pairs), int(pos_idx.numel()), int(neg_sel.numel())))
            pos_hard = pos_idx[torch.topk(s[pos_idx], k=k_rank, largest=False).indices]
            neg_hard = neg_sel[torch.topk(s[neg_sel], k=k_rank, largest=True).indices]
            rank_term = F.relu(float(rank_margin) - s[pos_hard] + s[neg_hard]).mean()
            losses.append(loss_bce + float(rank_alpha) * rank_term)
        else:
            losses.append(loss_bce)
    if not losses:
        return slog.new_zeros(())
    return torch.stack(losses).mean()


def _soft_dice_loss(logits, targets, eps=1e-6):
    prob = torch.sigmoid(logits)
    inter = (prob * targets).sum()
    denom = prob.sum() + targets.sum()
    return 1.0 - (2.0 * inter + eps) / (denom + eps)


def _residue_loss_by_protein(logits, targets, mask, pw):
    losses = []
    weights = []
    n_extreme = 0
    n_high = 0
    dice_terms = []
    B = logits.size(0)
    for b in range(B):
        m = mask[b].bool()
        if not bool(m.any()):
            continue
        lg = logits[b][m]
        yt = targets[b][m]
        n = float(yt.numel())
        pos = float((yt > 0.5).sum().item())
        frac = pos / max(1.0, n)
        wt = 1.0
        if pos <= 0.0 or pos >= n:
            wt = float(getattr(P, "l1_extreme_label_weight", 0.35))
            n_extreme += 1
        elif frac >= float(getattr(P, "l1_high_pos_frac", 0.95)):
            wt = float(getattr(P, "l1_high_pos_weight", 0.5))
            n_high += 1
        cur = _mixed_loss(
            lg, yt,
            bce_w=1 - P.l1_focal_w, focal_w=P.l1_focal_w,
            fg=P.l1_focal_gamma, fa=P.l1_focal_alpha, pw=pw,
            smooth=P.label_smoothing_l1
        )
        dice_w = float(getattr(P, "l1_dice_w", 0.0))
        if dice_w > 0.0 and pos > 0.0 and pos < n:
            dl = _soft_dice_loss(lg, yt)
            cur = cur + dice_w * dl
            dice_terms.append(dl.detach())
        losses.append(cur * wt)
        weights.append(logits.new_tensor(wt))
    if not losses:
        return logits.new_zeros(()), {"n": 0, "w_mean": 0.0, "n_extreme": 0, "n_high": 0, "dice": 0.0}
    loss = torch.stack(losses).sum() / torch.stack(weights).sum().clamp_min(1e-6)
    dice_val = float(torch.stack(dice_terms).mean()) if dice_terms else 0.0
    return loss, {
        "n": len(losses),
        "w_mean": float(torch.stack(weights).mean().detach()),
        "n_extreme": int(n_extreme),
        "n_high": int(n_high),
        "dice": dice_val,
    }


def _rank_loss(logits, labels, margin=0.2, n_pairs=256, hard_frac=0.75):
    pos = (labels > 0.5).nonzero(as_tuple=False).squeeze(-1)
    neg = (labels <= 0.5).nonzero(as_tuple=False).squeeze(-1)
    if pos.numel() == 0 or neg.numel() == 0:
        return torch.zeros((), device=logits.device)
    pos_frac = float(pos.numel()) / max(1.0, float(pos.numel() + neg.numel()))
    if pos_frac > float(getattr(P, "topk_rank_max_pos_frac", 0.80)):
        return torch.zeros((), device=logits.device)
    n_pairs = max(1, int(n_pairs))
    n_pos = min(n_pairs, pos.numel())
    pp = pos[torch.randperm(pos.numel(), device=logits.device)[:n_pos]]
    n_h = min(max(1, int(n_pairs * hard_frac)), neg.numel())
    _, ti = torch.topk(logits[neg], k=n_h, largest=True)
    hard_neg = neg[ti]
    n_r = min(max(0, n_pairs - n_h), neg.numel())
    rand_neg = neg[torch.randperm(neg.numel(), device=logits.device)[:n_r]]
    nn = torch.cat([hard_neg, rand_neg], dim=0).unique()
    if nn.numel() == 0:
        return torch.zeros((), device=logits.device)
    diff = logits[pp].view(-1, 1) - logits[nn].view(1, -1)
    return F.softplus(float(margin) - diff).mean()




def _hard_ap_rank_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.05,
    tau: float = 0.25,
    pos_cap: int = 128,
    neg_per_pos: int = 8,
):
    """Late-stage AUPRC-oriented hard ranking loss.

    This is intentionally optional and OFF by default. It should only be used
    with a small weight during second-stage fine-tuning, because the previous
    aggressive AP-rank + fragment-fusion run degraded validation AUPRC.
    """
    logits = logits.float()
    labels = labels.float()
    pos = logits[labels > 0.5]
    neg = logits[labels <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return logits.new_zeros(())

    pos_cap = int(max(1, pos_cap))
    if pos.numel() > pos_cap:
        # Focus on hard positives that the model currently scores too low.
        pos = torch.topk(pos, k=pos_cap, largest=False).values

    n_neg = min(int(max(16, int(neg_per_pos) * int(pos.numel()))), int(neg.numel()))
    if n_neg <= 0:
        return logits.new_zeros(())
    # Focus on hard negatives that are currently ranked too high.
    neg = torch.topk(neg, k=n_neg, largest=True).values

    diff = (neg.unsqueeze(0) - pos.unsqueeze(1) + float(margin)) / max(float(tau), 1e-6)
    return F.softplus(diff).mean()


def _pair_rank_loss(logits: torch.Tensor, labels: torch.Tensor, margin=0.20, tau=0.25, hard_frac=0.75):
    """Pair-level hard positive-vs-negative ranking loss for L3 AUPRC."""
    logits = logits.float().view(-1)
    labels = labels.float().view(-1)
    pos = logits[labels > 0.5]
    neg = logits[labels <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return logits.new_zeros(())
    # all positives are usually few; keep the hard/high-scoring negatives plus some random negatives
    n_hard = max(1, min(int(math.ceil(float(hard_frac) * float(neg.numel()))), int(neg.numel())))
    hard_neg = torch.topk(neg, k=n_hard, largest=True).values
    diff = (hard_neg.unsqueeze(0) - pos.unsqueeze(1) + float(margin)) / max(float(tau), 1e-6)
    return F.softplus(diff).mean()


def _pair_tau_margin_loss(logits: torch.Tensor, labels: torch.Tensor, tau=0.50, margin=0.05):
    """Differentiable threshold-margin loss around a fixed/learnable-threshold proxy."""
    prob = torch.sigmoid(logits.float().view(-1))
    y = labels.float().view(-1)
    tau = float(tau); margin = float(margin)
    pos_loss = y * F.relu((tau + margin) - prob)
    neg_loss = (1.0 - y) * F.relu(prob - (tau - margin))
    return (pos_loss + neg_loss).mean()


def _pair_disturb_loss(logits: torch.Tensor, labels: torch.Tensor, tau=0.50, target=0.15):
    """Encourage pair decisions to move away from the decision boundary after the classifier is stable."""
    prob = torch.sigmoid(logits.float().view(-1))
    disturb = torch.abs(prob - float(tau))
    return F.relu(float(target) - disturb).mean()


def _task_profile(site_mode: bool):
    """Return task-adaptive loss profile.

    L3/pair task: pair decision dominates; L1 evidence is auxiliary.
    L1/DEST task: residue-site decision dominates; L3 bag evidence is auxiliary.
    """
    prof = str(getattr(P, "task_loss_profile", "auto")).lower()
    if prof == "auto":
        prof = "l1_main" if site_mode else "l3_main"
    return prof


def _batch_pos_weight(labels: torch.Tensor, default: float, device=None):
    labels = labels.float().view(-1)
    if float(default) > 0:
        return torch.tensor(float(default), device=device or labels.device)
    pos = (labels > 0.5).sum().float()
    neg = (labels <= 0.5).sum().float()
    if pos.item() <= 0 or neg.item() <= 0:
        return torch.tensor(1.0, device=device or labels.device)
    return (neg / pos).clamp(1.0, 20.0).to(device or labels.device)


def _effective_rank_weight(ep):
    w = float(getattr(P, "l1_rank_w", 0.0))
    if w <= 0.0 or ep < int(getattr(P, "l1_rank_start_epoch", 0)):
        return 0.0
    if bool(getattr(P, "objective_weight_auto", True)) and PRIMARY_OBJ == "topk":
        primary = str(getattr(P, "topk_primary", "precision_K10")).lower()
        boost = float(getattr(P, "topk_rank_boost", 1.5))
        if "precision" in primary:
            w *= boost
        elif "score" in primary:
            w *= 0.90 * boost
        elif "recall" in primary:
            w *= 0.80 * boost
        else:
            w *= boost
    ramp = int(getattr(P, "topk_rank_ramp_epochs", 0))
    if ramp > 0:
        step = ep - int(getattr(P, "l1_rank_start_epoch", 0)) + 1
        w *= min(1.0, max(0.0, float(step) / float(ramp)))
    return float(w)


def _ramp(ep, start, ramp, max_w):
    if ep < start: return 0.0
    return max_w if ramp <= 0 else max_w * min(1.0, (ep - start) / ramp)


def _soft_clamp(loss, cap):
    if cap <= 0: return loss
    return torch.clamp(loss, max=float(cap))


def _soft_cap_with_grad(loss, cap):
    if cap <= 0:
        return loss
    cap_t = loss.new_tensor(float(cap))
    return cap_t * torch.tanh(loss / cap_t)


def _select_binary_threshold(prob, ytrue, mode=None):
    from sklearn.metrics import f1_score, matthews_corrcoef, recall_score
    mode = str(mode or getattr(P, 'val_thr_mode', 'auto_mcc')).lower()
    ts = np.linspace(float(P.val_thr_min), float(P.val_thr_max), int(P.val_thr_grid))
    if mode in ('fixed', 'manual'):
        return float(getattr(P, 'val_thr', 0.5)), 'fixed'
    best_t, best_score = 0.5, -1e9
    y_pos_rate = float(np.mean(ytrue)) if len(ytrue) else 0.0
    max_pred_rate = min(0.65, max(0.25, y_pos_rate * 3.0))
    min_pred_rate = max(0.001, min(0.02, y_pos_rate * 0.10))
    recall_floor = float(getattr(P, 'val_recall_floor', 0.65))
    beta = float(getattr(P, 'val_fbeta_beta', 1.5))
    beta2 = beta * beta
    fallback_t, fallback_recall = 0.5, -1.0
    for t in ts:
        pred = (prob >= t).astype(np.int32)
        try:
            rec = float(recall_score(ytrue, pred, zero_division=0))
            prec = float((ytrue[pred == 1].mean()) if pred.sum() > 0 else 0.0)
            f1 = float(f1_score(ytrue, pred, zero_division=0))
            mcc = float(matthews_corrcoef(ytrue, pred))
        except Exception:
            rec, prec, f1, mcc = 0.0, 0.0, 0.0, -1.0
        if rec > fallback_recall:
            fallback_recall, fallback_t = rec, float(t)
        if mode in ('auto_f1', 'f1', 'val_f1'):
            score = f1
        elif mode in ('auto_fbeta', 'fbeta', 'val_fbeta'):
            score = (1.0 + beta2) * prec * rec / max(1e-12, beta2 * prec + rec)
        elif mode in ('auto_recall_floor', 'recall_floor', 'val_recall_floor'):
            if rec + 1e-12 < recall_floor:
                continue
            score = mcc + 0.25 * f1
        elif mode in ('auto_posrate', 'posrate', 'prevalence'):
            score = -abs(float(pred.mean()) - float(ytrue.mean())) + 1e-3 * mcc
        else:
            pred_rate = float(pred.mean()) if pred.size else 0.0
            if pred_rate < min_pred_rate or pred_rate > max_pred_rate:
                continue
            score = mcc
        if score > best_score:
            best_score, best_t = float(score), float(t)
    if best_score <= -1e8:
        return float(fallback_t), f'{mode}_fallback_max_recall'
    return float(best_t), mode


def _site_evidence_pool_loss(logit_res, labels, mask, top_frac=0.05, pos_weight=2.0):
    """
    Lightweight L1->L3 bag-level link for mixed protein-level labels.
    It is skipped when the batch has only one bag class, which avoids pushing every
    RBP protein to an unconditional high score on all-positive site datasets.
    """
    B, L = logit_res.shape
    bag_logits, bag_labels = [], []
    for b in range(B):
        m = mask[b].bool()
        if not bool(m.any()):
            continue
        cur = logit_res[b][m]
        lab = labels[b][m]
        k = max(1, int(math.ceil(float(top_frac) * cur.numel())))
        pooled = cur.topk(k=k, largest=True).values.mean()
        bag_logits.append(pooled)
        bag_labels.append((lab > 0.5).any().float())
    if not bag_logits:
        return logit_res.new_zeros(()), 0
    bag_logits = torch.stack(bag_logits)
    bag_labels = torch.stack(bag_labels).to(dtype=bag_logits.dtype)
    if bag_labels.min().item() == bag_labels.max().item():
        return logit_res.new_zeros(()), int(bag_labels.numel())
    pw = logit_res.new_tensor(float(pos_weight))
    loss = F.binary_cross_entropy_with_logits(
        bag_logits, bag_labels, pos_weight=pw, reduction="mean"
    )
    return loss, int(bag_labels.numel())


# ============================================================
# Cross-level consistency loss
# ============================================================

def cross_level_consistency_loss(
        pair_logit, evi_score,
        logit_resA, logit_resB,
        pair_label, maskA, maskB,
        margin=0.5, tau=1.0, neg_suppress_w=0.3,
        pos_support_w=0.4, neg_pair_suppress_w=0.2,
        device="cuda"):
    """
    Encourage concentrated residue evidence for positive pairs, suppress
    evidence for negative pairs, and align pair logits with evidence scores.
    """
    pos_m = pair_label > 0.5
    neg_m = ~pos_m
    loss  = torch.zeros((), device=device)
    log   = {}

    def _masked_vals(v, m, idx):
        x = v[idx]
        if m is not None and idx < m.size(0):
            x = x[m[idx]]
        return x

    if pos_m.any():
        conc_terms = []
        pos_support_terms = []
        for b in range(pair_logit.size(0)):
            if not bool(pos_m[b]):
                continue
            chain_top_means = []
            for arr, mask in ((logit_resA, maskA), (logit_resB, maskB)):
                cur = _masked_vals(arr, mask, b)
                if cur.numel() < 4:
                    continue
                nq = max(1, cur.numel() // 4)
                top_m = cur.topk(nq).values.mean()
                bot_m = cur.topk(nq, largest=False).values.mean()
                conc_terms.append(F.relu(margin - (top_m - bot_m)))
                chain_top_means.append(top_m)
            # support terms must be recorded per positive pair; otherwise only
            # the final loop value of b/chain_top_means contributes.
            if chain_top_means:
                support_target = torch.stack(chain_top_means).mean()
                pos_support_terms.append(F.relu(support_target - evi_score[b]))
                pos_support_terms.append(F.relu(support_target - pair_logit[b]))
        if conc_terms:
            lc = torch.stack(conc_terms).mean()
            loss = loss + lc
            log["conc"] = float(lc.detach())
        if pos_support_terms:
            lps = torch.stack(pos_support_terms).mean() * float(pos_support_w)
            loss = loss + lps
            log["pos_support"] = float(lps.detach())

    if neg_m.any():
        neg_terms = [F.relu(evi_score[neg_m]).mean() * neg_suppress_w]
        neg_terms.append(F.relu(pair_logit[neg_m]).mean() * float(neg_pair_suppress_w))
        for arr, mask in ((logit_resA, maskA), (logit_resB, maskB)):
            vals = []
            for b in range(pair_logit.size(0)):
                if not bool(neg_m[b]):
                    continue
                cur = _masked_vals(arr, mask, b)
                if cur.numel() == 0:
                    continue
                vals.append(torch.sigmoid(cur).mean())
            if vals:
                neg_terms.append(torch.stack(vals).mean() * 0.5 * neg_suppress_w)
        ln = torch.stack(neg_terms).sum()
        loss = loss + ln
        log["neg"] = float(ln.detach())

    scale = max(float(tau), 1e-3)
    pp = torch.sigmoid(pair_logit / scale)
    pe = torch.sigmoid(evi_score / scale)
    la = F.mse_loss(pp, pe)
    loss = loss + la
    log["align"] = float(la.detach())
    return loss, log


# ============================================================
# forward_one
# ============================================================

def forward_one(model: L13PDBGVPModel, batch: Dict, ep: int
                ) -> Tuple[Dict, torch.Tensor, Dict]:
    """
    Run one forward pass and compute the complete training objective.
    """
    device = DEVICE
    aux    = {}

    resA  = torch.nan_to_num(batch["resA"],  nan=0.0).to(device).float()
    resB  = torch.nan_to_num(batch["resB"],  nan=0.0).to(device).float()
    coordsA = torch.nan_to_num(batch["coordsA"], nan=0.0).to(device).float()
    coordsB = torch.nan_to_num(batch["coordsB"], nan=0.0).to(device).float()
    maskA = (batch["maskA"] > 0.5).to(device)
    maskB = (batch["maskB"] > 0.5).to(device)
    chainA = batch.get("chainA", None)
    chainB = batch.get("chainB", None)
    if chainA is not None: chainA = torch.nan_to_num(chainA).to(device).float()
    if chainB is not None: chainB = torch.nan_to_num(chainB).to(device).float()
    site_mode = bool(torch.is_tensor(batch.get("site_mode", None)) and float(batch["site_mode"].max().item()) > 0.5)
    is_dips_pair = (
        (not site_mode)
        and str(getattr(P, "dataset_mode", "")).lower() == "dips"
        and str(getattr(P, "primary_objective", "")).lower() == "pair"
    )

    yA = batch["y_res_A"].to(device).float()
    yB = batch["y_res_B"].to(device).float()
    if yA.ndim == 1: yA = yA.unsqueeze(0)
    if yB.ndim == 1: yB = yB.unsqueeze(0)
    La, Lb = maskA.shape[1], maskB.shape[1]
    yA = yA[:, :La]
    yB = yB[:, :Lb]

    y_pair_pos = batch.get("y_pair", torch.ones(resA.size(0))).to(device).float()
    B = resA.size(0)

    (resA_neg, maskA_neg, coordsA_neg, resB_neg, maskB_neg, coordsB_neg,
     chainA_neg, chainB_neg, y_pair_neg) = make_inbatch_negatives(
        batch, device, neg_ratio=P.train_neg_ratio
    )
    n_neg = y_pair_neg.size(0)

    out_pos = model(resA, maskA, chainA, coordsA, resB, maskB, chainB, coordsB, site_mode=site_mode)

    out_neg = None
    if (not site_mode) and n_neg > 0:
        out_neg = model(resA_neg, maskA_neg, chainA_neg, coordsA_neg,
                        resB_neg, maskB_neg, chainB_neg, coordsB_neg, site_mode=False)

    pair_logits = [out_pos["pair_logit"]]
    evi_scores = [out_pos["evi_score"]]
    y_pairs = [y_pair_pos]
    if out_neg is not None:
        pair_logits.append(out_neg["pair_logit"])
        evi_scores.append(out_neg["evi_score"])
        y_pairs.append(y_pair_neg)

    pair_logit_all = torch.cat(pair_logits, dim=0)
    evi_score_all  = torch.cat(evi_scores, dim=0)
    y_pair_all     = torch.cat(y_pairs, dim=0)

    logit_resA  = out_pos["logit_resA"]
    logit_resB  = out_pos["logit_resB"]
    logit_fragA = out_pos["logit_fragA"]
    logit_fragB = out_pos["logit_fragB"]
    slog_pos = out_pos.get("S", None)
    l2_focus_mask = None
    l2_info = getattr(model, "_cache", {}).get("l2_info", None)
    if isinstance(l2_info, dict):
        l2_focus_mask = l2_info.get("focus_mask2d", None)

    loss_total = torch.zeros((), device=device)

    if site_mode:
        _l1_pw_labels = yA[maskA]
    else:
        _l1_pw_labels = torch.cat([yA[maskA], yB[maskB]], dim=0)
    pw_l1   = _batch_pos_weight(_l1_pw_labels, float(getattr(P, "l1_pos_weight", 0.0)), device=device)
    loss_l1 = torch.zeros((), device=device)
    l1_terms = [(logit_resA, yA, maskA)] if site_mode else [(logit_resA, yA, maskA), (logit_resB, yB, maskB)]
    l1_stat_n = 0
    l1_stat_extreme = 0
    l1_stat_high = 0
    l1_stat_w = []
    l1_stat_dice = []
    for lr, yr, mr in l1_terms:
        if mr.any():
            w = 1.0 if site_mode else 0.5
            if bool(getattr(P, "l1_per_protein_loss", True)) and lr.ndim == 2:
                cur_l1, st = _residue_loss_by_protein(lr, yr, mr, pw_l1)
                l1_stat_n += int(st.get("n", 0))
                l1_stat_extreme += int(st.get("n_extreme", 0))
                l1_stat_high += int(st.get("n_high", 0))
                l1_stat_w.append(float(st.get("w_mean", 0.0)))
                l1_stat_dice.append(float(st.get("dice", 0.0)))
                loss_l1 = loss_l1 + cur_l1 * w
            else:
                loss_l1 = loss_l1 + _mixed_loss(
                    lr[mr], yr[mr],
                    bce_w=1 - P.l1_focal_w, focal_w=P.l1_focal_w,
                    fg=P.l1_focal_gamma, fa=P.l1_focal_alpha, pw=pw_l1,
                    smooth=P.label_smoothing_l1
                ) * w
    aux["l1_n_prot"] = int(l1_stat_n)
    aux["l1_extreme"] = int(l1_stat_extreme)
    aux["l1_high"] = int(l1_stat_high)
    aux["l1_qw"] = float(np.mean(l1_stat_w)) if l1_stat_w else 1.0
    dice_vals = [x for x in l1_stat_dice if x > 0.0]
    aux["loss_dice"] = float(np.mean(dice_vals)) if dice_vals else 0.0

    loss_rank = torch.zeros((), device=device)
    rank_w_eff = _effective_rank_weight(ep)
    if is_dips_pair and ep < int(getattr(P, "dips_rank_start_epoch", 3)):
        rank_w_eff = 0.0
    if rank_w_eff > 0:
        rl = []
        for b in range(B):
            rank_terms = [(logit_resA[b], yA[b], maskA[b])] if site_mode else [
                (logit_resA[b], yA[b], maskA[b]),
                (logit_resB[b], yB[b], maskB[b]),
            ]
            for lr, yr, mr in rank_terms:
                lg, yt = lr[mr], yr[mr]
                if lg.numel() < 4:
                    continue
                rl.append(_rank_loss(lg, yt, P.l1_rank_margin,
                                     P.l1_rank_n_pairs, P.l1_rank_neg_hard_frac))
        if rl:
            loss_rank = _soft_clamp(torch.stack(rl).mean(), 5.0)
            loss_l1 = loss_l1 + loss_rank * rank_w_eff
    aux["loss_rank"] = float(loss_rank.detach())
    aux["rank_w_eff"] = float(rank_w_eff)

    # Optional second-stage AP-oriented hard ranking loss.
    # Default ap_rank_w=0.0, so this is inert in the stable main run.
    loss_ap_rank = torch.zeros((), device=device)
    ap_rank_w = float(getattr(P, "ap_rank_w", 0.0))
    ap_rank_start = int(getattr(P, "ap_rank_start_epoch", 999))
    if site_mode and ap_rank_w > 0.0 and ep >= ap_rank_start:
        ap_rl = []
        for b in range(B):
            lg = logit_resA[b][maskA[b]]
            yt = yA[b][maskA[b]]
            if lg.numel() < 4:
                continue
            ap_rl.append(_hard_ap_rank_loss(
                lg, yt,
                margin=float(getattr(P, "ap_rank_margin", 0.05)),
                tau=float(getattr(P, "ap_rank_tau", 0.25)),
                pos_cap=int(getattr(P, "ap_rank_pos_cap", 128)),
                neg_per_pos=int(getattr(P, "ap_rank_neg_per_pos", 8)),
            ))
        if ap_rl:
            loss_ap_rank = _soft_clamp(torch.stack(ap_rl).mean(), 5.0)
            loss_l1 = loss_l1 + loss_ap_rank * ap_rank_w
    aux["loss_ap_rank"] = float(loss_ap_rank.detach())
    aux["ap_rank_w_eff"] = float(ap_rank_w if (site_mode and ep >= ap_rank_start) else 0.0)

    loss_site_l3 = torch.zeros((), device=device)
    site_l3_n = 0
    site_l3_w = 0.0
    if site_mode:
        site_l3_w = _ramp(
            ep,
            getattr(P, "site_l3_pool_start_epoch", 999),
            1,
            getattr(P, "site_l3_pool_w", 0.0),
        )
        if site_l3_w > 0:
            loss_site_l3, site_l3_n = _site_evidence_pool_loss(
                logit_resA, yA, maskA,
                top_frac=getattr(P, "site_l3_pool_top_frac", 0.05),
                pos_weight=getattr(P, "site_l3_pool_pos_weight", 2.0),
            )
            loss_l1 = loss_l1 + _soft_clamp(loss_site_l3, P.max_l3_loss) * site_l3_w
    aux["loss_site_l3"] = float(loss_site_l3.detach())
    aux["site_l3_w"] = float(site_l3_w)
    aux["site_l3_n"] = int(site_l3_n)

    # Task-adaptive weighting: L3/pair task keeps pair classification primary;
    # L1/DEST task keeps residue-site prediction primary.
    task_profile = _task_profile(site_mode)
    if task_profile == "l3_main":
        eff_l1_w = _ramp(
            ep,
            int(getattr(P, "l3_aux_l1_start_epoch", 5)),
            int(getattr(P, "l3_aux_l1_ramp_epochs", 10)),
            float(getattr(P, "l3_aux_l1_w", 0.15)),
        )
        eff_l3_w = float(getattr(P, "l3_w", 1.0))
    else:
        eff_l1_w = float(getattr(P, "l1_w", 1.0))
        eff_l3_w = _ramp(
            ep,
            int(getattr(P, "l1_aux_l3_start_epoch", 5)),
            1,
            float(getattr(P, "l1_aux_l3_w", getattr(P, "site_l3_pool_w", 0.0))),
        ) if site_mode else 0.0
    loss_l1    = _soft_clamp(loss_l1, P.max_res_loss)
    loss_total = loss_total + loss_l1 * eff_l1_w
    aux["loss_l1"] = float(loss_l1.detach())
    aux["l1_w_eff"] = float(eff_l1_w)
    aux["task_profile"] = str(task_profile)

    loss_frag = torch.zeros((), device=device)
    if P.l15_w > 0 and ep >= P.l15_start_epoch:
        frag_terms = [(logit_fragA, yA, maskA)] if site_mode else [(logit_fragA, yA, maskA), (logit_fragB, yB, maskB)]
        for lf, yr, mr in frag_terms:
            if mr.any():
                w = 1.0 if site_mode else 0.5
                loss_frag = loss_frag + F.binary_cross_entropy_with_logits(
                    lf[mr], yr[mr], reduction="mean") * w
        loss_frag  = _soft_clamp(loss_frag, P.max_frag_loss)
        loss_total = loss_total + loss_frag * P.l15_w
    aux["loss_frag"] = float(loss_frag.detach())

    loss_l2_map = torch.zeros((), device=device)
    loss_l2_map_raw = torch.zeros((), device=device)
    l2_w_eff = 0.0
    if is_dips_pair and slog_pos is not None and ep >= int(getattr(P, "l2_map_start_epoch", 0)):
        y2d = batch.get("y2d", None)
        if y2d is not None:
            y2d = y2d.to(device).float()
            if y2d.ndim == 2:
                y2d = y2d.unsqueeze(0)
            valid2d = maskA.unsqueeze(-1) & maskB.unsqueeze(1)
            loss_l2_map_raw = _sampled_l2_map_loss(
                slog_pos, y2d, valid2d,
                pos_weight=float(getattr(P, "l2_map_pos_weight", P.l1_pos_weight)),
                neg_per_pos=int(getattr(P, "l2_neg_per_pos", 8)),
                neg_min=int(getattr(P, "l2_neg_min", 256)),
                neg_cap=int(getattr(P, "l2_neg_cap", 4096)),
                hardneg_frac=float(getattr(P, "l2_hardneg_frac", 0.75)),
                rank_alpha=float(getattr(P, "l2_rank_alpha", 0.20)),
                rank_margin=float(getattr(P, "l2_rank_margin", 0.20)),
                rank_pairs=int(getattr(P, "l2_rank_pairs", 256)),
            )
            loss_l2_map = _soft_cap_with_grad(loss_l2_map_raw, P.max_l3_loss)
            l2_w_eff = float(getattr(P, "l2_map_w", 1.0))
            if bool(getattr(P, "dips_hier_enable", True)):
                l2_start = float(getattr(P, "l2_map_w_start", 0.15))
                l2_ramp = max(0, int(getattr(P, "l2_map_w_ramp_epochs", 8)))
                l2_factor = 1.0 if l2_ramp <= 0 else min(1.0, float(ep + 1) / float(l2_ramp))
                l2_w_eff = float(getattr(P, "l2_map_w", 1.0)) * (l2_start + (1.0 - l2_start) * l2_factor)
            loss_total = loss_total + loss_l2_map * l2_w_eff
    aux["loss_l2_map"] = float(loss_l2_map.detach())
    aux["loss_l2_map_raw"] = float(loss_l2_map_raw.detach())
    aux["l2_w_eff"] = float(l2_w_eff)

    loss_l3 = torch.zeros((), device=device)
    loss_pair_rank = torch.zeros((), device=device)
    loss_tau_margin = torch.zeros((), device=device)
    loss_disturb = torch.zeros((), device=device)
    pair_rank_w_eff = 0.0
    tau_margin_w_eff = 0.0
    disturb_w_eff = 0.0
    if not site_mode:
        pw_l3 = _batch_pos_weight(y_pair_all, float(getattr(P, "l3_pos_weight", 0.0)), device=device)
        loss_l3 = _mixed_loss(
            pair_logit_all, y_pair_all,
            bce_w=1 - P.l3_focal_w, focal_w=P.l3_focal_w,
            fg=P.l3_focal_gamma, fa=P.l3_focal_alpha, pw=pw_l3
        )
        loss_l3 = _soft_clamp(loss_l3, P.max_l3_loss)
        loss_total = loss_total + loss_l3 * eff_l3_w

        if task_profile == "l3_main":
            pair_rank_w_eff = _ramp(
                ep, int(getattr(P, "l3_pair_rank_start_epoch", 3)),
                int(getattr(P, "l3_pair_rank_ramp_epochs", 8)),
                float(getattr(P, "l3_pair_rank_w", 0.15)),
            )
            if pair_rank_w_eff > 0.0:
                loss_pair_rank = _pair_rank_loss(
                    pair_logit_all, y_pair_all,
                    margin=float(getattr(P, "l3_pair_rank_margin", 0.20)),
                    tau=float(getattr(P, "l3_pair_rank_tau", 0.25)),
                    hard_frac=float(getattr(P, "l3_pair_rank_hard_frac", 0.75)),
                )
                loss_pair_rank = _soft_clamp(loss_pair_rank, P.max_l3_loss)
                loss_total = loss_total + loss_pair_rank * pair_rank_w_eff

            tau_margin_w_eff = _ramp(
                ep, int(getattr(P, "l3_tau_margin_start_epoch", 5)),
                int(getattr(P, "l3_tau_margin_ramp_epochs", 10)),
                float(getattr(P, "l3_tau_margin_w", 0.05)),
            )
            if tau_margin_w_eff > 0.0:
                loss_tau_margin = _pair_tau_margin_loss(
                    pair_logit_all, y_pair_all,
                    tau=float(getattr(P, "l3_tau", 0.50)),
                    margin=float(getattr(P, "l3_tau_margin", 0.05)),
                )
                loss_total = loss_total + _soft_clamp(loss_tau_margin, P.max_l3_loss) * tau_margin_w_eff

            disturb_w_eff = _ramp(
                ep, int(getattr(P, "l3_disturb_start_epoch", 5)),
                int(getattr(P, "l3_disturb_ramp_epochs", 10)),
                float(getattr(P, "l3_disturb_w", 0.05)),
            )
            if disturb_w_eff > 0.0:
                loss_disturb = _pair_disturb_loss(
                    pair_logit_all, y_pair_all,
                    tau=float(getattr(P, "l3_tau", 0.50)),
                    target=float(getattr(P, "l3_disturb_target", 0.15)),
                )
                loss_total = loss_total + _soft_clamp(loss_disturb, P.max_l3_loss) * disturb_w_eff
    aux["loss_l3"] = float(loss_l3.detach())
    aux["l3_w_eff"] = float(eff_l3_w)
    aux["loss_pair_rank"] = float(loss_pair_rank.detach())
    aux["pair_rank_w_eff"] = float(pair_rank_w_eff)
    aux["loss_tau_margin"] = float(loss_tau_margin.detach())
    aux["tau_margin_w_eff"] = float(tau_margin_w_eff)
    aux["loss_disturb"] = float(loss_disturb.detach())
    aux["disturb_w_eff"] = float(disturb_w_eff)

    loss_cons_l13 = torch.zeros((), device=device)
    cons_w = 0.0
    if (not site_mode) and task_profile == "l3_main":
        cons_w = _ramp(
            ep, int(getattr(P, "l3_cons_start_epoch", 8)),
            int(getattr(P, "l3_cons_ramp_epochs", 10)),
            float(getattr(P, "l3_cons_w", getattr(P, "consistency_w", 0.03))),
        )
    elif (not site_mode):
        cons_w = float(getattr(P, "consistency_w", 0.0))
    if (not site_mode) and cons_w > 0.0:
        strengths = []
        for out in ([out_pos] + ([out_neg] if out_neg is not None else [])):
            if out is None:
                continue
            ta = out.get("topk_evidence_A", out.get("topk_valA", None))
            tb = out.get("topk_evidence_B", out.get("topk_valB", None))
            if ta is None or tb is None:
                continue
            if ta.numel() == 0 or tb.numel() == 0:
                continue
            strengths.append(0.5 * (ta.float().mean(dim=1) + tb.float().mean(dim=1)))
        if strengths:
            evi_strength = torch.cat(strengths, dim=0)
            if evi_strength.numel() == y_pair_all.numel():
                loss_cons_l13 = F.binary_cross_entropy_with_logits(evi_strength, y_pair_all.float())
                loss_total = loss_total + _soft_clamp(loss_cons_l13, P.max_l3_loss) * cons_w
    aux["loss_cons_l13"] = float(loss_cons_l13.detach())
    aux["consistency_w"] = float(cons_w)

    l3_evi_w_eff = 0.0 if is_dips_pair else float(P.l3_evi_w)
    if (not site_mode) and l3_evi_w_eff > 0:
        pm = y_pair_all > 0.5
        nm = ~pm
        loss_evi = torch.zeros((), device=device)
        if pm.any():
            loss_evi = loss_evi + F.relu(P.l3_evi_margin - evi_score_all[pm]).mean() * 0.5
        if nm.any():
            loss_evi = loss_evi + F.relu(P.l3_evi_margin + evi_score_all[nm]).mean() * 0.5
        loss_evi   = _soft_clamp(loss_evi, P.max_l3_loss)
        loss_total = loss_total + loss_evi * l3_evi_w_eff
        aux["loss_evi"] = float(loss_evi.detach())
    else:
        aux["loss_evi"] = 0.0

    if site_mode:
        cl_w = 0.0
    elif task_profile == "l3_main":
        cl_w = _ramp(
            ep, int(getattr(P, "l3_cons_start_epoch", 8)),
            int(getattr(P, "l3_cons_ramp_epochs", 10)),
            float(getattr(P, "cl_cons_w", 0.03)),
        )
    else:
        cl_w = _ramp(ep, P.cl_cons_start_epoch, P.cl_cons_ramp_epochs, P.cl_cons_w)
    if cl_w > 0:
        lrA_all = [out_pos["logit_resA"]]
        lrB_all = [out_pos["logit_resB"]]
        mA_all = [maskA]
        mB_all = [maskB]
        if out_neg is not None:
            lrA_all.append(out_neg["logit_resA"])
            lrB_all.append(out_neg["logit_resB"])
            mA_all.append(maskA_neg)
            mB_all.append(maskB_neg)

        loss_cl, cl_log = cross_level_consistency_loss(
            pair_logit_all, evi_score_all,
            torch.cat(lrA_all, dim=0), torch.cat(lrB_all, dim=0),
            y_pair_all,
            torch.cat(mA_all, dim=0), torch.cat(mB_all, dim=0),
            P.cl_cons_margin, P.cl_cons_tau, P.cl_cons_neg_suppress_w,
            getattr(P, "cl_cons_pos_support_w", 0.4),
            getattr(P, "cl_cons_neg_pair_suppress_w", 0.2),
            device=device
        )
        loss_cl    = _soft_clamp(loss_cl, P.max_cons_loss)
        loss_total = loss_total + loss_cl * cl_w
        aux["loss_cl"] = float(loss_cl.detach())
        for k, v in cl_log.items():
            aux[f"cl_{k}"] = v
    else:
        aux["loss_cl"] = 0.0
    aux["cl_w_eff"] = float(cl_w)

    if not torch.isfinite(loss_total):
        loss_total = torch.zeros((), device=device, requires_grad=True)
    aux["loss_total"] = float(loss_total.detach())
    return out_pos, loss_total, aux


# ============================================================
# Evaluation
# ============================================================

def _expected_calibration_error(prob, labels, n_bins=15):
    prob = np.asarray(prob, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if prob.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, int(max(2, n_bins)) + 1)
    total = float(prob.size)
    ece = 0.0
    for i in range(len(edges) - 1):
        if i == len(edges) - 2:
            mask = (prob >= edges[i]) & (prob <= edges[i + 1])
        else:
            mask = (prob >= edges[i]) & (prob < edges[i + 1])
        if not np.any(mask):
            continue
        confidence = float(prob[mask].mean())
        accuracy = float(labels[mask].mean())
        ece += (float(mask.sum()) / total) * abs(confidence - accuracy)
    return float(ece)


def eval_binary(
    model,
    dl,
    device=DEVICE,
    fixed_thr=None,
    threshold_mode=None,
    prediction_out=None,
):
    from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, f1_score, precision_score, recall_score
    model.eval()
    base_model = model.module if hasattr(model, "module") else model
    l2_bridge = getattr(base_model, "l2_bridge", None)
    saved_l2_geom_prior_w = None
    if l2_bridge is not None and hasattr(l2_bridge, "geom_prior_w"):
        saved_l2_geom_prior_w = float(l2_bridge.geom_prior_w)
        # Validation geometry prior is controlled by config_L13.Params.eval_l2_geom_prior_w.
        # Environment variables EVAL_L2_GEOM_PRIOR_W / eval_l2_geom_prior_w can still override it.
        # MedAUC boost mode: validation geometry prior defaults to the training geometry prior.
        # This avoids the previous mismatch: train_EB_geom_w=1.00 but eval_EB_geom_w=0.00.
        eval_default = getattr(P, "eval_l2_geom_prior_w", getattr(P, "l2_geom_prior_w", 0.0))
        eval_prior = os.environ.get(
            "EVAL_L2_GEOM_PRIOR_W",
            os.environ.get("eval_l2_geom_prior_w", str(eval_default))
        )
        try:
            l2_bridge.geom_prior_w = float(eval_prior)
        except Exception:
            l2_bridge.geom_prior_w = float(eval_default)
    if saved_l2_geom_prior_w is not None and float(getattr(P, "l2_geom_prior_w", 0.0)) > 0 and float(getattr(l2_bridge, "geom_prior_w", 0.0)) <= 0:
        print("[eval][warn] l2_geom_prior_w is enabled for training but disabled for validation; MedAUC will be strict/no-geometry.", flush=True)
    site_detected = False
    all_prob, all_lab = [], []
    site_records = []  # per-protein arrays for macro/surface-aware protocol metrics
    topk_hits = {0.1: [], 0.2: []}
    auc_list = []
    geom_auc_list = []
    focus_pos_hits, focus_pos_totals = [], []
    focus_pair_counts, valid_pair_counts = [], []
    pair_prob_all, pair_lab_all = [], []
    pair_decision_prob_all, pair_decision_lab_all = [], []
    pair_decision_id_all = []
    collect_global_pair_metrics = bool(getattr(P, "dips_report_extra_metrics", False))

    def _sample_l2_pixels(prob_vec: np.ndarray, lab_vec: np.ndarray,
                          max_pixels: int = 4096, neg_pos_ratio: int = 20,
                          neg_min: int = 512):
        if prob_vec.size == 0 or lab_vec.size == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
        y = (lab_vec > 0.5).astype(np.int32)
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        if pos.size == 0 and neg.size == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
        if pos.size == 0:
            k = min(int(max_pixels), int(neg.size))
            sel = np.random.choice(neg, size=k, replace=False) if neg.size > k else neg
            return prob_vec[sel].astype(np.float32), lab_vec[sel].astype(np.float32)
        k_neg = max(int(neg_min), int(pos.size * neg_pos_ratio))
        k_neg = min(int(max_pixels - pos.size), k_neg)
        k_neg = max(0, min(k_neg, int(neg.size)))
        if k_neg > 0:
            neg_sel = np.random.choice(neg, size=k_neg, replace=False) if neg.size > k_neg else neg
            sel = np.concatenate([pos, neg_sel], axis=0)
        else:
            sel = pos
        return prob_vec[sel].astype(np.float32), lab_vec[sel].astype(np.float32)

    def _pair_l2_from_residue(logitA: torch.Tensor, logitB: torch.Tensor,
                              y2d: torch.Tensor, mA: torch.Tensor, mB: torch.Tensor):
        mA = mA.bool()
        mB = mB.bool()
        if (not bool(mA.any())) or (not bool(mB.any())):
            return None
        la = logitA[mA]
        lb = logitB[mB]
        yy = y2d[:la.numel(), :lb.numel()].float()
        if yy.numel() == 0:
            return None
        # Reconstruct a dense pairwise score map from the residue evidence.
        # For AUC-type metrics, any monotonic transform is acceptable; sigmoid
        # keeps the values easy to interpret in [0, 1].
        score = torch.sigmoid(0.5 * (la.unsqueeze(1) + lb.unsqueeze(0)))
        return score.detach().float().cpu().numpy().reshape(-1), yy.detach().float().cpu().numpy().reshape(-1)

    def _pair_l2_from_map(slog: torch.Tensor, y2d: torch.Tensor, mA: torch.Tensor, mB: torch.Tensor):
        mA = mA.bool()
        mB = mB.bool()
        if (not bool(mA.any())) or (not bool(mB.any())):
            return None
        ls = int(mA.sum().item())
        rs = int(mB.sum().item())
        score = torch.sigmoid(slog[:ls, :rs])
        yy = y2d[:ls, :rs].float()
        if yy.numel() == 0:
            return None
        return score.detach().float().cpu().numpy().reshape(-1), yy.detach().float().cpu().numpy().reshape(-1)

    def _safe_ap_np(y, p):
        y = np.asarray(y).astype(np.int32)
        p = np.asarray(p).astype(np.float32)
        ok = np.isfinite(y) & np.isfinite(p)
        y, p = y[ok], p[ok]
        if y.size == 0 or len(np.unique(y)) < 2:
            return None
        try:
            return float(average_precision_score(y, p))
        except Exception:
            return None

    def _macro_ap_records(records, mask_key=None):
        vals = []
        for r in records:
            y = r.get('lab')
            p = r.get('prob')
            if y is None or p is None:
                continue
            if mask_key is not None:
                m = r.get(mask_key)
                if m is None:
                    continue
                m = np.asarray(m).astype(bool)
                y = y[m]
                p = p[m]
            ap = _safe_ap_np((np.asarray(y) > 0.5).astype(np.int32), p)
            if ap is not None:
                vals.append(ap)
        return float(np.mean(vals)) if vals else 0.0

    def _global_ap_records(records, mask_key=None):
        ys, ps = [], []
        for r in records:
            y = r.get('lab')
            p = r.get('prob')
            if y is None or p is None:
                continue
            if mask_key is not None:
                m = r.get(mask_key)
                if m is None:
                    continue
                m = np.asarray(m).astype(bool)
                y = y[m]
                p = p[m]
            if len(y):
                ys.append((np.asarray(y) > 0.5).astype(np.int32))
                ps.append(np.asarray(p).astype(np.float32))
        if not ys:
            return 0.0
        return _safe_ap_np(np.concatenate(ys), np.concatenate(ps)) or 0.0

    def _load_surface_mask_for_pid(pid, L):
        sdir = str(getattr(P, 'surface_mask_dir', '') or '')
        if not sdir:
            return None
        root = str(getattr(P, 'rbp_root', ''))
        d = sdir if os.path.isabs(sdir) else os.path.join(root, sdir)
        for ext in ('.npy', '.npz'):
            fp = os.path.join(d, str(pid) + ext)
            if os.path.exists(fp):
                try:
                    if ext == '.npy':
                        m = np.load(fp, allow_pickle=True)
                    else:
                        z = np.load(fp, allow_pickle=True)
                        key = 'surface' if 'surface' in z.files else z.files[0]
                        m = z[key]
                    m = np.asarray(m).reshape(-1).astype(bool)
                    if m.shape[0] < L:
                        m = np.pad(m, (0, L - m.shape[0]), constant_values=False)
                    return m[:L]
                except Exception:
                    return None
        return None

    def _ca_neighbor_masks(coords_np):
        out = {}
        if coords_np is None or len(coords_np) <= 1:
            return out
        c = np.asarray(coords_np, dtype=np.float32)
        if c.ndim != 2 or c.shape[1] < 3:
            return out
        c = c[:, :3]
        radius = float(getattr(P, 'eval_ca_surface_radius', 10.0))
        dist = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
        cnt = ((dist < radius) & (dist > 1e-6)).sum(axis=1)
        cut_s = str(getattr(P, 'eval_ca_surface_cutoffs', '10,12,14,16,18,20,22,24'))
        for tok in cut_s.split(','):
            tok = tok.strip()
            if not tok:
                continue
            try:
                cutoff = int(float(tok))
            except Exception:
                continue
            out[f'ca{cutoff}_mask'] = (cnt <= cutoff)
        return out

    def _pair_l2_from_geometry(coordsA: torch.Tensor, coordsB: torch.Tensor,
                               y2d: torch.Tensor, mA: torch.Tensor, mB: torch.Tensor):
        mA = mA.bool()
        mB = mB.bool()
        if (not bool(mA.any())) or (not bool(mB.any())):
            return None
        ca = coordsA[mA].float()
        cb = coordsB[mB].float()
        yy = y2d[:mA.numel(), :mB.numel()].float()[mA][:, mB]
        if ca.numel() == 0 or cb.numel() == 0 or yy.numel() == 0:
            return None
        score = -torch.cdist(ca, cb)
        return score.detach().float().cpu().numpy().reshape(-1), yy.detach().float().cpu().numpy().reshape(-1)

    try:
        with torch.no_grad():
            for batch in dl:
                resA  = torch.nan_to_num(batch["resA"], nan=0.0).to(device).float()
                resB  = torch.nan_to_num(batch["resB"], nan=0.0).to(device).float()
                coordsA = torch.nan_to_num(batch["coordsA"], nan=0.0).to(device).float()
                coordsB = torch.nan_to_num(batch["coordsB"], nan=0.0).to(device).float()
                maskA = (batch["maskA"] > 0.5).to(device)
                maskB = (batch["maskB"] > 0.5).to(device)
                cA = batch.get("chainA", None); cB = batch.get("chainB", None)
                if cA is not None: cA = cA.to(device).float()
                if cB is not None: cB = cB.to(device).float()
                site_mode = bool(torch.is_tensor(batch.get('site_mode', None)) and float(batch['site_mode'].max().item()) > 0.5)
                with (autocast(dtype=AMP_DTYPE) if USE_AMP else nullcontext()):
                    out_pos = model(resA, maskA, cA, coordsA, resB, maskB, cB, coordsB, site_mode=site_mode)
                if site_mode:
                    site_detected = True
                    logits = out_pos['logit_resA']
                    yA = batch['y_res_A'].to(device).float()
                    if yA.ndim == 1: yA = yA.unsqueeze(0)
                    if logits.ndim == 1: logits = logits.unsqueeze(0)
                    for b in range(logits.shape[0]):
                        mA = maskA[b].bool()
                        if not bool(mA.any()):
                            continue
                        prob_t = torch.sigmoid(logits[b][mA])
                        if bool(getattr(P, "binary_ager_eval", False)) and bool(getattr(P, "ager_enable", True)):
                            c = coordsA[b][mA]
                            if c.ndim == 2 and c.size(0) == prob_t.numel():
                                prob_t = _ager_refine_scores(prob_t, c)
                        lab_t = yA[b][mA]
                        if bool(getattr(P, "report_topk_metrics", True)):
                            npos = int((lab_t > 0.5).sum().item())
                            if npos > 0:
                                for frac in topk_hits:
                                    k = max(1, int(math.ceil(float(frac) * prob_t.numel())))
                                    hit = int((lab_t[torch.topk(prob_t, k=k).indices] > 0.5).sum().item())
                                    topk_hits[frac].append(hit / max(1, npos))
                        prob = prob_t.detach().float().cpu().numpy()
                        lab = lab_t.detach().float().cpu().numpy()
                        all_prob.append(prob); all_lab.append(lab)
                        pid_list = batch.get('complex', None)
                        pid = pid_list[b] if isinstance(pid_list, (list, tuple)) and b < len(pid_list) else str(len(site_records))
                        coords_np = coordsA[b][mA].detach().float().cpu().numpy() if coordsA is not None else None
                        rec = {'pid': str(pid), 'prob': prob, 'lab': lab, 'coords': coords_np}
                        surf = _load_surface_mask_for_pid(str(pid), len(prob))
                        if surf is not None and len(surf) == len(prob):
                            rec['surface_mask'] = surf
                        rec.update(_ca_neighbor_masks(coords_np))
                        site_records.append(rec)
                else:
                    # L3 protein-pair decision metrics. Original batch pairs use y_pair;
                    # optional deterministic in-batch shuffled partners provide negatives.
                    y_pair_eval = batch.get("y_pair", batch.get("has_contact", torch.ones(resA.size(0)))).to(device).float()
                    pair_decision_prob_all.append(torch.sigmoid(out_pos["pair_logit"]).detach().float().cpu().numpy().reshape(-1))
                    pair_decision_lab_all.append(y_pair_eval.detach().float().cpu().numpy().reshape(-1))
                    batch_ids = batch.get("complex", None)
                    if isinstance(batch_ids, (list, tuple)):
                        current_pair_ids = [str(value) for value in batch_ids]
                    else:
                        start = len(pair_decision_id_all)
                        current_pair_ids = [
                            f"pair_{start + offset}" for offset in range(resA.size(0))
                        ]
                    pair_decision_id_all.extend(current_pair_ids)
                    if bool(getattr(P, "pair_eval_make_negatives", True)) and resA.size(0) > 1:
                        perm = torch.roll(torch.arange(resA.size(0), device=device), shifts=1, dims=0)
                        with (autocast(dtype=AMP_DTYPE) if USE_AMP else nullcontext()):
                            out_neg_eval = model(
                                resA, maskA, cA, coordsA,
                                resB[perm], maskB[perm], (cB[perm] if cB is not None else None), coordsB[perm],
                                site_mode=False,
                            )
                        pair_decision_prob_all.append(torch.sigmoid(out_neg_eval["pair_logit"]).detach().float().cpu().numpy().reshape(-1))
                        pair_decision_lab_all.append(np.zeros((resA.size(0),), dtype=np.float32))
                        pair_decision_id_all.extend(
                            f"{current_pair_ids[offset]}__shuffled"
                            for offset in range(resA.size(0))
                        )

                    y2d = batch.get("y2d", None)
                    if y2d is None:
                        continue
                    y2d = y2d.detach().cpu()
                    # Validation diagnostics: (1) geometry-oracle MedAUC from -cdist,
                    # (2) how much of the true contact map is covered by L2Bridge's focus window.
                    cache = getattr(base_model, "_cache", {}) if base_model is not None else {}
                    l2_info = cache.get("l2_info", {}) if isinstance(cache, dict) else {}
                    focus_mask2d = l2_info.get("focus_mask2d", None) if isinstance(l2_info, dict) else None
                    use_explicit_map = "S" in out_pos and out_pos["S"] is not None
                    for b in range(resA.size(0)):
                        geom_pack = _pair_l2_from_geometry(
                            coordsA[b].detach().cpu(), coordsB[b].detach().cpu(),
                            y2d[b], maskA[b].detach().cpu(), maskB[b].detach().cpu()
                        )
                        if geom_pack is not None:
                            gp, gy = geom_pack
                            gyb = (gy > 0.5).astype(np.int32)
                            if gyb.max() != gyb.min():
                                try:
                                    geom_auc_list.append(float(roc_auc_score(gyb, gp)))
                                except Exception:
                                    pass
                        if focus_mask2d is not None:
                            fm = focus_mask2d[b].detach().cpu().bool()
                            yy_full = y2d[b].bool()
                            ma = maskA[b].detach().cpu().bool()
                            mb = maskB[b].detach().cpu().bool()
                            yy = yy_full[:ma.numel(), :mb.numel()][ma][:, mb]
                            ff = fm[:ma.numel(), :mb.numel()][ma][:, mb]
                            pos_total = int(yy.sum().item())
                            if pos_total > 0:
                                focus_pos_hits.append(int((yy & ff).sum().item()))
                                focus_pos_totals.append(pos_total)
                            valid_pair_counts.append(int(yy.numel()))
                            focus_pair_counts.append(int(ff.sum().item()))
                        if use_explicit_map:
                            packed = _pair_l2_from_map(
                                out_pos["S"][b],
                                y2d[b],
                                maskA[b].detach().cpu(),
                                maskB[b].detach().cpu(),
                            )
                        else:
                            packed = _pair_l2_from_residue(
                                out_pos["logit_resA"][b],
                                out_pos["logit_resB"][b],
                                y2d[b],
                                maskA[b].detach().cpu(),
                                maskB[b].detach().cpu(),
                            )
                        if packed is None:
                            continue
                        prob_np, lab_np = packed
                        if prob_np.size == 0 or lab_np.size == 0:
                            continue
                        ybin = (lab_np > 0.5).astype(np.int32)
                        if ybin.max() != ybin.min():
                            try:
                                auc_list.append(float(roc_auc_score(ybin, prob_np)))
                            except Exception:
                                pass
                        if collect_global_pair_metrics:
                            p_sel, y_sel = _sample_l2_pixels(
                                prob_np, lab_np,
                                max_pixels=int(getattr(P, "pair_eval_max_pixels", 4096)),
                                neg_pos_ratio=int(getattr(P, "pair_eval_neg_pos_ratio", 20)),
                                neg_min=int(getattr(P, "pair_eval_neg_min", 512)),
                            )
                            if p_sel.size > 0:
                                pair_prob_all.append(p_sel)
                                pair_lab_all.append(y_sel)
        if site_detected:
            prob = np.concatenate(all_prob) if all_prob else np.zeros(0, dtype=np.float32)
            lab = np.concatenate(all_lab) if all_lab else np.zeros(0, dtype=np.float32)
            res = {
                'n_eval': int(lab.size),
                'prob_mean': float(prob.mean()) if prob.size else 0.0,
                'pos_rate': float((lab > 0.5).mean()) if lab.size else 0.0,
            }
            if lab.size and len(np.unique(lab)) > 1:
                ytrue = (lab > 0.5).astype(np.int32)
                if fixed_thr is not None:
                    best_t = float(fixed_thr)
                    thr_mode_used = "fixed_from_validation"
                else:
                    best_t, thr_mode_used = _select_binary_threshold(prob, ytrue, mode=threshold_mode)
                pred = (prob >= best_t).astype(np.int32)
                res.update({
                    'acc': float((pred == ytrue).mean()),
                    'precision': float(precision_score(ytrue, pred, zero_division=0)),
                    'recall': float(recall_score(ytrue, pred, zero_division=0)),
                    'f1': float(f1_score(ytrue, pred, zero_division=0)),
                    'mcc': float(matthews_corrcoef(ytrue, pred)),
                    'auroc': float(roc_auc_score(ytrue, prob)),
                    'auprc': float(average_precision_score(ytrue, prob)),
                    'thr': float(best_t),
                    'thr_mode': thr_mode_used,
                })
                res['ager'] = bool(getattr(P, "binary_ager_eval", False)) and bool(getattr(P, "ager_enable", True))
                # Protocol-aware metrics are computed on the same predictions.
                # They are now part of the normal validation/test output rather than
                # a separate post-hoc-only script.
                res['auprc_macro'] = _macro_ap_records(site_records)
                res['surface_auprc'] = _global_ap_records(site_records, 'surface_mask')
                res['surface_macro_auprc'] = _macro_ap_records(site_records, 'surface_mask')
                for cutoff in (10, 12, 14, 16, 18, 20, 22, 24):
                    key = f'ca{cutoff}_mask'
                    res[f'ca{cutoff}_auprc'] = _global_ap_records(site_records, key)
                    res[f'ca{cutoff}_macro_auprc'] = _macro_ap_records(site_records, key)
                primary_key = str(getattr(P, 'binary_primary_metric', 'auprc')).lower()
                res['primary'] = float(res.get(primary_key, res.get('auprc', 0.0)))
            else:
                res.update({'acc':0.0,'precision':0.0,'recall':0.0,'f1':0.0,'mcc':0.0,'auroc':0.0,'auprc':0.0,'auprc_macro':0.0,'surface_auprc':0.0,'surface_macro_auprc':0.0,'ca12_auprc':0.0,'ca12_macro_auprc':0.0,'thr':0.5,'primary':0.0})
            if bool(getattr(P, "report_topk_metrics", True)):
                for frac, vals in topk_hits.items():
                    res[f'topK_recall_L{int(frac*10)}'] = float(np.mean(vals)) if vals else 0.0
            return res
        pp = np.concatenate(pair_prob_all) if pair_prob_all else np.zeros(0, dtype=np.float32)
        lb = np.concatenate(pair_lab_all) if pair_lab_all else np.zeros(0, dtype=np.float32)
        pdp = np.concatenate(pair_decision_prob_all) if pair_decision_prob_all else np.zeros(0, dtype=np.float32)
        pdl = np.concatenate(pair_decision_lab_all) if pair_decision_lab_all else np.zeros(0, dtype=np.float32)
        medauc = float(np.median(np.asarray(auc_list, dtype=np.float64))) if auc_list else 0.0
        medauc_geom = float(np.median(np.asarray(geom_auc_list, dtype=np.float64))) if geom_auc_list else 0.0
        focus_pos_recall = (float(np.sum(focus_pos_hits)) / max(1.0, float(np.sum(focus_pos_totals)))) if focus_pos_totals else 0.0
        focus_frac = (float(np.sum(focus_pair_counts)) / max(1.0, float(np.sum(valid_pair_counts)))) if valid_pair_counts else 0.0
        res = {
            'prob_mean_l2': float(pp.mean()) if pp.size else 0.0,
            'pos_rate_l2': float((lb > 0.5).mean()) if lb.size else 0.0,
            'pair_prob_mean': float(pdp.mean()) if pdp.size else 0.0,
            'pair_pos_rate': float((pdl > 0.5).mean()) if pdl.size else 0.0,
            'n_eval': int(pdl.size if pdl.size else lb.size),
            'n_auc': int(len(auc_list)),
            'medauc': medauc,
            'medauc_geom': medauc_geom,
            'focus_pos_recall': focus_pos_recall,
            'focus_frac': focus_frac,
        }
        if pdl.size and len(np.unique((pdl > 0.5).astype(np.int32))) > 1:
            ypair = (pdl > 0.5).astype(np.int32)
            res['pair_auroc'] = float(roc_auc_score(ypair, pdp))
            res['pair_auprc'] = float(average_precision_score(ypair, pdp))
            if fixed_thr is not None:
                thr_pair = float(fixed_thr)
                thr_mode_pair = "fixed_from_validation"
            else:
                thr_pair, thr_mode_pair = _select_binary_threshold(
                    pdp, ypair, mode=getattr(P, 'val_thr_mode', 'auto_mcc')
                )
            pred_pair = (pdp >= thr_pair).astype(np.int32)
            res['pair_acc'] = float((pred_pair == ypair).mean())
            res['pair_precision'] = float(precision_score(ypair, pred_pair, zero_division=0))
            res['pair_recall'] = float(recall_score(ypair, pred_pair, zero_division=0))
            tn = int(((pred_pair == 0) & (ypair == 0)).sum())
            fp = int(((pred_pair == 1) & (ypair == 0)).sum())
            res['pair_specificity'] = float(tn / max(1, tn + fp))
            res['pair_f1'] = float(f1_score(ypair, pred_pair, zero_division=0))
            res['pair_mcc'] = float(matthews_corrcoef(ypair, pred_pair))
            res['pair_brier'] = float(np.mean((pdp - ypair) ** 2))
            res['pair_ece'] = _expected_calibration_error(
                pdp, ypair, n_bins=int(os.environ.get("ECE_BINS", "15"))
            )
            res['pair_thr'] = float(thr_pair)
            res['pair_thr_mode'] = thr_mode_pair
        else:
            res.update(dict(
                pair_auroc=0.0, pair_auprc=0.0, pair_acc=0.0,
                pair_precision=0.0, pair_recall=0.0, pair_specificity=0.0,
                pair_f1=0.0, pair_mcc=0.0, pair_brier=0.0,
                pair_ece=0.0, pair_thr=0.5,
            ))
        if prediction_out and pdp.size:
            prediction_path = os.path.abspath(str(prediction_out))
            os.makedirs(os.path.dirname(prediction_path), exist_ok=True)
            threshold = float(res.get("pair_thr", fixed_thr if fixed_thr is not None else 0.5))
            with open(prediction_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("pair_id", "label", "pair_prob", "threshold", "prediction"),
                    delimiter="\t",
                )
                writer.writeheader()
                for index, (label, probability) in enumerate(zip(pdl, pdp)):
                    pair_id = (
                        pair_decision_id_all[index]
                        if index < len(pair_decision_id_all)
                        else f"pair_{index}"
                    )
                    writer.writerow({
                        "pair_id": pair_id,
                        "label": int(label > 0.5),
                        "pair_prob": float(probability),
                        "threshold": threshold,
                        "prediction": int(probability >= threshold),
                    })
        if collect_global_pair_metrics and lb.size and len(np.unique(lb)) > 1:
            ybin = (lb > 0.5).astype(np.int32)
            res['auroc_l2'] = float(roc_auc_score(ybin, pp))
            res['auprc_l2'] = float(average_precision_score(ybin, pp))
        else:
            res.update(dict(auroc_l2=0.0, auprc_l2=0.0))
        primary_metric = str(getattr(P, 'pair_primary_metric', 'pair_auprc')).lower()
        if primary_metric == 'medauc':
            res['primary'] = res['medauc']
        else:
            res['primary'] = res.get('pair_auprc', 0.0)
        return res
    finally:
        if saved_l2_geom_prior_w is not None:
            l2_bridge.geom_prior_w = saved_l2_geom_prior_w


def _ager_refine_scores(scores: torch.Tensor, coords: torch.Tensor):
    """Adaptive Graph Evidence Refinement for ranking-only residue scores."""
    if scores.numel() <= 1:
        return scores
    alpha = float(getattr(P, "ager_alpha", 0.30))
    if alpha <= 0:
        return scores
    radius = float(getattr(P, "ager_radius", 10.0))
    top_m = max(1, int(getattr(P, "ager_top_m", 5)))
    coords = torch.nan_to_num(coords.float(), nan=0.0)
    scores = torch.nan_to_num(scores.float(), nan=0.0)
    dist = torch.cdist(coords.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)
    adj = (dist <= radius) & (dist > 1e-6)
    refined = scores.clone()
    for i in range(scores.numel()):
        idx = torch.nonzero(adj[i], as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        vals = scores[idx]
        k = min(top_m, int(vals.numel()))
        neigh = torch.topk(vals, k=k, largest=True).values.mean()
        refined[i] = (1.0 - alpha) * scores[i] + alpha * neigh
    return refined


def eval_topk_residue(model, dl, device=DEVICE, fracs=(0.1, 0.2)):
    model.eval()
    recall_L5, recall_L10, precision_K10, hit_K20, enrichment_K10 = [], [], [], [], []
    with torch.no_grad():
        for batch in dl:
            resA  = torch.nan_to_num(batch["resA"], nan=0.0).to(device).float()
            resB  = torch.nan_to_num(batch["resB"], nan=0.0).to(device).float()
            maskA = (batch["maskA"] > 0.5).to(device)
            maskB = (batch["maskB"] > 0.5).to(device)
            coordsA = torch.nan_to_num(batch["coordsA"]).to(device).float()
            coordsB = torch.nan_to_num(batch["coordsB"]).to(device).float()
            cA = batch.get("chainA", None)
            cB = batch.get("chainB", None)
            if cA is not None: cA = cA.to(device).float()
            if cB is not None: cB = cB.to(device).float()
            yA = batch["y_res_A"].to(device).float()
            yB = batch["y_res_B"].to(device).float()
            if yA.ndim == 1: yA = yA.unsqueeze(0)
            if yB.ndim == 1: yB = yB.unsqueeze(0)

            site_mode = bool(torch.is_tensor(batch.get('site_mode', None)) and float(batch['site_mode'].max().item()) > 0.5)
            out = model(resA, maskA, cA, coordsA, resB, maskB, cB, coordsB, site_mode=site_mode)
            B = resA.size(0)
            for b in range(B):
                prob = out["p_res_A"][b][maskA[b]]
                yt = yA[b][maskA[b]]
                c = coordsA[b][maskA[b]]
                npos = int((yt > 0.5).sum())
                if npos == 0 or prob.numel() == 0:
                    continue
                if bool(getattr(P, "ager_enable", True)):
                    prob_rank = _ager_refine_scores(prob, c)
                else:
                    prob_rank = prob
                L = int(prob_rank.numel())
                order = torch.argsort(prob_rank, descending=True)
                def _hit_at(k):
                    kk = max(1, min(int(k), L))
                    return int((yt[order[:kk]] > 0.5).sum().item()), kk
                h5, k5 = _hit_at(math.ceil(L / 5.0))
                h10, k10 = _hit_at(math.ceil(L / 10.0))
                hpk10, kk10 = _hit_at(10)
                hk20, kk20 = _hit_at(20)
                pos_rate = float(npos) / max(1.0, float(L))
                p10 = hpk10 / max(1, kk10)
                enrich = p10 / max(1e-6, pos_rate)
                enrich = min(float(getattr(P, "topk_enrichment_cap", 10.0)), float(enrich))
                recall_L5.append(h5 / max(1, npos))
                recall_L10.append(h10 / max(1, npos))
                precision_K10.append(p10)
                hit_K20.append(1.0 if hk20 > 0 else 0.0)
                enrichment_K10.append(enrich)

    res = {
        "recall_L5": float(np.mean(recall_L5)) if recall_L5 else 0.0,
        "recall_L10": float(np.mean(recall_L10)) if recall_L10 else 0.0,
        "precision_K10": float(np.mean(precision_K10)) if precision_K10 else 0.0,
        "hit_K20": float(np.mean(hit_K20)) if hit_K20 else 0.0,
        "enrichment_K10": float(np.mean(enrichment_K10)) if enrichment_K10 else 0.0,
        "ager": bool(getattr(P, "ager_enable", True)),
    }
    cap = max(1e-6, float(getattr(P, "topk_enrichment_cap", 10.0)))
    res["topk_score"] = (
        0.45 * res["precision_K10"] +
        0.25 * res["hit_K20"] +
        0.20 * min(1.0, res["enrichment_K10"] / cap) +
        0.10 * res["recall_L5"]
    )
    primary_key = str(getattr(P, "topk_primary", "precision_K10"))
    res["primary"] = float(res.get(primary_key, res.get("precision_K10", 0.0)))
    return res


# ============================================================
# Data loading
# ============================================================

def _read_complex_list(path):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"complex list not found: {path}")
    out, seen = [], set()
    with open(path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'): continue
            tok = re.sub(r"\.(npz|npy|pt|pkl|fa|fasta|tsv|txt)(\.gz)?$",
                         "", s.split()[0], flags=re.I)
            if tok and tok not in seen:
                out.append(tok); seen.add(tok)
    return out


def _resolve_complex_list_path(spec, split_name):
    """Resolve a DIPS split list from either a file path or the DIPS list directory."""
    split_name = str(split_name).lower()
    aliases = {
        "train": ("train", "tr"),
        "val": ("val", "valid", "validation", "dev"),
        "test": ("test", "te"),
    }.get(split_name, (split_name,))

    candidates = []
    spec = str(spec or "").strip()
    if spec and os.path.isfile(spec):
        return spec
    if spec and os.path.isdir(spec):
        candidates.append(spec)
    list_dir = os.path.join(str(DIPS_ROOT), "list")
    if os.path.isdir(list_dir):
        candidates.append(list_dir)

    seen_dirs = set()
    for d in candidates:
        d_abs = os.path.abspath(d)
        if d_abs in seen_dirs:
            continue
        seen_dirs.add(d_abs)
        try:
            files = [
                os.path.join(d, fn)
                for fn in os.listdir(d)
                if fn.lower().endswith((".txt", ".tsv", ".list", ".lst"))
            ]
        except Exception:
            files = []
        exact = []
        loose = []
        for fp in files:
            name = os.path.basename(fp).lower()
            stem = re.sub(r"\.(txt|tsv|list|lst)$", "", name)
            toks = re.split(r"[^a-z0-9]+", stem)
            if split_name in toks:
                exact.append(fp)
            elif any(a in toks or ("_" + a + "_") in ("_" + stem + "_") for a in aliases):
                loose.append(fp)
        ranked = sorted(exact, key=lambda x: (len(os.path.basename(x)), os.path.basename(x)))
        ranked += sorted(loose, key=lambda x: (len(os.path.basename(x)), os.path.basename(x)))
        if ranked:
            return ranked[0]

    if spec:
        return spec
    return os.path.join(list_dir, f"{split_name}.txt")


def _resolve_esm_cache_dir(for_dips=False):
    cfg_cache = str(getattr(P, "esm_cache_dir", "") or "").strip()
    if cfg_cache:
        return cfg_cache
    if for_dips:
        root_cache = os.path.join(DIPS_ROOT, "esm_cache")
        if os.path.isdir(root_cache):
            return root_cache
    return os.path.join(SAVE_DIR, "esm_cache")


def build_loaders():
    use_rbp_dataset = DATASET_MODE in ("auto", "rbp", "site", "rbp_site")
    use_dips_dataset = DATASET_MODE in ("dips", "pair", "ppi", "dips_plus", "dips-plus")
    if not (use_rbp_dataset or use_dips_dataset):
        raise ValueError(f"Unknown dataset_mode={DATASET_MODE}; expected auto/rbp/dips")

    emb = None
    seq_mode = str(getattr(P, "sequence_mode", "light")).lower()
    live_esm_on_miss = bool(getattr(P, "dips_use_embedder_on_miss", False))
    need_live_esm = seq_mode in ("esm", "hybrid") and (not use_dips_dataset or live_esm_on_miss)
    if need_live_esm:
        try:
            print(f"[ESM] live SiteEmbedder enabled | mode={seq_mode} | device={DEVICE}", flush=True)
            emb = SiteEmbedder(device=DEVICE, esm_local_dir=ESM_LOCAL_DIR)
        except Exception as e:
            if not getattr(P, "allow_zero_esm_fallback", False):
                raise RuntimeError("ESM requested but SiteEmbedder init failed") from e
            emb = None
            print(f"[warn] ESM init failed ({e}); using zero ESM fallback", flush=True)
    elif seq_mode in ("esm", "hybrid"):
        print("[ESM] DIPS cache-only mode: no live ESM inference during training", flush=True)

    pair_train_manifest = os.environ.get("PAIR_TRAIN_MANIFEST", "").strip()
    pair_val_manifest = os.environ.get("PAIR_VAL_MANIFEST", "").strip()
    pair_test_manifest = os.environ.get("PAIR_TEST_MANIFEST", "").strip()
    if pair_train_manifest or pair_val_manifest or pair_test_manifest:
        if not pair_train_manifest or not pair_val_manifest:
            raise ValueError(
                "PAIR_TRAIN_MANIFEST and PAIR_VAL_MANIFEST are both required for explicit pair training"
            )
        if str(PRIMARY_OBJ).lower() != "pair":
            raise ValueError("Explicit pair manifests require PRIMARY_OBJECTIVE=pair")
        P.train_neg_ratio = 0.0
        P.pair_eval_make_negatives = False
        esm_cache = _resolve_esm_cache_dir(for_dips=False)
        pair_kwargs = dict(
            rbp_root=DIPS_ROOT,
            embedder=emb,
            use_pssm=P.use_pssm,
            use_dssp=P.use_dssp,
            esm_cache_dir=esm_cache,
            verbose=False,
        )
        ds_tr = ExplicitPairDataset(manifest_path=pair_train_manifest, **pair_kwargs)
        ds_va = ExplicitPairDataset(manifest_path=pair_val_manifest, **pair_kwargs)
        ds_te = (
            ExplicitPairDataset(manifest_path=pair_test_manifest, **pair_kwargs)
            if pair_test_manifest else None
        )
        print(
            f"[data][EXPLICIT-PAIR] Train={len(ds_tr)} Val={len(ds_va)} "
            f"Test={len(ds_te) if ds_te is not None else 0} | root={DIPS_ROOT}",
            flush=True,
        )
        print(
            f"[data][EXPLICIT-PAIR] train={pair_train_manifest} "
            f"val={pair_val_manifest} test={pair_test_manifest or 'none'}",
            flush=True,
        )
    # RBP labels dir present -> use RBP site dataset unless dataset_mode forces DIPS pair mode.
    elif use_rbp_dataset and os.path.isdir(os.path.join(DIPS_ROOT, 'labels')) and (
        os.path.exists(P.rbp_id_list) or
        (getattr(P, 'rbp_train_list', '') and os.path.exists(str(P.rbp_train_list)))
    ):
        is_dest_prepared = os.path.basename(os.path.normpath(str(DIPS_ROOT))) == "Dest_prepared"
        if not is_dest_prepared:
            _ensure_rbp_split_files()
        tr_ids = _read_id_list_optional(getattr(P, 'rbp_train_list', ''))
        va_ids = _read_id_list_optional(getattr(P, 'rbp_val_list', ''))
        te_ids = _read_id_list_optional(getattr(P, 'rbp_test_list', ''))
        if tr_ids and va_ids:
            pass
        elif is_dest_prepared:
            raise RuntimeError(
                f"[Dest] expected prepared split files but did not find usable train/val lists: "
                f"train={getattr(P, 'rbp_train_list', '')} val={getattr(P, 'rbp_val_list', '')}"
            )
        else:
            all_ids = _read_rbp_id_list(P.rbp_id_list)
            rng = random.Random(getattr(P, 'split_seed', 42))
            rng.shuffle(all_ids)
            n_total = len(all_ids)
            n_tr = max(1, int(round(n_total * float(getattr(P, 'split_train', 0.80)))))
            n_va = max(1, int(round(n_total * float(getattr(P, 'split_val', 0.10)))))
            if n_tr + n_va >= n_total:
                n_va = max(1, min(n_va, n_total - n_tr - 1)) if n_total >= 3 else max(1, n_total - n_tr)
            tr_ids = all_ids[:n_tr]
            va_ids = all_ids[n_tr:n_tr+n_va]
            te_ids = all_ids[n_tr+n_va:]
        if P.small_tr_n > 0:
            tr_ids = tr_ids[:P.small_tr_n]
        if P.small_va_n > 0:
            va_ids = va_ids[:P.small_va_n]
        esm_cache = _resolve_esm_cache_dir(for_dips=False)
        ds_tr = RBP296Dataset(DIPS_ROOT, tr_ids, embedder=emb, use_pssm=P.use_pssm, use_dssp=P.use_dssp, esm_cache_dir=esm_cache, verbose=False)
        ds_va = RBP296Dataset(DIPS_ROOT, va_ids, embedder=emb, use_pssm=P.use_pssm, use_dssp=P.use_dssp, esm_cache_dir=esm_cache, verbose=False)
        ds_te = RBP296Dataset(DIPS_ROOT, te_ids, embedder=emb, use_pssm=P.use_pssm, use_dssp=P.use_dssp, esm_cache_dir=esm_cache, verbose=False) if te_ids else None
        if float(getattr(P, 'l1_pos_weight', 1.0)) <= 0:
            auto_pw, train_pos_rate, n_zero, n_full = _estimate_l1_balance_from_ids(
                DIPS_ROOT, getattr(ds_tr, 'ids', tr_ids)
            )
            P.l1_pos_weight = float(auto_pw)
            print(
                f"[loss][auto] l1_pos_weight={P.l1_pos_weight:.3f} "
                f"train_pos={train_pos_rate:.3f} zero_label={n_zero} full_label={n_full}",
                flush=True
            )
        fc = _feature_coverage(DIPS_ROOT, getattr(ds_tr, 'ids', tr_ids) + getattr(ds_va, 'ids', va_ids))
        print(
            f"[features] PSSM={fc['pssm']}/{fc['n']}  DSSP={fc['dssp']}/{fc['n']}  "
            f"geom={'on' if bool(getattr(P, 'use_geom', True)) else 'off'}",
            flush=True
        )
        print(
            f"[data][RBP] Train={len(ds_tr)} Val={len(ds_va)} Test={len(te_ids)} | "
            f"seq_mode={P.sequence_mode} | root={DIPS_ROOT} | struct_dir={getattr(P, 'rbp_structure_dir', 'structures_af')}",
            flush=True
        )
    else:
        tr_list = _resolve_complex_list_path(DIPS_TRAIN_LIST, "train")
        va_list = _resolve_complex_list_path(DIPS_VAL_LIST, "val")
        same_lists = ((not va_list) or (os.path.abspath(str(tr_list)) == os.path.abspath(str(va_list))) or (not os.path.exists(str(va_list))))
        if same_lists:
            all_ids = _read_complex_list(tr_list)
            rng = random.Random(getattr(P, 'split_seed', SEED))
            rng.shuffle(all_ids)
            n_total = len(all_ids)
            n_tr = max(1, int(round(n_total * float(getattr(P, 'split_train', 0.70)))))
            n_va = max(1, int(round(n_total * float(getattr(P, 'split_val', 0.15)))))
            if n_tr + n_va > n_total:
                n_va = max(1, n_total - n_tr)
            tr_ids = all_ids[:n_tr]
            va_ids = all_ids[n_tr:n_tr+n_va]
        else:
            tr_ids = _read_complex_list(tr_list)
            va_ids = _read_complex_list(va_list)
        if P.small_tr_n > 0:
            tr_ids = tr_ids[:P.small_tr_n]
        if P.small_va_n > 0:
            va_ids = va_ids[:P.small_va_n]
        esm_cache = _resolve_esm_cache_dir(for_dips=True)
        ds_kw = dict(contact_cutoff=CONTACT_CUTOFF, embedder=emb, use_pssm=P.use_pssm, use_dssp=P.use_dssp, esm_cache_dir=esm_cache, verbose=bool(int(getattr(P, 'dips_index_verbose', 0))), skip_filter=getattr(P, 'dips_skip_filter', True))
        ds_tr = DIPSIndexedPairs(DIPS_ROOT, tr_ids, **ds_kw)
        ds_va = DIPSIndexedPairs(DIPS_ROOT, va_ids, **ds_kw)
        ds_te = None
        print(f"[data][PAIR] Train={len(ds_tr)} Val={len(ds_va)} | seq_mode={P.sequence_mode} | structure={P.structure_source}", flush=True)
        print(f"[data][PAIR] train_list={tr_list} val_list={va_list}", flush=True)
        print(f"[features][PAIR] ESM_cache={esm_cache} live_esm_on_miss={live_esm_on_miss}", flush=True)

    nw = NUM_WORKERS_SITE
    if DEVICE.startswith('cuda') and nw > 0 and not bool(getattr(P, "allow_cuda_workers", False)):
        print('[data] DEVICE=cuda -> num_workers=0', flush=True)
        nw = 0
    eval_batch = int(getattr(P, "eval_batch_site", BATCH_SITE) or BATCH_SITE)
    eval_batch = max(1, min(eval_batch, BATCH_SITE))
    dl_kw = dict(batch_size=BATCH_SITE, num_workers=nw, collate_fn=dips_collate, pin_memory=DEVICE.startswith('cuda'), persistent_workers=(nw > 0), drop_last=False)
    if nw > 0:
        dl_kw['prefetch_factor'] = max(2, int(getattr(P, "prefetch_factor", 2)))
    print(
        f"[data] batch={BATCH_SITE} workers={nw} prefetch={dl_kw.get('prefetch_factor', 0)} "
        f"eval_batch={eval_batch} AMP={'on' if USE_AMP else 'off'} dtype={str(AMP_DTYPE).replace('torch.', '') if USE_AMP else 'fp32'}",
        flush=True
    )
    dl_tr = DataLoader(ds_tr, shuffle=True, **dl_kw)
    dl_va = DataLoader(ds_va, shuffle=False, **{**dl_kw, 'batch_size': eval_batch, 'drop_last': False})
    dl_te = DataLoader(ds_te, shuffle=False, **{**dl_kw, 'batch_size': eval_batch, 'drop_last': False}) if ds_te is not None else None
    sample = next(iter(dl_tr))
    d_res = int(sample['resA'].shape[-1])
    chainA = sample.get('chainA', None)
    d_chain = int(chainA.shape[-1]) if (chainA is not None and chainA.ndim in (1, 2, 3)) else 0
    return dl_tr, dl_va, dl_te, d_res, d_chain


# ============================================================
# Early Stopping
# ============================================================

class EarlyStopping:
    def __init__(self, patience=10, min_epochs=12, min_delta=1e-4, ema=0.6):
        self.patience = patience; self.min_epochs = min_epochs
        self.min_delta = min_delta; self.ema_decay = ema
        self.best = -1e9; self.ema_val = -1e9; self.wait = 0; self.best_epoch = 0

    def step(self, ep, val):
        self.ema_val = (val if self.ema_val < -1e8
                        else self.ema_decay * self.ema_val + (1 - self.ema_decay) * val)
        if self.ema_val > self.best + self.min_delta:
            self.best = self.ema_val; self.wait = 0; self.best_epoch = ep
        else:
            self.wait += 1
        return ep >= self.min_epochs and self.wait >= self.patience


def _load_global_best_score():
    if os.path.exists(GLOBAL_BEST_META):
        try:
            with open(GLOBAL_BEST_META, 'r', encoding='utf-8') as f:
                m = json.load(f)
            return float(m.get('score', -1e9)), m
        except Exception:
            pass
    if os.path.exists(GLOBAL_BEST_CKPT):
        try:
            ck = torch.load(GLOBAL_BEST_CKPT, map_location='cpu')
            score = ck.get('score', None)
            if score is None:
                score = (ck.get('val_metrics', {}) or {}).get('primary', -1e9)
            return float(score), {
                'score': float(score),
                'ckpt': GLOBAL_BEST_CKPT,
                'source': 'checkpoint',
            }
        except Exception:
            pass
    return -1e9, None


def _remove_if_exists(path: str) -> bool:
    try:
        if path and os.path.exists(path):
            os.remove(path)
            return True
    except Exception:
        pass
    return False


def _promote_run_best_if_needed(run_ckpt: str, run_score: float, run_epoch: int, run_metrics: Dict):
    old_score, old_meta = _load_global_best_score()
    if not os.path.exists(run_ckpt):
        print(f"[global-best][warn] run best checkpoint missing: {run_ckpt}", flush=True)
        return False
    improved = (old_meta is None) or (float(run_score) > float(old_score))
    if improved:
        shutil.copy2(run_ckpt, GLOBAL_BEST_CKPT)
        meta = {
            'metric': BEST_TAG,
            'score': float(run_score),
            'epoch': int(run_epoch),
            'run_stamp': RUN_STAMP,
            'updated_at': time.strftime("%Y-%m-%d %H:%M:%S"),
            'run_ckpt': run_ckpt,
            'global_ckpt': GLOBAL_BEST_CKPT,
            'val_metrics': run_metrics,
            'previous_score': None if old_meta is None else float(old_score),
        }
        with open(GLOBAL_BEST_META, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        if isinstance(run_metrics, dict) and run_metrics:
            _write_best_metrics_tsv(GLOBAL_BEST_METRICS_TSV, run_epoch, run_metrics, GLOBAL_BEST_CKPT)
        if old_meta is None:
            print(f"[global-best] initialized score={run_score:.4f} -> {GLOBAL_BEST_CKPT}", flush=True)
        else:
            print(f"[global-best] improved {old_score:.4f} -> {run_score:.4f}; updated {GLOBAL_BEST_CKPT}", flush=True)
        return True
    removed = []
    for pth, label in (
        (run_ckpt, "run-best"),
        (RUN_BEST_METRICS_TSV, "run-metrics"),
        (RUN_TEST_METRICS_TSV, "run-test-metrics"),
    ):
        if _remove_if_exists(pth):
            removed.append(label)
    removed_msg = (" removed " + ",".join(removed)) if removed else ""
    print(f"[global-best] not improved: run={run_score:.4f} <= global={old_score:.4f}; keep {GLOBAL_BEST_CKPT}.{removed_msg}", flush=True)
    return False


def _metric_value(metrics: Dict, key: str, default=0.0):
    try:
        return float(metrics.get(key, default))
    except Exception:
        return default


def _format_binary_metrics(metrics: Dict) -> str:
    return (
        f"ACC={_metric_value(metrics, 'acc'):.4f}  "
        f"Precision={_metric_value(metrics, 'precision'):.4f}  "
        f"Recall={_metric_value(metrics, 'recall'):.4f}  "
        f"F1={_metric_value(metrics, 'f1'):.4f}  "
        f"AUPRC={_metric_value(metrics, 'auprc'):.4f}  "
        f"MacroAP={_metric_value(metrics, 'auprc_macro'):.4f}  "
        f"CA12AP={_metric_value(metrics, 'ca12_auprc'):.4f}  "
        f"SurfAP={_metric_value(metrics, 'surface_auprc'):.4f}  "
        f"MCC={_metric_value(metrics, 'mcc'):.4f}"
    )


def _append_metrics_history(path: str, ep: int, loss: float, metrics: Dict):
    if PRIMARY_OBJ == "topk":
        header = ["run_stamp", "epoch", "split", "loss"] + TOPK_FIELDS + ["primary", "ager"]
    elif PRIMARY_OBJ == "binary":
        header = ["run_stamp", "epoch", "split", "loss"] + METRIC_FIELDS + ["thr", "pos_rate", "primary"]
    else:
        header = ["run_stamp", "epoch", "split", "loss"] + PAIR_FIELDS + ["primary"]
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if not exists:
            f.write("\t".join(header) + "\n")
        vals = [RUN_STAMP, str(int(ep)), "val", f"{float(loss):.6f}"]
        if PRIMARY_OBJ == "topk":
            vals += [f"{_metric_value(metrics, k):.6f}" for k in TOPK_FIELDS]
            vals += [f"{_metric_value(metrics, 'primary'):.6f}", str(int(bool(metrics.get('ager', False))))]
        elif PRIMARY_OBJ == "binary":
            vals += [f"{_metric_value(metrics, k):.6f}" for k in METRIC_FIELDS]
            vals += [
                f"{_metric_value(metrics, 'thr', 0.5):.6f}",
                f"{_metric_value(metrics, 'pos_rate'):.6f}",
                f"{_metric_value(metrics, 'primary'):.6f}",
            ]
        else:
            vals += [f"{_metric_value(metrics, k):.6f}" for k in PAIR_FIELDS]
            vals += [f"{_metric_value(metrics, 'primary'):.6f}"]
        f.write("\t".join(vals) + "\n")


def _write_best_metrics_tsv(path: str, epoch: int, metrics: Dict, ckpt_path: str):
    if PRIMARY_OBJ == "topk":
        header = ["run_stamp", "epoch", "checkpoint"] + TOPK_FIELDS + ["primary", "ager"]
    elif PRIMARY_OBJ == "binary":
        header = ["run_stamp", "epoch", "checkpoint"] + METRIC_FIELDS + ["thr", "pos_rate", "primary"]
    else:
        header = ["run_stamp", "epoch", "checkpoint"] + PAIR_FIELDS + ["primary"]
    vals = [RUN_STAMP, str(int(epoch)), ckpt_path]
    if PRIMARY_OBJ == "topk":
        vals += [f"{_metric_value(metrics, k):.6f}" for k in TOPK_FIELDS]
        vals += [f"{_metric_value(metrics, 'primary'):.6f}", str(int(bool(metrics.get('ager', False))))]
    elif PRIMARY_OBJ == "binary":
        vals += [f"{_metric_value(metrics, k):.6f}" for k in METRIC_FIELDS]
        vals += [
            f"{_metric_value(metrics, 'thr', 0.5):.6f}",
            f"{_metric_value(metrics, 'pos_rate'):.6f}",
            f"{_metric_value(metrics, 'primary'):.6f}",
        ]
    else:
        vals += [f"{_metric_value(metrics, k):.6f}" for k in PAIR_FIELDS]
        vals += [f"{_metric_value(metrics, 'primary'):.6f}"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        f.write("\t".join(vals) + "\n")


# ============================================================
# Main
# ============================================================

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    dl_tr, dl_va, dl_te, d_res, d_chain = build_loaders()

    cfg   = build_model_config(P, d_res, d_chain)
    model = L13PDBGVPModel(cfg).to(DEVICE)
    pair_head_type = str(getattr(cfg, "pair_head_type", "eb")).lower()
    model_provenance = {
        "site_mode": False if PRIMARY_OBJ == "pair" else None,
        "pair_head_type": pair_head_type,
        "l2_active": pair_head_type == "eb_l2",
        "gc_eb_active": pair_head_type == "gpeh",
        "native_eb_export": pair_head_type in ("eb", "eb_l2"),
        "proposal_mode": str(getattr(cfg, "eb_proposal_mode", "top")).lower(),
        "support_mode": str(getattr(cfg, "eb_support_mode", "top")).lower(),
        "dataset_mode": str(DATASET_MODE),
        "primary_objective": str(PRIMARY_OBJ),
    }
    if (
        PRIMARY_OBJ == "pair"
        and os.environ.get("MANUSCRIPT_PRIMARY", "0").strip().lower()
        not in ("0", "false", "no", "off", "")
    ):
        if pair_head_type != "eb":
            raise RuntimeError(
                "MANUSCRIPT_PRIMARY=1 requires pair_head_type='eb'; "
                f"received {pair_head_type!r}"
            )
        if model_provenance["l2_active"] or model_provenance["gc_eb_active"]:
            raise RuntimeError("Primary manuscript EB must disable L2 and GC-EB/GPEH")

    # Optional second-stage fine-tuning from an existing checkpoint.
    # Load before optimizer/EMA construction so both start from resumed weights.
    resume_ckpt = os.environ.get("RESUME_CKPT", "").strip()
    if resume_ckpt:
        if not os.path.exists(resume_ckpt):
            raise FileNotFoundError(f"RESUME_CKPT not found: {resume_ckpt}")
        ck = torch.load(resume_ckpt, map_location=DEVICE)
        state = ck.get("model_state", ck) if isinstance(ck, dict) else ck
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"[resume] loaded {resume_ckpt} | missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )

    ema   = EMA(model, decay=P.ema_decay)

    n_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] L13PDBGVPModel  params={n_p/1e6:.2f}M  device={DEVICE}", flush=True)
    print(f"[model] d_res={d_res}  d_chain={d_chain}", flush=True)
    print(
        f"[model-cfg] d_model={cfg.d_model} enc={cfg.n_encoder_layers} "
        f"cross={cfg.n_cross_layers} heads={cfg.n_heads} dropout={cfg.dropout:.2f} "
        f"site_self_cross={int(bool(getattr(cfg, 'site_self_cross', False)))} "
        f"site_graph={getattr(cfg, 'site_graph_layers', 0)}xk{getattr(cfg, 'site_graph_k', 0)}",
        flush=True,
    )
    print(
        f"[site-head] type={getattr(cfg, 'site_head_type', 'linear')} "
        f"channels={getattr(cfg, 'site_ms_channels', 0)} kernels=3,7,15 "
        f"delta_init={getattr(cfg, 'site_ms_delta_init', 0.0):.2f}",
        flush=True,
    )
    print(
        f"[bridge] explicit_EB={int(pair_head_type in ('eb', 'eb_l2'))} "
        f"proposal={getattr(cfg, 'eb_proposal_mode', 'top')} "
        f"supports={getattr(cfg, 'eb_support_mode', 'top')} "
        f"retained={getattr(cfg, 'eb_pair_topk', 0)}",
        flush=True,
    )
    print(
        f"[pair-head] type={pair_head_type} "
        f"layers={getattr(cfg, 'pair_transformer_layers', 0)} "
        f"heads={getattr(cfg, 'pair_transformer_heads', 0)} "
        f"consistency_w={getattr(cfg, 'consistency_w', 0.0):.3f}",
        flush=True,
    )
    if PRIMARY_OBJ == "topk":
        print(f"[objective] L1-L3 top-k primary={getattr(P, 'topk_primary', 'precision_K10')}; pairwise residue ranking loss is enabled when rank_w > 0", flush=True)
    elif PRIMARY_OBJ == "pair":
        print(f"[objective] L3 protein-pair primary task; EB/L2 contact evidence supports the pair-level decision", flush=True)
        if str(getattr(P, "dataset_mode", "")).lower() == "dips" and bool(getattr(P, "dips_hier_enable", True)):
            print(
                f"[hier] DIPS L1-L3 curriculum: l1_scale={getattr(P, 'dips_l1_w_scale', 1.0):.2f}  "
                f"l3_max={getattr(P, 'dips_l3_w_max', 0.05):.2f}  "
                f"cl_max={getattr(P, 'dips_cl_w_max', 0.0):.2f}  "
                f"rank_start={int(getattr(P, 'dips_rank_start_epoch', 3))}  "
                f"train_EB_geom_w={getattr(P, 'l2_geom_prior_w', 0.0):.2f}  "
                f"eval_EB_geom_w={float(os.environ.get('EVAL_L2_GEOM_PRIOR_W', getattr(P, 'eval_l2_geom_prior_w', getattr(P, 'l2_geom_prior_w', 0.0)))):.2f}",
                flush=True
            )
            print(
                f"[EB/MedAUC] l2_w_start={getattr(P, 'l2_map_w_start', 0.15):.2f} "
                f"ramp={int(getattr(P, 'l2_map_w_ramp_epochs', 8))}  "
                f"rank_alpha={getattr(P, 'l2_rank_alpha', 0.0):.2f} "
                f"margin={getattr(P, 'l2_rank_margin', 0.2):.2f}  "
                f"hardneg={getattr(P, 'l2_hardneg_frac', 0.0):.2f} "
                f"neg_cap={int(getattr(P, 'l2_neg_cap', 0))}  "
                f"focus={int(getattr(P, 'l2_focus_topk', 0))}/{getattr(P, 'l2_focus_frac', 0.0):.2f}",
                flush=True
            )
    else:
        print(f"[objective] L1-L3 binary primary; ranking evidence is kept for later export, not used as a training/selection metric", flush=True)
    if PRIMARY_OBJ == "pair":
        print(
            f"[loss-profile] L3-main task={getattr(P, 'task_loss_profile', 'auto')}  "
            f"pair_metric={getattr(P, 'pair_primary_metric', 'pair_auprc')}  "
            f"pair_cls={getattr(P, 'l3_w', 1.0):.2f}  "
            f"pair_rank={getattr(P, 'l3_pair_rank_w', 0.0):.2f}@{getattr(P, 'l3_pair_rank_start_epoch', 0)}  "
            f"tau={getattr(P, 'l3_tau_margin_w', 0.0):.2f}  "
            f"disturb={getattr(P, 'l3_disturb_w', 0.0):.2f}  "
            f"l1_aux={getattr(P, 'l3_aux_l1_w', 0.0):.2f}  "
            f"cons={getattr(P, 'l3_cons_w', 0.0):.2f}",
            flush=True,
        )
    else:
        print(
            f"[loss-profile] L1-main task={getattr(P, 'task_loss_profile', 'auto')}  "
            f"l1={getattr(P, 'l1_w', 1.0):.2f}  "
            f"rank={getattr(P, 'l1_rank_w', 0.0):.2f}@{getattr(P, 'l1_rank_start_epoch', 0)}  "
            f"site_l3_aux={getattr(P, 'site_l3_pool_w', 0.0):.2f}  "
            f"site_context_cross={int(bool(getattr(P, 'site_self_cross', False)))}  "
            f"site_graph={getattr(P, 'site_graph_layers', 0)}",
            flush=True,
        )
    print(
        f"[loss] l1_pos_weight={P.l1_pos_weight:.2f}  "
        f"focal_w={getattr(P, 'l1_focal_w', 0.0):.2f}  "
        f"focal_alpha={P.l1_focal_alpha:.2f}  "
        f"dice_w={getattr(P, 'l1_dice_w', 0.0):.2f}  "
        f"per_protein={int(bool(getattr(P, 'l1_per_protein_loss', False)))}  "
        f"l15_w={P.l15_w:.3f}",
        flush=True,
    )
    max_rank_w = float(P.l1_rank_w)
    if PRIMARY_OBJ == "topk" and bool(getattr(P, "objective_weight_auto", True)):
        max_rank_w *= float(getattr(P, "topk_rank_boost", 1.0))
    print(f"[loss] rank_w={P.l1_rank_w:.3f} effective_max={max_rank_w:.3f} start={P.l1_rank_start_epoch} ramp={getattr(P, 'topk_rank_ramp_epochs', 0)}  site_l3_pool_w={getattr(P, 'site_l3_pool_w', 0.0):.3f} start={getattr(P, 'site_l3_pool_start_epoch', 999)}", flush=True)
    print(
        f"[loss] ap_rank_w={float(getattr(P, 'ap_rank_w', 0.0)):.3f} "
        f"start={int(getattr(P, 'ap_rank_start_epoch', 999))} "
        f"margin={float(getattr(P, 'ap_rank_margin', 0.05)):.3f} "
        f"tau={float(getattr(P, 'ap_rank_tau', 0.25)):.3f} "
        f"frag_logit_w={float(getattr(P, 'site_frag_logit_w', 0.0)):.2f}",
        flush=True,
    )
    print(
        f"[eval] binary_ager={int(bool(getattr(P, 'binary_ager_eval', False)) and bool(getattr(P, 'ager_enable', True)))}  "
        f"radius={getattr(P, 'ager_radius', 0.0):.1f}  alpha={getattr(P, 'ager_alpha', 0.0):.2f}  top_m={getattr(P, 'ager_top_m', 0)}",
        flush=True,
    )
    print(f"[y_pair] source: has_contact (PDB CA-dist < {CONTACT_CUTOFF} A) "
          f"+ in-batch shuffle negatives", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=P.lr,
                            weight_decay=P.weight_decay, betas=(0.9, 0.999))
    spe   = int(math.ceil(len(dl_tr) / float(max(1, P.accum_steps))))
    total = spe * P.epochs
    sched, si = build_warmup_cosine_scheduler(opt, total, P.lr, P)
    print(f"[sched] {si}", flush=True)

    scaler  = GradScaler() if USE_AMP else None
    amp_ctx = lambda: autocast(dtype=AMP_DTYPE) if USE_AMP else nullcontext()

    es         = EarlyStopping(P.early_patience, P.early_min_epochs,
                               P.early_min_delta, P.early_ema)
    best_score = -1e9; best_epoch = 0; best_metrics = {}
    opt.zero_grad()

    print(f"[train] epochs={P.epochs}  steps/ep={spe}  primary={PRIMARY_OBJ}  dataset={DATASET_MODE}  force_dest={int(os.environ.get('FORCE_DEST', '0').strip().lower() not in ('0', 'false', 'no', 'off', ''))}", flush=True)
    old_global_score, _ = _load_global_best_score()
    if old_global_score > -1e8:
        print(f"[global-best] current best_{BEST_TAG}={old_global_score:.4f} ckpt={GLOBAL_BEST_CKPT}", flush=True)
    else:
        print(f"[global-best] no existing best_{BEST_TAG}; this run can initialize it", flush=True)
    print(f"[run-best] temporary checkpoint={RUN_BEST_CKPT}", flush=True)

    for ep in range(P.epochs):
        ep_label = ep + 1
        model.train(); model.training_epoch = ep
        t0 = time.time(); ep_loss = 0.0; n_bat = 0

        for step, batch in enumerate(dl_tr):
            with amp_ctx():
                out, loss, aux = forward_one(model, batch, ep)
            if not torch.isfinite(loss):
                opt.zero_grad(); continue

            if USE_AMP and scaler:
                scaler.scale(loss / P.accum_steps).backward()
            else:
                (loss / P.accum_steps).backward()
            ep_loss += float(loss.detach()); n_bat += 1

            accum_n = (step % max(1, P.accum_steps)) + 1
            is_update_step = (accum_n == max(1, P.accum_steps)) or ((step + 1) == len(dl_tr))
            if is_update_step:
                if USE_AMP and scaler:
                    scaler.unscale_(opt)
                    if accum_n < max(1, P.accum_steps):
                        grad_scale = float(max(1, P.accum_steps)) / float(accum_n)
                        for p in model.parameters():
                            if p.grad is not None:
                                p.grad.mul_(grad_scale)
                    nn.utils.clip_grad_norm_(model.parameters(), P.max_grad_norm)
                    scaler.step(opt); scaler.update()
                else:
                    if accum_n < max(1, P.accum_steps):
                        grad_scale = float(max(1, P.accum_steps)) / float(accum_n)
                        for p in model.parameters():
                            if p.grad is not None:
                                p.grad.mul_(grad_scale)
                    nn.utils.clip_grad_norm_(model.parameters(), P.max_grad_norm)
                    opt.step()
                opt.zero_grad(); ema.update(model)
                if sched: sched.step()

            if (step + 1) % P.print_every == 0:
                lr_now = opt.param_groups[0]["lr"]
                step_prefix = f"  ep{ep_label:02d} s{step+1:4d}"
                if PRIMARY_OBJ == "pair":
                    pair_terms = [
                        f"loss={float(loss.detach()):.4f}",
                        f"l2={aux.get('loss_l2_map', 0):.4f}",
                        f"res={aux.get('loss_l1', 0):.4f}",
                    ]
                    l3_w_eff = float(aux.get("l3_w_eff", getattr(P, "l3_w", 1.0)) or 0.0)
                    l1_w_eff = float(aux.get("l1_w_eff", getattr(P, "l1_w", 1.0)) or 0.0)
                    l2_w_eff = float(aux.get("l2_w_eff", getattr(P, "l2_map_w", 1.0)) or 0.0)
                    pair_terms.append(f"w_res={l1_w_eff:.2f}")
                    pair_terms.append(f"w_l2={l2_w_eff:.2f}")
                    if l3_w_eff > 0:
                        pair_terms.append(f"pair={aux.get('loss_l3', 0):.4f}")
                        pair_terms.append(f"w_pair={l3_w_eff:.2f}")
                    pr_loss = float(aux.get("loss_pair_rank", 0.0) or 0.0)
                    pr_w = float(aux.get("pair_rank_w_eff", 0.0) or 0.0)
                    if pr_loss > 0.0 or pr_w > 0.0:
                        pair_terms.append(f"prank={pr_loss:.4f}")
                        pair_terms.append(f"prw={pr_w:.3f}")
                    tm_w = float(aux.get("tau_margin_w_eff", 0.0) or 0.0)
                    if tm_w > 0.0:
                        pair_terms.append(f"tau={aux.get('loss_tau_margin', 0):.4f}")
                        pair_terms.append(f"tw={tm_w:.3f}")
                    dw = float(aux.get("disturb_w_eff", 0.0) or 0.0)
                    if dw > 0.0:
                        pair_terms.append(f"dist={aux.get('loss_disturb', 0):.4f}")
                        pair_terms.append(f"dw={dw:.3f}")
                    rank_loss = float(aux.get("loss_rank", 0.0) or 0.0)
                    rank_w_eff = float(aux.get("rank_w_eff", 0.0) or 0.0)
                    cl_loss = float(aux.get("loss_cl", 0.0) or 0.0)
                    if rank_loss > 0.0 or rank_w_eff > 0.0:
                        pair_terms.append(f"rank={rank_loss:.4f}")
                        pair_terms.append(f"rw={rank_w_eff:.3f}")
                    if cl_loss > 0.0:
                        pair_terms.append(f"cl={cl_loss:.4f}")
                        pair_terms.append(f"cw={float(aux.get('cl_w_eff', 0.0) or 0.0):.2f}")
                    pair_terms.append(f"lr={lr_now:.2e}")
                    print(f"{step_prefix}  " + "  ".join(pair_terms), flush=True)
                elif PRIMARY_OBJ == "topk":
                    topk_terms = [
                        f"loss={float(loss.detach()):.4f}",
                        f"l3={aux.get('loss_l3', 0):.4f}",
                        f"l1={aux.get('loss_l1', 0):.4f}",
                    ]
                    rank_loss = float(aux.get("loss_rank", 0.0) or 0.0)
                    rank_w_eff = float(aux.get("rank_w_eff", 0.0) or 0.0)
                    if rank_loss > 0.0 or rank_w_eff > 0.0:
                        topk_terms.append(f"rank={rank_loss:.4f}")
                        topk_terms.append(f"rw={rank_w_eff:.3f}")
                    topk_terms.extend([
                        f"frag={aux.get('loss_frag', 0):.4f}",
                        f"cl={aux.get('loss_cl', 0):.4f}",
                        f"lr={lr_now:.2e}",
                    ])
                    print(f"{step_prefix}  " + "  ".join(topk_terms), flush=True)
                else:
                    print(f"{step_prefix}  "
                          f"loss={float(loss.detach()):.4f}  "
                          f"l3={aux.get('loss_l3',0):.4f}  "
                          f"l1={aux.get('loss_l1',0):.4f}  "
                          f"dice={aux.get('loss_dice',0):.4f}  "
                          f"qw={aux.get('l1_qw',1):.2f}  "
                          f"rank={aux.get('loss_rank',0):.4f}  "
                          f"rw={aux.get('rank_w_eff',0):.3f}  "
                          f"ap={aux.get('loss_ap_rank',0):.4f}  "
                          f"apw={aux.get('ap_rank_w_eff',0):.3f}  "
                          f"pool={aux.get('loss_site_l3',0):.4f}  "
                          f"frag={aux.get('loss_frag',0):.4f}  "
                          f"cl={aux.get('loss_cl',0):.4f}  "
                          f"lr={lr_now:.2e}", flush=True)

        # ---- Validation (EMA weights) ----
        ema.apply_shadow(model); model.eval()
        vm = eval_topk_residue(model, dl_va) if PRIMARY_OBJ == "topk" \
             else eval_binary(model, dl_va)
        ema.restore(model)

        ps = vm["primary"]
        val_loss = ep_loss / max(1, n_bat)
        _append_metrics_history(METRICS_HISTORY, ep_label, val_loss, vm)
        if PRIMARY_OBJ == "binary" and "auprc" in vm:
            print(f"[ep {ep_label:02d}] loss={val_loss:.4f}  {_format_binary_metrics(vm)}  thr={vm.get('thr',0.5):.3f}  pos={vm.get('pos_rate',0):.3f}  time={time.time()-t0:.1f}s  best_ep={best_epoch}", flush=True)
        elif PRIMARY_OBJ == "topk":
            print(f"[ep {ep_label:02d}] loss={val_loss:.4f}  Recall@L/5={vm.get('recall_L5',0):.4f}  Recall@L/10={vm.get('recall_L10',0):.4f}  Precision@K10={vm.get('precision_K10',0):.4f}  Hit@K20={vm.get('hit_K20',0):.4f}  Enrich@K10={vm.get('enrichment_K10',0):.2f}  TopKScore={vm.get('topk_score',0):.4f}  AGER={int(bool(vm.get('ager', False)))}  time={time.time()-t0:.1f}s  best_ep={best_epoch}", flush=True)
        else:
            print(
                f"[ep {ep_label:02d}] loss={val_loss:.4f}  "
                f"PairAUPRC={vm.get('pair_auprc',0):.4f}  "
                f"PairMCC={vm.get('pair_mcc',0):.4f}  "
                f"PairF1={vm.get('pair_f1',0):.4f}  "
                f"MedAUC={vm.get('medauc',0):.4f}  "
                f"GeomAUC={vm.get('medauc_geom',0):.4f}  "
                f"FocusPos={vm.get('focus_pos_recall',0):.3f}  "
                f"FocusFrac={vm.get('focus_frac',0):.3f}  "
                f"n_auc={int(vm.get('n_auc',0))}  "
                f"time={time.time()-t0:.1f}s  best_ep={best_epoch}",
                flush=True,
            )

        if ps > best_score:
            best_score = ps; best_epoch = ep_label; best_metrics = dict(vm)
            ema.apply_shadow(model)
            torch.save({"epoch": ep_label, "model_state": model.state_dict(),
                        "cfg": cfg, "val_metrics": vm, "score": best_score,
                        "run_stamp": RUN_STAMP, "metric": BEST_TAG,
                        "model_provenance": model_provenance}, RUN_BEST_CKPT)
            ema.restore(model)
            print(f"  [run-best] score={best_score:.4f} -> {RUN_BEST_CKPT}", flush=True)
            if (PRIMARY_OBJ == "binary" and "auprc" in vm) or PRIMARY_OBJ == "topk" or PRIMARY_OBJ == "pair":
                _write_best_metrics_tsv(RUN_BEST_METRICS_TSV, best_epoch, best_metrics, RUN_BEST_CKPT)

        if bool(getattr(P, 'save_epoch_ckpts', False)) and ep_label % 10 == 0:
            torch.save({"epoch": ep_label, "model_state": model.state_dict(), "cfg": cfg,
                        "model_provenance": model_provenance},
                       os.path.join(CKPT_DIR, f"ep{ep_label:03d}.pt"))
        if es.step(ep_label, ps):
            print(f"[early stop] ep={ep_label} best_ep={es.best_epoch} best={es.best:.4f}")
            break

    promoted = _promote_run_best_if_needed(RUN_BEST_CKPT, best_score, best_epoch, best_metrics)

    test_metrics = {}
    if promoted and PRIMARY_OBJ in ("binary", "pair") and dl_te is not None and os.path.exists(GLOBAL_BEST_CKPT):
        try:
            ck = torch.load(GLOBAL_BEST_CKPT, map_location=DEVICE)
            model.load_state_dict(ck["model_state"])
            model.eval()
            threshold_key = "pair_thr" if PRIMARY_OBJ == "pair" else "thr"
            val_thr = float((ck.get("val_metrics", {}) or {}).get(threshold_key, 0.5))
            test_metrics = eval_binary(
                model,
                dl_te,
                fixed_thr=val_thr,
                prediction_out=RUN_TEST_PREDICTIONS_TSV,
            )
            _write_best_metrics_tsv(RUN_TEST_METRICS_TSV, best_epoch, test_metrics, GLOBAL_BEST_CKPT)
            if PRIMARY_OBJ == "pair":
                print(
                    f"[test-metrics] PairAUPRC={test_metrics.get('pair_auprc',0):.4f} "
                    f"PairMCC={test_metrics.get('pair_mcc',0):.4f} "
                    f"Brier={test_metrics.get('pair_brier',0):.4f} "
                    f"ECE={test_metrics.get('pair_ece',0):.4f} "
                    f"thr={test_metrics.get('pair_thr', val_thr):.3f} "
                    f"metrics_tsv={RUN_TEST_METRICS_TSV} "
                    f"predictions_tsv={RUN_TEST_PREDICTIONS_TSV}",
                    flush=True,
                )
            else:
                print(
                    f"[test-metrics] {_format_binary_metrics(test_metrics)}  "
                    f"thr={test_metrics.get('thr', val_thr):.3f} "
                    f"metrics_tsv={RUN_TEST_METRICS_TSV}",
                    flush=True,
                )
        except Exception as e:
            print(f"[test-metrics][warn] failed to evaluate test split: {e}", flush=True)
    if PRIMARY_OBJ == "binary" and best_metrics:
        metrics_path = GLOBAL_BEST_METRICS_TSV if promoted else RUN_BEST_METRICS_TSV
        print(f"[best-metrics] {_format_binary_metrics(best_metrics)}  metrics_tsv={metrics_path}", flush=True)
    elif PRIMARY_OBJ == "topk" and best_metrics:
        metrics_path = GLOBAL_BEST_METRICS_TSV if promoted else RUN_BEST_METRICS_TSV
        print(f"[best-metrics] Recall@L/5={best_metrics.get('recall_L5',0):.4f}  Recall@L/10={best_metrics.get('recall_L10',0):.4f}  Precision@K10={best_metrics.get('precision_K10',0):.4f}  Hit@K20={best_metrics.get('hit_K20',0):.4f}  Enrich@K10={best_metrics.get('enrichment_K10',0):.2f}  TopKScore={best_metrics.get('topk_score',0):.4f}  metrics_tsv={metrics_path}", flush=True)
    elif PRIMARY_OBJ == "pair" and best_metrics:
        metrics_path = GLOBAL_BEST_METRICS_TSV if promoted else RUN_BEST_METRICS_TSV
        print(
            f"[best-metrics] MedAUC={best_metrics.get('medauc',0):.4f}  "
            f"GeomAUC={best_metrics.get('medauc_geom',0):.4f}  "
            f"FocusPos={best_metrics.get('focus_pos_recall',0):.3f}  "
            f"FocusFrac={best_metrics.get('focus_frac',0):.3f}  "
            f"metrics_tsv={metrics_path}",
            flush=True,
        )
    print(f"[metrics] history={METRICS_HISTORY}", flush=True)
    print(f"[done] best_epoch={best_epoch}  run_best={best_score:.4f}  global_ckpt={GLOBAL_BEST_CKPT}")


if __name__ == "__main__":
    main()
