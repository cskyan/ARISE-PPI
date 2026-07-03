# -*- coding: utf-8 -*-
"""Prediction export for the L131 Dest binary residue-site model.

Default usage on the server:
    python predict_L131.py

Common overrides:
    python predict_L131.py --split test
    python predict_L131.py --split all
    python predict_L131.py --checkpoint /path/to/best_AUPRC.pt --split test
"""

import argparse
import csv
import json
import math
import os
import subprocess, sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


def _preparse_env():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--root", default=None)
    parser.add_argument("--id-list", default=None)
    parser.add_argument("--train-list", default=None)
    parser.add_argument("--val-list", default=None)
    parser.add_argument("--test-list", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--structure-dir", default=None)
    parser.add_argument("--esm-local-dir", default=None)
    args, _ = parser.parse_known_args()
    env_map = {
        "root": "rbp_root",
        "id_list": "rbp_id_list",
        "train_list": "rbp_train_list",
        "val_list": "rbp_val_list",
        "test_list": "rbp_test_list",
        "save_dir": "save_dir",
        "structure_dir": "rbp_structure_dir",
        "esm_local_dir": "esm_local_dir",
    }
    for attr, key in env_map.items():
        val = getattr(args, attr)
        if val:
            os.environ[key] = str(val)
            if attr == "root":
                os.environ["DEST_PREPARED_ROOT"] = str(val)
            elif attr == "save_dir":
                os.environ["DEST_SAVE_DIR"] = str(val)


_preparse_env()
os.environ.setdefault("FORCE_DEST", "1")

from config_L131 import Params, build_model_config  # noqa: E402
from model_L131 import L13PDBGVPModel  # noqa: E402
import train_L131 as trainlib  # noqa: E402


P = Params.from_env()
P = trainlib._force_dest_binary_params(P)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
METRIC_FIELDS = ["acc", "precision", "recall", "f1", "auprc", "mcc"]
TOPK_FIELDS = [
    "recall_L5", "recall_L10", "precision_K10", "hit_K20",
    "enrichment_K10", "topk_score",
]


def read_ids(path: str) -> List[str]:
    ids = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            ids.append(s.split()[0])
    seen, out = set(), []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def resolve_ids(args) -> Tuple[List[str], str]:
    if args.ids:
        return read_ids(args.ids), args.ids
    split = str(args.split).lower()
    if split == "train":
        path = P.rbp_train_list
    elif split == "val":
        path = P.rbp_val_list
    elif split == "test":
        path = P.rbp_test_list
    elif split == "all":
        path = P.rbp_id_list
    else:
        path = args.split
    if not path or not os.path.exists(path):
        raise FileNotFoundError("ID list not found for split=%s: %s" % (args.split, path))
    return read_ids(path), path


def make_embedder():
    if str(getattr(P, "sequence_mode", "esm")).lower() not in ("esm", "hybrid"):
        return None
    try:
        return trainlib.SiteEmbedder(device=DEVICE, esm_local_dir=P.esm_local_dir)
    except Exception as e:
        if not getattr(P, "allow_zero_esm_fallback", False):
            raise RuntimeError("ESM requested but SiteEmbedder init failed") from e
        print("[warn] ESM init failed (%s); using zero ESM fallback" % e, flush=True)
        return None


def build_predict_loader(ids: List[str], batch_size: int):
    esm_cache = os.path.join(P.save_dir, "esm_cache")
    ds = trainlib.RBP296Dataset(
        P.rbp_root,
        ids,
        embedder=make_embedder(),
        use_pssm=P.use_pssm,
        use_dssp=P.use_dssp,
        esm_cache_dir=esm_cache,
        verbose=False,
    )
    dl = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(getattr(P, "num_workers", 0)),
        collate_fn=trainlib.dips_collate,
        drop_last=False,
    )
    sample = next(iter(dl))
    d_res = int(sample["resA"].shape[-1])
    chainA = sample.get("chainA", None)
    d_chain = int(chainA.shape[-1]) if (chainA is not None and chainA.ndim in (1, 2, 3)) else 0
    return ds, dl, d_res, d_chain


def load_model(checkpoint: str, d_res: int, d_chain: int):
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = ckpt.get("cfg", None) if isinstance(ckpt, dict) else None
    if cfg is None:
        cfg = build_model_config(P, d_res, d_chain)
    model = L13PDBGVPModel(cfg).to(DEVICE)
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[ckpt][warn] missing keys: %d" % len(missing), flush=True)
    if unexpected:
        print("[ckpt][warn] unexpected keys: %d" % len(unexpected), flush=True)
    model.eval()
    val_metrics = ckpt.get("val_metrics", {}) if isinstance(ckpt, dict) else {}
    return model, val_metrics


def threshold_from_labels(probs: np.ndarray, labels: np.ndarray, mode: str = "mcc") -> float:
    valid = np.isfinite(probs) & np.isfinite(labels)
    probs = probs[valid]
    labels = (labels[valid] > 0.5).astype(np.int32)
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return float(getattr(P, "val_thr", 0.5))
    try:
        from sklearn.metrics import f1_score, matthews_corrcoef, recall_score
    except Exception:
        return float(np.quantile(probs, 0.70))
    mode = str(mode or "mcc").lower()
    target_pos_rate = float(labels.mean())
    recall_floor = float(getattr(P, "val_recall_floor", 0.65))
    beta = float(getattr(P, "val_fbeta_beta", 1.5))
    beta2 = beta * beta
    best_t, best_score = 0.5, -2.0
    fallback_t, fallback_recall = 0.5, -1.0
    for t in np.linspace(float(P.val_thr_min), float(P.val_thr_max), int(P.val_thr_grid)):
        pred = (probs >= t).astype(np.int32)
        rec = float(recall_score(labels, pred, zero_division=0))
        prec = float((labels[pred == 1].mean()) if pred.sum() > 0 else 0.0)
        mcc = float(matthews_corrcoef(labels, pred))
        if rec > fallback_recall:
            fallback_recall, fallback_t = rec, float(t)
        if mode in ("f1", "val_f1"):
            score = float(f1_score(labels, pred, zero_division=0))
        elif mode in ("fbeta", "val_fbeta", "auto_fbeta"):
            score = (1.0 + beta2) * prec * rec / max(1e-12, beta2 * prec + rec)
        elif mode in ("recall_floor", "val_recall_floor", "auto_recall_floor"):
            if rec + 1e-12 < recall_floor:
                continue
            score = mcc + 0.25 * float(f1_score(labels, pred, zero_division=0))
        elif mode in ("posrate", "val_posrate", "prevalence"):
            score = -abs(float(pred.mean()) - target_pos_rate)
            # Tie-break with MCC when prevalence matching is similar.
            score += 1e-3 * mcc
        else:
            score = mcc
        if score > best_score:
            best_score, best_t = score, float(t)
    if best_score <= -1.9:
        return float(fallback_t)
    return best_t


def auto_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    return threshold_from_labels(probs, labels, mode="mcc")


def collect_labeled_scores(model, dl) -> Tuple[np.ndarray, np.ndarray]:
    probs_all, labels_all = [], []
    model.eval()
    with torch.no_grad():
        for batch in dl:
            resA = torch.nan_to_num(batch["resA"], nan=0.0).to(DEVICE).float()
            resB = torch.nan_to_num(batch["resB"], nan=0.0).to(DEVICE).float()
            coordsA = torch.nan_to_num(batch["coordsA"], nan=0.0).to(DEVICE).float()
            coordsB = torch.nan_to_num(batch["coordsB"], nan=0.0).to(DEVICE).float()
            maskA = (batch["maskA"] > 0.5).to(DEVICE)
            maskB = (batch["maskB"] > 0.5).to(DEVICE)
            cA = batch.get("chainA", None)
            cB = batch.get("chainB", None)
            if cA is not None:
                cA = cA.to(DEVICE).float()
            if cB is not None:
                cB = cB.to(DEVICE).float()
            out = model(resA, maskA, cA, coordsA, resB, maskB, cB, coordsB, site_mode=True)
            probs = torch.sigmoid(out["logit_resA"]).detach().cpu()
            labels = batch["y_res_A"].detach().cpu().float()
            masks = batch["maskA"].detach().cpu() > 0.5
            for b in range(probs.shape[0]):
                m = masks[b]
                if bool(m.any()):
                    probs_all.append(probs[b][m].numpy().astype(float))
                    labels_all.append(labels[b][m].numpy().astype(float))
    p = np.concatenate(probs_all) if probs_all else np.zeros(0, dtype=np.float32)
    y = np.concatenate(labels_all) if labels_all else np.zeros(0, dtype=np.float32)
    return p, y


def binary_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float) -> Dict[str, float]:
    valid = np.isfinite(probs) & np.isfinite(labels)
    probs = probs[valid]
    labels = (labels[valid] > 0.5).astype(np.int32)
    out = {"n_eval": int(labels.size), "threshold": float(threshold)}
    if labels.size == 0:
        return out
    pred = (probs >= threshold).astype(np.int32)
    out["pos_rate"] = float(labels.mean())
    out["pred_pos_rate"] = float(pred.mean())
    try:
        from sklearn.metrics import (
            average_precision_score,
            f1_score,
            matthews_corrcoef,
            precision_score,
            recall_score,
        )
        out.update({
            "auprc": float(average_precision_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0,
            "acc": float((pred == labels).mean()),
            "f1": float(f1_score(labels, pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(labels, pred)) if len(np.unique(labels)) > 1 else 0.0,
            "precision": float(precision_score(labels, pred, zero_division=0)),
            "recall": float(recall_score(labels, pred, zero_division=0)),
        })
    except Exception as e:
        out["metric_error"] = str(e)
    return out


def write_metrics_tsv(path: str, metrics: Dict):
    header = ["ACC", "Precision", "Recall", "F1", "AUPRC", "MCC", "Threshold", "N_eval", "Pos_rate"]
    keys = ["acc", "precision", "recall", "f1", "auprc", "mcc"]
    vals = []
    for k in keys:
        try:
            vals.append("%.6f" % float(metrics.get(k, 0.0)))
        except Exception:
            vals.append("0.000000")
    vals += [
        "%.6f" % float(metrics.get("threshold", 0.5)),
        str(int(metrics.get("n_eval", 0))),
        "%.6f" % float(metrics.get("pos_rate", 0.0)),
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        f.write("\t".join(vals) + "\n")


def write_metrics_comparison_tsv(path: str, official: Dict, auto_current: Dict):
    rows = []
    for name, metrics in [("checkpoint_val_threshold", official), ("auto_current_label_threshold", auto_current)]:
        row = {"setting": name}
        for k in METRIC_FIELDS:
            row[k] = float(metrics.get(k, 0.0))
        row["threshold"] = float(metrics.get("threshold", 0.5))
        row["n_eval"] = int(metrics.get("n_eval", 0))
        row["pos_rate"] = float(metrics.get("pos_rate", 0.0))
        row["pred_pos_rate"] = float(metrics.get("pred_pos_rate", 0.0))
        rows.append(row)
    write_tsv(path, rows)


def label_exists(pid: str) -> bool:
    return os.path.exists(os.path.join(P.rbp_root, "labels", "%s.npy" % pid))


def residue_window(seq: str, pos0: int, radius: int = 5) -> str:
    if not seq:
        return ""
    lo = max(0, pos0 - radius)
    hi = min(len(seq), pos0 + radius + 1)
    return "%d-%d:%s" % (lo + 1, hi, seq[lo:hi])


def ager_refine_np(scores: np.ndarray, coords: np.ndarray,
                   enabled=None, alpha=None, radius=None, top_m=None) -> np.ndarray:
    if enabled is None:
        enabled = bool(getattr(P, "ager_enable", True))
    if scores.size <= 1 or not bool(enabled):
        return scores.astype(float)
    if alpha is None:
        alpha = float(getattr(P, "ager_alpha", 0.30))
    alpha = float(alpha)
    if alpha <= 0:
        return scores.astype(float)
    if radius is None:
        radius = float(getattr(P, "ager_radius", 10.0))
    if top_m is None:
        top_m = int(getattr(P, "ager_top_m", 5))
    radius = float(radius)
    top_m = max(1, int(top_m))
    coords = np.nan_to_num(coords.astype(float), nan=0.0)
    scores = np.nan_to_num(scores.astype(float), nan=0.0)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=-1))
    adj = (dist <= radius) & (dist > 1e-6)
    refined = scores.copy()
    for i in range(scores.shape[0]):
        idx = np.where(adj[i])[0]
        if idx.size == 0:
            continue
        vals = scores[idx]
        k = min(top_m, vals.size)
        neigh = np.sort(vals)[-k:].mean()
        refined[i] = (1.0 - alpha) * scores[i] + alpha * neigh
    return refined


def _parse_float_grid(spec: str, default):
    if spec is None or str(spec).strip() == "":
        return list(default)
    vals = []
    for x in str(spec).replace(";", ",").split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    return vals if vals else list(default)


def _parse_int_grid(spec: str, default):
    return [int(round(x)) for x in _parse_float_grid(spec, default)]


def topk_metrics_from_arrays(scores_list, labels_list) -> Dict[str, float]:
    recall_L5, recall_L10, precision_K10, hit_K20, enrichment_K10 = [], [], [], [], []
    for scores, labels in zip(scores_list, labels_list):
        scores = np.asarray(scores, dtype=float)
        labels = (np.asarray(labels, dtype=float) > 0.5).astype(np.int32)
        if scores.size == 0:
            continue
        npos = int(labels.sum())
        if npos <= 0:
            continue
        order = np.argsort(-scores)
        L = int(scores.size)
        def _hit_at(k):
            kk = max(1, min(int(k), L))
            return int(labels[order[:kk]].sum()), kk
        h5, k5 = _hit_at(math.ceil(L / 5.0))
        h10, k10 = _hit_at(math.ceil(L / 10.0))
        hp10, kk10 = _hit_at(10)
        hk20, kk20 = _hit_at(20)
        pos_rate = float(npos) / max(1.0, float(L))
        p10 = hp10 / max(1, kk10)
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
    return res


def run_ager_scan(raw_scores_list, coords_list, labels_list, args, out_dir: str):
    alphas = _parse_float_grid(args.ager_scan_alpha, [0.0, 0.15, 0.25, 0.30, 0.35, 0.45])
    radii = _parse_float_grid(args.ager_scan_radius, [8.0, 10.0, 12.0])
    top_ms = _parse_int_grid(args.ager_scan_topm, [3, 5, 7])
    primary = str(args.ager_scan_primary or getattr(P, "topk_primary", "precision_K10"))
    rows = []
    best = None
    for alpha in alphas:
        for radius in radii:
            for top_m in top_ms:
                refined = [
                    ager_refine_np(s, c, enabled=True, alpha=alpha, radius=radius, top_m=top_m)
                    for s, c in zip(raw_scores_list, coords_list)
                ]
                m = topk_metrics_from_arrays(refined, labels_list)
                score = float(m.get(primary, m.get("precision_K10", 0.0)))
                row = {
                    "ager_alpha": float(alpha),
                    "ager_radius": float(radius),
                    "ager_top_m": int(top_m),
                    "primary": primary,
                    "primary_score": score,
                    "Recall@L/5": float(m.get("recall_L5", 0.0)),
                    "Recall@L/10": float(m.get("recall_L10", 0.0)),
                    "Precision@K10": float(m.get("precision_K10", 0.0)),
                    "Hit@K20": float(m.get("hit_K20", 0.0)),
                    "Enrich@K10": float(m.get("enrichment_K10", 0.0)),
                    "TopKScore": float(m.get("topk_score", 0.0)),
                }
                rows.append(row)
                if best is None or (score, row["TopKScore"], row["Hit@K20"]) > (
                    best["primary_score"], best["TopKScore"], best["Hit@K20"]
                ):
                    best = row
    rows = sorted(rows, key=lambda r: (r["primary_score"], r["TopKScore"], r["Hit@K20"]), reverse=True)
    scan_tsv = os.path.join(out_dir, "topk_ager_scan.tsv")
    write_tsv(scan_tsv, rows)
    best_json = os.path.join(out_dir, "topk_ager_scan_best.json")
    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(best or {}, f, indent=2, ensure_ascii=False)
    if best:
        print("[predict][ager-scan] best alpha=%.3f radius=%.1f top_m=%d %s=%.4f P@K10=%.4f Hit@K20=%.4f Enrich@K10=%.2f" % (
            best["ager_alpha"], best["ager_radius"], best["ager_top_m"],
            primary, best["primary_score"], best["Precision@K10"], best["Hit@K20"], best["Enrich@K10"]
        ), flush=True)
    print("[predict][ager-scan] wrote: %s" % scan_tsv, flush=True)
    return best, rows


def write_topk_metrics_tsv(path: str, metrics: Dict):
    header = ["Recall@L/5", "Recall@L/10", "Precision@K10", "Hit@K20", "Enrich@K10", "TopKScore", "AGER"]
    vals = [
        "%.6f" % float(metrics.get("recall_L5", 0.0)),
        "%.6f" % float(metrics.get("recall_L10", 0.0)),
        "%.6f" % float(metrics.get("precision_K10", 0.0)),
        "%.6f" % float(metrics.get("hit_K20", 0.0)),
        "%.6f" % float(metrics.get("enrichment_K10", 0.0)),
        "%.6f" % float(metrics.get("topk_score", 0.0)),
        str(int(bool(metrics.get("ager", False)))),
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        f.write("\t".join(vals) + "\n")


def resolve_checkpoint(args) -> str:
    if args.checkpoint:
        return args.checkpoint
    tag = str(getattr(args, "checkpoint_tag", "auto") or "auto").lower()
    if tag == "auto":
        primary = str(getattr(P, "primary_objective", "binary")).lower()
        if primary == "topk":
            tag = "TOPK"
        elif primary == "pair":
            tag = "MedAUC"
        else:
            tag = "AUPRC"
    elif tag in ("topk", "best_topk"):
        tag = "TOPK"
    elif tag in ("pair", "ppi", "medauc", "best_pair", "best_medauc"):
        tag = "MedAUC"
    elif tag in ("pair_auprc", "best_pair_auprc"):
        tag = "PAIR_AUPRC"
    elif tag in ("auprc", "auc", "binary", "best_auprc"):
        tag = "AUPRC"
    ckpt_dir = os.path.join(P.save_dir, "checkpoints")
    ckpt = os.path.join(ckpt_dir, "best_%s.pt" % tag)
    if not os.path.exists(ckpt) and tag == "TOPK":
        fallback = os.path.join(P.save_dir, "checkpoints", "best_AUPRC.pt")
        if os.path.exists(fallback):
            print("[predict][warn] best_TOPK.pt not found; fallback to best_AUPRC.pt", flush=True)
            return fallback
    if not os.path.exists(ckpt) and tag == "MedAUC":
        for alt in ("best_PAIR_AUPRC.pt", "best_AUPRC.pt"):
            fallback = os.path.join(P.save_dir, "checkpoints", alt)
            if os.path.exists(fallback):
                print(f"[predict][warn] best_MedAUC.pt not found; fallback to {alt}", flush=True)
                return fallback
    return ckpt


def compact_topk_metrics(metrics: Dict) -> str:
    keys = [
        ("Recall@L/5", "recall_L5"),
        ("Recall@L/10", "recall_L10"),
        ("Precision@K10", "precision_K10"),
        ("Hit@K20", "hit_K20"),
        ("Enrich@K10", "enrichment_K10"),
        ("TopKScore", "topk_score"),
    ]
    parts = []
    for label, key in keys:
        if key in metrics:
            parts.append("%s=%.4f" % (label, float(metrics.get(key, 0.0))))
    return " ".join(parts)


def predict(args):
    ids, id_source = resolve_ids(args)
    if not ids:
        raise RuntimeError("No IDs to predict.")
    ckpt = resolve_checkpoint(args)
    if not os.path.exists(ckpt):
        raise FileNotFoundError("Checkpoint not found: %s" % ckpt)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join(P.save_dir, "predictions", stamp)
    os.makedirs(out_dir, exist_ok=True)
    residue_score_dir = os.path.join(out_dir, "residue_scores")
    os.makedirs(residue_score_dir, exist_ok=True)

    ds, dl, d_res, d_chain = build_predict_loader(ids, args.batch_size)
    model, val_metrics = load_model(ckpt, d_res, d_chain)
    print("[predict] ids=%d source=%s" % (len(ids), id_source), flush=True)
    print("[predict] checkpoint=%s" % ckpt, flush=True)
    print("[predict] d_res=%d d_chain=%d out=%s" % (d_res, d_chain, out_dir), flush=True)

    residue_rows = []
    ranking_rows = []
    protein_rows = []
    topk_evidence_rows = []
    bridge_summary_rows = []
    anchor_partner_rows = []
    all_probs, all_labels = [], []
    per_protein_rank_scores, per_protein_rank_labels = [], []
    per_protein_raw_scores, per_protein_coords = [], []

    with torch.no_grad():
        for batch in dl:
            resA = torch.nan_to_num(batch["resA"], nan=0.0).to(DEVICE).float()
            resB = torch.nan_to_num(batch["resB"], nan=0.0).to(DEVICE).float()
            coordsA = torch.nan_to_num(batch["coordsA"], nan=0.0).to(DEVICE).float()
            coordsB = torch.nan_to_num(batch["coordsB"], nan=0.0).to(DEVICE).float()
            maskA = (batch["maskA"] > 0.5).to(DEVICE)
            maskB = (batch["maskB"] > 0.5).to(DEVICE)
            cA = batch.get("chainA", None)
            cB = batch.get("chainB", None)
            if cA is not None:
                cA = cA.to(DEVICE).float()
            if cB is not None:
                cB = cB.to(DEVICE).float()
            out = model(resA, maskA, cA, coordsA, resB, maskB, cB, coordsB, site_mode=True)
            probs = torch.sigmoid(out["logit_resA"]).detach().cpu()
            labels = batch["y_res_A"].detach().cpu().float()
            masks = batch["maskA"].detach().cpu() > 0.5
            coords_cpu = batch["coordsA"].detach().cpu().float()
            for b, pid in enumerate(batch["complex"]):
                m = masks[b]
                p = probs[b][m].numpy().astype(float)
                y = labels[b][m].numpy().astype(float)
                c_np = coords_cpu[b][m].numpy().astype(float)
                p_rank = ager_refine_np(p, c_np)
                L = int(p.shape[0])
                seq = ds._read_seq_for_label(pid, L)
                has_label = label_exists(pid)
                if has_label and L > 0:
                    all_probs.append(p)
                    all_labels.append(y)
                    per_protein_raw_scores.append(p)
                    per_protein_coords.append(c_np)
                    per_protein_rank_scores.append(p_rank)
                    per_protein_rank_labels.append(y)
                order = np.argsort(-p_rank)
                k_rank = min(int(args.topk), L)
                top_probs = p_rank[order[:max(1, min(L, max(k_rank, int(math.ceil(0.05 * max(1, L))))))]]
                protein_score_topmean = float(top_probs.mean()) if top_probs.size else 0.0
                protein_rows.append({
                    "accession": pid,
                    "length": L,
                    "has_label": int(has_label),
                    "label_positive": int((y > 0.5).any()) if has_label else "",
                    "label_pos_rate": float((y > 0.5).mean()) if has_label and L else "",
                    "score_max": float(p.max()) if L else 0.0,
                    "score_mean": float(p.mean()) if L else 0.0,
                    "ager_score_max": float(p_rank.max()) if L else 0.0,
                    "score_top_mean": protein_score_topmean,
                    "top_residue_pos": int(order[0] + 1) if L else "",
                    "top_residue_aa": seq[order[0]] if seq and L and order[0] < len(seq) else "X",
                    "top_residue_prob": float(p[order[0]]) if L else 0.0,
                    "top_residue_ager_score": float(p_rank[order[0]]) if L else 0.0,
                })
                for r, idx in enumerate(order[:k_rank], start=1):
                    aa = seq[idx] if seq and idx < len(seq) else "X"
                    top_row = {
                        "accession": pid,
                        "rank": r,
                        "pos": int(idx + 1),
                        "aa": aa,
                        "prob": float(p[idx]),
                        "ager_score": float(p_rank[idx]),
                        "label": int(y[idx] > 0.5) if has_label and idx < len(y) else "",
                        "window": residue_window(seq, int(idx), radius=int(args.window_radius)),
                    }
                    ranking_rows.append(top_row)
                    topk_evidence_rows.append(dict(top_row))
                for idx in range(L):
                    aa = seq[idx] if seq and idx < len(seq) else "X"
                    residue_rows.append({
                        "accession": pid,
                        "pos": int(idx + 1),
                        "aa": aa,
                        "prob": float(p[idx]),
                        "ager_score": float(p_rank[idx]),
                        "label": int(y[idx] > 0.5) if has_label and idx < len(y) else "",
                    })
                write_tsv(os.path.join(residue_score_dir, f"{pid}.tsv"), [
                    r for r in residue_rows if r["accession"] == pid
                ])
                bridge_summary_rows.append({
                    "accession": pid,
                    "mode": "site",
                    "bridge_available": 0,
                    "note": "DEST site-mode exports L1-AMLEH residue evidence; GC-EB bridge is produced in pair-mode.",
                    "topk": int(k_rank),
                    "topk_mean_prob": float(np.mean([float(x["prob"]) for x in ranking_rows if x["accession"] == pid])) if k_rank > 0 else 0.0,
                })

    label_probs = np.concatenate(all_probs) if all_probs else np.zeros(0, dtype=np.float32)
    label_y = np.concatenate(all_labels) if all_labels else np.zeros(0, dtype=np.float32)
    ager_scan_best = {}
    if bool(getattr(args, "ager_scan", False)) and per_protein_raw_scores:
        ager_scan_best, _ = run_ager_scan(
            per_protein_raw_scores,
            per_protein_coords,
            per_protein_rank_labels,
            args,
            out_dir,
        )
        if bool(getattr(args, "ager_scan_apply_best", False)) and ager_scan_best:
            per_protein_rank_scores = [
                ager_refine_np(
                    s,
                    c,
                    enabled=True,
                    alpha=ager_scan_best["ager_alpha"],
                    radius=ager_scan_best["ager_radius"],
                    top_m=ager_scan_best["ager_top_m"],
                )
                for s, c in zip(per_protein_raw_scores, per_protein_coords)
            ]
    auto_thr = auto_threshold(label_probs, label_y)
    calibration_metrics = {}
    threshold_mode = str(getattr(args, "threshold_mode", "val_recall_floor") or "val_recall_floor").lower()
    if args.threshold is not None and float(args.threshold) >= 0:
        thr = float(args.threshold)
        threshold_source = "manual"
    elif threshold_mode in ("checkpoint", "checkpoint_val"):
        if isinstance(val_metrics, dict) and "thr" in val_metrics:
            thr = float(val_metrics["thr"])
        else:
            thr = auto_thr
            threshold_mode = "auto_current"
        threshold_source = "checkpoint_val"
    elif threshold_mode in ("val_f1", "val_mcc", "val_posrate", "val_fbeta", "val_recall_floor"):
        try:
            val_ids = read_ids(P.rbp_val_list)
            _, dl_cal, _, _ = build_predict_loader(val_ids, args.batch_size)
            cal_probs, cal_y = collect_labeled_scores(model, dl_cal)
            cal_mode = threshold_mode.replace("val_", "")
            thr = threshold_from_labels(cal_probs, cal_y, mode=cal_mode)
            calibration_metrics = binary_metrics(cal_probs, cal_y, thr)
            calibration_metrics["calibration_split"] = P.rbp_val_list
            calibration_metrics["threshold_mode"] = threshold_mode
            threshold_source = threshold_mode
            print("[predict][calibration] mode=%s threshold=%.4f val_metrics=%s" % (threshold_mode, thr, compact_metrics(calibration_metrics)), flush=True)
        except Exception as e:
            print("[predict][calibration][warn] failed to calibrate on validation split (%s); using checkpoint threshold" % e, flush=True)
            if isinstance(val_metrics, dict) and "thr" in val_metrics:
                thr = float(val_metrics["thr"])
                threshold_source = "checkpoint_val_fallback"
            else:
                thr = auto_thr
                threshold_source = "auto_current_fallback"
    elif threshold_mode in ("auto_current", "auto_current_labels"):
        thr = auto_thr
        threshold_source = "auto_current_labels"
    elif isinstance(val_metrics, dict) and "thr" in val_metrics:
        thr = float(val_metrics["thr"])
        threshold_source = "checkpoint_val"
    else:
        thr = auto_thr
        threshold_source = "auto_current_labels"
    for row in protein_rows:
        pid = row["accession"]
        vals = [x["prob"] for x in residue_rows if x["accession"] == pid]
        pred = np.asarray(vals, dtype=float) >= thr
        row["threshold"] = float(thr)
        row["n_pred_positive"] = int(pred.sum())
        row["pred_pos_rate"] = float(pred.mean()) if pred.size else 0.0
    for row in residue_rows:
        row["pred"] = int(float(row["prob"]) >= thr)
    for row in ranking_rows:
        row["pred"] = int(float(row["prob"]) >= thr)

    protein_rows = sorted(protein_rows, key=lambda x: (x["score_top_mean"], x["score_max"]), reverse=True)
    metrics = binary_metrics(label_probs, label_y, thr)
    metrics_auto = binary_metrics(label_probs, label_y, auto_thr)
    topk_metrics = topk_metrics_from_arrays(per_protein_rank_scores, per_protein_rank_labels)
    metrics.update({
        "checkpoint": ckpt,
        "id_source": id_source,
        "threshold_source": threshold_source,
        "threshold_mode": threshold_mode,
        "validation_calibration_metrics": calibration_metrics,
        "auto_threshold_current_labels": float(auto_thr),
        "auto_metrics_current_labels": metrics_auto,
        "topk_metrics": topk_metrics,
        "ager_scan_best": ager_scan_best,
        "n_proteins": len(ids),
        "n_labeled_proteins": int(sum(1 for r in protein_rows if int(r["has_label"]) == 1)),
        "out_dir": out_dir,
        "val_metrics_in_checkpoint": val_metrics,
    })

    write_tsv(os.path.join(out_dir, "protein_summary.tsv"), protein_rows)
    write_tsv(os.path.join(out_dir, "residue_predictions.tsv"), residue_rows)
    write_tsv(os.path.join(out_dir, "ranking_top_residues.tsv"), ranking_rows)
    write_tsv(os.path.join(out_dir, "topk_evidence.tsv"), topk_evidence_rows)
    write_tsv(os.path.join(out_dir, "bridge_summary.tsv"), bridge_summary_rows)
    write_tsv(os.path.join(out_dir, "anchor_partner_ranking.tsv"), anchor_partner_rows)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    write_metrics_tsv(os.path.join(out_dir, "metrics.tsv"), metrics)
    write_topk_metrics_tsv(os.path.join(out_dir, "topk_metrics.tsv"), topk_metrics)
    write_metrics_tsv(os.path.join(out_dir, "metrics_auto_current_labels.tsv"), metrics_auto)
    write_metrics_comparison_tsv(os.path.join(out_dir, "metrics_threshold_comparison.tsv"), metrics, metrics_auto)
    if calibration_metrics:
        write_metrics_tsv(os.path.join(out_dir, "metrics_validation_calibration.tsv"), calibration_metrics)
    write_case_study(out_dir, protein_rows, ranking_rows, metrics, args)
    print("[predict][Dest-binary] threshold=%.4f source=%s metrics=%s" % (thr, threshold_source, compact_metrics(metrics)), flush=True)
    # Integrated protocol evaluation: write all/macro/surface-proxy metrics automatically.
    try:
        proto = os.path.join(os.path.dirname(os.path.abspath(__file__)), "protocol_eval_dest.py")
        score_dir = os.path.join(out_dir, "residue_scores")
        root = getattr(P, "rbp_root", getattr(P, "dips_root", ""))
        proto_out = os.path.join(out_dir, "protocol_metrics.tsv")
        if os.path.exists(proto) and os.path.isdir(score_dir):
            cmd = [sys.executable, proto, "--score-dir", score_dir, "--root", str(root),
                   "--surface-mask-dir", str(getattr(P, "surface_mask_dir", "")),
                   "--ca-cutoffs", str(getattr(P, "eval_ca_surface_cutoffs", "10,12,14,16,18,20,22,24")),
                   "--out", proto_out]
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            with open(os.path.join(out_dir, "protocol_metrics.log"), "w", encoding="utf-8") as pf:
                pf.write(r.stdout or "")
            if r.returncode == 0:
                print("[predict][protocol] wrote: %s" % proto_out, flush=True)
                for line in (r.stdout or "").splitlines():
                    if any(k in line for k in ("all_global", "all_macro", "surface_mask", "ca_neighbor<=12")):
                        print("[predict][protocol] " + line, flush=True)
            else:
                print("[predict][protocol][warn] failed; see protocol_metrics.log", flush=True)
    except Exception as e:
        print("[predict][protocol][warn] %s" % e, flush=True)
    if label_y.size and abs(float(auto_thr) - float(thr)) > 1e-8:
        print("[predict][diagnostic] auto-threshold-on-current-labels=%.4f metrics=%s" % (auto_thr, compact_metrics(metrics_auto)), flush=True)
    print("[predict] wrote: %s" % out_dir, flush=True)


def write_tsv(path: str, rows: List[Dict]):
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    fields = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def compact_metrics(metrics: Dict) -> str:
    keys = ["acc", "precision", "recall", "f1", "auprc", "mcc", "n_eval"]
    parts = []
    for k in keys:
        if k in metrics:
            v = metrics[k]
            parts.append("%s=%.4f" % (k, v) if isinstance(v, float) else "%s=%s" % (k, v))
    return " ".join(parts)


def write_case_study(out_dir: str, protein_rows: List[Dict], ranking_rows: List[Dict], metrics: Dict, args):
    topn = int(args.case_topn)
    rank_by_pid: Dict[str, List[Dict]] = {}
    for r in ranking_rows:
        rank_by_pid.setdefault(r["accession"], []).append(r)
    case = {
        "purpose": "L1-L3 case-study candidate export for RBP residue-site validation.",
        "selection_rule": "rank proteins by score_top_mean, then inspect top predicted residues/windows.",
        "metrics": metrics,
        "candidates": [],
    }
    lines = []
    lines.append("# L1-L3 RBP Case Study Candidates\n")
    lines.append("## Model-Level Evidence\n")
    lines.append("- Checkpoint: `%s`\n" % metrics.get("checkpoint", ""))
    lines.append("- Proteins predicted: %s\n" % metrics.get("n_proteins", 0))
    if "auprc" in metrics:
        lines.append(
            "- Labeled-residue validation: AUPRC %.4f, F1 %.4f, MCC %.4f at threshold %.3f\n"
            % (
                metrics.get("auprc", 0.0),
                metrics.get("f1", 0.0),
                metrics.get("mcc", 0.0),
                metrics.get("threshold", 0.5),
            )
        )
    lines.append("\n## Candidate Priority List\n")
    for i, row in enumerate(protein_rows[:topn], start=1):
        pid = row["accession"]
        top_res = rank_by_pid.get(pid, [])[: int(args.case_top_residues)]
        cand = {
            "rank": i,
            "accession": pid,
            "length": row["length"],
            "score_top_mean": row["score_top_mean"],
            "score_max": row["score_max"],
            "pred_pos_rate": row["pred_pos_rate"],
            "top_residues": top_res,
        }
        case["candidates"].append(cand)
        lines.append(
            "%d. `%s` | L=%s | top-mean=%.4f | max=%.4f | predicted-site-rate=%.3f\n"
            % (i, pid, row["length"], row["score_top_mean"], row["score_max"], row["pred_pos_rate"])
        )
        for rr in top_res:
            lines.append(
                "   - residue %s%s, prob=%.4f, window `%s`\n"
                % (rr.get("aa", "X"), rr.get("pos", ""), rr.get("prob", 0.0), rr.get("window", ""))
            )
    lines.append("\n## Case Study Interpretation Template\n")
    lines.append("- Biological hypothesis: high-confidence residue windows indicate potential RNA-contact or RNA-regulated functional regions.\n")
    lines.append("- Validation path: prioritize top windows for motif/domain overlap, conservation check, mutagenesis design, and RNA-binding assay follow-up.\n")
    lines.append("- Clinical translation framing: use the model as a candidate-prioritization layer for hospital-cohort biomarker discovery, not as a direct diagnostic decision system.\n")
    lines.append("- Reserved ranking output: `ranking_top_residues.tsv` keeps residue-level priority for later L1/L2/L3 fusion and top-k case-study analysis.\n")
    with open(os.path.join(out_dir, "case_study_candidates.md"), "w", encoding="utf-8") as f:
        f.write("".join(lines))
    with open(os.path.join(out_dir, "case_study_candidates.json"), "w", encoding="utf-8") as f:
        json.dump(case, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=P.rbp_root)
    parser.add_argument("--id-list", default=P.rbp_id_list)
    parser.add_argument("--train-list", default=P.rbp_train_list)
    parser.add_argument("--val-list", default=P.rbp_val_list)
    parser.add_argument("--test-list", default=P.rbp_test_list)
    parser.add_argument("--save-dir", default=P.save_dir)
    parser.add_argument("--structure-dir", default=P.rbp_structure_dir)
    parser.add_argument("--esm-local-dir", default=P.esm_local_dir)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-tag", default="auto",
                        choices=["auto", "topk", "pair", "ppi", "medauc", "pair_auprc", "auprc", "auc", "binary"],
                        help="Default checkpoint selector when --checkpoint is omitted. auto uses the checkpoint matching primary_objective.")
    parser.add_argument("--split", default="test", help="train|val|test|all|/path/to/id_list.txt")
    parser.add_argument("--ids", default=None, help="Explicit ID file; overrides --split.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=max(1, int(getattr(P, "batch_site", 4))))
    parser.add_argument("--threshold", type=float, default=-1.0, help="<0: checkpoint threshold or auto; otherwise fixed threshold.")
    parser.add_argument("--threshold-mode", default="checkpoint",
                        choices=["val_recall_floor", "val_fbeta", "val_f1", "val_mcc", "val_posrate", "checkpoint", "auto_current"],
                        help="How to choose the binary threshold when --threshold < 0. Default uses validation recall-floor calibration.")
    parser.add_argument("--topk", type=int, default=30, help="Top residues per protein for ranking output.")
    parser.add_argument("--ager-scan", action="store_true",
                        help="Scan AGER parameters for top-k metrics on the selected split; use validation split for tuning.")
    parser.add_argument("--ager-scan-alpha", default="0,0.15,0.25,0.30,0.35,0.45",
                        help="Comma-separated alpha values for --ager-scan.")
    parser.add_argument("--ager-scan-radius", default="8,10,12",
                        help="Comma-separated radius values for --ager-scan.")
    parser.add_argument("--ager-scan-topm", default="3,5,7",
                        help="Comma-separated top_m values for --ager-scan.")
    parser.add_argument("--ager-scan-primary", default="precision_K10",
                        help="Metric key used to select best AGER scan setting, e.g. precision_K10 or topk_score.")
    parser.add_argument("--case-topn", type=int, default=12)
    parser.add_argument("--case-top-residues", type=int, default=8)
    parser.add_argument("--window-radius", type=int, default=5)
    args = parser.parse_args()
    predict(args)


if __name__ == "__main__":
    main()
