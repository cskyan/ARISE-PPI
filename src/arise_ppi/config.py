# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass
from typing import Optional


def _env_bool(key: str, default: bool) -> bool:
    v = os.environ.get(key, os.environ.get(str(key).lower(), None))
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip().lower() not in ("0", "false", "no", "off")


def _env_int(key: str, default: int) -> int:
    v = os.environ.get(key, os.environ.get(str(key).lower(), None))
    if v is None or str(v).strip() == "":
        return default
    try:
        return int(float(v))
    except Exception:
        return default


def _env_float(key: str, default: float) -> float:
    v = os.environ.get(key, os.environ.get(str(key).lower(), None))
    if v is None or str(v).strip() == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


def _env_str(key: str, default: str) -> str:
    v = os.environ.get(key, os.environ.get(str(key).lower(), None))
    if v is None or str(v).strip() == "":
        return default
    return str(v)


@dataclass
class Params:
    # ---- objective / data root ----
    primary_objective: str = "topk"
    rbp_root: str = "data/RBP400"
    rbp_id_list: str = "data/RBP400_full_accessions.txt"
    rbp_train_list: str = "data/RBP400_split_train.txt"
    rbp_val_list: str = "data/RBP400_split_val.txt"
    rbp_test_list: str = "data/RBP400_split_test.txt"
    dips_train_list: str = "data/RBP400_full_accessions.txt"
    dips_val_list: str = "data/RBP400_full_accessions.txt"
    save_dir: str = "runs/rbp400"
    save_epoch_ckpts: bool = False
    esm_local_dir: str = "resources/esm"
    rbp_structure_dir: str = "structures"

    # ---- split ----
    split_train: float = 0.80
    split_val: float = 0.10
    split_test: float = 0.10
    split_seed: int = 42
    split_strategy: str = "stratified_label_v2"
    split_search_trials: int = 256

    # ---- feature toggles ----
    use_pssm: bool = True
    use_dssp: bool = True
    sequence_mode: str = "esm"          # esm | light | hybrid
    allow_zero_esm_fallback: bool = False
    structure_source: str = "pdb"       # auto | pdb | coords
    use_chain_geom: bool = True
    use_geom: bool = True
    dips_skip_filter: bool = True
    dips_index_verbose: int = 0

    # ---- optimization ----
    seed: int = 1337
    epochs: int = 80
    batch_site: int = 4
    num_workers: int = 0
    lr: float = 1.0e-5
    weight_decay: float = 2e-4
    max_grad_norm: float = 0.5
    accum_steps: int = 1
    print_every: int = 25
    bad_loss_thr: float = 1e5

    # ---- residue / fragment ----
    label_smoothing_l1: float = 0.02
    l1_w: float = 1.0
    # <=0 means estimate neg/pos from the current training split at runtime.
    l1_pos_weight: float = 0.0
    l1_focal_w: float = 0.0
    l1_focal_gamma: float = 2.0
    l1_focal_alpha: float = 0.50
    l1_per_protein_loss: bool = False
    l1_dice_w: float = 0.0
    l1_extreme_label_weight: float = 0.35
    l1_high_pos_frac: float = 0.95
    l1_high_pos_weight: float = 0.50
    l1_rank_w: float = 0.05
    l1_rank_start_epoch: int = 1
    l1_rank_margin: float = 0.2
    l1_rank_n_pairs: int = 256
    l1_rank_neg_hard_frac: float = 0.75
    objective_weight_auto: bool = False
    topk_rank_boost: float = 1.00
    topk_rank_ramp_epochs: int = 0
    site_l3_pool_w: float = 0.0
    site_l3_pool_start_epoch: int = 2
    site_l3_pool_top_frac: float = 0.05
    site_l3_pool_pos_weight: float = 2.0
    # L1-L3 mainline: keep the fragment head for compatibility, but do not
    # train an L1.5 auxiliary objective by default.
    l15_w: float = 0.0
    l15_start_epoch: int = 999

    # ---- innovation modules kept, but pair-level losses default off on current RBP labels ----
    eb_topk_k: int = 512
    eb_topk_frac: float = 0.05
    eb_pool_mode: str = "topk"
    eb_d_proj: int = 64
    eb_start_epoch: int = 3
    eb_ramp_epochs: int = 5
    eb_pair_topk: int = 128

    l3_w: float = 1.0
    l3_pos_weight: float = 3.0
    l3_focal_w: float = 0.3
    l3_focal_gamma: float = 2.0
    l3_focal_alpha: float = 0.25
    l3_evi_w: float = 0.15
    l3_evi_margin: float = 0.3
    cl_cons_w: float = 0.10
    cl_cons_start_epoch: int = 5
    cl_cons_ramp_epochs: int = 8
    cl_cons_tau: float = 1.0
    cl_cons_margin: float = 0.5
    cl_cons_neg_suppress_w: float = 0.3

    # ---- loss caps ----
    max_res_loss: float = 5.0
    max_frag_loss: float = 10.0
    max_l3_loss: float = 10.0
    max_cons_loss: float = 10.0

    # ---- scheduler / EMA / early stop ----
    lr_sched: bool = True
    lr_warmup_frac: float = 0.05
    lr_warmup_start_ratio: float = 0.0
    lr_warmup_min_updates: int = 40
    lr_warmup_max_updates: int = 160
    lr_min_ratio: float = 0.15
    ema_decay: float = 0.995
    early_min_epochs: int = 25
    early_patience: int = 14
    early_min_delta: float = 2e-4
    early_ema: float = 0.6

    # ---- validation / negatives ----
    val_thr_mode: str = "auto_recall_floor"
    val_thr: float = 0.5
    val_thr_grid: int = 181
    val_thr_min: float = 0.05
    val_thr_max: float = 0.95
    val_recall_floor: float = 0.65
    val_fbeta_beta: float = 1.5
    eval_tta_swap: bool = False
    report_topk_metrics: bool = False
    contact_cutoff: float = 8.0
    train_neg_ratio: float = 1.0
    eval_neg_ratio: float = 1.0

    # ---- top-k ranking / AGER ----
    topk_primary: str = "precision_K10"
    topk_rank_max_pos_frac: float = 0.80
    topk_enrichment_cap: float = 10.0
    ager_enable: bool = True
    ager_radius: float = 10.0
    ager_alpha: float = 0.30
    ager_top_m: int = 5

    # ---- GVP ----
    gvp_node_s_dim: int = 6
    gvp_node_v_dim: int = 2
    gvp_hidden_s: int = 64
    gvp_hidden_v: int = 8
    gvp_layers: int = 2
    gvp_k: int = 16
    gvp_dropout: float = 0.10
    gvp_chain_dim: int = 32

    # ---- misc ----
    small_tr_n: int = 0
    small_va_n: int = 0

    @classmethod
    def from_env(cls, p=None):
        if p is None:
            p = cls()
        spec = [
            ("primary_objective", _env_str),
            ("rbp_root", _env_str), ("rbp_id_list", _env_str),
            ("rbp_train_list", _env_str), ("rbp_val_list", _env_str), ("rbp_test_list", _env_str),
            ("dips_train_list", _env_str), ("dips_val_list", _env_str),
            ("save_dir", _env_str), ("esm_local_dir", _env_str), ("rbp_structure_dir", _env_str),
            ("split_train", _env_float), ("split_val", _env_float), ("split_test", _env_float), ("split_seed", _env_int), ("split_search_trials", _env_int),
            ("use_pssm", _env_bool), ("use_dssp", _env_bool),
            ("sequence_mode", _env_str), ("allow_zero_esm_fallback", _env_bool),
            ("structure_source", _env_str), ("use_chain_geom", _env_bool), ("use_geom", _env_bool),
            ("dips_skip_filter", _env_bool), ("dips_index_verbose", _env_int),
            ("seed", _env_int), ("epochs", _env_int), ("batch_site", _env_int), ("num_workers", _env_int),
            ("lr", _env_float), ("weight_decay", _env_float), ("max_grad_norm", _env_float), ("accum_steps", _env_int),
            ("print_every", _env_int), ("bad_loss_thr", _env_float),
            ("label_smoothing_l1", _env_float), ("l1_w", _env_float), ("l1_pos_weight", _env_float),
            ("l1_focal_w", _env_float), ("l1_focal_gamma", _env_float), ("l1_focal_alpha", _env_float),
            ("l1_per_protein_loss", _env_bool), ("l1_dice_w", _env_float),
            ("l1_extreme_label_weight", _env_float), ("l1_high_pos_frac", _env_float),
            ("l1_high_pos_weight", _env_float),
            ("l1_rank_w", _env_float), ("l1_rank_start_epoch", _env_int), ("l1_rank_margin", _env_float),
            ("l1_rank_n_pairs", _env_int), ("l1_rank_neg_hard_frac", _env_float),
            ("objective_weight_auto", _env_bool), ("topk_rank_boost", _env_float), ("topk_rank_ramp_epochs", _env_int),
            ("site_l3_pool_w", _env_float), ("site_l3_pool_start_epoch", _env_int),
            ("site_l3_pool_top_frac", _env_float), ("site_l3_pool_pos_weight", _env_float),
            ("l15_w", _env_float), ("l15_start_epoch", _env_int),
            ("eb_topk_k", _env_int), ("eb_topk_frac", _env_float), ("eb_pool_mode", _env_str),
            ("eb_d_proj", _env_int), ("eb_start_epoch", _env_int), ("eb_ramp_epochs", _env_int), ("eb_pair_topk", _env_int),
            ("l3_w", _env_float), ("l3_pos_weight", _env_float), ("l3_focal_w", _env_float),
            ("l3_focal_gamma", _env_float), ("l3_focal_alpha", _env_float), ("l3_evi_w", _env_float), ("l3_evi_margin", _env_float),
            ("cl_cons_w", _env_float), ("cl_cons_start_epoch", _env_int), ("cl_cons_ramp_epochs", _env_int),
            ("cl_cons_tau", _env_float), ("cl_cons_margin", _env_float), ("cl_cons_neg_suppress_w", _env_float),
            ("max_res_loss", _env_float), ("max_frag_loss", _env_float), ("max_l3_loss", _env_float), ("max_cons_loss", _env_float),
            ("lr_sched", _env_bool), ("lr_warmup_frac", _env_float), ("lr_warmup_start_ratio", _env_float),
            ("lr_warmup_min_updates", _env_int), ("lr_warmup_max_updates", _env_int), ("lr_min_ratio", _env_float),
            ("ema_decay", _env_float), ("early_min_epochs", _env_int), ("early_patience", _env_int), ("early_min_delta", _env_float), ("early_ema", _env_float),
            ("val_thr_mode", _env_str), ("val_thr", _env_float), ("val_thr_grid", _env_int), ("val_thr_min", _env_float), ("val_thr_max", _env_float),
            ("val_recall_floor", _env_float), ("val_fbeta_beta", _env_float),
            ("eval_tta_swap", _env_bool), ("report_topk_metrics", _env_bool), ("contact_cutoff", _env_float),
            ("split_strategy", _env_str), ("save_epoch_ckpts", _env_bool),
            ("train_neg_ratio", _env_float), ("eval_neg_ratio", _env_float),
            ("topk_primary", _env_str), ("ager_enable", _env_bool), ("ager_radius", _env_float),
            ("ager_alpha", _env_float), ("ager_top_m", _env_int),
            ("topk_rank_max_pos_frac", _env_float), ("topk_enrichment_cap", _env_float),
            ("gvp_hidden_s", _env_int), ("gvp_hidden_v", _env_int), ("gvp_layers", _env_int), ("gvp_k", _env_int),
            ("gvp_dropout", _env_float), ("gvp_chain_dim", _env_int),
            ("small_tr_n", _env_int), ("small_va_n", _env_int),
        ]
        for name, fn in spec:
            setattr(p, name, fn(name.upper(), getattr(p, name)))
        p.primary_objective = str(p.primary_objective).lower()
        p.sequence_mode = str(p.sequence_mode).lower()
        p.structure_source = str(p.structure_source).lower()
        return p


@dataclass
class ModelConfig:
    d_seq_in: int = 1309
    d_chain_in: int = 12
    d_model: int = 384
    n_encoder_layers: int = 8
    n_cross_layers: int = 8
    n_heads: int = 8
    dropout: float = 0.20
    use_layerscale: bool = True
    layerscale_init: float = 1e-4
    use_swiglu: bool = True
    ffn_mult: float = 4.0
    eb_topk_k: int = 512
    eb_topk_frac: float = 0.05
    eb_d_proj: int = 64
    eb_n_heads: int = 4
    eb_pair_topk: int = 128
    eb_start_epoch: int = 3
    eb_ramp_epochs: int = 5
    l3_d_hidden: int = 256
    l3_n_layers: int = 2
    prior_pos_pix: float = 0.05
    l1_pos_weight: float = 5.0
    cl_cons_tau: float = 1.0
    cl_cons_margin: float = 0.5
    explain_topk_k: int = 512
    gvp_node_s_dim: int = 6
    gvp_node_v_dim: int = 2
    gvp_hidden_s: int = 64
    gvp_hidden_v: int = 8
    gvp_layers: int = 2
    gvp_k: int = 16
    gvp_dropout: float = 0.10
    gvp_chain_dim: int = 32

    def get(self, key, default=None):
        return getattr(self, key, default)


def build_model_config(p: Params, d_seq_in: int, d_chain_in: int) -> ModelConfig:
    return ModelConfig(
        d_seq_in=int(d_seq_in),
        d_chain_in=int(d_chain_in),
        eb_topk_k=int(p.eb_topk_k),
        eb_topk_frac=float(p.eb_topk_frac),
        eb_d_proj=int(p.eb_d_proj),
        eb_pair_topk=int(p.eb_pair_topk),
        eb_start_epoch=int(p.eb_start_epoch),
        eb_ramp_epochs=int(p.eb_ramp_epochs),
        l1_pos_weight=float(p.l1_pos_weight),
        cl_cons_tau=float(p.cl_cons_tau),
        cl_cons_margin=float(p.cl_cons_margin),
        explain_topk_k=int(p.eb_topk_k),
        gvp_node_s_dim=int(p.gvp_node_s_dim),
        gvp_node_v_dim=int(p.gvp_node_v_dim),
        gvp_hidden_s=int(p.gvp_hidden_s),
        gvp_hidden_v=int(p.gvp_hidden_v),
        gvp_layers=int(p.gvp_layers),
        gvp_k=int(p.gvp_k),
        gvp_dropout=float(p.gvp_dropout),
        gvp_chain_dim=int(p.gvp_chain_dim),
    )
