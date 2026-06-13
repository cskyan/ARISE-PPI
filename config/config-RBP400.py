# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass
from typing import Optional

# GraphRBF-PP L1 site benchmark configuration.
# Baseline uses light sequence features + PSSM + DSSP/SS + CA geometry from
# PP/prepared. HMM and extra SS switches are reserved for follow-up ablations.

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
    dataset_mode: str = "rbp"           # rbp | pair | custom; NEVER defaults to DIPS in this branch
    primary_objective: str = "binary"   # binary | topk | pair; PP is an L1 residue-site benchmark by default
    # Keep legacy dips_* names only for backward compatibility; default root follows rbp/custom data.
    dips_root: str = "..."
    rbp_root: str = "..."
    rbp_id_list: str = "..."
    rbp_train_list: str = "..."
    rbp_val_list: str = "..."
    rbp_test_list: str = "..."
    dips_train_list: str = "..."
    dips_val_list: str = "..."
    dips_test_list: str = "..."
    save_dir: str = "..."
    save_epoch_ckpts: bool = False
    esm_local_dir: str = "/esm"
    rbp_structure_dir: str = "coords"
    # PP prepared data uses legacy 9-dim DSSP/SS one-hot + ASA from feature/SS.zip.
    rbp_dssp_dir: str = "dssp"
    dssp_dim: int = 9
    surface_mask_dir: str = "surface_mask_rsa05"
    binary_primary_metric: str = "auroc"  # PP paper-style primary checkpoint metric
    eval_ca_surface_radius: float = 10.0
    eval_ca_surface_cutoffs: str = "10,12,14,16,18,20,22,24"

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
    use_hmm: bool = False               # reserved for PP feature/HMM.zip; off for baseline
    use_ss: bool = False                # reserved switch; PP SS is already included in dssp by default
    sequence_mode: str = "light"        # esm | light | hybrid
    esm_cache_dir: str = ""             # empty => DIPS_ROOT/esm_cache for DIPS, SAVE_DIR/esm_cache otherwise
    allow_zero_esm_fallback: bool = False
    dips_use_embedder_on_miss: bool = False
    structure_source: str = "coords"    # auto | pdb | coords; MedAUC boost uses coords explicitly
    use_chain_geom: bool = True
    use_geom: bool = True
    site_self_cross: bool = False       # stable baseline: disabled
    dips_skip_filter: bool = False
    dips_filter_cache: bool = True
    dips_index_verbose: int = 1
    dips_esm_miss_log_limit: int = 50

    # ---- optimization ----
    seed: int = 1337
    epochs: int = 140
    batch_site: int = 4
    eval_batch_site: int = 4
    num_workers: int = 0
    allow_cuda_workers: bool = False
    prefetch_factor: int = 2
    lr: float = 2.0e-5
    weight_decay: float = 3e-5
    max_grad_norm: float = 0.5
    accum_steps: int = 4
    print_every: int = 100
    bad_loss_thr: float = 1e5

    # ---- residue / fragment ----
    label_smoothing_l1: float = 0.00
    l1_w: float = 1.0
    # <=0 means estimate neg/pos from the current training split at runtime.
    l1_pos_weight: float = 0.0
    l1_focal_w: float = 0.15
    l1_focal_gamma: float = 2.0
    l1_focal_alpha: float = 0.75
    l1_per_protein_loss: bool = True
    l1_dice_w: float = 0.05
    l1_extreme_label_weight: float = 0.60
    l1_high_pos_frac: float = 0.95
    l1_high_pos_weight: float = 0.70
    l1_rank_w: float = 0.10
    l1_rank_start_epoch: int = 5
    l1_rank_margin: float = 0.20
    l1_rank_n_pairs: int = 1024
    l1_rank_neg_hard_frac: float = 0.75
    objective_weight_auto: bool = False
    topk_rank_boost: float = 1.00
    topk_rank_ramp_epochs: int = 10
    site_l3_pool_w: float = 0.0
    site_l3_pool_start_epoch: int = 2
    site_l3_pool_top_frac: float = 0.05
    site_l3_pool_pos_weight: float = 2.0
    # L1-L3 mainline: keep the fragment head for compatibility, but do not
    # train an L1.5 auxiliary objective by default.
    l15_w: float = 0.0
    l15_start_epoch: int = 999

    # Optional late-stage AP-oriented hard ranking.
    # Default is OFF because the aggressive version degraded validation AUPRC.
    # Use only for low-LR second-stage fine-tuning, e.g. ap_rank_w=0.03.
    ap_rank_w: float = 0.0
    ap_rank_start_epoch: int = 999
    ap_rank_margin: float = 0.05
    ap_rank_tau: float = 0.25
    ap_rank_pos_cap: int = 128
    ap_rank_neg_per_pos: int = 8

    # Keep fragment head as an auxiliary/export branch. Do NOT fuse it into
    # residue logits by default; direct fusion degraded the previous run.
    site_frag_logit_w: float = 0.0

    # ---- task-adaptive loss profile ----
    # auto: L3/pair task uses pair-decision-first loss; site/binary uses L1-first loss.
    task_loss_profile: str = "auto"        # auto | l3_main | l1_main

    # L3 protein-pair primary terms. These are active only in non-site pair mode.
    l3_pair_rank_w: float = 0.15
    l3_pair_rank_start_epoch: int = 3
    l3_pair_rank_ramp_epochs: int = 8
    l3_pair_rank_margin: float = 0.20
    l3_pair_rank_tau: float = 0.25
    l3_pair_rank_hard_frac: float = 0.75

    # L3 decision-boundary / disturbance terms. They support MCC/F1 and prioritisation.
    l3_tau_margin_w: float = 0.05
    l3_tau_margin_start_epoch: int = 5
    l3_tau_margin_ramp_epochs: int = 10
    l3_tau: float = 0.50
    l3_tau_margin: float = 0.05
    l3_disturb_w: float = 0.05
    l3_disturb_start_epoch: int = 5
    l3_disturb_ramp_epochs: int = 10
    l3_disturb_target: float = 0.15

    # In L3-main mode, L1 evidence is auxiliary rather than primary.
    l3_aux_l1_w: float = 0.15
    l3_aux_l1_start_epoch: int = 5
    l3_aux_l1_ramp_epochs: int = 10

    # Cross-level consistency: light bridge constraint, not the main decision loss.
    l3_cons_w: float = 0.03
    l3_cons_start_epoch: int = 8
    l3_cons_ramp_epochs: int = 10

    # In L1-main site mode, an optional bag-level L3 auxiliary can be used.
    l1_aux_l3_w: float = 0.03
    l1_aux_l3_start_epoch: int = 5

    # Validation selection for L3 pair mode.
    pair_primary_metric: str = "pair_auprc"   # pair_auprc | medauc
    pair_eval_make_negatives: bool = True

    # ---- innovation modules kept, but pair-level losses default off on current RBP labels ----
    eb_topk_k: int = 512
    eb_topk_frac: float = 0.10
    eb_pool_mode: str = "topk"
    eb_d_proj: int = 64
    eb_start_epoch: int = 3
    eb_ramp_epochs: int = 5
    eb_pair_topk: int = 256

    l3_w: float = 1.0
    l3_pos_weight: float = 0.0   # <=0 means estimate neg/pos from current pair batch
    l3_focal_w: float = 0.10
    l3_focal_gamma: float = 2.0
    l3_focal_alpha: float = 0.50
    l3_evi_w: float = 0.15
    l3_evi_margin: float = 0.3
    cl_cons_w: float = 0.03
    cl_cons_start_epoch: int = 5
    cl_cons_ramp_epochs: int = 8
    cl_cons_tau: float = 1.0
    cl_cons_margin: float = 0.5
    cl_cons_neg_suppress_w: float = 0.3
    cl_cons_pos_support_w: float = 0.4
    cl_cons_neg_pair_suppress_w: float = 0.2

    # ---- DIPS-specific MedAUC curriculum ----
    dips_hier_enable: bool = True
    dips_l1_w_scale: float = 1.0  # legacy only; task profile now controls L1 aux weight
    dips_l3_w_start: float = 0.0
    dips_l3_w_ramp_epochs: int = 0
    dips_l3_w_max: float = 1.0  # legacy only; task profile keeps L3 active
    dips_cl_w_start: float = 0.0
    dips_cl_w_ramp_epochs: int = 0
    dips_cl_w_max: float = 0.03
    dips_rank_start_epoch: int = 3
    l2_map_w: float = 0.0          # L3-pair primary default: keep L2/contact-map auxiliary OFF unless explicitly enabled
    l2_neg_per_pos: int = 24       # MedAUC boost: more sampled negatives per positive
    l2_neg_min: int = 512
    l2_neg_cap: int = 12000
    l2_hardneg_frac: float = 0.85  # keep some random negatives for stability
    l2_map_start_epoch: int = 0
    # 注意: l2_map_w_start 是 l2_map_w 的乘数(0~1)，不是绝对值！
    # l2_w_eff = l2_map_w * (l2_start + (1 - l2_start) * ramp_factor)
    # ep00(factor=1/6): 3.0*(0.15+0.85*0.167)=3.0*0.292=0.88  ← 安全起点
    # ep06+(factor=1):  3.0*(0.15+0.85*1.0)=3.0*1.0=3.0       ← 目标值
    l2_map_w_start: float = 0.15   # warm up from 15% of l2_map_w
    l2_map_w_ramp_epochs: int = 6
    l2_map_pos_weight: float = 3.0 # stronger positive contact weighting
    l2_rank_alpha: float = 1.0     # ranking signal for MedAUC
    l2_rank_margin: float = 0.50
    l2_rank_pairs: int = 1024
    l2_focus_topk: int = 768       # larger focus region for contact-map ranking
    l2_focus_frac: float = 0.65
    # MedAUC boost mode: enable inter-chain geometric prior.
    # For strict ablation/generalization, override with L2_GEOM_PRIOR_W=0.0.
    l2_geom_prior_w: float = 1.0
    l2_geom_prior_sigma: float = 2.0
    # Validation MedAUC uses the same geometry prior by default.
    # Override with EVAL_L2_GEOM_PRIOR_W=0.0 for strict no-geometry validation.
    eval_l2_geom_prior_w: float = 1.0

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
    ema_decay: float = 0.999
    early_min_epochs: int = 50
    early_patience: int = 28
    early_min_delta: float = 2e-4
    early_ema: float = 0.6

    # ---- validation / negatives ----
    val_thr_mode: str = "val_mcc"
    val_thr: float = 0.5
    val_thr_grid: int = 181
    val_thr_min: float = 0.05
    val_thr_max: float = 0.95
    val_recall_floor: float = 0.65
    val_fbeta_beta: float = 1.5
    eval_tta_swap: bool = False
    report_topk_metrics: bool = True
    contact_cutoff: float = 8.0
    train_neg_ratio: float = 1.0
    eval_neg_ratio: float = 1.0

    # ---- top-k ranking / AGER ----
    topk_primary: str = "precision_K10"
    topk_rank_max_pos_frac: float = 0.80
    topk_enrichment_cap: float = 10.0
    ager_enable: bool = False
    binary_ager_eval: bool = False
    ager_radius: float = 10.0
    ager_alpha: float = 0.25
    ager_top_m: int = 5

    # ---- GVP ----
    gvp_node_s_dim: int = 6
    gvp_node_v_dim: int = 2
    d_model: int = 256
    n_encoder_layers: int = 3
    n_cross_layers: int = 0             # stable baseline: disabled
    n_heads: int = 8
    dropout: float = 0.15
    gvp_hidden_s: int = 64
    gvp_hidden_v: int = 8
    gvp_layers: int = 2
    gvp_k: int = 16
    gvp_dropout: float = 0.10
    gvp_chain_dim: int = 32
    site_graph_layers: int = 0          # stable baseline: disabled
    site_graph_k: int = 12
    site_graph_radius: float = 12.0
    site_head_type: str = "amleh"
    site_ms_channels: int = 64
    site_ms_dropout: float = 0.10
    site_ms_delta_init: float = 0.03
    site_ms_use_scale_gate: bool = True
    site_ms_use_channel_gate: bool = True

    # ---- compact L1->L3 evidence chain ----
    eb_enable: bool = True
    eb_topk: int = 16
    eb_dropout: float = 0.10
    eb_use_pair_context: bool = True
    pair_head_type: str = "gpeh"
    pair_transformer_layers: int = 2
    pair_transformer_heads: int = 4
    pair_dropout: float = 0.10
    pair_use_product_token: bool = True
    pair_use_diff_token: bool = True
    consistency_w: float = 0.03

    # ---- misc ----
    small_tr_n: int = 0
    small_va_n: int = 0

    @classmethod
    def from_env(cls, p=None):
        if p is None:
            p = cls()
        spec = [
            ("dataset_mode", _env_str), ("primary_objective", _env_str),
            ("dips_root", _env_str), ("rbp_root", _env_str), ("rbp_id_list", _env_str),
            ("rbp_train_list", _env_str), ("rbp_val_list", _env_str), ("rbp_test_list", _env_str),
            ("rbp_official_test_list", _env_str),
            ("dips_train_list", _env_str), ("dips_val_list", _env_str), ("dips_test_list", _env_str),
            ("save_dir", _env_str), ("esm_local_dir", _env_str), ("rbp_structure_dir", _env_str),
            ("rbp_dssp_dir", _env_str), ("dssp_dim", _env_int), ("surface_mask_dir", _env_str),
            ("binary_primary_metric", _env_str), ("eval_ca_surface_radius", _env_float), ("eval_ca_surface_cutoffs", _env_str),
            ("split_train", _env_float), ("split_val", _env_float), ("split_test", _env_float), ("split_seed", _env_int), ("split_search_trials", _env_int),
            ("use_pssm", _env_bool), ("use_dssp", _env_bool), ("use_hmm", _env_bool), ("use_ss", _env_bool),
            ("sequence_mode", _env_str), ("esm_cache_dir", _env_str),
            ("allow_zero_esm_fallback", _env_bool), ("dips_use_embedder_on_miss", _env_bool),
            ("structure_source", _env_str), ("use_chain_geom", _env_bool), ("use_geom", _env_bool), ("site_self_cross", _env_bool),
            ("dips_skip_filter", _env_bool), ("dips_filter_cache", _env_bool),
            ("dips_index_verbose", _env_int), ("dips_esm_miss_log_limit", _env_int),
            ("seed", _env_int), ("epochs", _env_int), ("batch_site", _env_int), ("eval_batch_site", _env_int), ("num_workers", _env_int),
            ("allow_cuda_workers", _env_bool), ("prefetch_factor", _env_int),
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
            ("ap_rank_w", _env_float), ("ap_rank_start_epoch", _env_int),
            ("ap_rank_margin", _env_float), ("ap_rank_tau", _env_float),
            ("ap_rank_pos_cap", _env_int), ("ap_rank_neg_per_pos", _env_int),
            ("site_frag_logit_w", _env_float),
            ("task_loss_profile", _env_str),
            ("l3_pair_rank_w", _env_float), ("l3_pair_rank_start_epoch", _env_int),
            ("l3_pair_rank_ramp_epochs", _env_int), ("l3_pair_rank_margin", _env_float),
            ("l3_pair_rank_tau", _env_float), ("l3_pair_rank_hard_frac", _env_float),
            ("l3_tau_margin_w", _env_float), ("l3_tau_margin_start_epoch", _env_int),
            ("l3_tau_margin_ramp_epochs", _env_int), ("l3_tau", _env_float), ("l3_tau_margin", _env_float),
            ("l3_disturb_w", _env_float), ("l3_disturb_start_epoch", _env_int),
            ("l3_disturb_ramp_epochs", _env_int), ("l3_disturb_target", _env_float),
            ("l3_aux_l1_w", _env_float), ("l3_aux_l1_start_epoch", _env_int), ("l3_aux_l1_ramp_epochs", _env_int),
            ("l3_cons_w", _env_float), ("l3_cons_start_epoch", _env_int), ("l3_cons_ramp_epochs", _env_int),
            ("l1_aux_l3_w", _env_float), ("l1_aux_l3_start_epoch", _env_int),
            ("pair_primary_metric", _env_str), ("pair_eval_make_negatives", _env_bool),
            ("eb_topk_k", _env_int), ("eb_topk_frac", _env_float), ("eb_pool_mode", _env_str),
            ("eb_d_proj", _env_int), ("eb_start_epoch", _env_int), ("eb_ramp_epochs", _env_int), ("eb_pair_topk", _env_int),
            ("l3_w", _env_float), ("l3_pos_weight", _env_float), ("l3_focal_w", _env_float),
            ("l3_focal_gamma", _env_float), ("l3_focal_alpha", _env_float), ("l3_evi_w", _env_float), ("l3_evi_margin", _env_float),
            ("cl_cons_w", _env_float), ("cl_cons_start_epoch", _env_int), ("cl_cons_ramp_epochs", _env_int),
            ("cl_cons_tau", _env_float), ("cl_cons_margin", _env_float), ("cl_cons_neg_suppress_w", _env_float),
            ("cl_cons_pos_support_w", _env_float), ("cl_cons_neg_pair_suppress_w", _env_float),
            ("dips_hier_enable", _env_bool), ("dips_l1_w_scale", _env_float),
            ("dips_l3_w_start", _env_float), ("dips_l3_w_ramp_epochs", _env_int),
            ("dips_l3_w_max", _env_float),
            ("dips_cl_w_start", _env_float), ("dips_cl_w_ramp_epochs", _env_int),
            ("dips_cl_w_max", _env_float),
            ("dips_rank_start_epoch", _env_int),
            ("l2_map_w", _env_float), ("l2_neg_per_pos", _env_int), ("l2_neg_min", _env_int),
            ("l2_neg_cap", _env_int), ("l2_hardneg_frac", _env_float), ("l2_map_start_epoch", _env_int),
            ("l2_map_w_start", _env_float), ("l2_map_w_ramp_epochs", _env_int), ("l2_map_pos_weight", _env_float),
            ("l2_rank_alpha", _env_float), ("l2_rank_margin", _env_float), ("l2_rank_pairs", _env_int),
            ("l2_focus_topk", _env_int), ("l2_focus_frac", _env_float),
            ("l2_geom_prior_w", _env_float), ("l2_geom_prior_sigma", _env_float),
            ("eval_l2_geom_prior_w", _env_float),
            ("max_res_loss", _env_float), ("max_frag_loss", _env_float), ("max_l3_loss", _env_float), ("max_cons_loss", _env_float),
            ("lr_sched", _env_bool), ("lr_warmup_frac", _env_float), ("lr_warmup_start_ratio", _env_float),
            ("lr_warmup_min_updates", _env_int), ("lr_warmup_max_updates", _env_int), ("lr_min_ratio", _env_float),
            ("ema_decay", _env_float), ("early_min_epochs", _env_int), ("early_patience", _env_int), ("early_min_delta", _env_float), ("early_ema", _env_float),
            ("val_thr_mode", _env_str), ("val_thr", _env_float), ("val_thr_grid", _env_int), ("val_thr_min", _env_float), ("val_thr_max", _env_float),
            ("val_recall_floor", _env_float), ("val_fbeta_beta", _env_float),
            ("eval_tta_swap", _env_bool), ("report_topk_metrics", _env_bool), ("contact_cutoff", _env_float),
            ("split_strategy", _env_str), ("save_epoch_ckpts", _env_bool),
            ("train_neg_ratio", _env_float), ("eval_neg_ratio", _env_float),
            ("topk_primary", _env_str), ("ager_enable", _env_bool), ("binary_ager_eval", _env_bool), ("ager_radius", _env_float),
            ("ager_alpha", _env_float), ("ager_top_m", _env_int),
            ("topk_rank_max_pos_frac", _env_float), ("topk_enrichment_cap", _env_float),
            ("d_model", _env_int), ("n_encoder_layers", _env_int), ("n_cross_layers", _env_int), ("n_heads", _env_int), ("dropout", _env_float),
            ("gvp_hidden_s", _env_int), ("gvp_hidden_v", _env_int), ("gvp_layers", _env_int), ("gvp_k", _env_int),
            ("gvp_dropout", _env_float), ("gvp_chain_dim", _env_int),
            ("site_graph_layers", _env_int), ("site_graph_k", _env_int), ("site_graph_radius", _env_float),
            ("site_head_type", _env_str), ("site_ms_channels", _env_int), ("site_ms_dropout", _env_float),
            ("site_ms_delta_init", _env_float), ("site_ms_use_scale_gate", _env_bool), ("site_ms_use_channel_gate", _env_bool),
            ("eb_enable", _env_bool), ("eb_topk", _env_int), ("eb_dropout", _env_float), ("eb_use_pair_context", _env_bool),
            ("pair_head_type", _env_str), ("pair_transformer_layers", _env_int), ("pair_transformer_heads", _env_int),
            ("pair_dropout", _env_float), ("pair_use_product_token", _env_bool), ("pair_use_diff_token", _env_bool),
            ("consistency_w", _env_float),
            ("small_tr_n", _env_int), ("small_va_n", _env_int),
        ]
        for name, fn in spec:
            setattr(p, name, fn(name.upper(), getattr(p, name)))
        p.dataset_mode = str(p.dataset_mode).lower()
        p.primary_objective = str(p.primary_objective).lower()
        p.task_loss_profile = str(getattr(p, "task_loss_profile", "auto")).lower()
        p.pair_primary_metric = str(getattr(p, "pair_primary_metric", "pair_auprc")).lower()
        p.sequence_mode = str(p.sequence_mode).lower()
        p.structure_source = str(p.structure_source).lower()

        # ---- custom/current data override ----
        # Data source, task type and loss profile must be independent.
        # DATA_ROOT/CUSTOM_DATA_ROOT changes the current dataset root without
        # implicitly switching to DIPS or changing primary_objective.
        data_root = (
            os.environ.get("DATA_ROOT")
            or os.environ.get("CUSTOM_DATA_ROOT")
            or os.environ.get("PP_PREPARED_ROOT")
            or ""
        )
        if str(data_root).strip():
            root = os.path.abspath(str(data_root).strip())
            p.rbp_root = root
            p.dips_root = root
            p.rbp_id_list = os.environ.get("ID_LIST", os.path.join(root, "all_ids.txt"))
            p.rbp_train_list = os.environ.get("TRAIN_LIST", os.path.join(root, "train.txt"))
            p.rbp_val_list = os.environ.get("VAL_LIST", os.path.join(root, "val.txt"))
            p.rbp_test_list = os.environ.get("TEST_LIST", os.path.join(root, "test.txt"))
            p.rbp_official_test_list = os.environ.get("OFFICIAL_TEST_LIST", os.path.join(root, "test_all.txt"))
            p.dips_train_list = p.rbp_train_list
            p.dips_val_list = p.rbp_val_list
            p.dips_test_list = p.rbp_test_list

        # train_L13_medauc_diagnostic.py reads this value from the environment during evaluation.
        # Setting it here makes the config file self-contained.
        os.environ.setdefault("EVAL_L2_GEOM_PRIOR_W", str(p.eval_l2_geom_prior_w))
        return p


@dataclass
class ModelConfig:
    d_seq_in: int = 1309
    d_chain_in: int = 12
    d_model: int = 256
    n_encoder_layers: int = 3
    n_cross_layers: int = 0             # stable baseline: disabled
    n_heads: int = 8
    dropout: float = 0.15
    use_layerscale: bool = True
    layerscale_init: float = 1e-4
    use_swiglu: bool = True
    ffn_mult: float = 4.0
    eb_topk_k: int = 512
    eb_topk_frac: float = 0.10
    eb_d_proj: int = 64
    eb_n_heads: int = 4
    eb_pair_topk: int = 256
    eb_start_epoch: int = 3
    eb_ramp_epochs: int = 5
    l2_d_proj: int = 64
    l2_pair_topk: int = 256
    l2_focus_topk: int = 768
    l2_focus_frac: float = 0.65
    l2_geom_prior_w: float = 1.0   # MedAUC boost default; Params can override
    l2_geom_prior_sigma: float = 2.0
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
    site_self_cross: bool = False       # stable baseline: disabled
    site_graph_layers: int = 0          # stable baseline: disabled
    site_graph_k: int = 12
    site_graph_radius: float = 12.0
    site_frag_logit_w: float = 0.0
    site_head_type: str = "amleh"
    site_ms_channels: int = 64
    site_ms_dropout: float = 0.10
    site_ms_delta_init: float = 0.03
    site_ms_use_scale_gate: bool = True
    site_ms_use_channel_gate: bool = True
    eb_enable: bool = True
    eb_topk: int = 16
    eb_dropout: float = 0.10
    eb_use_pair_context: bool = True
    pair_head_type: str = "gpeh"
    pair_transformer_layers: int = 2
    pair_transformer_heads: int = 4
    pair_dropout: float = 0.10
    pair_use_product_token: bool = True
    pair_use_diff_token: bool = True
    consistency_w: float = 0.03

    def get(self, key, default=None):
        return getattr(self, key, default)


def build_model_config(p: Params, d_seq_in: int, d_chain_in: int) -> ModelConfig:
    return ModelConfig(
        d_seq_in=int(d_seq_in),
        d_chain_in=int(d_chain_in),
        d_model=int(getattr(p, "d_model", 256)),
        n_encoder_layers=int(getattr(p, "n_encoder_layers", 3)),
        n_cross_layers=int(getattr(p, "n_cross_layers", 0)),
        n_heads=int(getattr(p, "n_heads", 8)),
        dropout=float(getattr(p, "dropout", 0.15)),
        eb_topk_k=int(p.eb_topk_k),
        eb_topk_frac=float(p.eb_topk_frac),
        eb_d_proj=int(p.eb_d_proj),
        eb_pair_topk=int(p.eb_pair_topk),
        eb_start_epoch=int(p.eb_start_epoch),
        eb_ramp_epochs=int(p.eb_ramp_epochs),
        l2_d_proj=int(p.eb_d_proj),
        l2_pair_topk=int(p.eb_pair_topk),
        l2_focus_topk=int(p.l2_focus_topk),
        l2_focus_frac=float(p.l2_focus_frac),
        l2_geom_prior_w=float(p.l2_geom_prior_w),
        l2_geom_prior_sigma=float(p.l2_geom_prior_sigma),
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
        site_self_cross=bool(getattr(p, "site_self_cross", False)),
        site_graph_layers=int(getattr(p, "site_graph_layers", 0)),
        site_graph_k=int(getattr(p, "site_graph_k", 12)),
        site_graph_radius=float(getattr(p, "site_graph_radius", 12.0)),
        site_frag_logit_w=float(getattr(p, "site_frag_logit_w", 0.0)),
        site_head_type=str(getattr(p, "site_head_type", "linear")).lower(),
        site_ms_channels=int(getattr(p, "site_ms_channels", 64)),
        site_ms_dropout=float(getattr(p, "site_ms_dropout", 0.10)),
        site_ms_delta_init=float(getattr(p, "site_ms_delta_init", 0.03)),
        site_ms_use_scale_gate=bool(getattr(p, "site_ms_use_scale_gate", True)),
        site_ms_use_channel_gate=bool(getattr(p, "site_ms_use_channel_gate", True)),
        eb_enable=bool(getattr(p, "eb_enable", True)),
        eb_topk=int(getattr(p, "eb_topk", 16)),
        eb_dropout=float(getattr(p, "eb_dropout", 0.10)),
        eb_use_pair_context=bool(getattr(p, "eb_use_pair_context", True)),
        pair_head_type=str(getattr(p, "pair_head_type", "gpeh")).lower(),
        pair_transformer_layers=int(getattr(p, "pair_transformer_layers", 2)),
        pair_transformer_heads=int(getattr(p, "pair_transformer_heads", 4)),
        pair_dropout=float(getattr(p, "pair_dropout", 0.10)),
        pair_use_product_token=bool(getattr(p, "pair_use_product_token", True)),
        pair_use_diff_token=bool(getattr(p, "pair_use_diff_token", True)),
        consistency_w=float(getattr(p, "consistency_w", 0.03)),
    )
