# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from typing import Optional, Dict, Tuple
from .config import ModelConfig

def _bmask(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if x.dtype == torch.bool:
        return x
    return x > 0.5


class FiLM(nn.Module):
    """"""

    def __init__(self, d_model: int, d_chain: int):
        super().__init__()
        dc = max(8, d_chain if d_chain > 0 else 8)
        self.proj = nn.Sequential(
            nn.Linear(dc, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * 2),
        )

    def forward(self, x: torch.Tensor, chain_pooled: Optional[torch.Tensor]) -> torch.Tensor:
        if chain_pooled is None:
            return x
        B, L, D = x.shape
        g = self.proj(chain_pooled)
        gamma, beta = g.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1).expand(-1, L, -1)
        beta = beta.unsqueeze(1).expand(-1, L, -1)
        gamma = torch.tanh(gamma) * 0.5
        return x * (1.0 + gamma) + beta * 0.1


class LayerScale(nn.Module):
    def __init__(self, d: int, init_value: float = 1e-4):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d) * float(init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class SwiGLUFFN(nn.Module):
    def __init__(self, d: int, mult: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = max(32, int(d * float(mult) * 2.0 / 3.0))
        self.w1 = nn.Linear(d, hidden)
        self.w2 = nn.Linear(d, hidden)
        self.w3 = nn.Linear(hidden, d)
        self.drop = nn.Dropout(float(drop))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w1(x) * F.silu(self.w2(x))
        x = self.drop(x)
        return self.w3(x)


class SelfEnc(nn.Module):
    """"""

    def __init__(self, d: int, n_layers: int, h: int, drop: float):
        super().__init__()
        if n_layers <= 0:
            self.mod = nn.Identity()
        else:
            layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=h, dim_feedforward=d * 4,
                dropout=drop, activation="gelu",
                batch_first=True, norm_first=True,
            )
            self.mod = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if isinstance(self.mod, nn.Identity):
            return x
        kpm = None
        if mask is not None:
            kpm = ~_bmask(mask)
        return self.mod(x, src_key_padding_mask=kpm)


class CrossBlock(nn.Module):
    """"""

    def __init__(self, d, h, drop, use_layerscale=False, layerscale_init=1e-4,
                 use_swiglu=False, ffn_mult=4.0):
        super().__init__()
        self.use_layerscale = bool(use_layerscale)
        self.use_swiglu = bool(use_swiglu)

        self.qA_kB = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.qB_kA = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.saA = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.saB = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)

        if self.use_swiglu:
            self.ffA = SwiGLUFFN(d, mult=ffn_mult, drop=drop)
            self.ffB = SwiGLUFFN(d, mult=ffn_mult, drop=drop)
        else:
            self.ffA = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))
            self.ffB = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))

        self.norm1A = nn.LayerNorm(d); self.norm1B = nn.LayerNorm(d)
        self.norm2A = nn.LayerNorm(d); self.norm2B = nn.LayerNorm(d)
        self.norm3A = nn.LayerNorm(d); self.norm3B = nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)

        if self.use_layerscale:
            self.res_scale = 1.0
            self.ls_crossA = LayerScale(d, init_value=layerscale_init)
            self.ls_crossB = LayerScale(d, init_value=layerscale_init)
            self.ls_selfA  = LayerScale(d, init_value=layerscale_init)
            self.ls_selfB  = LayerScale(d, init_value=layerscale_init)
            self.ls_ffA    = LayerScale(d, init_value=layerscale_init)
            self.ls_ffB    = LayerScale(d, init_value=layerscale_init)
        else:
            self.res_scale = 0.5
            self.ls_crossA = self.ls_crossB = nn.Identity()
            self.ls_selfA  = self.ls_selfB  = nn.Identity()
            self.ls_ffA    = self.ls_ffB    = nn.Identity()

    def forward(self, xA, xB, mA=None, mB=None):
        kpmA = None if mA is None else ~_bmask(mA)
        kpmB = None if mB is None else ~_bmask(mB)

        nA, nB = self.norm1A(xA), self.norm1B(xB)
        hA, _ = self.qA_kB(nA, nB, nB, key_padding_mask=kpmB, need_weights=False)
        hB, _ = self.qB_kA(nB, nA, nA, key_padding_mask=kpmA, need_weights=False)
        xA = xA + self.drop(self.ls_crossA(hA)) * self.res_scale
        xB = xB + self.drop(self.ls_crossB(hB)) * self.res_scale

        nA, nB = self.norm2A(xA), self.norm2B(xB)
        zA, _ = self.saA(nA, nA, nA, key_padding_mask=kpmA, need_weights=False)
        zB, _ = self.saB(nB, nB, nB, key_padding_mask=kpmB, need_weights=False)
        xA = xA + self.drop(self.ls_selfA(zA)) * self.res_scale
        xB = xB + self.drop(self.ls_selfB(zB)) * self.res_scale

        uA = self.ffA(self.norm3A(xA))
        uB = self.ffB(self.norm3B(xB))
        xA = xA + self.drop(self.ls_ffA(uA)) * self.res_scale
        xB = xB + self.drop(self.ls_ffB(uB)) * self.res_scale

        return xA, xB


class CrossEnc(nn.Module):
    def __init__(self, d, n_layers, h, drop, use_layerscale=False,
                 layerscale_init=1e-4, use_swiglu=False, ffn_mult=4.0):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossBlock(d, h, drop, use_layerscale=use_layerscale,
                       layerscale_init=layerscale_init, use_swiglu=use_swiglu,
                       ffn_mult=ffn_mult)
            for _ in range(n_layers)
        ])

    def forward(self, xA, xB, mA=None, mB=None):
        def run_block(blk, xA, xB, mA, mB):
            return blk(xA, xB, mA, mB)

        for blk in self.layers:
            if self.training and xA.requires_grad:
                xA, xB = checkpoint.checkpoint(run_block, blk, xA, xB, mA, mB, use_reentrant=False)
            else:
                xA, xB = blk(xA, xB, mA, mB)
        return xA, xB


class FragmentHead(nn.Module):
    """Depthwise Separable Conv head for fragment/segment prediction.

    Uses GroupNorm instead of BatchNorm1d so that it works correctly with
    any batch size (including B=1) and any sequence length (including L=1),
    and is consistent between train and eval modes.
    """

    def __init__(self, d_model: int, kernel_size: int = 9, dropout: float = 0.1):
        super().__init__()
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size,
                                   padding=kernel_size // 2, groups=d_model, bias=False)
        # Bug C fix: replace BatchNorm1d with GroupNorm.
        # train/eval behaviour differs. GroupNorm normalises per-channel within
        # each sample independently, which is always stable regardless of B or L.
        # num_groups=min(32, d_model) is a good default for d_model=384.
        n_groups = min(32, d_model)
        self.gn1 = nn.GroupNorm(num_groups=n_groups, num_channels=d_model)
        self.pointwise = nn.Conv1d(d_model, d_model, 1, bias=True)
        self.gn2 = nn.GroupNorm(num_groups=n_groups, num_channels=d_model)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.proj = nn.Linear(d_model, 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.depthwise.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.pointwise.weight)
        nn.init.zeros_(self.pointwise.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        B, L, D = x.shape
        xt = x.transpose(1, 2)          # [B, D, L]
        xt = F.gelu(self.gn1(self.depthwise(xt)))
        xt = F.gelu(self.gn2(self.pointwise(xt)))
        xt = self.drop(xt)
        xout = xt.transpose(1, 2)       # [B, L, D]
        if mask is not None:
            xout = xout * mask.unsqueeze(-1).float()
        return self.proj(xout).squeeze(-1)


# ============================================================
# ============================================================


class EvidenceBridge(nn.Module):
    """
     L1 residue evidence  residue-pair evidence
     L3  pair-aware evidence vector

    :
      1.  Top-K residue 
      2.  Top-K  Top-K  residue-pair 
      3.  residue-pair ranking case study 
    """

    def __init__(self, d_model: int, d_proj: int = 64, n_heads: int = 4,
                 k: int = 512, frac: float = 0.05, pair_k: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.k = int(k)
        self.frac = float(frac)
        self.d_proj = int(d_proj)
        self.pair_k = int(pair_k)

        self.projA = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_proj))
        self.projB = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_proj))
        self.gate_normA = nn.LayerNorm(d_proj)
        self.gate_normB = nn.LayerNorm(d_proj)

        self.pair_norm = nn.LayerNorm(d_proj * 2)
        self.pair_proj = nn.Sequential(
            nn.Linear(d_proj * 2, d_proj * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.evi_scalar = nn.Sequential(
            nn.Linear(d_proj * 2, d_proj),
            nn.GELU(),
            nn.Linear(d_proj, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _topk_select(
            logits: torch.Tensor,
            feat: torch.Tensor,
            mask: Optional[torch.Tensor],
            k: int,
            frac: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L = logits.shape
        K = int(k)
        if frac > 0:
            Kf = max(1, int(round(frac * L)))
            K = min(K, Kf) if K > 0 else Kf
        K = max(1, min(K, L))

        masked = logits.clone()
        if mask is not None:
            masked = masked.masked_fill(~_bmask(mask), -1e9)

        val, idx = torch.topk(masked, k=K, dim=1, largest=True, sorted=False)
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, feat.size(-1))
        topk_feat = feat.gather(1, idx_exp)
        return topk_feat, val, idx

    def forward(
            self,
            logit_A: torch.Tensor,
            logit_B: torch.Tensor,
            xA: torch.Tensor,
            xB: torch.Tensor,
            maskA: Optional[torch.Tensor] = None,
            maskB: Optional[torch.Tensor] = None,
            gate_factor: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        if gate_factor is None:
            gate_factor = xA.new_tensor(1.0)
        if not torch.is_tensor(gate_factor):
            gate_factor = xA.new_tensor(float(gate_factor))
        gate_factor = gate_factor.to(dtype=xA.dtype, device=xA.device)

        fA = self.projA(xA)
        fB = self.projB(xB)

        gateA = torch.sigmoid(logit_A).unsqueeze(-1)
        gateB = torch.sigmoid(logit_B).unsqueeze(-1)
        fA_gated = self.gate_normA(fA * gateA * gate_factor)
        fB_gated = self.gate_normB(fB * gateB * gate_factor)

        topk_fA, topk_valA, topk_idxA = self._topk_select(
            logit_A, fA_gated, maskA, self.k, self.frac
        )
        topk_fB, topk_valB, topk_idxB = self._topk_select(
            logit_B, fB_gated, maskB, self.k, self.frac
        )

        nA = F.normalize(topk_fA, dim=-1)
        nB = F.normalize(topk_fB, dim=-1)
        compat = torch.einsum("bkd,bqd->bkq", nA, nB) / math.sqrt(max(1, self.d_proj))
        unary = 0.5 * (topk_valA.unsqueeze(2) + topk_valB.unsqueeze(1))
        pair_score = compat + unary

        B, KA, KB = pair_score.shape
        flat = pair_score.reshape(B, KA * KB)
        pair_k = max(1, min(self.pair_k, flat.size(1)))
        top_pair_val, top_pair_flat = torch.topk(flat, k=pair_k, dim=1, largest=True, sorted=True)

        pair_idxA_local = torch.div(top_pair_flat, KB, rounding_mode="floor")
        pair_idxB_local = top_pair_flat % KB

        gatherA = pair_idxA_local.unsqueeze(-1).expand(-1, -1, topk_fA.size(-1))
        gatherB = pair_idxB_local.unsqueeze(-1).expand(-1, -1, topk_fB.size(-1))
        pair_featA = topk_fA.gather(1, gatherA)
        pair_featB = topk_fB.gather(1, gatherB)
        pair_feat = torch.cat([pair_featA, pair_featB], dim=-1)

        pair_weight = torch.softmax(top_pair_val, dim=1).unsqueeze(-1)
        evi_vec = (pair_feat * pair_weight).sum(dim=1)
        evi_vec = self.pair_proj(self.pair_norm(evi_vec))

        evi_score = self.evi_scalar(evi_vec).squeeze(-1)
        evi_score = torch.nan_to_num(evi_score, nan=0.0, posinf=10.0, neginf=-10.0)

        residue_idxA = topk_idxA.gather(1, pair_idxA_local)
        residue_idxB = topk_idxB.gather(1, pair_idxB_local)
        compactness = (top_pair_val[:, 0] - top_pair_val.mean(dim=1)).detach()

        topk_info = {
            "topk_idxA": topk_idxA,
            "topk_idxB": topk_idxB,
            "topk_valA": topk_valA,
            "topk_valB": topk_valB,
            "pair_idxA_local": pair_idxA_local,
            "pair_idxB_local": pair_idxB_local,
            "pair_residue_idxA": residue_idxA,
            "pair_residue_idxB": residue_idxB,
            "pair_score": top_pair_val,
            "pair_compactness": compactness,
            "evi_score": evi_score.detach(),
        }
        return evi_vec, evi_score, topk_info


class L2Bridge(nn.Module):
    """
    Explicit L2 bridge: build a dense residue-pair interaction map S [B, La, Lb]
    from cross-encoded residue features, then compress the top interaction pairs
    into a vector that can be passed upward to L3.
    """

    def __init__(self, d_model: int, d_proj: int = 64, pair_k: int = 128,
                 focus_k: int = 192, focus_frac: float = 0.20, dropout: float = 0.1,
                 geom_prior_w: float = 0.0, geom_prior_sigma: float = 4.0):
        super().__init__()
        self.d_proj = int(d_proj)
        self.pair_k = int(pair_k)
        self.focus_k = int(focus_k)
        self.focus_frac = float(focus_frac)
        self.geom_prior_w = float(geom_prior_w)
        self.geom_prior_sigma = float(max(1e-3, geom_prior_sigma))
        self.projA = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_proj))
        self.projB = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_proj))
        self.fuse = nn.Sequential(
            nn.LayerNorm(d_proj * 2),
            nn.Linear(d_proj * 2, d_proj * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pair_score_head = nn.Sequential(
            nn.LayerNorm(d_proj * 2 + 3),
            nn.Linear(d_proj * 2 + 3, d_proj),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_proj, 1),
        )
        self.score_head = nn.Sequential(
            nn.Linear(d_proj * 2, d_proj),
            nn.GELU(),
            nn.Linear(d_proj, 1),
        )

    @staticmethod
    def _topk_select(logits: torch.Tensor, feat: torch.Tensor, mask: Optional[torch.Tensor],
                     k: int, frac: float):
        B, L = logits.shape
        K = int(k)
        if frac > 0:
            Kf = max(1, int(round(frac * L)))
            K = min(K, Kf) if K > 0 else Kf
        K = max(1, min(K, L))
        masked = logits.clone()
        if mask is not None:
            masked = masked.masked_fill(~_bmask(mask), -1e9)
        val, idx = torch.topk(masked, k=K, dim=1, largest=True, sorted=False)
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, feat.size(-1))
        top_feat = feat.gather(1, idx_exp)
        return top_feat, val, idx

    def _geom_prior(self, coordsA, coordsB, maskA=None, maskB=None):
        if self.geom_prior_w <= 0.0 or coordsA is None or coordsB is None:
            return None
        if coordsA.dim() == 2:
            coordsA = coordsA.unsqueeze(0)
        if coordsB.dim() == 2:
            coordsB = coordsB.unsqueeze(0)
        dist = torch.cdist(coordsA.float(), coordsB.float()).to(coordsA.dtype)
        prior = (8.0 - dist) / self.geom_prior_sigma
        prior = prior.clamp(-4.0, 4.0) * self.geom_prior_w
        if maskA is not None and maskB is not None:
            valid2d = _bmask(maskA).unsqueeze(-1) & _bmask(maskB).unsqueeze(1)
            prior = prior.masked_fill(~valid2d, 0.0)
        return prior

    def forward(self, logit_A, logit_B, xA, xB, maskA=None, maskB=None, coordsA=None, coordsB=None):
        pA = self.projA(xA)
        pB = self.projB(xB)
        unary_full = 0.5 * (logit_A.unsqueeze(2) + logit_B.unsqueeze(1))
        geom_full = self._geom_prior(coordsA, coordsB, maskA, maskB)
        if geom_full is not None:
            unary_full = unary_full + geom_full.to(dtype=unary_full.dtype)
        topA, topA_val, idxA = self._topk_select(logit_A, pA, maskA, self.focus_k, self.focus_frac)
        topB, topB_val, idxB = self._topk_select(logit_B, pB, maskB, self.focus_k, self.focus_frac)
        nA = F.normalize(topA, dim=-1)
        nB = F.normalize(topB, dim=-1)
        compat = torch.einsum("bid,bjd->bij", nA, nB) / math.sqrt(max(1, self.d_proj))
        unary = 0.5 * (topA_val.unsqueeze(2) + topB_val.unsqueeze(1))
        pair_feat_full = torch.cat([
            topA.unsqueeze(2).expand(-1, -1, topB.size(1), -1),
            topB.unsqueeze(1).expand(-1, topA.size(1), -1, -1),
            topA_val.unsqueeze(2).unsqueeze(-1).expand(-1, -1, topB.size(1), -1),
            topB_val.unsqueeze(1).unsqueeze(-1).expand(-1, topA.size(1), -1, -1),
            compat.unsqueeze(-1),
        ], dim=-1)
        S_focus = self.pair_score_head(pair_feat_full).squeeze(-1) + 0.25 * unary
        if geom_full is not None:
            bidx = torch.arange(geom_full.size(0), device=geom_full.device).view(-1, 1, 1)
            gA = idxA.unsqueeze(2).expand(-1, -1, idxB.size(1))
            gB = idxB.unsqueeze(1).expand(-1, idxA.size(1), -1)
            S_focus = S_focus + geom_full[bidx, gA, gB].to(dtype=S_focus.dtype)

        B, Ka, Kb = S_focus.shape
        La = xA.size(1)
        Lb = xB.size(1)
        S = unary_full.clone()
        if maskA is not None and maskB is not None:
            valid2d = _bmask(maskA).unsqueeze(-1) & _bmask(maskB).unsqueeze(1)
            S = S.masked_fill(~valid2d, -20.0)
        else:
            valid2d = None
        focus2d = torch.zeros((B, La, Lb), dtype=torch.bool, device=xA.device)
        batch_idx = torch.arange(B, device=xA.device).view(B, 1, 1).expand(-1, Ka, Kb)
        srcA = idxA.unsqueeze(2).expand(-1, -1, Kb)
        srcB = idxB.unsqueeze(1).expand(-1, Ka, -1)
        S[batch_idx, srcA, srcB] = S_focus.to(dtype=S.dtype)
        focus2d[batch_idx, srcA, srcB] = True

        flat = S_focus.reshape(B, Ka * Kb)
        pair_k = max(1, min(self.pair_k, flat.size(1)))
        top_val, top_flat = torch.topk(flat, k=pair_k, dim=1, largest=True, sorted=True)
        pair_idxA_local = torch.div(top_flat, Kb, rounding_mode="floor")
        pair_idxB_local = top_flat % Kb
        gA = pair_idxA_local.unsqueeze(-1).expand(-1, -1, topA.size(-1))
        gB = pair_idxB_local.unsqueeze(-1).expand(-1, -1, topB.size(-1))
        pair_feat = torch.cat([topA.gather(1, gA), topB.gather(1, gB)], dim=-1)
        pair_w = torch.softmax(top_val, dim=1).unsqueeze(-1)
        l2_vec = self.fuse((pair_feat * pair_w).sum(dim=1))
        l2_score = self.score_head(l2_vec).squeeze(-1)
        l2_score = torch.nan_to_num(l2_score, nan=0.0, posinf=10.0, neginf=-10.0)
        info = {
            "focus_idxA": idxA,
            "focus_idxB": idxB,
            "pair_idxA_local": pair_idxA_local,
            "pair_idxB_local": pair_idxB_local,
            "pair_score": top_val,
            "l2_score": l2_score.detach(),
            "focus_mask2d": focus2d,
        }
        return S, l2_vec, l2_score, info


# ============================================================
# ============================================================

class L3Head(nn.Module):
    """
    Protein-level interaction head.

    :
      global_A:  [B, D]        chain A global representation (mean pool)
      global_B:  [B, D]        chain B global representation (mean pool)
      evi_vec:   [B, 2*d_proj] pair-aware evidence vector from EvidenceBridge

    :
      pair_logit: [B]   protein-level interaction logit ()
      : evi_score  EvidenceBridge ()
    """

    def __init__(self, d_model: int, d_proj: int = 64, d_hidden: int = 256,
                 n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        d_in = d_model * 2 + d_proj * 2
        layers = []
        d_cur = d_in
        for i in range(n_layers):
            d_next = d_hidden if i < n_layers - 1 else d_hidden // 2
            layers += [nn.Linear(d_cur, d_next), nn.GELU(), nn.Dropout(dropout)]
            d_cur = d_next
        layers += [nn.Linear(d_cur, 1)]
        self.mlp = nn.Sequential(*layers)
        self.norm_in = nn.LayerNorm(d_in)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
            self,
            global_A: torch.Tensor,
            global_B: torch.Tensor,
            evi_vec: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([global_A, global_B, evi_vec], dim=-1)
        x = self.norm_in(x)
        logit = self.mlp(x).squeeze(-1)
        return torch.nan_to_num(logit, nan=0.0, posinf=10.0, neginf=-10.0)


class AdaptiveMultiScaleLocalEvidenceHead(nn.Module):
    """Adaptive multi-scale local evidence head for residue-site scoring."""

    def __init__(self, d_model: int, channels: int = 64, dropout: float = 0.10,
                 delta_init: float = 0.03, use_scale_gate: bool = True,
                 use_channel_gate: bool = True):
        super().__init__()
        c = int(max(8, channels))
        self.use_scale_gate = bool(use_scale_gate)
        self.use_channel_gate = bool(use_channel_gate)
        self.base = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, c)
        self.convs = nn.ModuleList([
            nn.Conv1d(c, c, kernel_size=k, padding=k // 2)
            for k in (3, 7, 15)
        ])
        self.scale_gate = nn.Linear(d_model, len(self.convs))
        self.channel_gate = nn.Sequential(nn.Linear(d_model, c), nn.Sigmoid())
        self.out_proj = nn.Sequential(
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(c, d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(d_model, 1),
        )
        self.delta_scale = nn.Parameter(torch.tensor(float(delta_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        h = self.in_proj(self.norm(x)).transpose(1, 2)
        scales = torch.stack([conv(h).transpose(1, 2) for conv in self.convs], dim=2)
        if self.use_scale_gate:
            sw = torch.softmax(self.scale_gate(x), dim=-1).unsqueeze(-1)
            local = (scales * sw).sum(dim=2)
        else:
            local = scales.mean(dim=2)
        if self.use_channel_gate:
            local = local * self.channel_gate(x)
        delta = self.out_proj(local)
        return base + torch.tanh(self.delta_scale) * delta


class GlobalContextPooler(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.10):
        super().__init__()
        self.seg_norm = nn.LayerNorm(d_model)
        self.seg = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=9, padding=4, groups=math.gcd(d_model, max(1, d_model // 16))),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Conv1d(d_model, d_model, kernel_size=1),
        )

    @staticmethod
    def _masked_mean(x, mask):
        if mask is None:
            return x.mean(dim=1)
        m = _bmask(mask).float().unsqueeze(-1)
        return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _masked_max(x, mask):
        if mask is None:
            return x.max(dim=1).values
        return x.masked_fill(~_bmask(mask).unsqueeze(-1), -1e4).max(dim=1).values

    def forward(self, x: torch.Tensor, logits: torch.Tensor, mask: Optional[torch.Tensor]):
        mean = self._masked_mean(x, mask)
        maxv = self._masked_max(x, mask)
        if mask is not None:
            logits = logits.masked_fill(~_bmask(mask), -20.0)
        w = torch.softmax(logits, dim=1).unsqueeze(-1)
        evi = (x * w).sum(dim=1)
        seg_seq = self.seg(self.seg_norm(x).transpose(1, 2)).transpose(1, 2)
        seg = self._masked_mean(seg_seq, mask)
        ctx = 0.25 * (mean + maxv + evi + seg)
        return {"mean": mean, "max": maxv, "evi": evi, "seg": seg, "ctx": ctx}


class GlobalContextConditionedEvidenceBridge(nn.Module):
    def __init__(self, d_model: int, topk: int = 16, dropout: float = 0.10,
                 use_pair_context: bool = True):
        super().__init__()
        self.topk = int(max(1, topk))
        self.use_pair_context = bool(use_pair_context)
        self.ctx = nn.Sequential(
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.weight = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(d_model * 5),
            nn.Linear(d_model * 5, d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(d_model, d_model),
        )
        self.score = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def _topk(self, x, logits, mask):
        B, L, D = x.shape
        k = max(1, min(self.topk, L))
        scores = logits
        if mask is not None:
            scores = scores.masked_fill(~_bmask(mask), -1e9)
        vals, idx = torch.topk(scores, k=k, dim=1, largest=True, sorted=True)
        tok = x.gather(1, idx.unsqueeze(-1).expand(-1, -1, D))
        return tok, vals, idx

    def _weighted(self, tok, cond):
        c = cond.unsqueeze(1).expand(-1, tok.size(1), -1)
        w = self.weight(torch.cat([tok, c], dim=-1)).squeeze(-1)
        w = torch.softmax(w, dim=1).unsqueeze(-1)
        return (tok * w).sum(dim=1), w.squeeze(-1)

    def forward(self, xA, xB, logitA, logitB, maskA, maskB, ctxA, ctxB):
        gA = ctxA["ctx"]
        gB = ctxB["ctx"]
        pair_ctx = self.ctx(torch.cat([gA, gB, (gA - gB).abs(), gA * gB], dim=-1))
        tokA, valA, idxA = self._topk(xA, logitA, maskA)
        tokB, valB, idxB = self._topk(xB, logitB, maskB)
        eA, wA = self._weighted(tokA, pair_ctx)
        eB, wB = self._weighted(tokB, pair_ctx)
        bridge = self.fuse(torch.cat([eA, eB, (eA - eB).abs(), eA * eB, pair_ctx], dim=-1))
        evi_score = self.score(bridge).squeeze(-1)
        info = {
            "idxA": idxA, "idxB": idxB,
            "scoreA": valA.detach(), "scoreB": valB.detach(),
            "weightA": wA.detach(), "weightB": wB.detach(),
        }
        return bridge, evi_score, info


class GlobalPairEvidenceHead(nn.Module):
    def __init__(self, d_model: int, n_layers: int = 2, n_heads: int = 4,
                 dropout: float = 0.10, use_product: bool = True, use_diff: bool = True):
        super().__init__()
        self.use_product = bool(use_product)
        self.use_diff = bool(use_diff)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 2,
            dropout=float(dropout), activation="gelu", batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=max(1, int(n_layers)))
        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, ctxA, ctxB, bridge):
        toks = [
            ctxA["mean"], ctxB["mean"], ctxA["evi"], ctxB["evi"],
            ctxA["seg"], ctxB["seg"], bridge,
        ]
        if self.use_diff:
            toks.append((ctxA["evi"] - ctxB["evi"]).abs())
        if self.use_product:
            toks.append(ctxA["evi"] * ctxB["evi"])
        x = torch.stack(toks, dim=1)
        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        h = self.enc(x)[:, 0]
        return torch.nan_to_num(self.out(h).squeeze(-1), nan=0.0, posinf=10.0, neginf=-10.0)


def _safe_unit(v, dim=-1, eps=1e-6):
    return v / v.norm(dim=dim, keepdim=True).clamp_min(eps)

def _geom_scalar_feats(coords):
    L = coords.shape[0]
    out = coords.new_zeros((L, 6))
    if L > 1:
        d = (coords[1:] - coords[:-1]).norm(dim=-1)
        out[1:,0] = d; out[:-1,1] = d
    if L > 2:
        v1 = _safe_unit(coords[1:-1] - coords[:-2])
        v2 = _safe_unit(coords[2:] - coords[1:-1])
        out[1:-1,2] = (v1 * v2).sum(-1)
    center = coords.mean(dim=0, keepdim=True)
    out[:,3] = (coords-center).norm(dim=-1)
    dist = torch.cdist(coords, coords)
    out[:,4] = (dist < 8.0).float().sum(dim=1) - 1.0
    out[:,5] = torch.linspace(0.0, 1.0, steps=L, device=coords.device, dtype=coords.dtype)
    if L > 0:
        out[:,0:2] /= 4.0
        out[:,3] /= out[:,3].max().clamp_min(1.0)
        out[:,4] /= out[:,4].max().clamp_min(1.0)
    return out

def _geom_vector_feats(coords):
    L = coords.shape[0]
    out = coords.new_zeros((L, 2, 3))
    if L > 1:
        out[:-1,0] = _safe_unit(coords[1:] - coords[:-1])
        out[1:,1] = _safe_unit(coords[:-1] - coords[1:])
    return out

class GVP(nn.Module):
    def __init__(self, s_in, v_in, s_out, v_out, dropout=0.1):
        super().__init__()
        self.v_out = v_out
        self.lin_s = nn.Linear(s_in + v_in, s_out)
        self.lin_v = nn.Linear(v_in * 3, v_out * 3)
        self.gate = nn.Linear(s_out, v_out)
        self.norm_s = nn.LayerNorm(s_out)
        self.drop = nn.Dropout(dropout)
    def forward(self, s, v):
        s_out = self.drop(F.gelu(self.norm_s(self.lin_s(torch.cat([s, v.norm(dim=-1)], dim=-1)))))
        v_out = self.lin_v(v.reshape(*v.shape[:-2], -1)).reshape(*v.shape[:-2], self.v_out, 3)
        v_out = v_out * torch.sigmoid(self.gate(s_out)).unsqueeze(-1)
        return s_out, v_out

class GVPGraphBlock(nn.Module):
    def __init__(self, s_dim, v_dim, k=16, dropout=0.1):
        super().__init__()
        self.k = int(k)
        self.msg = GVP(s_dim + 1, v_dim + 1, s_dim, v_dim, dropout)
        self.upd = GVP(s_dim * 2, v_dim * 2, s_dim, v_dim, dropout)
        self.norm_s = nn.LayerNorm(s_dim)
    def forward(self, s, v, coords):
        L = s.shape[0]
        if L <= 1: return s, v
        k = min(self.k, max(1, L-1))
        dist = torch.cdist(coords, coords)
        dist = dist.masked_fill(torch.eye(L, device=coords.device, dtype=torch.bool), 1e9)
        nn_dist, nn_idx = torch.topk(dist, k=k, dim=-1, largest=False)
        nbr_s = s[nn_idx]; nbr_v = v[nn_idx]
        rel = coords[nn_idx] - coords.unsqueeze(1)
        rel_u = _safe_unit(rel, dim=-1)
        msg_s, msg_v = self.msg(torch.cat([nbr_s, nn_dist.unsqueeze(-1)], dim=-1), torch.cat([nbr_v, rel_u.unsqueeze(-2)], dim=-2))
        agg_s, agg_v = msg_s.mean(dim=1), msg_v.mean(dim=1)
        upd_s, upd_v = self.upd(torch.cat([s, agg_s], dim=-1), torch.cat([v, agg_v], dim=-2))
        return self.norm_s(s + upd_s), v + upd_v

class GVPStructureEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.in_gvp = GVP(cfg.gvp_node_s_dim, cfg.gvp_node_v_dim, cfg.gvp_hidden_s, cfg.gvp_hidden_v, cfg.gvp_dropout)
        self.layers = nn.ModuleList([GVPGraphBlock(cfg.gvp_hidden_s, cfg.gvp_hidden_v, cfg.gvp_k, cfg.gvp_dropout) for _ in range(cfg.gvp_layers)])
        self.out_proj = nn.Linear(cfg.gvp_hidden_s + cfg.gvp_hidden_v, cfg.d_model)
        self.chain_proj = nn.Linear(cfg.gvp_hidden_s + cfg.gvp_hidden_v, cfg.gvp_chain_dim)
        self.out_norm = nn.LayerNorm(cfg.d_model)
    def _encode_one(self, coords):
        s, v = self.in_gvp(_geom_scalar_feats(coords), _geom_vector_feats(coords))
        for blk in self.layers: s, v = blk(s, v, coords)
        vn = v.norm(dim=-1)
        node = self.out_norm(self.out_proj(torch.cat([s, vn], dim=-1)))
        chain = self.chain_proj(torch.cat([s.mean(dim=0), vn.mean(dim=0)], dim=-1))
        return node, chain
    def forward(self, coords, mask):
        if coords.dim() == 2: coords = coords.unsqueeze(0)
        B, L, _ = coords.shape
        nodes, chains = [], []
        for b in range(B):
            valid = torch.ones(L, device=coords.device, dtype=torch.bool) if mask is None else _bmask(mask[b])
            xyz = coords[b][valid]
            if xyz.numel() == 0:
                nodes.append(coords.new_zeros((L, self.out_proj.out_features)))
                chains.append(coords.new_zeros((self.chain_proj.out_features,)))
                continue
            node_valid, chain = self._encode_one(xyz)
            node = coords.new_zeros((L, self.out_proj.out_features))
            node[valid] = node_valid
            nodes.append(node); chains.append(chain)
        return torch.stack(nodes,0), torch.stack(chains,0)


class SiteGraphBlock(nn.Module):
    """Structure-aware residue refinement for single-chain site prediction."""

    def __init__(self, d_model: int, k: int = 12, radius: float = 12.0, dropout: float = 0.1):
        super().__init__()
        self.k = int(k)
        self.radius = float(max(1e-3, radius))
        self.norm = nn.LayerNorm(d_model)
        self.msg = nn.Sequential(
            nn.Linear(d_model * 3 + 2, d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(d_model, d_model),
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(d_model * 2, d_model),
        )
        self.gate = nn.Parameter(torch.tensor(-0.5))

    def _one(self, x: torch.Tensor, coords: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        L, D = x.shape
        valid = torch.ones(L, device=x.device, dtype=torch.bool) if mask is None else _bmask(mask)
        idx_valid = valid.nonzero(as_tuple=False).squeeze(-1)
        if idx_valid.numel() <= 1:
            return x
        xv = self.norm(x[idx_valid])
        cv = coords[idx_valid].float()
        n = int(xv.size(0))
        k = max(1, min(self.k, n - 1))
        dist = torch.cdist(cv.unsqueeze(0), cv.unsqueeze(0)).squeeze(0)
        dist = dist.masked_fill(torch.eye(n, device=x.device, dtype=torch.bool), 1e6)
        nn_dist, nn_idx = torch.topk(dist, k=k, dim=-1, largest=False, sorted=False)
        nbr = xv[nn_idx]
        center = xv.unsqueeze(1).expand(-1, k, -1)
        weight = torch.exp(-nn_dist / self.radius).unsqueeze(-1)
        nbr_mean = (nbr * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1e-6)
        edge = torch.stack([
            (nn_dist / self.radius).clamp(0.0, 4.0).mean(dim=1),
            (nn_dist < self.radius).float().mean(dim=1),
        ], dim=-1)
        delta = self.msg(torch.cat([xv, nbr_mean, xv - nbr_mean, edge], dim=-1))
        refined = x[idx_valid] + torch.sigmoid(self.gate) * delta
        refined = refined + self.ff(refined)
        out = x.clone()
        out[idx_valid] = refined
        return out

    def forward(self, x: torch.Tensor, coords: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if x.dim() == 2:
            return self._one(x, coords, mask)
        outs = []
        for b in range(x.size(0)):
            mb = None if mask is None else mask[b]
            outs.append(self._one(x[b], coords[b], mb))
        return torch.stack(outs, dim=0)


class SiteGraphRefiner(nn.Module):
    def __init__(self, d_model: int, n_layers: int = 2, k: int = 12, radius: float = 12.0, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            SiteGraphBlock(d_model, k=k, radius=radius, dropout=dropout)
            for _ in range(max(0, int(n_layers)))
        ])

    def forward(self, x: torch.Tensor, coords: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        for blk in self.layers:
            x = blk(x, coords, mask)
        return x


class L13PDBGVPModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__(); self.cfg = cfg
        D, H, DR = cfg.d_model, cfg.n_heads, cfg.dropout
        self.seqA = nn.Linear(cfg.d_seq_in, D); self.seqB = nn.Linear(cfg.d_seq_in, D)
        self.struct = GVPStructureEncoder(cfg)
        self.fuseA = nn.Sequential(nn.Linear(D*2, D), nn.GELU(), nn.LayerNorm(D))
        self.fuseB = nn.Sequential(nn.Linear(D*2, D), nn.GELU(), nn.LayerNorm(D))
        self.filmA = FiLM(D, cfg.d_chain_in + cfg.gvp_chain_dim)
        self.filmB = FiLM(D, cfg.d_chain_in + cfg.gvp_chain_dim)
        self.encA = SelfEnc(D, cfg.n_encoder_layers, H, DR)
        self.encB = SelfEnc(D, cfg.n_encoder_layers, H, DR)
        self.cross = CrossEnc(D, cfg.n_cross_layers, H, DR, use_layerscale=cfg.use_layerscale, layerscale_init=cfg.layerscale_init, use_swiglu=cfg.use_swiglu, ffn_mult=cfg.ffn_mult)
        self.site_cross_gate = nn.Parameter(torch.tensor(0.0))
        self.site_graph = SiteGraphRefiner(
            D,
            n_layers=getattr(cfg, "site_graph_layers", 2),
            k=getattr(cfg, "site_graph_k", 12),
            radius=getattr(cfg, "site_graph_radius", 12.0),
            dropout=DR,
        )
        if str(getattr(cfg, "site_head_type", "linear")).lower() == "amleh":
            self.head_resA = AdaptiveMultiScaleLocalEvidenceHead(
                D, channels=getattr(cfg, "site_ms_channels", 64),
                dropout=getattr(cfg, "site_ms_dropout", 0.10),
                delta_init=getattr(cfg, "site_ms_delta_init", 0.03),
                use_scale_gate=getattr(cfg, "site_ms_use_scale_gate", True),
                use_channel_gate=getattr(cfg, "site_ms_use_channel_gate", True),
            )
            self.head_resB = AdaptiveMultiScaleLocalEvidenceHead(
                D, channels=getattr(cfg, "site_ms_channels", 64),
                dropout=getattr(cfg, "site_ms_dropout", 0.10),
                delta_init=getattr(cfg, "site_ms_delta_init", 0.03),
                use_scale_gate=getattr(cfg, "site_ms_use_scale_gate", True),
                use_channel_gate=getattr(cfg, "site_ms_use_channel_gate", True),
            )
        else:
            self.head_resA = nn.Sequential(nn.LayerNorm(D), nn.Linear(D,1))
            self.head_resB = nn.Sequential(nn.LayerNorm(D), nn.Linear(D,1))
        self.frag_head = FragmentHead(d_model=D, kernel_size=9, dropout=0.1)
        self.evidence_bridge = EvidenceBridge(d_model=D, d_proj=cfg.eb_d_proj, n_heads=cfg.eb_n_heads, k=cfg.eb_topk_k, frac=cfg.eb_topk_frac, pair_k=cfg.eb_pair_topk, dropout=DR)
        self.l2_bridge = L2Bridge(
            d_model=D, d_proj=cfg.l2_d_proj, pair_k=cfg.l2_pair_topk,
            focus_k=cfg.l2_focus_topk, focus_frac=cfg.l2_focus_frac, dropout=DR,
            geom_prior_w=getattr(cfg, "l2_geom_prior_w", 0.0),
            geom_prior_sigma=getattr(cfg, "l2_geom_prior_sigma", 4.0),
        )
        self.l2_to_l3 = nn.Sequential(
            nn.LayerNorm(cfg.eb_d_proj * 2 + cfg.l2_d_proj * 2),
            nn.Linear(cfg.eb_d_proj * 2 + cfg.l2_d_proj * 2, cfg.eb_d_proj * 2),
            nn.GELU(),
            nn.Dropout(DR),
        )
        self.l2_pair_scale = nn.Parameter(torch.tensor(0.5))
        self.eb_gate_scale = nn.Parameter(torch.tensor(0.0))
        self.l3_head = L3Head(d_model=D, d_proj=cfg.eb_d_proj, d_hidden=cfg.l3_d_hidden, n_layers=cfg.l3_n_layers, dropout=DR)
        self.global_pooler = GlobalContextPooler(D, dropout=getattr(cfg, "eb_dropout", 0.10))
        self.gc_eb = GlobalContextConditionedEvidenceBridge(
            D, topk=getattr(cfg, "eb_topk", 16),
            dropout=getattr(cfg, "eb_dropout", 0.10),
            use_pair_context=getattr(cfg, "eb_use_pair_context", True),
        )
        self.gpeh = GlobalPairEvidenceHead(
            D, n_layers=getattr(cfg, "pair_transformer_layers", 2),
            n_heads=getattr(cfg, "pair_transformer_heads", 4),
            dropout=getattr(cfg, "pair_dropout", 0.10),
            use_product=getattr(cfg, "pair_use_product_token", True),
            use_diff=getattr(cfg, "pair_use_diff_token", True),
        )
        self.training_epoch = 0
        self._cache = {}
    @staticmethod
    def _pool_chain(chain, mask):
        if chain is None: return None
        if chain.dim() == 2: return chain
        if chain.dim() == 3:
            if mask is None: return chain.mean(dim=1)
            m = _bmask(mask).float().unsqueeze(-1)
            return (chain*m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        raise ValueError(chain.shape)
    @staticmethod
    def _global_pool(x, mask):
        if mask is None: return x.mean(dim=1)
        m = _bmask(mask).float().unsqueeze(-1)
        return (x*m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
    def forward(self, resA, maskA, chainA, coordsA, resB, maskB, chainB, coordsB, site_mode: bool = False):
        if resA.dim()==2: resA=resA.unsqueeze(0)
        if resB.dim()==2: resB=resB.unsqueeze(0)
        if coordsA.dim()==2: coordsA=coordsA.unsqueeze(0)
        if coordsB.dim()==2: coordsB=coordsB.unsqueeze(0)
        maskA = _bmask(maskA) if maskA is not None else None
        maskB = _bmask(maskB) if maskB is not None else None
        seqA = self.seqA(resA)
        strA, gA = self.struct(coordsA, maskA)
        xA = self.fuseA(torch.cat([seqA, strA], dim=-1))
        cA = self._pool_chain(chainA, maskA)
        if cA is None: cA = gA.new_zeros((gA.size(0),0))
        xA = self.filmA(xA, torch.cat([cA, gA], dim=-1))
        xA = self.encA(xA, maskA)

        if bool(site_mode):
            if bool(getattr(self.cfg, "site_self_cross", True)):
                xA_cross, xA_mirror = self.cross(xA, xA, maskA, maskA)
                xA_deep = 0.5 * (xA_cross + xA_mirror)
                xA = xA + torch.sigmoid(self.site_cross_gate) * (xA_deep - xA)
            xA = self.site_graph(xA, coordsA, maskA)
            B = xA.size(0)
            Lb = resB.size(1)
            zA = torch.nan_to_num(self.head_resA(xA), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0,20.0)
            if maskA is not None: zA = zA.masked_fill(~maskA.unsqueeze(-1), -30.0)
            logit_resA = zA.squeeze(-1)
            logit_fragA = torch.nan_to_num(self.frag_head(xA, maskA), nan=0.0).clamp(-10.0,10.0)
            logit_resB = resB.new_zeros((B, Lb))
            logit_fragB = resB.new_zeros((B, Lb))
            if maskB is not None:
                logit_resB = logit_resB.masked_fill(~maskB, -30.0)
            pair_logit = xA.new_zeros((B,))
            evi_score = xA.new_zeros((B,))
            eb_gate = xA.new_tensor(0.0)
            L = logit_resA.size(1)
            K = int(getattr(self.cfg, "explain_topk_k", 512))
            frac = float(getattr(self.cfg, "eb_topk_frac", 0.05))
            if frac > 0:
                Kf = max(1, int(round(frac * L)))
                K = min(K, Kf) if K > 0 else Kf
            K = max(1, min(K, L))
            masked_resA = logit_resA.clone()
            if maskA is not None:
                masked_resA = masked_resA.masked_fill(~maskA, -1e9)
            topk_valA, topk_idxA = torch.topk(masked_resA, k=K, dim=1, largest=True, sorted=True)
            topk_long = torch.empty((B, 0), dtype=torch.long, device=xA.device)
            topk_float = xA.new_empty((B, 0))
            topk = {
                "topk_idxA": topk_idxA,
                "topk_idxB": topk_long,
                "topk_valA": topk_valA,
                "topk_valB": topk_float,
                "pair_residue_idxA": topk_long,
                "pair_residue_idxB": topk_long,
                "pair_score": topk_float,
                "pair_compactness": xA.new_zeros((B,)),
            }
            self._cache = {'logit_resA':logit_resA,'logit_resB':logit_resB,'logit_fragA':logit_fragA,'logit_fragB':logit_fragB,'evi_vec':None,'evi_score':evi_score,'eb_gate':eb_gate.detach(),'pair_logit':pair_logit,'topk_info':topk}
            return {'pair_logit':pair_logit,'pair_prob':torch.sigmoid(pair_logit),'evi_score':evi_score,'evi_prob':torch.sigmoid(evi_score),'logit_resA':logit_resA,'logit_resB':logit_resB,'p_res_A':torch.sigmoid(logit_resA),'p_res_B':torch.sigmoid(logit_resB),'logit_fragA':logit_fragA,'logit_fragB':logit_fragB,'p_frag_A':torch.sigmoid(logit_fragA),'p_frag_B':torch.sigmoid(logit_fragB),'topk_idxA':topk['topk_idxA'],'topk_idxB':topk['topk_idxB'],'topk_valA':topk['topk_valA'],'topk_valB':topk['topk_valB'],'pair_idxA':topk['pair_residue_idxA'],'pair_idxB':topk['pair_residue_idxB'],'pair_score':topk['pair_score'],'pair_compactness':topk['pair_compactness']}

        seqB = self.seqB(resB)
        strB, gB = self.struct(coordsB, maskB)
        xB = self.fuseB(torch.cat([seqB, strB], dim=-1))
        cB = self._pool_chain(chainB, maskB)
        if cB is None: cB = gB.new_zeros((gB.size(0),0))
        xB = self.filmB(xB, torch.cat([cB, gB], dim=-1))
        xB = self.encB(xB, maskB)
        xA, xB = self.cross(xA, xB, maskA, maskB)
        zA = torch.nan_to_num(self.head_resA(xA), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0,20.0)
        zB = torch.nan_to_num(self.head_resB(xB), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0,20.0)
        if maskA is not None: zA = zA.masked_fill(~maskA.unsqueeze(-1), -30.0)
        if maskB is not None: zB = zB.masked_fill(~maskB.unsqueeze(-1), -30.0)
        logit_resA, logit_resB = zA.squeeze(-1), zB.squeeze(-1)
        logit_fragA = torch.nan_to_num(self.frag_head(xA, maskA), nan=0.0).clamp(-10.0,10.0)
        logit_fragB = torch.nan_to_num(self.frag_head(xB, maskB), nan=0.0).clamp(-10.0,10.0)
        if bool(getattr(self.cfg, "eb_enable", True)) and str(getattr(self.cfg, "pair_head_type", "gpeh")).lower() == "gpeh":
            ctxA = self.global_pooler(xA, logit_resA, maskA)
            ctxB = self.global_pooler(xB, logit_resB, maskB)
            bridge_token, evi_score, gc_info = self.gc_eb(xA, xB, logit_resA, logit_resB, maskA, maskB, ctxA, ctxB)
            pair_logit = self.gpeh(ctxA, ctxB, bridge_token)
            topk = {
                "topk_idxA": gc_info["idxA"],
                "topk_idxB": gc_info["idxB"],
                "topk_valA": gc_info["scoreA"],
                "topk_valB": gc_info["scoreB"],
                "pair_residue_idxA": gc_info["idxA"],
                "pair_residue_idxB": gc_info["idxB"],
                "pair_score": 0.5 * (gc_info["scoreA"].mean(dim=1, keepdim=True) + gc_info["scoreB"].mean(dim=1, keepdim=True)),
                "pair_compactness": torch.zeros((xA.size(0),), dtype=xA.dtype, device=xA.device),
            }
            self._cache = {
                'logit_resA': logit_resA, 'logit_resB': logit_resB,
                'logit_fragA': logit_fragA, 'logit_fragB': logit_fragB,
                'evi_vec': bridge_token, 'evi_score': evi_score,
                'bridge_token': bridge_token, 'gc_info': gc_info,
                'pair_logit': pair_logit, 'topk_info': topk,
            }
            return {
                'pair_logit': pair_logit, 'pair_prob': torch.sigmoid(pair_logit),
                'evi_score': evi_score, 'evi_prob': torch.sigmoid(evi_score),
                'bridge_token': bridge_token,
                'logit_resA': logit_resA, 'logit_resB': logit_resB,
                'p_res_A': torch.sigmoid(logit_resA), 'p_res_B': torch.sigmoid(logit_resB),
                'logit_fragA': logit_fragA, 'logit_fragB': logit_fragB,
                'p_frag_A': torch.sigmoid(logit_fragA), 'p_frag_B': torch.sigmoid(logit_fragB),
                'topk_idxA': topk['topk_idxA'], 'topk_idxB': topk['topk_idxB'],
                'topk_valA': topk['topk_valA'], 'topk_valB': topk['topk_valB'],
                'pair_idxA': topk['pair_residue_idxA'], 'pair_idxB': topk['pair_residue_idxB'],
                'pair_score': topk['pair_score'], 'pair_compactness': topk['pair_compactness'],
                'topk_evidence_A': gc_info["scoreA"], 'topk_evidence_B': gc_info["scoreB"],
            }
        ep_now = int(getattr(self, 'training_epoch', 0)); eb_start = int(getattr(self.cfg, 'eb_start_epoch', 0)); eb_ramp = int(getattr(self.cfg, 'eb_ramp_epochs', 0))
        if ep_now < eb_start: eb_ramp_factor = xA.new_tensor(0.0)
        elif eb_ramp <= 0: eb_ramp_factor = xA.new_tensor(1.0)
        else: eb_ramp_factor = xA.new_tensor(min(1.0, float(ep_now - eb_start + 1) / float(max(1, eb_ramp))))
        eb_gate = torch.sigmoid(self.eb_gate_scale) * eb_ramp_factor
        evi_vec, evi_score, topk = self.evidence_bridge(logit_resA, logit_resB, xA, xB, maskA, maskB, eb_gate)
        S, l2_vec, l2_score, l2_info = self.l2_bridge(logit_resA, logit_resB, xA, xB, maskA, maskB, coordsA, coordsB)
        evi_vec_fused = self.l2_to_l3(torch.cat([evi_vec, l2_vec], dim=-1))
        pair_logit = self.l3_head(self._global_pool(xA, maskA), self._global_pool(xB, maskB), evi_vec_fused)
        pair_logit = pair_logit + torch.tanh(self.l2_pair_scale) * l2_score
        self._cache = {'logit_resA':logit_resA,'logit_resB':logit_resB,'logit_fragA':logit_fragA,'logit_fragB':logit_fragB,'evi_vec':evi_vec_fused,'evi_score':evi_score,'l2_score':l2_score,'S':S,'eb_gate':eb_gate.detach(),'pair_logit':pair_logit,'topk_info':topk,'l2_info':l2_info}
        return {'pair_logit':pair_logit,'pair_prob':torch.sigmoid(pair_logit),'evi_score':evi_score,'evi_prob':torch.sigmoid(evi_score),'l2_score':l2_score,'l2_prob':torch.sigmoid(l2_score),'S':S,'logit_resA':logit_resA,'logit_resB':logit_resB,'p_res_A':torch.sigmoid(logit_resA),'p_res_B':torch.sigmoid(logit_resB),'logit_fragA':logit_fragA,'logit_fragB':logit_fragB,'p_frag_A':torch.sigmoid(logit_fragA),'p_frag_B':torch.sigmoid(logit_fragB),'topk_idxA':topk['topk_idxA'],'topk_idxB':topk['topk_idxB'],'topk_valA':topk['topk_valA'],'topk_valB':topk['topk_valB'],'pair_idxA':topk['pair_residue_idxA'],'pair_idxB':topk['pair_residue_idxB'],'pair_score':topk['pair_score'],'pair_compactness':topk['pair_compactness']}
