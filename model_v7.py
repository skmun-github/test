#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------- #
#  model_v7.py – 2025-08-10                                                   #
#  • v5 계열 유지 + v7 실무성 강화                                            #
#     - 입력 길이 T 가변 (윈도우 길이 assert X)                                #
#     - 작은 해상도에서도 안전한 패치 추출(커널 자동 축소)                      #
#     - MaskAware static + Country embedding + horizon별 head                  #
# ---------------------------------------------------------------------------- #
from __future__ import annotations
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------- Patch-Encoder -------------------------------- #
class PatchEncoder(nn.Module):
    """
    2-D conv 전처리 후 패치 추출 → (B, N_patch, E)
    패치 크기 : patch_size,  stride = max(1, patch_size//2) (50% overlap)
    • 입력 (H,W) 보다 큰 kernel이 들어오면 자동 축소
    """
    def __init__(self, in_ch: int,
                 patch_size: int = 32,
                 embed_ch: int  = 8,
                 img_h: int | None = None,
                 img_w: int | None = None) -> None:
        super().__init__()
        if (img_h is not None) and (img_w is not None):
            patch_size = min(patch_size, img_h, img_w)
            if patch_size < 1:
                patch_size = 1
        self.patch = patch_size
        self.stride = max(1, patch_size // 2)

        self.dw1 = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.dw2 = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.pw  = nn.Conv2d(in_ch, embed_ch, 1)
        self.unfold = nn.Unfold(kernel_size=self.patch, stride=self.stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, H, W)
        → (B, N_patch, embed_ch)
        """
        x = F.relu(self.dw2(F.relu(self.dw1(x.float()))))
        x = self.pw(x)                                   # (B,E,H,W)
        patches = self.unfold(x)                         # (B,E*P²,Np)
        patches = patches.view(x.size(0), -1, self.patch**2).mean(-1)
        n_patch = patches.size(1) // x.size(1)           # = Np
        return patches.view(x.size(0), n_patch, x.size(1))

# --------------------------- Temporal Backbone ----------------------------- #
class BahdanauAttention(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.proj  = nn.Linear(hidden, hidden, bias=False)
        self.score = nn.Linear(hidden, 1,      bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h : (B, T, H) → ctx : (B, H)
        w = torch.softmax(self.score(torch.tanh(self.proj(h))).squeeze(-1), dim=1)
        ctx = torch.bmm(w.unsqueeze(1), h).squeeze(1)
        return ctx

class TemporalBackbone(nn.Module):
    def __init__(self, in_dim: int,
                 hidden: int = 256,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.gru  = nn.GRU(in_dim, hidden, 2,
                           dropout=dropout, batch_first=True)
        self.attn = BahdanauAttention(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, T, in_dim)  (T 가변)
        h, _  = self.gru(self.norm(x.float()))
        ctx   = self.attn(h)
        return self.head(ctx)

# --------------------------- Mask-Aware Linear ----------------------------- #
class MaskAwareLinear(nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        self.bias   = nn.Parameter(torch.zeros(d_out))
        nn.init.xavier_uniform_(self.weight)
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return F.linear(x * (1 - mask), self.weight, self.bias)

# ------------------------------ Static Block ------------------------------- #
class StaticBlock(nn.Module):
    def __init__(self, d_in: int,
                 d_hidden: int = 256,
                 p_drop: float = 0.5) -> None:
        super().__init__()
        self.lin = MaskAwareLinear(d_in, d_hidden)
        self.act = nn.ReLU()
        self.dp  = nn.Dropout(p_drop)
    def forward(self, s: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.dp(self.act(self.lin(s, mask)))

# ------------------------------ Binary Head -------------------------------- #
class BinaryHead(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z).squeeze(-1)   # (B,)

# ------------------------------- Full Model -------------------------------- #
class ModelV7(nn.Module):
    """
    Dynamic X → PatchEncoder → TemporalBackbone(64)
               + StaticBlock(256) + CountryEmb
               → horizon-별 BinaryHead
    """
    def __init__(self,
                 # ───────── Data-dependent params ───────── #
                 in_ch: int,                 # 해역 채널 수
                 img_h: int,
                 img_w: int,
                 d_s_val: int,               # static value dim
                 d_s_mask: int,              # static mask dim
                 month_dim: int,             # 12(one-hot) or 2(cyclical)
                 n_country: int,
                 horizons: List[int],        # ex: [1,3]
                 # ───────── Arch hyperparams ───────────── #
                 patch_size: int      = 32,
                 embed_ch: int        = 8,
                 country_emb_dim: int = 16,
                 dropout_static: float= 0.5) -> None:

        super().__init__()
        self.horizons   = sorted(horizons)

        # ① Patch encoder & dynamic feature dim
        self.patch_enc = PatchEncoder(in_ch=in_ch,
                                      patch_size=patch_size,
                                      embed_ch=embed_ch,
                                      img_h=img_h, img_w=img_w)
        with torch.no_grad():
            n_patch = self.patch_enc(torch.zeros(1,in_ch,img_h,img_w)).shape[1]
        dyn_dim = n_patch * embed_ch

        # ② Temporal backbone
        self.temporal = TemporalBackbone(dyn_dim)

        # ③ Static block (S_val + S_mask + month_rep)
        d_static_in = d_s_val + d_s_mask + month_dim
        self.static_block = StaticBlock(d_in=d_static_in,
                                        d_hidden=256,
                                        p_drop=dropout_static)

        # ④ Country embedding
        self.cty_emb = nn.Embedding(n_country, country_emb_dim)

        # ⑤ Heads (horizon-별)
        fuse_dim = 64 + 256 + country_emb_dim
        self.heads = nn.ModuleDict({
            f"h{h}": BinaryHead(fuse_dim) for h in self.horizons
        })

    def forward(self,
                x_dyn: torch.Tensor,         # (B,T,C,H,W)  (T 가변)
                s_val: torch.Tensor,         # (B,D_val)
                s_mask: torch.Tensor,        # (B,D_mask)
                month_rep: torch.Tensor,     # (B,month_dim)
                country_idx: torch.Tensor    # (B,)
                ) -> Dict[str, torch.Tensor]:

        B,T,C,H,W = x_dyn.shape
        # 1) Dynamic branch
        patches = self.patch_enc(x_dyn.view(B*T, C, H, W)).flatten(1).view(B, T, -1)
        z_time  = self.temporal(patches)                     # (B,64)

        # 2) Static branch
        x_static = torch.cat([s_val, s_mask.float(), month_rep], dim=1)
        mask_full = torch.cat([
            s_mask,                        # 결측 플래그
            torch.zeros_like(s_mask),      # value 자리 (항상 관측)
            torch.zeros(B, month_rep.size(1), device=s_mask.device, dtype=s_mask.dtype)
        ], dim=1)
        z_static = self.static_block(x_static, mask_full)    # (B,256)

        # 3) Country embedding
        z_cty = self.cty_emb(country_idx)                    # (B,emb)

        # 4) Fusion & heads
        z_all = torch.cat([z_time, z_static, z_cty], dim=1)
        return {name: head(z_all) for name, head in self.heads.items()}

# --------------------------- Helper Builder -------------------------------- #
def build_model_v7(device: str | torch.device = "cpu",
                   **kwargs) -> ModelV7:
    model = ModelV7(**kwargs)
    return model.to(device)
