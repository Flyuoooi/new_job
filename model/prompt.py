# text_branch.py
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utils
# -----------------------------
def l2n(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.normalize(x, dim=-1, eps=eps)


@dataclass
class TextBranchOutput:
    # 原始 ctx token（用于你之后做“亮点残差”）
    ctx_full: torch.Tensor          # [B, n_ctx, 512]
    ctx_masked: torch.Tensor        # [B, n_ctx, 512] or same as full if mask disabled

    # 你要的 2D 向量（走 meta 路线最方便）
    txt_full: torch.Tensor          # [B, 512]
    txt_masked: torch.Tensor        # [B, 512]

    # 两段注入用（可选）
    txt_full_1: Optional[torch.Tensor] = None   # [B, 512]
    txt_full_2: Optional[torch.Tensor] = None   # [B, 512]
    txt_masked_1: Optional[torch.Tensor] = None # [B, 512]
    txt_masked_2: Optional[torch.Tensor] = None # [B, 512]

    # 残差端口：你后续单独文件实现 residual recovery
    resid_port: Optional[Dict[str, torch.Tensor]] = None


# -----------------------------
# Context token generator: CoOp
# -----------------------------
class CoOpContextLearner(nn.Module):
    """
    CoOp：每个 class / pid 一个 ctx（class-specific ctx）
    输出 ctx_tokens: [B, n_ctx, 512]
    """
    def __init__(self, num_classes: int, n_ctx: int, ctx_dim: int = 512):
        super().__init__()
        self.num_classes = int(num_classes)
        self.n_ctx = int(n_ctx)
        self.ctx_dim = int(ctx_dim)

        self.ctx = nn.Parameter(torch.empty(self.num_classes, self.n_ctx, self.ctx_dim))
        nn.init.normal_(self.ctx, std=0.02)

    def forward(self, pids: torch.Tensor) -> torch.Tensor:
        # pids: [B]
        return self.ctx[pids]  # [B, n_ctx, 512]


# -----------------------------
# Context token generator: CoCoOp
# -----------------------------
class CoCoOpContextLearner(nn.Module):
    """
    CoCoOp：ctx + meta_net(image_feature) 的 bias，得到 instance-conditioned ctx tokens
    对应原版：bias = meta_net(im_features) -> (B, ctx_dim), ctx_shifted = ctx + bias -> (B, n_ctx, ctx_dim):contentReference[oaicite:3]{index=3}
    """
    def __init__(
        self,
        n_ctx: int,
        cond_in_dim: int = 512,
        ctx_dim: int = 512,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.n_ctx = int(n_ctx)
        self.ctx_dim = int(ctx_dim)

        # generic ctx: [n_ctx, 512]
        self.ctx = nn.Parameter(torch.empty(self.n_ctx, self.ctx_dim))
        nn.init.normal_(self.ctx, std=0.02)

        # meta-net: (B, cond_in_dim) -> (B, ctx_dim)
        self.meta_net = nn.Sequential(
            nn.Linear(cond_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, ctx_dim),
        )

    def forward(self, cond_feat: torch.Tensor) -> torch.Tensor:
        """
        cond_feat: [B, cond_in_dim]（你可以给 CLIP image embed、或你自己的 img->txt proj）
        return: [B, n_ctx, 512]
        """
        bias = self.meta_net(cond_feat)          # [B, 512]
        bias = bias.unsqueeze(1)                 # [B, 1, 512]
        ctx = self.ctx.unsqueeze(0)              # [1, n_ctx, 512]
        ctx_shifted = ctx + bias                 # [B, n_ctx, 512]
        return ctx_shifted


# -----------------------------
# Attn-guided masking on ctx tokens
# -----------------------------
class CtxAttnGuidedMask(nn.Module):
    """
    只在 ctx token 区间做 mask（这里 ctx 本身就是区间了）
    思路等价于你之前在 prompts 上做的 apply_attn_guided_mask:contentReference[oaicite:4]{index=4}
    """
    def __init__(self, ctx_dim: int = 512):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(ctx_dim))

    @torch.no_grad()
    def _topk_indices(self, scores: torch.Tensor, k: int) -> torch.Tensor:
        # scores: [B, n_ctx]
        return torch.topk(scores, k=k, dim=1).indices

    def forward(self, ctx_tokens: torch.Tensor, cond_feat: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        """
        ctx_tokens: [B, n_ctx, 512]
        cond_feat:  [B, 512] (建议已归一化)
        """
        if mask_ratio <= 0:
            return ctx_tokens

        B, n_ctx, D = ctx_tokens.shape
        k = max(1, int(mask_ratio * n_ctx))

        # similarity: token vs cond_feat
        p = l2n(ctx_tokens.float())                     # [B, n_ctx, D]
        i = l2n(cond_feat.float()).unsqueeze(1)         # [B, 1, D]
        scores = (p * i).sum(dim=-1)                    # [B, n_ctx]

        sel = self._topk_indices(scores, k=k)           # [B, k]
        masked = ctx_tokens.clone()
        mask_vec = self.mask_token.view(1, 1, D).to(masked.device, masked.dtype)
        masked.scatter_(1, sel.unsqueeze(-1).expand(-1, -1, D), mask_vec.expand(B, k, D))
        return masked


# -----------------------------
# Pooling + optional split
# -----------------------------
class CtxPooler(nn.Module):
    def __init__(self, mode: str = "mean"):
        super().__init__()
        assert mode in ["mean"], "当前先支持 mean（你要更复杂的 pooling 我们后面再加）"
        self.mode = mode

    def forward(self, ctx_tokens: torch.Tensor) -> torch.Tensor:
        # [B, n_ctx, 512] -> [B, 512]
        return ctx_tokens.mean(dim=1)

    def split_half_and_pool(self, ctx_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # [B, n_ctx, 512] -> ([B,512], [B,512]) using first half / second half
        B, n_ctx, D = ctx_tokens.shape
        half = n_ctx // 2
        if half == 0:
            # 极端情况 n_ctx=1
            v = ctx_tokens.mean(dim=1)
            return v, v
        v1 = ctx_tokens[:, :half, :].mean(dim=1)
        v2 = ctx_tokens[:, half:, :].mean(dim=1)
        return v1, v2


# -----------------------------
# Main Text Branch (规范命名版)
# -----------------------------
class PromptText(nn.Module):
    """
    你要的“meta 路线”text_branch：
      - 生成 ctx tokens（CoOp 或 CoCoOp）
      - 可选 attn-guided mask（只在 ctx 上）
      - 在 text 生成阶段就 pooling 成 [B,512]
      - 残差不做，只留 resid_port（把输入/目标/中间量都吐出去）
    """
    def __init__(
        self,
        num_classes: int,
        n_ctx: int = 4,
        method: str = "coop",                # "coop" or "cocoop"
        ctx_dim: int = 512,
        cocoop_cond_in_dim: int = 512,
        mask_ratio: float = 0.5,
        enable_mask: bool = True,
        enable_split_inject: bool = True,    # 是否输出两段注入向量
        enable_resid_port: bool = True,             # 是否输出 residual 端口（不计算loss）
    ):
        super().__init__()
        self.method = method.lower()
        assert self.method in ["coop", "cocoop"]

        self.n_ctx = int(n_ctx)
        self.ctx_dim = int(ctx_dim)

        if self.method == "coop":
            self.ctx_learner = CoOpContextLearner(num_classes=num_classes, n_ctx=n_ctx, ctx_dim=ctx_dim)
        else:
            self.ctx_learner = CoCoOpContextLearner(
                n_ctx=n_ctx, cond_in_dim=cocoop_cond_in_dim, ctx_dim=ctx_dim, hidden_dim=ctx_dim
            )

        self.enable_mask = bool(enable_mask)
        self.mask_ratio = float(mask_ratio)
        self.ctx_masker = CtxAttnGuidedMask(ctx_dim=ctx_dim)

        self.pooler = CtxPooler(mode="mean")

        self.enable_split_inject = bool(enable_split_inject)
        self.enable_resid_port = bool(enable_resid_port)

    def forward(
        self,
        pids: torch.Tensor,                       # [B]
        cond_feat: Optional[torch.Tensor] = None, # [B,512] 用于：cocoop 的 meta_net + masking 的相似度引导
    ) -> TextBranchOutput:
        """
        - CoOp: cond_feat 可以只用于 mask（也可以不给）
        - CoCoOp: cond_feat 必须给（用来生成 ctx_shifted），同时也用来 mask
        """
        if self.method == "cocoop":
            assert cond_feat is not None, "CoCoOp 模式下必须提供 cond_feat（给 meta_net 产生 bias）"
            ctx_full = self.ctx_learner(cond_feat)                 # [B, n_ctx, 512]
        else:
            ctx_full = self.ctx_learner(pids)                      # [B, n_ctx, 512]

        if self.enable_mask and (cond_feat is not None) and (self.mask_ratio > 0):
            ctx_masked = self.ctx_masker(ctx_full, cond_feat, self.mask_ratio)
        else:
            ctx_masked = ctx_full

        # 你要的：在 text 生成阶段就 mean-pool 成 [B,512]
        txt_full = self.pooler(ctx_full)                           # [B, 512]
        txt_masked = self.pooler(ctx_masked)                       # [B, 512]

        out = TextBranchOutput(
            ctx_full=ctx_full,
            ctx_masked=ctx_masked,
            txt_full=txt_full,
            txt_masked=txt_masked,
        )

        # 残差端口：你后续单独写 residual_recovery.py 来吃这些张量
        if self.enable_resid_port:
            out.resid_port = {
                "ctx_full": ctx_full,          # [B,n_ctx,512]
                "ctx_masked": ctx_masked,      # [B,n_ctx,512]
                "txt_full": txt_full,          # [B,512]
                "txt_masked": txt_masked,      # [B,512]
            }
            if self.enable_split_inject:
                out.resid_port.update({
                    "txt_full_1": out.txt_full_1,
                    "txt_full_2": out.txt_full_2,
                    "txt_masked_1": out.txt_masked_1,
                    "txt_masked_2": out.txt_masked_2,
                })

        return out
