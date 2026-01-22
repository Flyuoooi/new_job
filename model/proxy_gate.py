import torch
import torch.nn as nn


class ProxyFeatureGate(nn.Module):
    """Feature-wise keep-top + soft gate driven by a per-sample proxy.

    Pattern: select top-k indices via torch.topk, then scatter_ into dense gate.
    (Commonly used in top-k gating / MoE routing, adapted here to feature dims.)

    Args:
        keep_ratio: fraction of dims to hard-keep (gate=1.0).
        temperature: sigmoid temperature for non-top-k soft gate.
        min_gate: lower bound for soft gate on non-top-k dims.
        detach_proxy: detach proxy when computing gate (stabilize training).
        detach_score: detach score (gate becomes constant w.r.t. token features).
    """

    def __init__(
        self,
        keep_ratio: float = 0.5,
        temperature: float = 1.0,
        min_gate: float = 0.0,
        detach_proxy: bool = True,
        detach_score: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.keep_ratio = float(keep_ratio)
        self.temperature = float(temperature)
        self.min_gate = float(min_gate)
        self.detach_proxy = bool(detach_proxy)
        self.detach_score = bool(detach_score)
        self.eps = float(eps)

    def forward(self, token: torch.Tensor, proxy: torch.Tensor) -> torch.Tensor:
        # token: [B,C] or [B,1,C]
        if token.dim() == 3:
            tok = token.squeeze(1)
            squeeze_back = True
        elif token.dim() == 2:
            tok = token
            squeeze_back = False
        else:
            raise ValueError(f"ProxyFeatureGate expects token dim 2/3, got {token.shape}")

        if proxy.dim() != 2 or tok.shape != proxy.shape:
            raise ValueError(f"token/proxy shape mismatch: {tok.shape} vs {proxy.shape}")

        px = proxy.detach() if self.detach_proxy else proxy

        # S1: per-dim contribution score
        score = (tok * px).abs()
        if self.detach_score:
            score = score.detach()

        B, C = score.shape
        k = max(1, int(round(C * self.keep_ratio)))

        # M2: keep-top-K dims as identity evidence (hard keep)
        _, topk_idx = torch.topk(score, k=k, dim=-1, largest=True, sorted=False)
        hard_mask = torch.zeros_like(score)
        hard_mask.scatter_(-1, topk_idx, 1.0)

        # soft gate on non-top-k dims
        mu = score.mean(dim=-1, keepdim=True)
        sigma = score.std(dim=-1, keepdim=True).clamp_min(self.eps)
        z = (score - mu) / sigma
        z = z / max(self.temperature, self.eps)
        soft_gate = torch.sigmoid(z)

        if self.min_gate > 0:
            soft_gate = soft_gate * (1.0 - self.min_gate) + self.min_gate

        gate = soft_gate * (1.0 - hard_mask) + hard_mask  # top-k => 1, rest => soft

        out = tok * gate
        return out.unsqueeze(1) if squeeze_back else out
