# NEW/loss/proxy_anchor_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProxyAnchorLoss(nn.Module):
    """
    Proxy-Anchor Loss (CVPR 2020), Eq.(4).
    We use cosine similarity between normalized embeddings and normalized proxies.

    loss = (1/|P+|) * sum_{p in P+} log(1 + sum_{x in X_p+} exp(-alpha*(s(x,p)-delta)))
         + (1/|P| ) * sum_{p in P } log(1 + sum_{x in X_p-} exp( alpha*(s(x,p)+delta)))
    """
    def __init__(self, alpha: float = 32.0, delta: float = 0.1):
        super().__init__()
        self.alpha = float(alpha)
        self.delta = float(delta)

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor, proxies: torch.Tensor) -> torch.Tensor:
        # embeddings: [B,D] or [B,1,D]
        if embeddings.dim() == 3 and embeddings.size(1) == 1:
            embeddings = embeddings[:, 0, :]
        elif embeddings.dim() != 2:
            embeddings = embeddings.view(embeddings.size(0), -1)

        targets = targets.long()
        B, D = embeddings.shape
        C, Dp = proxies.shape
        assert D == Dp, f"ProxyAnchorLoss: dim mismatch emb {D} vs proxy {Dp}"

        # normalize for cosine similarity
        emb = F.normalize(embeddings, p=2, dim=1)
        prx = F.normalize(proxies, p=2, dim=1)

        # cosine sim: [B,C]
        sim = emb @ prx.t()

        # masks
        pos_mask = F.one_hot(targets, num_classes=C).bool()  # [B,C]
        neg_mask = ~pos_mask

        # exp sums per proxy
        pos_exp = torch.exp(-self.alpha * (sim - self.delta)) * pos_mask
        neg_exp = torch.exp( self.alpha * (sim + self.delta)) * neg_mask

        pos_sum = pos_exp.sum(dim=0)  # [C]
        neg_sum = neg_exp.sum(dim=0)  # [C]

        # P+ proxies that appear in batch
        pos_present = pos_mask.any(dim=0)  # [C]
        if pos_present.any():
            loss_pos = torch.log1p(pos_sum[pos_present]).mean()
        else:
            loss_pos = sim.new_tensor(0.0)

        loss_neg = torch.log1p(neg_sum).mean()
        return loss_pos + loss_neg
