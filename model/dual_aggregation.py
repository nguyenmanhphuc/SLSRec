"""Dual Aggregation from FiCoRec (Zhang et al., RecSys 2026, Sec. 4.3) -- an extension,
not part of SLSRec.

FiCoRec replaces "the hidden state of the last item" by a gated mix of
  Tail Aggregation    s_j = w_tail' H_j - alpha * D_j  over the last K real positions, (6)-(7)
  Global Aggregation  s_j = q' H_j  with an L2-normalised learnable query q, all real positions (8)
  Gating Fusion       g = sigmoid(W_g [H_tail || H_glob] + b_g),
                      H_final = g (*) H_tail + (1 - g) (*) H_glob                          (9)-(10)

FiCoRec aggregates once per sequence and predicts from the last position.  This
harness trains every position t to predict item t+1, so the aggregation is made
*causal*: position t pools only positions j <= t, with distance D = t - j and the
tail window (t-K, t].  At the last position this is exactly eqs. (6)-(10).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualAggregation(nn.Module):
    def __init__(self, D, K=5):
        super().__init__()
        self.K = K
        self.w_tail = nn.Linear(D, 1, bias=False)
        self.alpha_raw = nn.Parameter(torch.zeros(()))           # alpha = softplus(.) > 0
        self.query = nn.Parameter(torch.randn(D) / D ** 0.5)
        self.gate = nn.Linear(2 * D, D)
        self.last_g = None  # exposed for logging

    @staticmethod
    def _pool(scores, allowed, H):
        """softmax over allowed j of scores [B, L, L], rows with nothing allowed -> 0."""
        has = allowed.any(-1, keepdim=True)
        s = scores.masked_fill(~allowed, float('-inf')).masked_fill(~has, 0.0)
        a = torch.softmax(s, dim=-1) * has.to(H.dtype)
        return torch.bmm(a, H)

    def forward(self, H, mask):
        B, L, _ = H.shape
        m = mask.bool()
        idx = torch.arange(L, device=H.device)
        dist = (idx.view(L, 1) - idx.view(1, L)).to(H.dtype)          # t - j
        causal = dist >= 0
        allowed = causal.view(1, L, L) & m.view(B, 1, L) & m.view(B, L, 1)

        # Tail: the last K *real* positions up to t (counted over real items, not raw slots)
        rank = torch.cumsum(m.long(), dim=1)                          # real-item index of each slot
        in_tail = (rank.view(B, L, 1) - rank.view(B, 1, L)) < self.K
        alpha = F.softplus(self.alpha_raw)
        s_tail = self.w_tail(H).squeeze(-1).view(B, 1, L) - alpha * dist.view(1, L, L)
        h_tail = self._pool(s_tail, allowed & in_tail, H)            # (6)-(7)

        q = F.normalize(self.query, dim=0)
        s_glob = (H @ q).view(B, 1, L).expand(B, L, L)
        h_glob = self._pool(s_glob, allowed, H)                       # (8)

        g = torch.sigmoid(self.gate(torch.cat([h_tail, h_glob], dim=-1)))   # (9)
        self.last_g = g.detach()
        out = g * h_tail + (1.0 - g) * h_glob                          # (10)
        return out * m.unsqueeze(-1).to(out.dtype)
