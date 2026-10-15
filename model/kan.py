"""KAN fusion of the two SG-LRU branches -- a research extension, NOT in the SLSRec paper.

SLSRec fuses the gate branch G and the recurrent branch H by an element-wise
product (eq. 6): L3 = (G (*) H) W3 + b3.  Here that product is replaced by a
Kolmogorov-Arnold network (Liu et al., 2024) over the concatenation [G; H]:

  kan      fused = KAN2(LN([G; H]))            two KAN layers, 2d -> hidden -> d.
                   Two layers are needed: a single KAN layer is a sum of
                   univariate functions phi(g_i) + psi(h_i) and cannot express
                   the product g_i * h_i that eq. (6) relies on.
  kan_res  fused = G (*) H + KAN1(LN([G; H]))   one KAN layer added on top of the
                   product, zero-initialised, so the model starts out exactly as
                   eq. (6) and only learns a correction.

KANLinear follows the efficient-KAN formulation: out = W_b silu(x) + sum of
learnable cubic B-splines on a fixed grid.  Inputs are layer-normalised so they
fall on the spline grid.  Everything is position-wise, so causality and padding
invariance are preserved (padded positions are zeroed explicitly).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, grid_range=(-2.0, 2.0)):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.grid_size, self.spline_order = grid_size, spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h + grid_range[0]
        self.register_buffer('grid', grid.expand(in_features, -1).contiguous())
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size + spline_order))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, std=0.1 / math.sqrt(self.in_features))

    @torch.no_grad()
    def zero_(self):
        self.base_weight.zero_()
        self.spline_weight.zero_()

    def b_splines(self, x):
        """x: [N, in] -> B-spline bases [N, in, grid_size + spline_order]."""
        g = self.grid.to(x.dtype)
        x = x.unsqueeze(-1)
        bases = ((x >= g[:, :-1]) & (x < g[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - g[:, :-(k + 1)]) / (g[:, k:-1] - g[:, :-(k + 1)]) * bases[:, :, :-1]
                     + (g[:, k + 1:] - x) / (g[:, k + 1:] - g[:, 1:(-k)]) * bases[:, :, 1:])
        return bases

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, self.in_features)
        base = F.linear(F.silu(x), self.base_weight)
        spline = F.linear(self.b_splines(x).reshape(x.size(0), -1),
                          self.spline_weight.reshape(self.out_features, -1))
        return (base + spline).reshape(*shape[:-1], self.out_features)


class KANFusion(nn.Module):
    """Fusion of G and H.  'kan' / 'kan_res' replace or correct the product;
    'kan_gate' / 'mlp_gate' are KAPTRec C1, a learned *branch selection*:

        o = sigmoid(f(LN([G; H]))),   F = o (*) H + (1 - o) (*) G

    with f a single KAN layer 2d -> d (kan_gate) or a two-layer MLP (mlp_gate, the
    efficiency ablation).  f starts at zero, i.e. o = 0.5 and F = (G + H) / 2.
    """

    MODES = ('kan', 'kan_res', 'kan_gate', 'mlp_gate')

    def __init__(self, mode, d, grid_size=5, spline_order=3, hidden=0):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f'unknown KAN fusion mode: {mode}')
        self.mode = mode
        self.last_o = None  # exposed for logging (gate modes)
        # named layer_norm so the model's global truncated-normal init leaves it alone
        self.layer_norm = nn.LayerNorm(2 * d)
        if mode == 'kan':
            hid = hidden or d
            self.layers = nn.ModuleList([KANLinear(2 * d, hid, grid_size, spline_order),
                                         KANLinear(hid, d, grid_size, spline_order)])
        elif mode == 'mlp_gate':
            hid = hidden or d
            self.layers = nn.ModuleList([nn.Linear(2 * d, hid), nn.GELU(), nn.Linear(hid, d)])
        else:  # kan_res, kan_gate
            self.layers = nn.ModuleList([KANLinear(2 * d, d, grid_size, spline_order)])
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        """Called again by the model after its global init, which would overwrite
        these weights.  kan_res starts at exactly G (*) H; the gate modes at o = 0.5."""
        self.layer_norm.reset_parameters()
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        if self.mode in ('kan_res', 'kan_gate'):
            self.layers[-1].zero_()
        elif self.mode == 'mlp_gate':
            self.layers[-1].weight.zero_()
            self.layers[-1].bias.zero_()

    def forward(self, x, gate, h, mask=None):
        z = self.layer_norm(torch.cat([gate, h], dim=-1))
        for layer in self.layers:
            z = layer(z)
        if self.mode == 'kan_res':
            out = gate * h + z
        elif self.mode in ('kan_gate', 'mlp_gate'):
            o = torch.sigmoid(z)
            self.last_o = o.detach()
            out = o * h + (1.0 - o) * gate
        else:
            out = z
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out
