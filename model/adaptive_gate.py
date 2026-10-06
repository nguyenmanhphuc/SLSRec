"""Adaptive gating for SG-LRU -- a research extension, NOT part of the SLSRec paper.

SLSRec fuses its two branches with two *fixed* rules (eq. 3 and eq. 6):

    G  = SAM(L1) + Conv1-D1(L1)          # unweighted sum, fixed 50/50
    L3 = (G (*) H) W3 + b3               # G always gates H at full strength

Neither rule can vary with the input, although the paper's own motivation is
that some positions need global context and others need strict ordering
(the "Janet" example in Sec. I).  This module makes both rules adaptive.

Two orthogonal axes, each selectable on its own:

  axis A -- how G is composed        (``--gate_mode``)
      G = 2 * [ g (*) SAM + (1-g) (*) Conv ],      g = sigmoid(z)
      z = 0  =>  g = 0.5  =>  G = SAM + Conv   (exactly eq. 3)

  axis B -- how strongly G gates H   (``--fusion_mode``)
      out = (1 + a (*) (G - 1)) (*) H,             a = 2 * sigmoid(z)
      z = 0  =>  a = 1      =>  out = G (*) H    (exactly eq. 6)

Both parameterisations are built so that **neutral initialisation reproduces the
original SLSRec layer bit-for-bit**.  Any measured difference is therefore
attributable to what the gate learns, not to a change of scale, of
initialisation, or of the optimisation problem at step 0.
"""

import torch
import torch.nn as nn


def _make_projection(in_dim, out_dim, hidden=0):
    if hidden and hidden > 0:
        return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
    return nn.Linear(in_dim, out_dim)


def _final_linear(module):
    return module[-1] if isinstance(module, nn.Sequential) else module


class AdaptiveGate(nn.Module):
    """Axis A: mixing coefficient g between the SAM branch and the conv branch.

    modes
        scalar   one learnable logit per layer          (static, 1 param)
        channel  one learnable logit per channel        (static, d params)
        token    g = sigmoid(W [X^(l-1)])               (input dependent)
        context  g = sigmoid(W [SAM ; Conv])            (input dependent, sees both branches)

    ``granularity`` picks between one coefficient per position (``scalar``) and
    one per position *and* channel (``channel``) for the input-dependent modes.
    """

    MODES = ('scalar', 'channel', 'token', 'context')

    def __init__(self, mode, d_in, d, granularity='channel', hidden=0):
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f'unknown gate_mode: {mode}')
        self.mode = mode
        self.d = d
        out = d if granularity == 'channel' else 1

        if mode == 'scalar':
            self.gate_logit = nn.Parameter(torch.zeros(1))
        elif mode == 'channel':
            self.gate_logit = nn.Parameter(torch.zeros(d))
        elif mode == 'token':
            self.proj = _make_projection(d_in, out, hidden)
        elif mode == 'context':
            self.proj = _make_projection(2 * d, out, hidden)

        self.last_g = None  # detached copy for analysis / logging

    @torch.no_grad()
    def reset_neutral(self):
        """Force g = 0.5 so the layer starts out identical to eq. (3)."""
        if self.mode in ('scalar', 'channel'):
            self.gate_logit.zero_()
        else:
            lin = _final_linear(self.proj)
            lin.weight.zero_()
            if lin.bias is not None:
                lin.bias.zero_()

    def coefficient(self, x, sam_out, conv_out):
        if self.mode == 'scalar':
            return torch.sigmoid(self.gate_logit).view(1, 1, 1)
        if self.mode == 'channel':
            return torch.sigmoid(self.gate_logit).view(1, 1, -1)
        if self.mode == 'token':
            return torch.sigmoid(self.proj(x))                     # [B, L, out]
        return torch.sigmoid(self.proj(torch.cat([sam_out, conv_out], dim=-1)))

    def forward(self, x, sam_out, conv_out, mask=None):
        g = self.coefficient(x, sam_out, conv_out)
        self.last_g = g.detach()  # cheap, detached -- for gate_statistics()
        # factor 2 keeps the scale of eq. (3): at g = 0.5 this is SAM + Conv
        return 2.0 * (g * sam_out + (1.0 - g) * conv_out)


class FusionGate(nn.Module):
    """Axis B: how strongly G is allowed to gate H.

        out = (1 + a (*) (G - 1)) (*) H,    a = 2 * sigmoid(z) in (0, 2)

    a = 1 gives eq. (6) exactly; a -> 0 removes the gating and passes H through;
    a > 1 sharpens it.  ``source`` decides what a is conditioned on.
    """

    SOURCES = ('token', 'context')

    def __init__(self, source, d_in, d, granularity='channel', hidden=0):
        super().__init__()
        if source not in self.SOURCES:
            raise ValueError(f'unknown fusion source: {source}')
        self.source = source
        out = d if granularity == 'channel' else 1
        in_dim = d_in if source == 'token' else 2 * d
        self.proj = _make_projection(in_dim, out, hidden)
        self.last_a = None

    @torch.no_grad()
    def reset_neutral(self):
        """Force a = 1 so the layer starts out identical to eq. (6)."""
        lin = _final_linear(self.proj)
        lin.weight.zero_()
        if lin.bias is not None:
            lin.bias.zero_()

    def forward(self, x, gate, h):
        src = x if self.source == 'token' else torch.cat([gate, h], dim=-1)
        a = 2.0 * torch.sigmoid(self.proj(src))
        self.last_a = a.detach()
        return (1.0 + a * (gate - 1.0)) * h


def gate_statistics(model):
    """Per-layer summary of what the gates actually learned.

    A near-zero std across positions means the gate collapsed to a static
    value, i.e. the extra capacity is not being used adaptively.
    """
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, AdaptiveGate) and module.last_g is not None:
            g = module.last_g.float()
            stats[name] = {
                'mode': module.mode,
                'mean': g.mean().item(),
                'std_over_positions': g.mean(-1).std().item() if g.dim() == 3 and g.size(1) > 1 else 0.0,
                'std_over_channels': g.std(-1).mean().item() if g.size(-1) > 1 else 0.0,
                'min': g.min().item(), 'max': g.max().item(),
            }
        elif isinstance(module, FusionGate) and module.last_a is not None:
            a = module.last_a.float()
            stats[name] = {
                'mode': 'fusion:' + module.source,
                'mean': a.mean().item(),
                'std_over_positions': a.mean(-1).std().item() if a.dim() == 3 and a.size(1) > 1 else 0.0,
                'std_over_channels': a.std(-1).mean().item() if a.size(-1) > 1 else 0.0,
                'min': a.min().item(), 'max': a.max().item(),
            }
    return stats
