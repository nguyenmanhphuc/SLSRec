"""Behavior-dependent linear recurrent unit (BD-LRU) for the LRURec harness.

Port of RecBLR's ``GatedRecurrentLayer`` recurrence (Liu et al., CIKM'24) into the
LRURec tensor interface, parameterised exactly as SLSRec eq. (9)-(12):

    r_t = Sigmoid(W4 x_t + b4)                                          (9)
    i_t = Sigmoid(W5 x_t + b5)                                          (10)
    a_t = Exp(c * Log(Sigmoid(lambda)) (*) r_t)                         (11)
    h_t = a_t (*) h_{t-1} + sqrt(1 - a_t^2) (*) (i_t (*) x_t)           (12)

Interface
    input : [batch, seq_len, hidden]
    mask  : [batch, seq_len]  (True == real token, False == left padding)
    output: [batch, seq_len, hidden]
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Scans for h_t = a_t * h_{t-1} + b_t  (h_{-1} = 0)
# --------------------------------------------------------------------------- #
def sequential_scan(a, b):
    """Reference O(L) loop. a, b: [B, L, D] -> [B, L, D]."""
    outputs, h = [], torch.zeros_like(b[:, 0])
    for t in range(b.size(1)):
        h = a[:, t] * h + b[:, t]
        outputs.append(h)
    return torch.stack(outputs, dim=1)


def hillis_steele_scan(a, b):
    """Work-efficient-enough O(L log L) associative scan in pure PyTorch.

    Handles any sequence length (no power-of-two padding needed) and is fully
    differentiable.  After round k:
        h_t = sum_{j=t-2^k+1}^{t} (prod_{m=j+1}^{t} a_m) b_j
        A_t = prod_{m=t-2^k+1}^{t} a_m
    """
    length = b.size(1)
    h, cum_a = b, a
    step = 1
    while step < length:
        h = torch.cat([h[:, :step], h[:, step:] + cum_a[:, step:] * h[:, :-step]], dim=1)
        if step * 2 < length:  # last round never uses cum_a again
            cum_a = torch.cat([cum_a[:, :step], cum_a[:, step:] * cum_a[:, :-step]], dim=1)
        step *= 2
    return h


def _triton_scan(a, b):
    """RecBLR's Triton kernel.  Requires CUDA + triton and a power-of-two length."""
    from .triton_scan import parallel_scan as _ps  # local import: optional dep

    length = b.size(1)
    pad_len = 2 ** max(0, (length - 1).bit_length()) - length
    if pad_len:
        a = F.pad(a, (0, 0, pad_len, 0), value=0.0)
        b = F.pad(b, (0, 0, pad_len, 0), value=0.0)
    h = _ps(a.mT.contiguous(), b.mT.contiguous()).mT
    return h[:, pad_len:] if pad_len else h


def parallel_scan(a, b, impl='torch'):
    if impl == 'torch':
        return hillis_steele_scan(a, b)
    if impl == 'sequential':
        return sequential_scan(a, b)
    if impl == 'triton':
        return _triton_scan(a, b)
    raise ValueError(f'unknown scan_impl: {impl}')


# --------------------------------------------------------------------------- #
# Causal depth-wise convolution
# --------------------------------------------------------------------------- #
class CausalDepthwiseConv1d(nn.Module):
    """Depth-wise 1-D convolution that only looks at the current and past steps.

    Padded (left-padding) positions are zeroed *before* convolving so that the
    number of pad tokens cannot change the output at any real position.
    """

    def __init__(self, d_model, kernel_size=4, activation='silu'):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                              groups=d_model, padding=kernel_size - 1, bias=True)
        self.activation = activation

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        length = x.size(1)
        out = self.conv(x.transpose(1, 2))[..., :length].transpose(1, 2)  # drop right pad -> causal
        if self.activation == 'silu':
            out = F.silu(out)
        elif self.activation not in ('none', None):
            raise ValueError(f'unknown conv activation: {self.activation}')
        return out


# --------------------------------------------------------------------------- #
# BD-LRU
# --------------------------------------------------------------------------- #
class BDLRU(nn.Module):
    """Behaviour-dependent linear recurrent unit, SLSRec eq. (9)-(12).

    ``perturb=True`` applies SLSRec eq. (17): every *real* position is replaced,
    independently with probability ``perturb_prob``, by a vector whose entries
    are drawn i.i.d. from U(0, 1).  The replacement happens on ``x`` *before*
    eq. (9), so r_t, i_t and the state update all see the perturbed input.
    """

    def __init__(self, d_model, c=8.0, r_min=0.9, r_max=0.999,
                 lambda_init='linspace', scan_impl='torch',
                 perturb_prob=0.1, perturb_low=0.0, perturb_high=1.0):
        super().__init__()
        self.d_model = d_model
        self.c = c
        self.scan_impl = scan_impl
        self.perturb_prob = perturb_prob
        self.perturb_low = perturb_low
        self.perturb_high = perturb_high

        # r_t and i_t from a single fused linear (equivalent to W4/b4, W5/b5)
        self.gates = nn.Linear(d_model, 2 * d_model, bias=True)

        # lambda such that, with r_t == 1, a_t = exp(c*log(sigmoid(lambda)))
        # is spread over [r_min, r_max]  ->  lambda = logit(a^(1/c))
        if lambda_init == 'linspace':
            a0 = torch.linspace(r_min, r_max, d_model)
        elif lambda_init == 'uniform':
            a0 = torch.rand(d_model) * (r_max - r_min) + r_min
        else:
            raise ValueError(f'unknown lambda_init: {lambda_init}')
        s0 = a0.pow(1.0 / c).clamp(1e-6, 1 - 1e-6)
        self.Lambda = nn.Parameter(torch.log(s0) - torch.log1p(-s0))

        self.last_perturb_mask = None  # exposed for tests / logging

    def _perturb(self, x, mask):
        """SLSRec eq. (17).  Returns the perturbed x and the boolean mask used."""
        keep = torch.rand(x.shape[:2], device=x.device, dtype=x.dtype)
        replace = keep <= self.perturb_prob
        if mask is not None:  # never perturb padding
            replace = replace & mask.bool()
        noise = torch.rand_like(x) * (self.perturb_high - self.perturb_low) + self.perturb_low
        replace_f = replace.unsqueeze(-1).to(x.dtype)
        return x * (1.0 - replace_f) + noise * replace_f, replace

    def forward(self, x, mask=None, perturb=False):
        if perturb and self.perturb_prob > 0:
            x, self.last_perturb_mask = self._perturb(x, mask)
        else:
            self.last_perturb_mask = None

        recurrence, inp = self.gates(x).chunk(2, dim=-1)      # eq. (9), (10)
        r_t, i_t = torch.sigmoid(recurrence), torch.sigmoid(inp)

        log_sigmoid_lambda = F.logsigmoid(self.Lambda)         # <= 0  -> a_t in (0, 1]
        a_t = torch.exp(self.c * log_sigmoid_lambda * r_t)     # eq. (11)
        b_t = torch.sqrt((1.0 - a_t.pow(2)).clamp_min(1e-8)) * (i_t * x)   # eq. (12) input term

        if mask is not None:  # padded steps contribute nothing to the state
            b_t = b_t * mask.unsqueeze(-1).to(b_t.dtype)

        h = parallel_scan(a_t, b_t, impl=self.scan_impl)        # eq. (12)
        if mask is not None:
            h = h * mask.unsqueeze(-1).to(h.dtype)
        return h
