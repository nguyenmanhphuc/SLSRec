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
# Perturbation injection -- eq. (17) and its alternative readings
# --------------------------------------------------------------------------- #
def _borrow_from_other_sequences(x, mask):
    """For every (b, t): the vector at a random real position of another sequence.

    Assumes the harness's left padding (real items right-aligned).  Detached, so
    the borrowed vector is pure noise to the perturbed pass.
    """
    B, L, _ = x.shape
    dev = x.device
    if mask is None:
        mask = torch.ones(B, L, dtype=torch.bool, device=dev)
    lengths = mask.long().sum(1)
    if B > 1:
        src_b = (torch.arange(B, device=dev).view(B, 1) + torch.randint(1, B, (B, L), device=dev)) % B
    else:
        src_b = torch.zeros(B, L, dtype=torch.long, device=dev)
    n = lengths[src_b].clamp_min(1)
    offset = torch.minimum((torch.rand(B, L, device=dev) * n).long(), n - 1)
    return x.detach()[src_b, L - 1 - offset]


def perturb_tokens(x, mask, prob, low=0.0, high=1.0, noise='uniform', granularity='position'):
    """SLSRec eq. (17).  Returns the perturbed x and the boolean replacement mask.

    granularity='position' is eq. (17): one draw p per (sequence, step) replaces the
    whole vector; 'element' replaces each component independently with probability
    ``prob``.  noise='uniform' is eq. (17): i.i.d. U(low, high); 'shuffle' uses the
    vector at a random real position of a different sequence in the batch -- noise
    whose scale always matches the activations, so the network cannot spot the
    replacement by its range.  Padding is never replaced.

    The default (position, uniform) draws its random numbers in the same order as
    the original implementation, so default runs are bit-for-bit unchanged.
    """
    if granularity == 'position':
        replace = torch.rand(x.shape[:2], device=x.device, dtype=x.dtype) <= prob
        if mask is not None:  # never perturb padding
            replace = replace & mask.bool()
        replace_f = replace.unsqueeze(-1).to(x.dtype)
    elif granularity == 'element':
        replace = torch.rand_like(x) <= prob
        if mask is not None:
            replace = replace & mask.bool().unsqueeze(-1)
        replace_f = replace.to(x.dtype)
    else:
        raise ValueError(f'unknown perturbation granularity: {granularity}')

    if noise == 'uniform':
        fill = torch.rand_like(x) * (high - low) + low
    elif noise == 'shuffle':
        fill = _borrow_from_other_sequences(x, mask)
    else:
        raise ValueError(f'unknown perturbation noise: {noise}')
    return x * (1.0 - replace_f) + fill * replace_f, replace


class TokenPerturbation(nn.Module):
    """eq. (17) at a site other than the BD-LRU input (``--perturbation_site``).

    No parameters, so checkpoints are interchangeable across sites.
    """

    def __init__(self, prob, low=0.0, high=1.0, noise='uniform', granularity='position'):
        super().__init__()
        self.prob, self.low, self.high = prob, low, high
        self.noise, self.granularity = noise, granularity
        self.last_perturb_mask = None  # exposed for tests / logging

    def forward(self, x, mask, perturb=True):
        if perturb and self.prob > 0:
            x, self.last_perturb_mask = perturb_tokens(
                x, mask, self.prob, self.low, self.high, self.noise, self.granularity)
        else:
            self.last_perturb_mask = None
        return x


# --------------------------------------------------------------------------- #
# BD-LRU
# --------------------------------------------------------------------------- #
class BDLRU(nn.Module):
    """Behaviour-dependent linear recurrent unit, SLSRec eq. (9)-(12).

    ``perturb=True`` applies SLSRec eq. (17): every *real* position is replaced,
    independently with probability ``perturb_prob``, by a vector whose entries
    are drawn i.i.d. from U(0, 1).  The replacement happens on ``x`` *before*
    eq. (9), so r_t, i_t and the state update all see the perturbed input.
    ``perturb_noise`` / ``perturb_granularity`` select the alternative readings
    of ``perturb_tokens``.

    ``perturb_gates='clean'`` computes r_t and i_t from the clean x_t and uses
    the replacement only in the input term of eq. (12).  With the literal
    reading the gates see the noise, and a trained model learns to recognise the
    U(0,1) vector and close r_t on it (a_t -> 1: the step is skipped), which is
    what makes L_per collapse onto L_rec; clean gates take that escape away.
    """

    def __init__(self, d_model, c=8.0, r_min=0.9, r_max=0.999,
                 lambda_init='linspace', scan_impl='torch',
                 perturb_prob=0.1, perturb_low=0.0, perturb_high=1.0,
                 perturb_noise='uniform', perturb_granularity='position',
                 perturb_gates='perturbed'):
        super().__init__()
        self.d_model = d_model
        self.c = c
        self.scan_impl = scan_impl
        self.perturb_prob = perturb_prob
        self.perturb_low = perturb_low
        self.perturb_high = perturb_high
        self.perturb_noise = perturb_noise
        self.perturb_granularity = perturb_granularity
        if perturb_gates not in ('perturbed', 'clean'):
            raise ValueError(f'unknown perturb_gates: {perturb_gates}')
        self.perturb_gates = perturb_gates

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
        return perturb_tokens(x, mask, self.perturb_prob, self.perturb_low, self.perturb_high,
                              self.perturb_noise, self.perturb_granularity)

    def forward(self, x, mask=None, perturb=False):
        gate_x = x
        if perturb and self.perturb_prob > 0:
            x, self.last_perturb_mask = self._perturb(x, mask)
            if self.perturb_gates == 'perturbed':
                gate_x = x
        else:
            self.last_perturb_mask = None

        recurrence, inp = self.gates(gate_x).chunk(2, dim=-1)  # eq. (9), (10)
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
