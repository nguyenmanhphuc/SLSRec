"""SLSRec -- Self-Attention Gated Linear Recurrent Units for Sequential Recommendation.

Faithful reproduction of Yang, Bi, Huang & Li, IEEE TCSS (DOI 10.1109/TCSS.2026.3679563),
built on the LRURec training/evaluation harness and reusing RecBLR's BD-LRU recurrence.

Equation map (paper -> code)
    (1)  X^(0) = LayerNorm(Dropout(E + P))            -> SLSRecEmbedding
    (2)  L1 = X W1 + b1                               -> SGLRULayer.w1
    (3)  G  = SAM(L1) + Conv1-D1(L1)                  -> SGLRULayer.forward
    (4)  L2 = X W2 + b2                               -> SGLRULayer.w2
    (5)  H  = BD-LRU(Conv1-D2(L2))                    -> SGLRULayer.conv2 + BDLRU
    (6)  L3 = (G (*) H) W3 + b3                       -> SGLRULayer.w3
    (7)  O  = LayerNorm(Dropout(L3) + X^(l-1))        -> SGLRULayer.forward
    (8)  SAM(L1) = Softmax(QK'/sqrt(d) + S) V         -> CausalSelfAttention
    (9)-(12) BD-LRU recurrence                        -> model/bdlru.py::BDLRU
    (13) PFFN(O) = (GeLU(O W6) (*) (O W7)) W8         -> PFFN
    (14) X^(l) = LayerNorm(Dropout(PFFN(O)) + O)      -> PFFNBlock
    (15) y = Softmax(M x^(L))                         -> SLSRecModel.forward (logits)
    (16) L_Rec = -Log(y_ui)                           -> trainer/slsrec.py
    (17) perturbation x_t -> U(0,1) w.p. theta        -> model/bdlru.py::BDLRU._perturb
    (18) L_Per = -Log(ybar_ui)                        -> trainer/slsrec.py
    (19) ytilde = Softmax((1/sigma^2) M x^(L))        -> trainer/slsrec.py (exact mode)
    (20) Log(sigma^2) = Mc[i,:]' LN(Drop(PFFN(xc))+xc)-> UncertaintyEstimator
    (21) L_Unc = -Log(ytilde_ui)                      -> trainer/slsrec.py
    (22) L_Unc ~ (1/sigma^2) L_Rec + beta Log(sigma)  -> trainer/slsrec.py (approx mode)
    (23)/(24) L = (1 + alpha/sigma^2) L_Rec + L_Per   -> trainer/slsrec.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bdlru import BDLRU, CausalDepthwiseConv1d


# --------------------------------------------------------------------------- #
# Embedding -- eq. (1)
# --------------------------------------------------------------------------- #
class SLSRecEmbedding(nn.Module):
    """X^(0) = LayerNorm(Dropout(E_u + P)).

    Sequences are *left* padded by the LRURec dataloader, so positions are
    anchored to the right end of the sequence: the last item always receives
    P[max_len-1].  For the fixed-length batches the harness produces this is
    identical to plain left-to-right indexing, and it additionally makes every
    representation invariant to the amount of left padding.
    """

    def __init__(self, num_items, hidden, max_len, dropout):
        super().__init__()
        self.token = nn.Embedding(num_items + 1, hidden)
        self.position = nn.Embedding(max_len, hidden)
        self.layer_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

    def forward(self, x):
        mask = x > 0                                       # padding item id is 0
        length = x.size(1)
        pos = torch.arange(self.max_len - length, self.max_len, device=x.device)
        e = self.token(x) + self.position(pos).unsqueeze(0)
        e = self.layer_norm(self.dropout(e))
        return e * mask.unsqueeze(-1).to(e.dtype), mask


# --------------------------------------------------------------------------- #
# Self-attention -- eq. (8)
# --------------------------------------------------------------------------- #
class CausalSelfAttention(nn.Module):
    """SAM(L1) = Softmax((L1 WQ)(L1 WK)' / sqrt(d) + S)(L1 WV).

    S is the causal mask; padded *keys* are additionally masked out so that the
    result at a real position does not depend on how much padding precedes it.
    """

    def __init__(self, d, num_heads=1, dropout=0.0):
        super().__init__()
        assert d % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        self.w_q = nn.Linear(d, d, bias=False)
        self.w_k = nn.Linear(d, d, bias=False)
        self.w_v = nn.Linear(d, d, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split(self, t, B, L):
        return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, mask=None):
        B, L, _ = x.shape
        q = self._split(self.w_q(x), B, L)
        k = self._split(self.w_k(x), B, L)
        v = self._split(self.w_v(x), B, L)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = torch.ones(L, L, device=x.device, dtype=torch.bool).tril()
        allowed = causal.view(1, 1, L, L)
        if mask is not None:
            allowed = allowed & mask.view(B, 1, 1, L).bool()
        # A fully padded query row would give softmax(-inf) = NaN; let it attend
        # to itself and zero the row afterwards instead.
        allowed = allowed | torch.eye(L, device=x.device, dtype=torch.bool).view(1, 1, L, L)
        scores = scores.masked_fill(~allowed, float('-inf'))

        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, L, -1)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out


# --------------------------------------------------------------------------- #
# SG-LRU -- eq. (2)-(7)
# --------------------------------------------------------------------------- #
class SGLRULayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        D = args.bert_hidden_units
        d = int(D * args.slsrec_expand)
        self.args = args
        self.use_sam_gate = args.use_sam_gate
        self.sam_conv = args.sam_conv
        self.use_bdlru = args.use_bdlru

        self.w1 = nn.Linear(D, d, bias=True)                                  # (2)
        self.w2 = nn.Linear(D, d, bias=True)                                  # (4)
        self.w3 = nn.Linear(d, D, bias=True)                                  # (6)

        if self.use_sam_gate:
            self.sam = CausalSelfAttention(d, args.sam_num_heads, args.sam_attn_dropout)
        if self.sam_conv:
            self.conv1 = CausalDepthwiseConv1d(d, args.slsrec_d_conv, args.sam_conv_activation)

        self.conv2 = CausalDepthwiseConv1d(d, args.slsrec_d_conv, args.bdlru_conv_activation)
        if self.use_bdlru:
            self.bdlru = BDLRU(
                d, c=args.bdlru_c, r_min=args.bdlru_r_min, r_max=args.bdlru_r_max,
                lambda_init=args.bdlru_lambda_init, scan_impl=args.scan_impl,
                perturb_prob=args.perturbation_probability,
                perturb_low=args.perturbation_low, perturb_high=args.perturbation_high,
            )

        self.dropout = nn.Dropout(args.bert_dropout)
        self.layer_norm = nn.LayerNorm(D)

    def forward(self, x, mask, perturb=False):
        x = x * mask.unsqueeze(-1).to(x.dtype)  # keep padding inert at every layer

        l1 = self.w1(x)                                                        # (2)
        gate_terms = []
        if self.use_sam_gate:
            gate_terms.append(self.sam(l1, mask))
        if self.sam_conv:
            gate_terms.append(self.conv1(l1, mask))
        if gate_terms:
            g = gate_terms[0] if len(gate_terms) == 1 else gate_terms[0] + gate_terms[1]  # (3)
        else:
            g = F.silu(l1)  # RecBLR-style linear gate branch (ablation A)

        l2 = self.w2(x)                                                        # (4)
        h = self.conv2(l2, mask)                                               # (5) Conv1-D2
        if self.use_bdlru:
            h = self.bdlru(h, mask, perturb=perturb)                           # (5) BD-LRU

        l3 = self.w3(g * h)                                                    # (6)
        return self.layer_norm(self.dropout(l3) + x)                           # (7)


# --------------------------------------------------------------------------- #
# PFFN -- eq. (13)/(14)
# --------------------------------------------------------------------------- #
class PFFN(nn.Module):
    """(GeLU(O W6) (*) (O W7)) W8 -- eq. (13).  No biases are given in the paper."""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w6 = nn.Linear(d_model, d_ff, bias=False)
        self.w7 = nn.Linear(d_model, d_ff, bias=False)
        self.w8 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w8(F.gelu(self.w6(x)) * self.w7(x))


class PFFNBlock(nn.Module):
    """X^(l) = LayerNorm(Dropout(PFFN(O)) + O) -- eq. (14)."""

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.pffn = PFFN(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.layer_norm(self.dropout(self.pffn(x)) + x)


class SLSRecBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        D = args.bert_hidden_units
        self.sg_lru = SGLRULayer(args)
        self.pffn_block = PFFNBlock(D, D * 4, args.bert_dropout)

    def forward(self, x, mask, perturb=False):
        return self.pffn_block(self.sg_lru(x, mask, perturb=perturb))


# --------------------------------------------------------------------------- #
# Uncertainty estimator -- eq. (20)
# --------------------------------------------------------------------------- #
class UncertaintyEstimator(nn.Module):
    """Log(sigma_u^2) = Mc[i,:]' LayerNorm(Dropout(PFFN(xc_u)) + xc_u).

    Runs on its own item/position embedding tables (Mc, Pc) and is purely
    position-wise, so it never mixes information across time steps.
    """

    def __init__(self, num_items, hidden, max_len, dropout):
        super().__init__()
        self.embedding = SLSRecEmbedding(num_items, hidden, max_len, dropout)
        self.pffn_block = PFFNBlock(hidden, hidden * 4, dropout)

    def forward(self, seqs, labels):
        """seqs, labels: [B, L] -> log sigma^2: [B, L]."""
        xc, _ = self.embedding(seqs)
        z = self.pffn_block(xc)                                   # [B, L, D]
        mc = self.embedding.token(labels)                         # Mc[i, :], [B, L, D]
        return (mc * z).sum(-1)


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #
class SLSRecModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        D = args.bert_hidden_units
        self.embedding = SLSRecEmbedding(args.num_items, D, args.bert_max_len, args.bert_dropout)
        self.blocks = nn.ModuleList([SLSRecBlock(args) for _ in range(args.bert_num_blocks)])
        self.use_item_bias = args.use_item_bias
        if self.use_item_bias:
            self.bias = nn.Parameter(torch.zeros(args.num_items + 1))
        self.uncertainty = (
            UncertaintyEstimator(args.num_items, D, args.bert_max_len, args.bert_dropout)
            if args.use_uncertainty else None
        )
        self.truncated_normal_init()

    def truncated_normal_init(self, mean=0, std=0.02, lower=-0.04, upper=0.04):
        """Identical to LRURec's initialisation (layer norms and the recurrence
        parameter lambda keep their own initialisation)."""
        with torch.no_grad():
            l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
            u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.
            for n, p in self.named_parameters():
                if 'layer_norm' in n or 'Lambda' in n:
                    continue
                p.uniform_(2 * l - 1, 2 * u - 1)
                p.erfinv_()
                p.mul_(std * math.sqrt(2.))
                p.add_(mean)

    def encode(self, seqs, perturb=False):
        x, mask = self.embedding(seqs)
        for block in self.blocks:
            x = block(x, mask, perturb=perturb)
        return x, mask

    def logits_from_hidden(self, x):
        scores = torch.matmul(x, self.embedding.token.weight.transpose(0, 1))   # (15)
        if self.use_item_bias:
            scores = scores + self.bias
        return scores

    def forward(self, seqs, perturb=False):
        """seqs: [B, L] -> logits [B, L, |I|+1] for every position."""
        x, _ = self.encode(seqs, perturb=perturb)
        return self.logits_from_hidden(x)

    def log_variance(self, seqs, labels):
        """Log(sigma_u^2) per position -- eq. (20).  [B, L]."""
        assert self.uncertainty is not None
        logvar = self.uncertainty(seqs, labels)
        clamp = self.args.uncertainty_logvar_clamp
        if clamp is not None and clamp > 0:
            logvar = logvar.clamp(-clamp, clamp)
        return logvar


class SLSRec(nn.Module):
    """Thin wrapper matching the LRURec ``LRU`` module signature."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = SLSRecModel(args)

    def forward(self, seqs, labels=None, perturb=False):
        return self.model(seqs, perturb=perturb), None

    def log_variance(self, seqs, labels):
        return self.model.log_variance(seqs, labels)
