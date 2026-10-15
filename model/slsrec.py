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
    (20) Log(sigma^2) = Mc[i,:]' LN(Drop(PFFN(xc))+xc)-> UncertaintyEstimator (Fig. 2)
    (21) L_Unc = -Log(ytilde_ui)                      -> trainer/slsrec.py
    (22) L_Unc ~ (1/sigma^2) L_Rec + beta Log(sigma)  -> trainer/slsrec.py (approx mode)
    (23)/(24) L = (1 + alpha/sigma^2) L_Rec + L_Per   -> trainer/slsrec.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adaptive_gate import AdaptiveGate, FusionGate, gate_statistics
from .bdlru import BDLRU, CausalDepthwiseConv1d, TokenPerturbation
from .entmax import entmax_bisect
from .kan import KANFusion


def _perturbation_options(args):
    """eq. (17) and its alternative readings (--perturbation_site/_noise/_granularity/_gates)."""
    return dict(site=getattr(args, 'perturbation_site', 'bdlru'),
                noise=getattr(args, 'perturbation_noise', 'uniform'),
                granularity=getattr(args, 'perturbation_granularity', 'position'),
                gates=getattr(args, 'perturbation_gates', 'perturbed'))


def _token_perturbation(args, opts):
    return TokenPerturbation(args.perturbation_probability, args.perturbation_low,
                             args.perturbation_high, opts['noise'], opts['granularity'])


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

    ATTENTIONS = ('softmax', 'entmax15', 'sparsemax', 'entmax_adaptive')

    def __init__(self, d, num_heads=1, dropout=0.0, attention='softmax', entmax_iters=50):
        super().__init__()
        assert d % num_heads == 0
        if attention not in self.ATTENTIONS:
            raise ValueError(f'unknown sam_attention: {attention}')
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        self.w_q = nn.Linear(d, d, bias=False)
        self.w_k = nn.Linear(d, d, bias=False)
        self.w_v = nn.Linear(d, d, bias=False)
        self.dropout = nn.Dropout(dropout)
        # --- research extension (SAFA): sparse attention, not in the SLSRec paper --- #
        # entmax_adaptive learns alpha in (1, 2) as SAFA eq. (7), alpha = sigmoid(w.c + b) + 1,
        # but with c the *causal* running mean of the inputs up to each position, so a
        # future item cannot change how sparse the attention of an earlier one is.
        self.attention = attention
        self.entmax_iters = entmax_iters
        self.last_alpha = None  # exposed for logging
        if attention == 'entmax_adaptive':
            self.alpha_proj = nn.Linear(d, 1)

    def _normalise(self, scores, x, mask):
        """softmax of eq. (8), or alpha-entmax (SAFA) -- masked scores are -inf."""
        if self.attention == 'softmax':
            return torch.softmax(scores, dim=-1)
        B, H, L, _ = scores.shape
        if self.attention == 'entmax15':
            alpha = 1.5
        elif self.attention == 'sparsemax':
            alpha = 2.0
        else:
            m = (mask if mask is not None else torch.ones(B, L, device=x.device)).to(x.dtype)
            running = torch.cumsum(x * m.unsqueeze(-1), dim=1) / torch.cumsum(m, 1).clamp_min(1).unsqueeze(-1)
            a = torch.sigmoid(self.alpha_proj(running)) + 1.0            # [B, L, 1]
            a = a.clamp(1.01, 2.0)    # alpha -> 1 makes 1/(alpha-1) explode
            self.last_alpha = a.detach()
            alpha = a.view(B, 1, L, 1)
        return entmax_bisect(scores, alpha, self.entmax_iters)

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

        attn = self.dropout(self._normalise(scores, x, mask))
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
            self.sam = CausalSelfAttention(d, args.sam_num_heads, args.sam_attn_dropout,
                                           attention=getattr(args, 'sam_attention', 'softmax'),
                                           entmax_iters=getattr(args, 'entmax_iters', 50))
        if self.sam_conv:
            self.conv1 = CausalDepthwiseConv1d(d, args.slsrec_d_conv, args.sam_conv_activation)

        # --- research extension: adaptive gating (not in the SLSRec paper) --- #
        # Only meaningful when both gate branches exist; otherwise there is
        # nothing to mix and eq. (3) degenerates to a single term.
        self.gate_mode = getattr(args, 'gate_mode', 'sum')
        self.adaptive_gate = None
        if self.gate_mode != 'sum' and self.use_sam_gate and self.sam_conv:
            self.adaptive_gate = AdaptiveGate(
                self.gate_mode, d_in=D, d=d,
                granularity=getattr(args, 'gate_granularity', 'channel'),
                hidden=getattr(args, 'gate_hidden', 0))

        self.fusion_mode = getattr(args, 'fusion_mode', 'mul')
        self.fusion_gate = None
        if self.fusion_mode in KANFusion.MODES:                  # KAN fusion (extension)
            self.fusion_gate = KANFusion(self.fusion_mode, d,
                                         grid_size=getattr(args, 'kan_grid', 5),
                                         spline_order=getattr(args, 'kan_order', 3),
                                         hidden=getattr(args, 'kan_hidden', 0))
        elif self.fusion_mode != 'mul':
            self.fusion_gate = FusionGate(
                getattr(args, 'fusion_source', 'token'), d_in=D, d=d,
                granularity=getattr(args, 'gate_granularity', 'channel'),
                hidden=getattr(args, 'gate_hidden', 0))

        self.conv2 = CausalDepthwiseConv1d(d, args.slsrec_d_conv, args.bdlru_conv_activation)
        # eq. (17): 'bdlru' (the literal reading) perturbs x_t inside the BD-LRU;
        # 'layer' perturbs this layer's input X^(l-1) (gate branch, BD-LRU branch and
        # residual); 'bdlru_conv' perturbs L2 before Conv1-D2.  'embedding' lives in
        # SLSRecModel.  No parameters either way, so checkpoints are interchangeable.
        p = _perturbation_options(args)
        self.perturbation_site = p['site']
        self.perturbation = (_token_perturbation(args, p)
                             if p['site'] in ('layer', 'bdlru_conv') else None)
        if self.use_bdlru:
            self.bdlru = BDLRU(
                d, c=args.bdlru_c, r_min=args.bdlru_r_min, r_max=args.bdlru_r_max,
                lambda_init=args.bdlru_lambda_init, scan_impl=args.scan_impl,
                perturb_prob=args.perturbation_probability,
                perturb_low=args.perturbation_low, perturb_high=args.perturbation_high,
                perturb_noise=p['noise'], perturb_granularity=p['granularity'],
                perturb_gates=p['gates'],
            )

        # eq. (7).  --sglru_dropout overrides the shared rate here only: LRURec, which
        # SLSRec says it builds on, drives the matching position with bert_attn_dropout.
        sglru_p = getattr(args, 'sglru_dropout', None)
        self.dropout = nn.Dropout(args.bert_dropout if sglru_p is None else sglru_p)
        self.layer_norm = nn.LayerNorm(D)

    def forward(self, x, mask, perturb=False):
        x = x * mask.unsqueeze(-1).to(x.dtype)  # keep padding inert at every layer
        if self.perturbation_site == 'layer':
            x = self.perturbation(x, mask, perturb=perturb)

        l1 = self.w1(x)                                                        # (2)
        sam_out = self.sam(l1, mask) if self.use_sam_gate else None
        conv_out = self.conv1(l1, mask) if self.sam_conv else None

        if self.adaptive_gate is not None:                     # axis A (extension)
            g = self.adaptive_gate(x, sam_out, conv_out, mask)
        elif sam_out is not None and conv_out is not None:
            g = sam_out + conv_out                                             # (3)
        elif sam_out is not None or conv_out is not None:
            g = sam_out if sam_out is not None else conv_out
        else:
            g = F.silu(l1)  # RecBLR-style linear gate branch (ablation A)

        l2 = self.w2(x)                                                        # (4)
        if self.perturbation_site == 'bdlru_conv':
            l2 = self.perturbation(l2, mask, perturb=perturb)
        h = self.conv2(l2, mask)                                               # (5) Conv1-D2
        if self.use_bdlru:
            h = self.bdlru(h, mask, perturb=perturb and self.perturbation_site == 'bdlru')  # (5)

        if isinstance(self.fusion_gate, KANFusion):
            fused = self.fusion_gate(x, g, h, mask)
        else:
            fused = self.fusion_gate(x, g, h) if self.fusion_gate is not None else g * h
        l3 = self.w3(fused)                                                    # (6)
        return self.layer_norm(self.dropout(l3) + x)                           # (7)


# --------------------------------------------------------------------------- #
# PFFN -- eq. (13)/(14)
# --------------------------------------------------------------------------- #
class PFFN(nn.Module):
    """(GeLU(O W6) (*) (O W7)) W8 -- eq. (13).  No biases are given in the paper;
    ``bias=True`` (--pffn_bias) adds them, as the LRURec and RecBLR FFNs have."""

    def __init__(self, d_model, d_ff, bias=False):
        super().__init__()
        self.w6 = nn.Linear(d_model, d_ff, bias=bias)
        self.w7 = nn.Linear(d_model, d_ff, bias=bias)
        self.w8 = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):
        return self.w8(F.gelu(self.w6(x)) * self.w7(x))


class PFFNBlock(nn.Module):
    """X^(l) = LayerNorm(Dropout(PFFN(O)) + O) -- eq. (14)."""

    def __init__(self, d_model, d_ff, dropout, bias=False):
        super().__init__()
        self.pffn = PFFN(d_model, d_ff, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.layer_norm(self.dropout(self.pffn(x)) + x)


class UncertaintyPFFNBlock(PFFNBlock):
    """LayerNorm(Dropout(PFFN(xc) + xc)) -- eq. (20) exactly as printed.

    Only used with ``--unc_dropout_form=eq20``.  The default follows Fig. 2,
    which draws PFFN -> Dropout -> Add & Norm with the residual entering Add &
    Norm directly, i.e. plain PFFNBlock -- the same placement as every other
    block (Fig. 1, eq. 7, eq. 14).  The two forms agree exactly at eval time
    and differ only in whether training-time dropout also hits the residual.

    Reuses the module names of PFFNBlock, so checkpoints load under either form.
    """

    def forward(self, x):
        return self.layer_norm(self.dropout(self.pffn(x) + x))


class SLSRecBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        D = args.bert_hidden_units
        self.sg_lru = SGLRULayer(args)
        self.pffn_block = PFFNBlock(D, D * 4, args.bert_dropout,
                                    bias=getattr(args, 'pffn_bias', False))

    def forward(self, x, mask, perturb=False):
        return self.pffn_block(self.sg_lru(x, mask, perturb=perturb))


# --------------------------------------------------------------------------- #
# Uncertainty estimator -- eq. (20)
# --------------------------------------------------------------------------- #
class UncertaintyEstimator(nn.Module):
    """Log(sigma_u^2) = Mc[i,:]' LayerNorm(Dropout(PFFN(xc_u)) + xc_u)  (Fig. 2).

    ``dropout_form='eq20'`` moves the residual inside the dropout, as eq. (20)
    is printed: LayerNorm(Dropout(PFFN(xc_u) + xc_u)).  See UncertaintyPFFNBlock.

    Runs on its own item/position embedding tables (Mc, Pc) and is purely
    position-wise, so it never mixes information across time steps.
    """

    FORMS = {'fig2': PFFNBlock, 'eq20': UncertaintyPFFNBlock}

    def __init__(self, num_items, hidden, max_len, dropout, dropout_form='fig2', pffn_bias=False):
        super().__init__()
        if dropout_form not in self.FORMS:
            raise ValueError(f'unknown unc_dropout_form: {dropout_form}')
        self.dropout_form = dropout_form
        self.embedding = SLSRecEmbedding(num_items, hidden, max_len, dropout)
        self.pffn_block = self.FORMS[dropout_form](hidden, hidden * 4, dropout, bias=pffn_bias)

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
        p = _perturbation_options(args)
        self.embedding_perturbation = (_token_perturbation(args, p)
                                       if p['site'] == 'embedding' else None)
        self.blocks = nn.ModuleList([SLSRecBlock(args) for _ in range(args.bert_num_blocks)])
        self.use_item_bias = args.use_item_bias
        if self.use_item_bias:
            self.bias = nn.Parameter(torch.zeros(args.num_items + 1))
        self.uncertainty = (
            UncertaintyEstimator(args.num_items, D, args.bert_max_len, args.bert_dropout,
                                 dropout_form=getattr(args, 'unc_dropout_form', 'fig2'),
                                 pffn_bias=getattr(args, 'pffn_bias', False))
            if args.use_uncertainty else None
        )
        self.truncated_normal_init()
        # the global init above also overwrites the KAN splines; give them their own back
        for module in self.modules():
            if isinstance(module, KANFusion):
                module.reset_parameters()
        if getattr(args, 'gate_neutral_init', True):
            self.reset_gates_neutral()

    def reset_gates_neutral(self):
        """Put every adaptive gate back at its neutral point *after* the global
        truncated-normal init, so the model starts out bit-identical to the
        original SLSRec layer (g = 0.5 -> eq. 3, a = 1 -> eq. 6)."""
        for module in self.modules():
            if isinstance(module, (AdaptiveGate, FusionGate)):
                module.reset_neutral()

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
        if self.embedding_perturbation is not None:              # --perturbation_site=embedding
            x = self.embedding_perturbation(x, mask, perturb=perturb)
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
