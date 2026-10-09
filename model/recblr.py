"""RecBLR -- Behavior-Dependent Linear Recurrent Units (Liu et al., CIKM'24).

Port of the official implementation (https://github.com/chengkai-liu/RecBLR,
``RecBLR.py``) into the LRURec harness, so that it trains and evaluates under
exactly the same data, schedule and full-sort protocol as SLSRec.  The SLSRec
paper's RecBLR column (Table II, ML-1M Recall@20 = 0.44979) was produced the
same way ("all of them except for STOSA are implemented using the LRURec code
repository"); RecBLR's own paper reports a RecBole number (HR@20 = 0.4336) that
is not comparable.

Kept from the official code, line for line:
    * item embedding only -- no positional embedding (the recurrence carries order)
    * X^0 = LayerNorm(Dropout(E)), LayerNorm eps = 1e-12
    * input projection D -> 2d without bias, split into the recurrence input x and
      the output gate z
    * causal depth-wise conv (kernel d_conv) + SiLU on x
    * alpha = exp(-softplus(Lambda) * sigmoid(W_r x)) -- no c = 8 (SLSRec eq. 11 has it)
    * beta  = sqrt(1 - alpha^2 + 1e-8) * sigmoid(W_i x)
    * h = scan(alpha, beta * x);  out = W_o(SiLU(z) * h), W_o without bias
    * Lambda = linspace(softplus^-1(-log r_min), softplus^-1(-log r_max), d)
    * LayerNorm(Dropout(out) + x), then FFN: LN(Dropout(W2 Dropout(SiLU(W1 x))) + x)
    * init: Linear/Embedding ~ N(0, 0.02), biases 0, LayerNorm (1, 0)
    * logits = x E^T (no item bias)

The one adaptation: RecBole right-pads, so in the official code padding can
never reach a real position of a causal model.  The LRURec harness left-pads,
so padded steps are zeroed before the convolution and removed from the scan
input -- the state at the first real item starts from 0, as it does there.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bdlru import parallel_scan


def softplus_inverse(x):
    return torch.log(torch.exp(x) - 1)


class GatedRecurrentLayer(nn.Module):
    """RecBLR's behaviour-modelling layer (conv -> BD-LRU, gated by SiLU(z))."""

    def __init__(self, d_model=64, expansion_factor=2, kernel_size=4, scan_impl='torch',
                 r_min=0.9, r_max=0.999):
        super().__init__()
        l = softplus_inverse(torch.tensor(-math.log(r_min))).item()
        r = softplus_inverse(torch.tensor(-math.log(r_max))).item()

        hidden = int(d_model * expansion_factor)
        self.scan_impl = scan_impl
        self.input = nn.Linear(d_model, 2 * hidden, bias=False)
        self.conv1d = nn.Conv1d(in_channels=hidden, out_channels=hidden, bias=True,
                                kernel_size=kernel_size, groups=hidden, padding=kernel_size - 1)
        self.gates = nn.Linear(hidden, 2 * hidden, bias=True)
        self.Lambda = nn.Parameter(torch.linspace(l, r, hidden))
        self.output = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x, mask):
        seq_len = x.size(1)
        m = mask.unsqueeze(-1).to(x.dtype)

        x, z = self.input(x).chunk(2, dim=-1)
        x = x * m  # left padding -> zeros, like RecBLR's own power-of-two left pad
        x = F.silu(self.conv1d(x.transpose(1, 2))[..., :seq_len].transpose(1, 2))

        recurrence, inp = self.gates(x).chunk(2, dim=-1)
        alpha = torch.exp(-F.softplus(self.Lambda) * torch.sigmoid(recurrence))
        beta = torch.sqrt(1 - alpha.pow(2) + 1e-8) * torch.sigmoid(inp)
        h = parallel_scan(alpha, beta * x * m, impl=self.scan_impl)

        return self.output(F.silu(z) * h)


class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, dropout=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.dropout(F.silu(self.w_1(input_tensor)))
        hidden_states = self.dropout(self.w_2(hidden_states))
        return self.layer_norm(hidden_states + input_tensor)


class RecurrentLayer(nn.Module):
    def __init__(self, d_model, d_conv, expand, dropout, scan_impl='torch'):
        super().__init__()
        self.behavior_modeling = GatedRecurrentLayer(
            d_model=d_model, expansion_factor=expand, kernel_size=d_conv, scan_impl=scan_impl)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.ffn = FeedForward(d_model=d_model, inner_size=d_model * 4, dropout=dropout)

    def forward(self, input_tensor, mask):
        hidden_states = self.behavior_modeling(input_tensor, mask)
        hidden_states = self.layer_norm(self.dropout(hidden_states) + input_tensor)
        return self.ffn(hidden_states)


class RecBLR(nn.Module):
    """Matches the LRURec ``LRU`` module signature: forward(seqs) -> (logits, None).

    Hyperparameters map onto the shared flags: ``--bert_hidden_units`` (64),
    ``--bert_num_blocks`` (2), ``--bert_dropout`` (0.2), ``--slsrec_expand`` (2),
    ``--slsrec_d_conv`` (4), ``--scan_impl`` -- RecBLR's config.yaml values.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        D = args.bert_hidden_units
        self.item_embedding = nn.Embedding(args.num_items + 1, D, padding_idx=0)
        self.layer_norm = nn.LayerNorm(D, eps=1e-12)
        self.dropout = nn.Dropout(args.bert_dropout)
        self.recurrent_layers = nn.ModuleList([
            RecurrentLayer(d_model=D, d_conv=args.slsrec_d_conv, expand=args.slsrec_expand,
                           dropout=args.bert_dropout, scan_impl=args.scan_impl)
            for _ in range(args.bert_num_blocks)
        ])
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def encode(self, seqs):
        mask = seqs > 0
        x = self.layer_norm(self.dropout(self.item_embedding(seqs)))
        for layer in self.recurrent_layers:
            x = layer(x, mask)
        return x

    def forward(self, seqs, labels=None):
        x = self.encode(seqs)
        return torch.matmul(x, self.item_embedding.weight.transpose(0, 1)), None
