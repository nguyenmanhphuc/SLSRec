"""Ablation A as RecBLR (Liu et al., CIKM'24) inside the LRURec harness.

RecBLR differs from the SLSRec skeleton with the SAM gate removed in three
places, each behind a flag that defaults to SLSRec:
    --slsrec_pos_emb=false   RecBLR eq. (1): no positional embedding
    --slsrec_ffn=mlp         RecBLR eq. (4): SiLU(H W1 + b1) W2 + b2
    --bdlru_c=1              RecBLR eq. (8): a_t = exp(-softplus(Lambda) r_t)
"""

import pytest
import torch
import torch.nn.functional as F

from conftest import make_args, make_batch, seed_all
from model.bdlru import BDLRU
from model.slsrec import SLSRec, PFFNBlock

RECBLR = dict(use_sam_gate=False, sam_conv=False, use_perturbation=False,
              use_uncertainty=False, slsrec_pos_emb=False, slsrec_ffn='mlp', bdlru_c=1.0)


def _build(**over):
    seed_all()
    return SLSRec(make_args(**over))


def test_defaults_are_unchanged():
    """Explicit SLSRec values for the new flags give the very same model as omitting them."""
    seqs, _ = make_batch(make_args())
    implicit, explicit = _build(), _build(slsrec_pos_emb=True, slsrec_ffn='geglu')
    assert list(implicit.state_dict()) == list(explicit.state_dict())
    implicit.eval(), explicit.eval()
    with torch.no_grad():
        assert torch.equal(implicit(seqs)[0], explicit(seqs)[0])


def test_no_positional_embedding():
    args = make_args(**RECBLR)
    with_p = _build(**{**RECBLR, 'slsrec_pos_emb': True})
    without_p = _build(**RECBLR)
    names = [n for n, _ in without_p.named_parameters()]
    assert not any('embedding.position' in n for n in names)
    n_with = sum(p.numel() for p in with_p.parameters())
    n_without = sum(p.numel() for p in without_p.parameters())
    assert n_with - n_without == args.bert_max_len * args.bert_hidden_units

    # the uncertainty estimator keeps its own Pc (SLSRec eq. 20)
    unc = _build(slsrec_pos_emb=False, use_uncertainty=True)
    assert any('uncertainty.embedding.position' in n for n, _ in unc.named_parameters())
    assert not any(n.startswith('model.embedding.position') for n, _ in unc.named_parameters())


def test_mlp_ffn_is_recblr_eq4():
    D = 16
    block = PFFNBlock(D, 4 * D, dropout=0.0, ffn='mlp')
    ffn = block.pffn
    assert ffn.fc1.weight.shape == (4 * D, D) and ffn.fc1.bias.shape == (4 * D,)
    assert ffn.fc2.weight.shape == (D, 4 * D) and ffn.fc2.bias.shape == (D,)
    x = torch.randn(3, 5, D)
    ref = F.silu(x @ ffn.fc1.weight.T + ffn.fc1.bias) @ ffn.fc2.weight.T + ffn.fc2.bias
    assert torch.allclose(ffn(x), ref, atol=1e-6)
    assert torch.allclose(block(x), F.layer_norm(ref + x, (D,), block.layer_norm.weight,
                                                 block.layer_norm.bias), atol=1e-6)
    with pytest.raises(ValueError):
        PFFNBlock(D, 4 * D, dropout=0.0, ffn='swiglu')


def _recblr_reference(layer, x):
    """RecBLR eq. (5)-(8), written from the paper, with Lambda = -lambda."""
    w, b = layer.gates.weight, layer.gates.bias
    d = x.size(-1)
    r = torch.sigmoid(x @ w[:d].T + b[:d])                        # eq. (6)
    i = torch.sigmoid(x @ w[d:].T + b[d:])                        # eq. (7)
    big_lambda = -layer.Lambda
    alpha = torch.exp(-F.softplus(big_lambda) * r)                # eq. (8)
    beta = torch.sqrt(1 - alpha ** 2) * i
    h, out = torch.zeros_like(x[:, 0]), []
    for t in range(x.size(1)):
        h = alpha[:, t] * h + beta[:, t] * x[:, t]                # eq. (5)
        out.append(h)
    return torch.stack(out, dim=1)


def test_bdlru_c1_is_recblr_parameterisation():
    seed_all()
    d = 8
    layer = BDLRU(d, c=1.0, r_min=0.9, r_max=0.999, lambda_init='linspace')
    with torch.no_grad():
        layer.gates.weight.normal_(0, 0.5)
        layer.gates.bias.normal_(0, 0.5)

    # RecBLR's custom init: exp(-softplus(Lambda)) spread over [0.9, 0.999]
    a0 = torch.exp(-F.softplus(-layer.Lambda.detach()))
    assert torch.allclose(a0[[0, -1]], torch.tensor([0.9, 0.999]), atol=1e-5)

    x = torch.randn(2, 11, d)
    with torch.no_grad():
        assert torch.allclose(layer(x), _recblr_reference(layer, x), atol=1e-5)


def test_recblr_variant_trains_end_to_end():
    from trainer.slsrec import slsrec_losses
    args = make_args(**RECBLR)
    model = _build(**RECBLR)
    seqs, labels = make_batch(args)
    losses = slsrec_losses(args, model, seqs, labels, training=True)
    assert losses['L_perturb'].item() == 0.0 and losses['L_uncertainty'].item() == 0.0
    losses['L_total'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)
