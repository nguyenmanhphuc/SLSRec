"""--sam_attention: alpha-entmax in the SAM (SAFA extension, not in the SLSRec paper)."""

import pytest
import torch

from conftest import make_args, make_batch, seed_all
from model.entmax import EntmaxBisectFunction, entmax_bisect
from model.slsrec import SLSRec
from trainer.slsrec import slsrec_losses

SPARSE = ['entmax15', 'sparsemax', 'entmax_adaptive']


def test_entmax_limits_and_masking():
    torch.manual_seed(0)
    X = torch.randn(4, 6, 9, dtype=torch.double)
    assert torch.allclose(entmax_bisect(X, 1.0001, 80), torch.softmax(X, -1), atol=1e-3)
    p = entmax_bisect(X, 2.0, 80)
    assert (p == 0).any() and torch.allclose(p.sum(-1), torch.ones(4, 6, dtype=torch.double))
    X[..., 6:] = float('-inf')
    p = entmax_bisect(X, 1.5)
    assert torch.equal(p[..., 6:], torch.zeros_like(p[..., 6:]))


def test_entmax_gradients_including_alpha():
    torch.manual_seed(1)
    X = torch.randn(2, 3, 6, dtype=torch.double, requires_grad=True)
    alpha = (torch.rand(2, 3, 1, dtype=torch.double) * 0.8 + 1.15).requires_grad_()
    assert torch.autograd.gradcheck(lambda x, a: EntmaxBisectFunction.apply(x, a, 100),
                                    (X, alpha), eps=1e-6, atol=1e-4)


def test_default_is_softmax_and_keeps_the_parameter_set():
    seed_all()
    base = SLSRec(make_args()).eval()
    for att in ['entmax15', 'sparsemax']:
        seed_all()
        m = SLSRec(make_args(sam_attention=att)).eval()
        assert list(base.state_dict()) == list(m.state_dict())
    seed_all()
    ada = SLSRec(make_args(sam_attention='entmax_adaptive'))
    assert any('alpha_proj' in n for n in ada.state_dict())


@pytest.mark.parametrize('att', SPARSE)
def test_sparse_attention_changes_the_sam_and_is_sparse(att):
    """Same weights, softmax vs entmax: the SAM output differs and some causal
    (allowed) weights are exactly zero.  Q/K are scaled up because at the default
    init all scores are ~0 and any normaliser is close to uniform."""
    seed_all()
    dense = SLSRec(make_args(bert_dropout=0.0)).eval()
    seed_all()
    sparse = SLSRec(make_args(sam_attention=att, bert_dropout=0.0)).eval()
    sparse.load_state_dict(dense.state_dict(), strict=False)
    with torch.no_grad():
        for m in (dense, sparse):
            for blk in m.model.blocks:
                blk.sg_lru.sam.w_q.weight.mul_(100.0)
                blk.sg_lru.sam.w_k.weight.mul_(100.0)
    seqs, _ = make_batch(make_args())
    outs, weights = {}, []
    hooks = [dense.model.blocks[0].sg_lru.sam.register_forward_hook(lambda m, i, o: outs.__setitem__('d', o)),
             sparse.model.blocks[0].sg_lru.sam.register_forward_hook(lambda m, i, o: outs.__setitem__('s', o)),
             sparse.model.blocks[0].sg_lru.sam.dropout.register_forward_hook(lambda m, i, o: weights.append(o))]
    with torch.no_grad():
        dense(seqs)
        sparse(seqs)
    for h in hooks:
        h.remove()
    assert not torch.allclose(outs['d'], outs['s'])
    L = seqs.size(1)
    causal = torch.ones(L, L, dtype=torch.bool).tril()
    real_keys = (seqs > 0).view(seqs.size(0), 1, 1, L)
    allowed = causal.view(1, 1, L, L) & real_keys
    assert (weights[0][allowed] == 0).any(), 'no exact zeros among the allowed weights'


@pytest.mark.parametrize('att', SPARSE)
def test_sparse_attention_stays_causal(att):
    """A change at position t must not affect positions < t (adaptive alpha included)."""
    seed_all()
    args = make_args(sam_attention=att)
    model = SLSRec(args).eval()
    seqs, _ = make_batch(args)
    alt = seqs.clone()
    t = args.bert_max_len - 3
    alt[:, t] = (alt[:, t] % args.num_items) + 1
    with torch.no_grad():
        a, _ = model.model.encode(seqs)
        b, _ = model.model.encode(alt)
    assert torch.allclose(a[:, :t], b[:, :t], atol=1e-6)


@pytest.mark.parametrize('att', SPARSE)
def test_sparse_attention_trains(att):
    seed_all()
    args = make_args(sam_attention=att, use_uncertainty=False)
    model = SLSRec(args).train()
    seqs, labels = make_batch(args, batch_size=8, real_len=args.bert_max_len - 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    first = None
    for _ in range(30):
        opt.zero_grad()
        l = slsrec_losses(args, model, seqs, labels, training=True)
        assert torch.isfinite(l['L_total'])
        l['L_total'].backward()
        opt.step()
        first = first if first is not None else l['L_rec'].item()
    assert l['L_rec'].item() < first
    if att == 'entmax_adaptive':
        alpha = model.model.blocks[0].sg_lru.sam.last_alpha
        assert alpha is not None and (alpha >= 1.01).all() and (alpha <= 2.0).all()
