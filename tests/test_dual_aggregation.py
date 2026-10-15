"""--dual_agg: FiCoRec Dual Aggregation, causal per position (extension, not in SLSRec)."""

import torch

from conftest import make_args, make_batch, seed_all
from model.dual_aggregation import DualAggregation
from model.slsrec import SLSRec
from trainer.slsrec import slsrec_losses


def _padded(B=3, L=10, real=6, D=8):
    H = torch.randn(B, L, D)
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[:, L - real:] = True
    return H * mask.unsqueeze(-1), mask


def test_k1_with_the_gate_open_returns_the_current_state():
    """K = 1 and g -> 1: H_final is the position's own hidden state (the usual readout)."""
    seed_all()
    agg = DualAggregation(8, K=1)
    with torch.no_grad():
        agg.gate.weight.zero_()
        agg.gate.bias.fill_(50.0)
    H, mask = _padded()
    out = agg(H, mask)
    assert torch.allclose(out, H, atol=1e-6)


def test_last_position_matches_the_sequence_level_formula():
    """At the last position the causal version is FiCoRec eqs. (6)-(10) verbatim."""
    seed_all()
    agg = DualAggregation(8, K=3)
    H, mask = _padded(B=1, L=10, real=6)
    out = agg(H, mask)[0, -1]
    h = H[0, -6:]                                          # the real items
    L = h.size(0)
    alpha = torch.nn.functional.softplus(agg.alpha_raw)
    s_tail = agg.w_tail(h).squeeze(-1) - alpha * (L - 1 - torch.arange(L).float())
    s_tail[:L - 3] = float('-inf')
    h_tail = torch.softmax(s_tail, 0) @ h
    q = torch.nn.functional.normalize(agg.query, dim=0)
    h_glob = torch.softmax(h @ q, 0) @ h
    g = torch.sigmoid(agg.gate(torch.cat([h_tail, h_glob])))
    assert torch.allclose(out, g * h_tail + (1 - g) * h_glob, atol=1e-5)


def test_padding_rows_are_zero_and_finite():
    seed_all()
    agg = DualAggregation(8, K=5)
    H, mask = _padded()
    mask[0] = False                                        # a row with no real item at all
    H[0] = 0
    out = agg(H, mask)
    assert torch.isfinite(out).all()
    assert torch.equal(out[~mask], torch.zeros_like(out[~mask]))


def test_model_with_dual_agg_is_causal_and_padding_invariant():
    seed_all()
    args = make_args(dual_agg=True, dual_agg_k=3, bert_max_len=24)
    model = SLSRec(args).eval()
    seqs, _ = make_batch(args)
    alt = seqs.clone()
    t = args.bert_max_len - 3
    alt[:, t] = (alt[:, t] % args.num_items) + 1
    with torch.no_grad():
        a = model.model.represent(seqs)
        b = model.model.represent(alt)
    assert torch.allclose(a[:, :t], b[:, :t], atol=1e-6)

    body = torch.randint(1, args.num_items + 1, (3, 10))

    def rep(total):
        s = torch.zeros(3, total, dtype=torch.long)
        s[:, -10:] = body
        with torch.no_grad():
            return model.model.represent(s)[:, -10:]

    assert torch.allclose(rep(10), rep(24), atol=1e-5)


def test_dual_agg_trains_alone_and_with_kaptrec():
    for extra in (dict(), dict(kap_perturb=True)):
        seed_all()
        args = make_args(dual_agg=True, use_perturbation=False, use_uncertainty=False, **extra)
        model = SLSRec(args).train()
        seqs, labels = make_batch(args, batch_size=8, real_len=args.bert_max_len - 2)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
        first = None
        for _ in range(25):
            opt.zero_grad()
            l = slsrec_losses(args, model, seqs, labels, training=True)
            assert torch.isfinite(l['L_total'])
            l['L_total'].backward()
            opt.step()
            first = first if first is not None else l['L_rec'].item()
        assert l['L_rec'].item() < first
