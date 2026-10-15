"""KAPTRec extensions: C1 branch-selection gate, C2 transition-aware LRU,
C3 adaptive perturbation + history-aware uncertainty.  Not in the SLSRec paper."""

import pytest
import torch

from conftest import make_args, make_batch, seed_all
from model.kan import KANFusion
from model.kaptrec import AdaptivePerturbation, HistoryUncertainty, grad_reverse
from model.slsrec import SLSRec
from trainer.slsrec import slsrec_losses

VARIANTS = [dict(fusion_mode='kan_gate'), dict(fusion_mode='mlp_gate'), dict(bdlru_transition=True)]


# --------------------------------------------------------------------------- #
# C1 -- branch selection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('mode', ['kan_gate', 'mlp_gate'])
def test_gate_starts_at_the_branch_average(mode):
    f = KANFusion(mode, 8)
    g, h = torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    assert torch.allclose(f(None, g, h), 0.5 * (g + h), atol=1e-6)
    assert torch.allclose(f.last_o, torch.full_like(f.last_o, 0.5))


def test_gate_is_a_convex_selection_after_training_steps():
    seed_all()
    f = KANFusion('kan_gate', 8)
    with torch.no_grad():
        f.layers[-1].spline_weight.normal_(std=1.0)
    g, h = torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    out = f(None, g, h)
    o = f.last_o
    assert (o > 0).all() and (o < 1).all() and not torch.allclose(o, torch.full_like(o, 0.5))
    assert torch.allclose(out, o * h + (1 - o) * g, atol=1e-6)


# --------------------------------------------------------------------------- #
# C2 -- transition gate
# --------------------------------------------------------------------------- #
def test_transition_gate_changes_the_recurrence():
    seed_all()
    base = SLSRec(make_args(bert_dropout=0.0)).eval()
    seed_all()
    tr = SLSRec(make_args(bdlru_transition=True, bert_dropout=0.0)).eval()
    tr.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        for blk in tr.model.blocks:
            blk.sg_lru.bdlru.transition_gate[-1].weight.mul_(50.0)
    seqs, _ = make_batch(make_args())
    with torch.no_grad():
        assert not torch.allclose(base(seqs)[0], tr(seqs)[0])


# --------------------------------------------------------------------------- #
# shared invariants for C1/C2
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('over', VARIANTS)
def test_variants_are_causal(over):
    seed_all()
    args = make_args(**over)
    model = SLSRec(args).eval()
    seqs, _ = make_batch(args)
    alt = seqs.clone()
    t = args.bert_max_len - 3
    alt[:, t] = (alt[:, t] % args.num_items) + 1
    with torch.no_grad():
        a, _ = model.model.encode(seqs)
        b, _ = model.model.encode(alt)
    assert torch.allclose(a[:, :t], b[:, :t], atol=1e-6)


@pytest.mark.parametrize('over', VARIANTS)
def test_variants_ignore_padding_amount(over):
    seed_all()
    args = make_args(bert_max_len=24, **over)
    model = SLSRec(args).eval()
    body = torch.randint(1, args.num_items + 1, (3, 10))

    def hidden(total):
        s = torch.zeros(3, total, dtype=torch.long)
        s[:, -10:] = body
        with torch.no_grad():
            return model.model.encode(s)[0][:, -10:]

    assert torch.allclose(hidden(10), hidden(24), atol=1e-5)


# --------------------------------------------------------------------------- #
# C3 -- adaptive perturbation
# --------------------------------------------------------------------------- #
def _controller_inputs(D=16, d=32, B=3, L=12, real=8):
    x, z = torch.randn(B, L, D), torch.randn(B, L, D)
    g, h = torch.randn(B, L, d), torch.randn(B, L, d)
    y = torch.randn(B, L, D)
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[:, L - real:] = True
    return x * mask.unsqueeze(-1), z, g, h, y, mask


def test_perturbation_is_bounded_and_spares_padding():
    seed_all()
    kap = AdaptivePerturbation(16, pi_max=0.9, eps_min=0.1, eps_max=0.4)
    x, z, g, h, y, mask = _controller_inputs()
    delta, pi = kap(x, z, g, h, y, mask)
    assert torch.equal(delta[~mask], torch.zeros_like(delta[~mask]))
    assert (pi >= 0).all() and (pi <= 0.9).all()
    rel = delta.norm(dim=-1)[mask] / x.norm(dim=-1)[mask]
    hit = rel > 0
    assert hit.any(), 'nothing was perturbed'
    assert (rel[hit] >= 0.1 - 1e-5).all() and (rel[hit] <= 0.4 + 1e-5).all()


def test_mask_is_hard_forward_and_follows_pi():
    seed_all()
    kap = AdaptivePerturbation(16, pi_max=0.3)
    p = torch.full((20000, 1), 0.3)
    m = kap._st_bernoulli(p)
    assert set(m.unique().tolist()) <= {0.0, 1.0}
    assert abs(m.mean().item() - 0.3) < 0.02


def test_adv_controller_receives_the_reversed_gradient():
    x = torch.randn(4, requires_grad=True)
    grad_reverse(x).sum().backward()
    assert torch.equal(x.grad, -torch.ones(4))


def test_history_uncertainty_is_positive_and_detached():
    seed_all()
    unc = HistoryUncertainty(16, 32)
    x, z, g, h, y, mask = _controller_inputs()
    z.requires_grad_(True)
    sigma = unc(z, g, h, y[:, -1], mask)
    assert sigma.shape == (3,) and (sigma > 0).all()
    sigma.sum().backward()
    assert z.grad is None, 'uncertainty head must not backprop into the encoder features'


def test_rows_without_history_stay_finite():
    """The harness produces rows whose input is all padding (one-item users)."""
    seed_all()
    args = make_args(use_perturbation=False, use_uncertainty=False, kap_perturb=True, kap_uncertainty=True)
    model = SLSRec(args).train()
    seqs, labels = make_batch(args, batch_size=4, real_len=6)
    seqs[0] = 0
    labels[0] = 0
    labels[0, -1] = 3                      # a target with an empty history
    l = slsrec_losses(args, model, seqs, labels, training=True)
    assert all(torch.isfinite(v) for v in l.values())
    l['L_total'].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.parametrize('extra', [dict(), dict(kap_uncertainty=True), dict(kap_controller='min'),
                                   dict(fusion_mode='kan_gate', bdlru_transition=True, kap_uncertainty=True)])
def test_kaptrec_trains_and_logs(extra):
    seed_all()
    args = make_args(use_perturbation=False, use_uncertainty=False, kap_perturb=True, **extra)
    model = SLSRec(args).train()
    seqs, labels = make_batch(args, batch_size=8, real_len=args.bert_max_len - 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    first = None
    for _ in range(25):
        opt.zero_grad()
        l = slsrec_losses(args, model, seqs, labels, training=True)
        assert torch.isfinite(l['L_total'])
        for k in ('L_kap_pert', 'L_kap_js', 'L_kap_budget', 'kap_pi_mean'):
            assert k in l
        assert l['L_kap_js'] >= 0
        l['L_total'].backward()
        opt.step()
        first = first if first is not None else l['L_rec'].item()
    assert l['L_rec'].item() < first


def test_kaptrec_is_training_only():
    """Evaluation never perturbs: the model output does not depend on --kap_perturb."""
    seed_all()
    base = SLSRec(make_args(use_uncertainty=False)).eval()
    seed_all()
    kap = SLSRec(make_args(use_uncertainty=False, kap_perturb=True, kap_uncertainty=True)).eval()
    kap.load_state_dict(base.state_dict(), strict=False)
    seqs, _ = make_batch(make_args())
    with torch.no_grad():
        assert torch.equal(base(seqs)[0], kap(seqs)[0])
