"""Perturbation injection, SLSRec eq. (17)/(18)."""

import torch

from conftest import make_args, make_batch, seed_all
from model.bdlru import BDLRU
from model.slsrec import SLSRec


def test_disabled_perturbation_is_deterministic():
    seed_all()
    args = make_args()
    model = SLSRec(args).eval()
    seqs, _ = make_batch(args)
    with torch.no_grad():
        a = model(seqs, perturb=False)[0]
        b = model(seqs, perturb=False)[0]
    assert torch.equal(a, b)
    for block in model.model.blocks:
        assert block.sg_lru.bdlru.last_perturb_mask is None


def test_perturbation_changes_the_recurrence_input():
    seed_all()
    mod = BDLRU(8, perturb_prob=1.0)
    x = torch.randn(2, 12, 8)
    mask = torch.ones(2, 12, dtype=torch.bool)
    clean = mod(x, mask, perturb=False)
    dirty = mod(x, mask, perturb=True)
    assert not torch.allclose(clean, dirty)
    assert mod.last_perturb_mask.all()      # prob 1.0 -> every real step replaced


def test_perturbation_replacement_is_uniform_0_1():
    """The replacement vector components come from U(0, 1)."""
    seed_all()
    mod = BDLRU(64, perturb_prob=1.0)
    x = torch.full((64, 32, 64), -5.0)
    mask = torch.ones(64, 32, dtype=torch.bool)
    xbar, replaced = mod._perturb(x, mask)
    vals = xbar[replaced]
    assert replaced.all()
    assert vals.min() >= 0.0 and vals.max() <= 1.0
    assert abs(vals.mean().item() - 0.5) < 0.01     # E[U(0,1)] = 0.5
    assert abs(vals.var().item() - 1 / 12) < 0.005  # Var[U(0,1)] = 1/12


def test_perturbation_mask_respects_padding():
    seed_all()
    mod = BDLRU(8, perturb_prob=1.0)
    x = torch.randn(4, 20, 8)
    mask = torch.zeros(4, 20, dtype=torch.bool)
    mask[:, 7:] = True
    xbar, replaced = mod._perturb(x, mask)
    assert not replaced[:, :7].any(), 'padding was perturbed'
    assert replaced[:, 7:].all()
    assert torch.equal(xbar[:, :7], x[:, :7])


def test_empirical_rate_matches_configured_probability():
    seed_all()
    theta = 0.1
    mod = BDLRU(4, perturb_prob=theta)
    x = torch.randn(256, 200, 4)
    mask = torch.ones(256, 200, dtype=torch.bool)
    _, replaced = mod._perturb(x, mask)
    rate = replaced.float().mean().item()
    assert abs(rate - theta) < 0.01, f'empirical rate {rate} vs theta {theta}'


def test_same_seed_reproduces_perturbation_different_seed_does_not():
    args = make_args(perturbation_probability=0.3)
    seed_all()
    model = SLSRec(args).eval()
    seqs, _ = make_batch(args)

    def run(seed):
        torch.manual_seed(seed)
        with torch.no_grad():
            out = model(seqs, perturb=True)[0]
        masks = [b.sg_lru.bdlru.last_perturb_mask.clone() for b in model.model.blocks]
        return out, masks

    out42a, m42a = run(42)
    out42b, m42b = run(42)
    out7, m7 = run(7)

    assert torch.equal(out42a, out42b)
    assert all(torch.equal(a, b) for a, b in zip(m42a, m42b))
    assert not torch.equal(out42a, out7)
    assert any(not torch.equal(a, b) for a, b in zip(m42a, m7))


def test_each_layer_draws_its_own_perturbation():
    """eq. (17) perturbs 'an item representation in each layer'."""
    seed_all()
    args = make_args(perturbation_probability=0.3, bert_num_blocks=2)
    model = SLSRec(args).eval()
    seqs, _ = make_batch(args)
    with torch.no_grad():
        model(seqs, perturb=True)
    m0, m1 = (b.sg_lru.bdlru.last_perturb_mask for b in model.model.blocks)
    assert m0.shape == m1.shape == seqs.shape
    assert not torch.equal(m0, m1)


def test_perturbation_off_by_flag():
    seed_all()
    args = make_args(perturbation_probability=0.0)
    model = SLSRec(args).eval()
    seqs, _ = make_batch(args)
    with torch.no_grad():
        assert torch.equal(model(seqs, perturb=True)[0], model(seqs, perturb=False)[0])
