"""Alternative readings of perturbation (eq. 17) and uncertainty (eq. 19-24).

--perturbation_site / _noise / _granularity and --uncertainty_granularity /
_weight_norm.  Each reading must do what its flag says, keep padding and
causality intact, and leave the parameter set (hence checkpoints) unchanged.
That the defaults are bit-for-bit the original code is pinned by the rest of
the suite, which runs on the defaults.
"""

import pytest
import torch

from conftest import make_args, make_batch, seed_all
from model.bdlru import perturb_tokens
from model.slsrec import SLSRec
from trainer.slsrec import slsrec_losses

SITES = ['bdlru', 'bdlru_conv', 'layer', 'embedding']


GROUP = {'embedding': 'embedding', 'layer': 'sg_lru', 'bdlru_conv': 'sg_lru', 'bdlru': 'bdlru'}


def _perturbers(model):
    """Every module that can hold a perturbation mask ('layer' and 'bdlru_conv'
    share SGLRULayer.perturbation and differ only in where it is applied)."""
    m = model.model
    return {
        'embedding': [m.embedding_perturbation],
        'sg_lru': [b.sg_lru.perturbation for b in m.blocks],
        'bdlru': [b.sg_lru.bdlru for b in m.blocks],
    }


# --------------------------------------------------------------------------- #
# perturb_tokens
# --------------------------------------------------------------------------- #
def test_default_reading_matches_the_original_implementation():
    """position + uniform must consume the RNG exactly as the original _perturb."""
    x = torch.randn(3, 20, 8)
    mask = torch.zeros(3, 20, dtype=torch.bool)
    mask[:, 5:] = True
    torch.manual_seed(3)
    keep = torch.rand(x.shape[:2])
    replace = (keep <= 0.3) & mask
    noise = torch.rand_like(x)
    ref = x * (1 - replace.unsqueeze(-1).float()) + noise * replace.unsqueeze(-1).float()
    torch.manual_seed(3)
    out, rep = perturb_tokens(x, mask, 0.3)
    assert torch.equal(rep, replace) and torch.equal(out, ref)


def test_element_granularity_replaces_components_independently():
    seed_all()
    x = torch.full((64, 50, 16), -5.0)
    mask = torch.zeros(64, 50, dtype=torch.bool)
    mask[:, 10:] = True
    out, rep = perturb_tokens(x, mask, 0.2, granularity='element')
    assert rep.shape == x.shape
    assert not rep[:, :10].any(), 'padding was perturbed'
    assert abs(rep[:, 10:].float().mean().item() - 0.2) < 0.01
    # a replaced position is no longer all-or-nothing
    per_pos = rep[:, 10:].float().mean(-1)
    assert ((per_pos > 0) & (per_pos < 1)).float().mean() > 0.5
    assert torch.equal(out[~rep], x[~rep])


def test_shuffle_noise_borrows_real_vectors_from_other_sequences():
    seed_all()
    B, L, D = 6, 12, 4
    # every real vector is unique and tagged with its sequence id in component 0
    x = torch.randn(B, L, D)
    x[..., 0] = torch.arange(B).view(B, 1).float()
    mask = torch.zeros(B, L, dtype=torch.bool)
    lengths = [12, 3, 7, 1, 12, 5]
    for b, n in enumerate(lengths):
        mask[b, L - n:] = True
    out, rep = perturb_tokens(x, mask, 1.0, noise='shuffle')
    assert torch.equal(rep, mask)
    pool = {tuple(v.tolist()) for v in x[mask]}
    for b in range(B):
        for t in range(L):
            if rep[b, t]:
                v = out[b, t]
                assert tuple(v.tolist()) in pool, 'not a real vector of the batch'
                assert int(v[0].item()) != b, 'borrowed from its own sequence'
    assert torch.equal(out[~mask], x[~mask])


def test_shuffle_noise_is_detached():
    x = torch.randn(4, 6, 3, requires_grad=True)
    mask = torch.ones(4, 6, dtype=torch.bool)
    out, rep = perturb_tokens(x, mask, 1.0, noise='shuffle')
    out.sum().backward()
    assert torch.equal(x.grad, torch.zeros_like(x))


# --------------------------------------------------------------------------- #
# perturbation sites in the model
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('site', SITES)
def test_each_site_perturbs_only_where_it_says(site):
    seed_all()
    args = make_args(perturbation_site=site, perturbation_probability=0.5)
    model = SLSRec(args).eval()
    seqs, _ = make_batch(args)
    with torch.no_grad():
        clean = model(seqs, perturb=False)[0]
        dirty = model(seqs, perturb=True)[0]
    assert not torch.allclose(clean, dirty), f'{site}: perturbation had no effect'
    for group, mods in _perturbers(model).items():
        for m in mods:
            if m is None:
                assert group != GROUP[site], f'{site}: no perturbation module'
                continue
            if group == GROUP[site]:
                assert m.last_perturb_mask is not None and m.last_perturb_mask.any()
            else:
                assert m.last_perturb_mask is None, f'{site}: {group} perturbed as well'


@pytest.mark.parametrize('site,gate_sees_noise', [('layer', True), ('bdlru_conv', False),
                                                   ('bdlru', False)])
def test_layer_site_reaches_the_gate_branch_and_the_others_do_not(site, gate_sees_noise):
    """'layer' replaces X^(l-1) itself, so the SAM/conv gate branch (W1) sees the noise;
    'bdlru_conv' and 'bdlru' only touch the recurrence branch."""
    seed_all()
    args = make_args(perturbation_site=site, perturbation_probability=0.5, bert_num_blocks=1)
    model = SLSRec(args).eval()
    seqs, _ = make_batch(args)
    seen = []
    hook = model.model.blocks[0].sg_lru.w1.register_forward_hook(lambda m, i, o: seen.append(i[0]))
    with torch.no_grad():
        model(seqs, perturb=False)
        model(seqs, perturb=True)
    hook.remove()
    assert (not torch.equal(seen[0], seen[1])) == gate_sees_noise


@pytest.mark.parametrize('site', SITES)
def test_every_site_is_inert_without_perturb_and_keeps_the_parameter_set(site):
    seed_all()
    base = SLSRec(make_args()).eval()
    seed_all()
    model = SLSRec(make_args(perturbation_site=site, perturbation_noise='shuffle')).eval()
    assert list(base.state_dict()) == list(model.state_dict())
    model.load_state_dict(base.state_dict())
    seqs, _ = make_batch(make_args())
    with torch.no_grad():
        assert torch.equal(base(seqs)[0], model(seqs)[0])


@pytest.mark.parametrize('site', ['bdlru_conv', 'layer', 'embedding'])
def test_alternative_sites_stay_causal(site):
    """Uniform noise is position-wise, so a future item cannot reach the past."""
    seed_all()
    args = make_args(perturbation_site=site, perturbation_probability=0.5)
    model = SLSRec(args).eval()
    seqs, _ = make_batch(args)
    alt = seqs.clone()
    t = args.bert_max_len - 3
    alt[:, t] = (alt[:, t] % args.num_items) + 1
    with torch.no_grad():
        torch.manual_seed(5)
        a, _ = model.model.encode(seqs, perturb=True)
        torch.manual_seed(5)
        b, _ = model.model.encode(alt, perturb=True)
    assert torch.allclose(a[:, :t], b[:, :t], atol=1e-6)


@pytest.mark.parametrize('gates', ['perturbed', 'clean'])
def test_perturbation_gates_choose_what_r_and_i_see(gates):
    """perturbed: W4/W5 of eq. (9)/(10) see x_bar; clean: they see x, and only the
    input term of eq. (12) is replaced -- the state still changes."""
    from model.bdlru import BDLRU
    seed_all()
    mod = BDLRU(8, perturb_prob=0.5, perturb_gates=gates)
    x = torch.randn(3, 10, 8)
    mask = torch.ones(3, 10, dtype=torch.bool)
    seen = []
    hook = mod.gates.register_forward_hook(lambda m, i, o: seen.append(i[0].clone()))
    clean = mod(x, mask, perturb=False)
    torch.manual_seed(11)
    dirty = mod(x, mask, perturb=True)
    hook.remove()
    assert mod.last_perturb_mask.any()
    assert torch.equal(seen[1], x) == (gates == 'clean')
    assert not torch.allclose(clean, dirty)


def test_layer_site_keeps_perturbation_alive_without_bdlru():
    """With the literal reading P lives inside the BD-LRU, so w/o BD-LRU has no P."""
    seed_all()
    seqs, _ = make_batch(make_args())
    for site, expect_effect in [('bdlru', False), ('layer', True)]:
        seed_all()
        model = SLSRec(make_args(use_bdlru=False, perturbation_site=site,
                                 perturbation_probability=0.5)).eval()
        with torch.no_grad():
            changed = not torch.equal(model(seqs, perturb=False)[0], model(seqs, perturb=True)[0])
        assert changed == expect_effect, site


# --------------------------------------------------------------------------- #
# uncertainty readings
# --------------------------------------------------------------------------- #
def test_sequence_granularity_uses_the_last_row_for_every_position():
    seed_all()
    args = make_args(uncertainty_granularity='sequence', bert_dropout=0.0)
    model = SLSRec(args).eval()
    seqs, labels = make_batch(args)
    with torch.no_grad():
        lv = model.log_variance(seqs, labels)
        model.model.uncertainty_granularity = 'position'
        per_pos = model.log_variance(seqs, labels)
    assert lv.shape == labels.shape
    assert torch.equal(lv, lv[:, -1:].expand_as(lv))
    assert torch.allclose(lv[:, -1], per_pos[:, -1], atol=1e-6)


def test_global_granularity_is_one_parameter_starting_at_sigma_one():
    seed_all()
    args = make_args(uncertainty_granularity='global')
    model = SLSRec(args)
    assert model.model.uncertainty is None
    assert model.model.global_logvar.numel() == 1 and model.model.global_logvar.item() == 0.0
    names = [n for n, _ in model.named_parameters()]
    assert not any('uncertainty' in n for n in names)
    seqs, labels = make_batch(args)
    l = slsrec_losses(args, model, seqs, labels, training=True)
    l['L_total'].backward()
    assert model.model.global_logvar.grad is not None and model.model.global_logvar.grad != 0


def test_batch_weight_norm_cannot_switch_uncertainty_off():
    """Without it a huge sigma^2 drives alpha/sigma^2 to 0; with it L_unc = L_rec."""
    seed_all()
    seqs, labels = make_batch(make_args())
    for norm, expect_equal in [('none', False), ('batch', True)]:
        args = make_args(uncertainty_granularity='global', uncertainty_weight_norm=norm,
                         use_perturbation=False)
        seed_all()
        model = SLSRec(args)
        with torch.no_grad():
            model.model.global_logvar.fill_(7.0)
        l = slsrec_losses(args, model, seqs, labels, training=True)
        same = torch.allclose(l['L_uncertainty'], l['L_rec'], atol=1e-5)
        assert same == expect_equal, (norm, l['L_uncertainty'].item(), l['L_rec'].item())


def test_batch_weight_norm_has_unit_mean_weight():
    seed_all()
    args = make_args(uncertainty_weight_norm='batch', use_perturbation=False, uncertainty_logvar_clamp=0)
    model = SLSRec(args).train()
    seqs, labels = make_batch(args)
    valid = (labels != 0).float()
    inv = torch.exp(-model.log_variance(seqs, labels))
    w = inv / ((inv * valid).sum() / valid.sum())
    assert abs(((w * valid).sum() / valid.sum()).item() - 1.0) < 1e-5
    l = slsrec_losses(args, model, seqs, labels, training=True)
    assert torch.isfinite(l['L_total'])


@pytest.mark.parametrize('over', [
    dict(perturbation_site='layer', perturbation_noise='shuffle'),
    dict(perturbation_site='embedding', perturbation_granularity='element'),
    dict(uncertainty_granularity='sequence', uncertainty_beta=1.0),
    dict(uncertainty_granularity='global', uncertainty_beta=1.0, uncertainty_loss='exact'),
    dict(uncertainty_weight_norm='batch', uncertainty_loss='exact'),
])
def test_alternative_readings_train(over):
    """A few steps on every combination family: finite, decreasing L_rec."""
    seed_all()
    args = make_args(**over)
    model = SLSRec(args).train()
    seqs, labels = make_batch(args, batch_size=8, real_len=args.bert_max_len - 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    first = None
    for _ in range(40):
        opt.zero_grad()
        l = slsrec_losses(args, model, seqs, labels, training=True)
        assert torch.isfinite(l['L_total'])
        l['L_total'].backward()
        opt.step()
        first = first if first is not None else l['L_rec'].item()
    assert l['L_rec'].item() < first
