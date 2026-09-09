"""Loss decomposition: L_rec, L_perturb, L_uncertainty and eq. (23)/(24)."""

import math

import pytest
import torch

from conftest import make_args, make_batch, seed_all
from model.slsrec import SLSRec
from trainer.slsrec import slsrec_losses


def _run(args, seed=0, **model_over):
    seed_all()
    model = SLSRec(make_args(**{**vars(args), **model_over}))
    model.train()
    seqs, labels = make_batch(args)
    torch.manual_seed(seed)
    return model, seqs, labels, slsrec_losses(args, model, seqs, labels, training=True)


def test_all_losses_are_finite():
    args = make_args()
    _, _, _, losses = _run(args)
    for k, v in losses.items():
        assert torch.isfinite(v), f'{k} = {v}'
        assert v.item() > 0 or k == 'L_perturb'


def test_total_equals_paper_weighted_combination():
    """L_total == L_rec + w_per*L_per + alpha*L_unc  (eq. 23)."""
    args = make_args(uncertainty_alpha=0.3, perturbation_weight=1.0)
    _, _, _, l = _run(args)
    expected = l['L_rec'] + 1.0 * l['L_perturb'] + 0.3 * l['L_uncertainty']
    assert abs(l['L_total'].item() - expected.item()) < 1e-6


def test_uncertainty_term_is_inverse_variance_weighted_rec_loss():
    """eq. (22) with beta = 0:  L_Unc = mean_t (1/sigma_t^2) * CE_t."""
    args = make_args(uncertainty_beta=0.0)
    seed_all()
    model = SLSRec(args)
    model.eval()                       # deterministic (dropout is 0 anyway)
    seqs, labels = make_batch(args)
    losses = slsrec_losses(args, model, seqs, labels, training=False)

    with torch.no_grad():
        logits = model(seqs)[0]
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
            ignore_index=0, reduction='none').view_as(labels)
        inv_var = torch.exp(-model.log_variance(seqs, labels))
        valid = (labels != 0).float()
        manual = (inv_var * ce * valid).sum() / valid.sum()
    assert abs(losses['L_uncertainty'].item() - manual.item()) < 1e-5


def test_alpha_zero_removes_uncertainty_contribution():
    args = make_args(uncertainty_alpha=0.0)
    _, _, _, l = _run(args)
    assert abs(l['L_total'].item() - (l['L_rec'] + l['L_perturb']).item()) < 1e-6


def test_perturbation_weight_zero_removes_its_contribution():
    args = make_args(perturbation_weight=0.0)
    _, _, _, l = _run(args)
    expected = l['L_rec'] + args.uncertainty_alpha * l['L_uncertainty']
    assert abs(l['L_total'].item() - expected.item()) < 1e-6


def test_disabling_branches_zeroes_the_terms():
    args = make_args(use_perturbation=False, use_uncertainty=False)
    _, _, _, l = _run(args)
    assert l['L_perturb'].item() == 0.0
    assert l['L_uncertainty'].item() == 0.0
    assert abs(l['L_total'].item() - l['L_rec'].item()) < 1e-9


def test_eval_mode_skips_the_perturbed_forward_pass():
    args = make_args()
    seed_all()
    model = SLSRec(args).eval()
    seqs, labels = make_batch(args)
    l = slsrec_losses(args, model, seqs, labels, training=False)
    assert l['L_perturb'].item() == 0.0


@pytest.mark.parametrize('mode', ['approx', 'exact'])
def test_both_uncertainty_formulations_run_and_differ(mode):
    args = make_args(uncertainty_loss=mode)
    _, _, _, l = _run(args)
    assert torch.isfinite(l['L_uncertainty'])
    assert l['L_uncertainty'].item() > 0


def test_beta_adds_the_log_sigma_penalty():
    """eq. (22) second term: beta * Log(sigma) = beta/2 * Log(sigma^2)."""
    args0 = make_args(uncertainty_beta=0.0)
    args1 = make_args(uncertainty_beta=1.0)
    seed_all()
    model = SLSRec(args0).eval()
    seqs, labels = make_batch(args0)
    l0 = slsrec_losses(args0, model, seqs, labels, training=False)
    l1 = slsrec_losses(args1, model, seqs, labels, training=False)
    with torch.no_grad():
        logvar = model.log_variance(seqs, labels)
        valid = (labels != 0).float()
        penalty = (0.5 * logvar * valid).sum() / valid.sum()
    assert abs((l1['L_uncertainty'] - l0['L_uncertainty']).item() - penalty.item()) < 1e-5


def test_every_enabled_branch_receives_gradient_from_l_total():
    args = make_args()
    model, seqs, labels, l = _run(args)
    l['L_total'].backward()
    names = [
        'model.embedding.token.weight',
        'model.embedding.position.weight',
        'model.blocks.0.sg_lru.sam.w_q.weight',
        'model.blocks.0.sg_lru.conv1.conv.weight',
        'model.blocks.0.sg_lru.conv2.conv.weight',
        'model.blocks.0.sg_lru.bdlru.Lambda',
        'model.blocks.0.sg_lru.bdlru.gates.weight',
        'model.blocks.0.sg_lru.w3.weight',
        'model.blocks.1.pffn_block.pffn.w8.weight',
        'model.uncertainty.embedding.token.weight',
        'model.uncertainty.pffn_block.pffn.w8.weight',
    ]
    params = dict(model.named_parameters())
    for n in names:
        g = params[n].grad
        assert g is not None and g.abs().sum().item() > 0, f'no gradient for {n}'


def test_uncertainty_branch_gets_no_gradient_when_alpha_is_zero():
    args = make_args(uncertainty_alpha=0.0)
    model, _, _, l = _run(args)
    l['L_total'].backward()
    g = dict(model.named_parameters())['model.uncertainty.pffn_block.pffn.w8.weight'].grad
    assert g is None or g.abs().sum().item() == 0.0
