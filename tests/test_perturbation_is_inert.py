"""Falsifiers for SLSRec's perturbation-injection claim (eq. 17/18, Sec. III-D).

Unlike the rest of `tests/`, these do **not** test the reproduction -- they test the
*paper*.  Each one asserts what must be true if perturbation injection is the
anti-overfitting regulariser the paper says it is.  A FAILURE here is the finding.

They are marked `claim` and deselected by the default `pytest.ini` addopts, so the
reproduction suite stays green.  Run them on purpose:

    python -m pytest -m claim -q
    python -m pytest -m claim -q -rA          # show the measured numbers
"""

import os

import pytest
import torch
import torch.nn.functional as F

from conftest import make_args, make_batch, seed_all
from model.slsrec import SLSRec
from trainer.slsrec import slsrec_losses

pytestmark = pytest.mark.claim

STEPS = int(os.environ.get('SLSREC_CLAIM_STEPS', 220))


def _train_tiny(args, steps=STEPS, lr=1e-2, batch_size=8):
    seed_all()
    model = SLSRec(args)
    model.train()
    seqs, labels = make_batch(args, batch_size=batch_size, real_len=args.bert_max_len - 2)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    for _ in range(steps):
        opt.zero_grad()
        slsrec_losses(args, model, seqs, labels, training=True)['L_total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    return model, seqs, labels


def _train_accuracy(model, seqs, labels):
    model.eval()
    with torch.no_grad():
        logits = model(seqs)[0]
        valid = labels != 0
        return (logits.argmax(-1)[valid] == labels[valid]).float().mean().item()


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('theta,weight', [(1.0, 1.0), (1.0, 5.0)])
def test_claim_perturbation_blocks_memorisation(theta, weight):
    """CLAIM (Sec. III-D): perturbation injection prevents overfitting.

    The strongest possible dose -- every position of every layer replaced by
    U(0,1), with L_Per weighted 5x -- must cost the model *something* on a
    memorisation task that the un-perturbed model solves outright.
    """
    args = make_args(bert_dropout=0.0, use_uncertainty=False,
                     perturbation_probability=theta, perturbation_weight=weight)
    model, seqs, labels = _train_tiny(args)
    acc = _train_accuracy(model, seqs, labels)
    assert acc < 0.9, (
        f'theta={theta}, w_per={weight}: 8 sequences still memorised to acc={acc:.3f}. '
        'L_Per imposes no capacity cost, so it cannot regularise.')


def test_claim_perturbed_loss_is_a_distinct_signal():
    """CLAIM (eq. 18): L_Per is a second, different objective from L_Rec.

    If it is, its gradient must point somewhere else.  Near-collinear gradients
    mean L_total = L_Rec + L_Per is just L_Rec rescaled -- and AdamW is invariant
    to a loss rescale.
    """
    args = make_args(bert_dropout=0.0, use_uncertainty=False,
                     perturbation_probability=1.0)
    model, seqs, labels = _train_tiny(args, steps=60)
    model.train()
    losses = slsrec_losses(args, model, seqs, labels, training=True)
    params = [p for p in model.parameters() if p.requires_grad]
    g_rec = torch.autograd.grad(losses['L_rec'], params, retain_graph=True, allow_unused=True)
    g_per = torch.autograd.grad(losses['L_perturb'], params, retain_graph=True, allow_unused=True)
    dot = sum((a * b).sum() for a, b in zip(g_rec, g_per) if a is not None and b is not None)
    na = torch.sqrt(sum((a * a).sum() for a in g_rec if a is not None))
    nb = torch.sqrt(sum((b * b).sum() for b in g_per if b is not None))
    cos = (dot / (na * nb)).item()
    assert cos < 0.5, (
        f'cos(grad L_Rec, grad L_Per) = {cos:.3f} at theta=1.0 -- the perturbed pass '
        'reproduces the reconstruction gradient instead of adding a new one.')


def test_claim_theta_actually_perturbs_the_prediction():
    """CLAIM (eq. 17): theta = 0.1 is a meaningful corruption rate.

    Measured at the output, not inside the recurrence: perturbing 10% of the
    positions must move the next-item distribution.
    """
    args = make_args(bert_dropout=0.0, use_uncertainty=False,
                     perturbation_probability=0.1)
    model, seqs, _ = _train_tiny(args, steps=120)
    model.eval()
    torch.manual_seed(0)
    with torch.no_grad():
        clean = model(seqs, perturb=False)[0][:, -1, :]
        dirty = model(seqs, perturb=True)[0][:, -1, :]
        kl = F.kl_div(F.log_softmax(dirty, -1), F.log_softmax(clean, -1),
                      log_target=True, reduction='batchmean').item()
        rel = ((dirty - clean).norm() / clean.norm()).item()
    assert kl > 0.1, (
        f'KL(clean || perturbed) = {kl:.4f} (relative logit change {rel:.4f}) at theta=0.1 -- '
        'the paper-default perturbation rate is invisible at the output.')


def test_claim_scale_of_the_replacement_does_not_dominate_theta():
    """CLAIM (Sec. IV-E2): theta is the hyperparameter that matters.

    The paper tunes theta over {0.05 .. 0.25} and fixes the replacement range at
    U(0,1) without ablating it.  If the *range* moves the output more than a 10x
    change in theta does, the tuned knob is the wrong one.
    """
    args = make_args(bert_dropout=0.0, use_uncertainty=False,
                     perturbation_probability=0.1)
    model, seqs, _ = _train_tiny(args, steps=120)
    model.eval()

    def rel_change(theta, low, high):
        for b in model.model.blocks:
            b.sg_lru.bdlru.perturb_prob = theta
            b.sg_lru.bdlru.perturb_low = low
            b.sg_lru.bdlru.perturb_high = high
        torch.manual_seed(0)
        with torch.no_grad():
            clean = model(seqs, perturb=False)[0][:, -1, :]
            dirty = model(seqs, perturb=True)[0][:, -1, :]
        return ((dirty - clean).norm() / clean.norm()).item()

    by_theta = rel_change(1.0, 0.0, 1.0)     # 10x the paper's theta
    by_range = rel_change(0.1, 0.0, 10.0)    # paper's theta, 10x the range
    assert by_theta > by_range, (
        f'theta 0.1->1.0 moves the logits by {by_theta:.4f}, but keeping theta=0.1 and '
        f'widening U(0,1)->U(0,10) moves them by {by_range:.4f}. The un-ablated range '
        'dominates the ablated rate.')


@pytest.mark.skipif(
    not os.path.exists(os.path.join('experiments', 'slsrec', 'smoke3ep_D_full_slsrec',
                                    'models', 'best_acc_model.pth')),
    reason='needs a trained SLSRec checkpoint; run scripts/kill_tests.py --probe routing')
def test_claim_bdlru_output_is_load_bearing():
    """CLAIM (Sec. III-C): the BD-LRU recurrence carries the sequence signal.

    Delegated to the routing probe, which needs the real dataloaders.
    """
    pytest.skip('run: python scripts/kill_tests.py --probe routing --dataset_code=ml-1m '
                '--num_workers=0')
