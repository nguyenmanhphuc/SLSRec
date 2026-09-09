"""Tiny overfit: eight synthetic sequences must be memorised, for every ablation."""

import pytest
import torch

from conftest import make_args, make_batch, seed_all
from model.slsrec import SLSRec
from trainer.slsrec import slsrec_losses

ABLATIONS = {
    # A: BD-LRU + CE (RecBLR-style gate branch, no SAM, no P, no U)
    'A_bdlru_ce': dict(use_sam_gate=False, sam_conv=False,
                       use_perturbation=False, use_uncertainty=False),
    # B: + self-attention gate
    'B_sam_gate': dict(use_perturbation=False, use_uncertainty=False),
    # C: + uniform perturbation
    'C_perturbation': dict(use_perturbation=True, use_uncertainty=False),
    # D: + Gibbs uncertainty = full SLSRec
    'D_full_slsrec': dict(use_perturbation=True, use_uncertainty=True),
}


def _overfit(args, steps=220, lr=1e-2):
    seed_all()
    model = SLSRec(args)
    model.train()
    seqs, labels = make_batch(args, batch_size=8, real_len=args.bert_max_len - 2)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    first = None
    for step in range(steps):
        opt.zero_grad()
        losses = slsrec_losses(args, model, seqs, labels, training=True)
        assert torch.isfinite(losses['L_total']), f'non-finite loss at step {step}'
        losses['L_total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if first is None:
            first = {k: v.item() for k, v in losses.items()}

    model.eval()
    with torch.no_grad():
        logits = model(seqs)[0]
        final_ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=0).item()
        valid = labels != 0
        acc = (logits.argmax(-1)[valid] == labels[valid]).float().mean().item()
    return first, final_ce, acc


@pytest.mark.parametrize('name', list(ABLATIONS))
def test_tiny_overfit_for_every_ablation(name):
    args = make_args(bert_dropout=0.0, **ABLATIONS[name])
    first, final_ce, acc = _overfit(args)

    assert final_ce < 0.2 * first['L_rec'], \
        f'{name}: CE only fell {first["L_rec"]:.3f} -> {final_ce:.3f}'
    assert acc > 0.9, f'{name}: train accuracy only {acc:.3f}'


def test_ablations_are_selectable_and_wire_the_right_terms():
    """Each ablation must be reachable purely through configuration."""
    seed_all()
    for name, over in ABLATIONS.items():
        args = make_args(bert_dropout=0.0, **over)
        model = SLSRec(args)
        seqs, labels = make_batch(args, batch_size=8)
        l = slsrec_losses(args, model, seqs, labels, training=True)

        has_sam = any('sam.' in n for n, _ in model.named_parameters())
        has_unc = any('uncertainty' in n for n, _ in model.named_parameters())
        assert has_sam == args.use_sam_gate, name
        assert has_unc == args.use_uncertainty, name
        assert (l['L_perturb'].item() != 0.0) == args.use_perturbation, name
        assert (l['L_uncertainty'].item() != 0.0) == args.use_uncertainty, name
        assert torch.isfinite(l['L_total'])
