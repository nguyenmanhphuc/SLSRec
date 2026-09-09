"""Causality: nothing after position t may influence the hidden state at t."""

import torch

from conftest import make_args, make_batch, seed_all
from model.slsrec import SLSRec


def _hidden(model, seqs):
    with torch.no_grad():
        h, _ = model.model.encode(seqs)
    return h


def test_future_item_does_not_change_earlier_states():
    seed_all()
    args = make_args()
    model = SLSRec(args).eval()
    seqs, _ = make_batch(args, batch_size=4)

    L = seqs.size(1)
    change_at = L - 2                      # perturb one *future* position
    seqs_b = seqs.clone()
    seqs_b[:, change_at] = (seqs_b[:, change_at] % args.num_items) + 1
    assert (seqs_b[:, change_at] != seqs[:, change_at]).all()

    ha, hb = _hidden(model, seqs), _hidden(model, seqs_b)
    err = (ha[:, :change_at] - hb[:, :change_at]).abs().max().item()
    assert err < 1e-6, f'future leakage: max abs diff {err} before position {change_at}'
    # and the change must actually be visible from that position onwards
    assert (ha[:, change_at:] - hb[:, change_at:]).abs().max().item() > 1e-6


def test_causality_holds_at_every_position():
    seed_all()
    args = make_args(bert_max_len=12, bert_num_blocks=2)
    model = SLSRec(args).eval()
    seqs, _ = make_batch(args, batch_size=2, real_len=12)
    base = _hidden(model, seqs)

    for t in range(1, seqs.size(1)):
        alt = seqs.clone()
        alt[:, t] = (alt[:, t] % args.num_items) + 1
        h = _hidden(model, alt)
        err = (base[:, :t] - h[:, :t]).abs().max().item()
        assert err < 1e-6, f'leakage from position {t} into 0..{t - 1} (err={err})'


def test_causality_holds_with_perturbation_disabled_and_enabled():
    """The perturbation must not open a backwards path either."""
    seed_all()
    args = make_args(bert_max_len=12)
    model = SLSRec(args).eval()
    seqs, _ = make_batch(args, batch_size=2, real_len=12)
    t = 8

    torch.manual_seed(0)
    with torch.no_grad():
        ha, _ = model.model.encode(seqs, perturb=True)
    alt = seqs.clone()
    alt[:, t] = (alt[:, t] % args.num_items) + 1
    torch.manual_seed(0)                    # identical perturbation draw
    with torch.no_grad():
        hb, _ = model.model.encode(alt, perturb=True)

    err = (ha[:, :t] - hb[:, :t]).abs().max().item()
    assert err < 1e-6, f'future leakage under perturbation: {err}'
