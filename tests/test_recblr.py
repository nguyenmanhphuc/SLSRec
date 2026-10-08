"""RecBLR (model/recblr.py): the official architecture, inside the LRURec harness.

Numerical parity with the official RecBLR.py was checked once against a clone of
https://github.com/chengkai-liu/RecBLR (identical state_dict keys, max abs
difference 0.0 on a power-of-two length); that repository needs RecBole, so the
check is not part of this suite.  These tests pin the properties the port has
to keep in the left-padded harness.
"""

import math
import os
import sys

import torch
import torch.nn as nn

from conftest import make_args, make_batch, seed_all
from model import model_factory
from model.recblr import RecBLR, GatedRecurrentLayer
from trainer.recblr import RecBLRTrainer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _args(**over):
    base = dict(model_code='recblr', bert_dropout=0.0, scan_impl='sequential')
    base.update(over)
    return make_args(**base)


def test_parameter_count_matches_the_official_layout():
    """ML-1M scale, D=64, 2 layers, expand 2, d_conv 4 -- counted by hand from
    RecBLR.py: embedding 3417*64 + LN, per layer input 64*256, conv 128*4+128,
    gates 128*256+256, Lambda 128, output 128*64, LN, FFN 64*256+256 + 256*64+64, LN."""
    args = _args(num_items=3416, bert_hidden_units=64, bert_max_len=200)
    per_layer = 64 * 256 + (128 * 4 + 128) + (128 * 256 + 256) + 128 + 128 * 64 + 128 \
        + (64 * 256 + 256) + (256 * 64 + 64) + 128
    expected = 3417 * 64 + 128 + 2 * per_layer
    assert sum(p.numel() for p in RecBLR(args).parameters()) == expected == 402240


def test_lambda_init_spans_r_min_r_max_at_full_recurrence_gate():
    """alpha = exp(-softplus(Lambda) * sigmoid(.)); at sigmoid = 1 it must cover
    [0.9, 0.999] exactly (RecBLR's softplus-inverse initialisation)."""
    layer = GatedRecurrentLayer(d_model=8, expansion_factor=2)
    alpha = torch.exp(-nn.functional.softplus(layer.Lambda.detach()))
    assert math.isclose(alpha.min().item(), 0.9, rel_tol=1e-5)
    assert math.isclose(alpha.max().item(), 0.999, rel_tol=1e-5)


def test_no_positional_embedding_and_no_item_bias():
    names = [n for n, _ in RecBLR(_args()).named_parameters()]
    assert not any('position' in n for n in names), names
    assert not any(n == 'bias' for n in names)


def test_future_items_do_not_change_the_past():
    seed_all()
    args = _args()
    model = RecBLR(args).eval()
    seqs, _ = make_batch(args, batch_size=3)
    changed = seqs.clone()
    t = args.bert_max_len - 4
    changed[:, t] = (changed[:, t] % args.num_items) + 1
    with torch.no_grad():
        a, b = model.encode(seqs), model.encode(changed)
    assert torch.allclose(a[:, :t], b[:, :t], atol=1e-6), (a[:, :t] - b[:, :t]).abs().max().item()
    assert not torch.allclose(a[:, t:], b[:, t:])


def test_extra_left_padding_does_not_change_scores():
    seed_all()
    args = _args(bert_max_len=32)
    model = RecBLR(args).eval()
    real_len = 9
    body = torch.randint(1, args.num_items + 1, (2, real_len))

    def scores(total_len):
        s = torch.zeros(2, total_len, dtype=torch.long)
        s[:, -real_len:] = body
        with torch.no_grad():
            return model(s)[0][:, -real_len:]

    for n in (9, 20, 32):
        err = (scores(n) - scores(9)).abs().max().item()
        assert err < 1e-5, f'padding to {n} moved the scores by {err}'


def test_torch_and_sequential_scans_agree():
    seed_all()
    a = RecBLR(_args(scan_impl='sequential')).eval()
    b = RecBLR(_args(scan_impl='torch')).eval()
    b.load_state_dict(a.state_dict())
    seqs, _ = make_batch(_args())
    with torch.no_grad():
        assert torch.allclose(a(seqs)[0], b(seqs)[0], atol=1e-4)


def test_tiny_overfit():
    seed_all()
    args = _args(scan_impl='torch')
    model = RecBLR(args).train()
    seqs, labels = make_batch(args, batch_size=8, real_len=args.bert_max_len - 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    ce = nn.CrossEntropyLoss(ignore_index=0)
    first = None
    for _ in range(220):
        opt.zero_grad()
        logits = model(seqs)[0]
        loss = ce(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        first = first if first is not None else loss.item()
    model.eval()
    with torch.no_grad():
        pred = model(seqs)[0].argmax(-1)
    valid = labels != 0
    acc = (pred[valid] == labels[valid]).float().mean().item()
    assert loss.item() < 0.2 * first, (first, loss.item())
    assert acc > 0.9, acc


def test_factories_and_trainer_loss():
    args = _args()
    model = model_factory(args)
    assert isinstance(model, RecBLR)
    seqs, labels = make_batch(args)
    trainer = RecBLRTrainer.__new__(RecBLRTrainer)          # loss only: no loaders/loggers
    trainer.model, trainer.ce = model, nn.CrossEntropyLoss(ignore_index=0)
    loss = trainer.calculate_loss((seqs, labels))
    logits = model(seqs)[0]
    ref = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=0)
    assert torch.allclose(loss, ref)


def _resolve(flags):
    saved = sys.argv
    try:
        sys.argv = ['x']
        import config as cfg
        a = cfg.parser.parse_args(flags)
        cfg.set_template(a)
    finally:
        sys.argv = saved
    return a


def test_recblr_gets_the_slsrec_paper_schedule():
    """SLSRec Sec. IV-A4 trains its re-run baselines like SLSRec itself."""
    a = _resolve(['--dataset_code=ml-1m', '--model_code=recblr'])
    assert (a.num_epochs, a.val_strategy, a.val_epochs, a.early_stopping_patience) == (1000, 'epoch', 5, 10)
    assert (a.lr, a.weight_decay, a.train_batch_size, a.bert_dropout) == (0.001, 0.01, 128, 0.2)
    assert (a.bert_hidden_units, a.bert_num_blocks, a.slsrec_expand, a.slsrec_d_conv) == (64, 2, 2, 4)
