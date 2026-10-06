"""Validation cadence and epoch budget -- SLSRec paper Sec. IV-A4.

    "training is limited to a maximum of 1000 epochs.  Validation is conducted
     every 5 epochs, and an early stopping mechanism is activated if there is no
     improvement in the validation Recall@10 metric for 10 consecutive
     validation epochs."

The LRURec harness validates every ``val_iterations`` *steps* instead, so the
paper-literal schedule needs ``--val_strategy=epoch --val_epochs=5``.  These
tests drive the real trainer on tiny synthetic data and count the validations,
so the flag is verified rather than assumed.
"""

import os
import tempfile

import pytest
import torch
import torch.utils.data as data_utils

from conftest import make_args, seed_all
from model.slsrec import SLSRec
from trainer.slsrec import SLSRecTrainer

BATCHES_PER_EPOCH = 2


def _harness_args(**over):
    args = make_args(
        num_items=40, bert_max_len=16, bert_hidden_units=16, bert_num_blocks=1,
        bert_dropout=0.0, use_perturbation=False, use_uncertainty=False,
        # harness plumbing that BaseTrainer needs
        device='cpu', num_epochs=11, metric_ks=[1, 5, 10], best_metric='Recall@10',
        val_strategy='iteration', val_iterations=1000, val_epochs=1,
        early_stopping=True, early_stopping_patience=10_000,   # never fire here
        max_grad_norm=5.0, enable_lr_schedule=False, enable_lr_warmup=False,
        optimizer='AdamW', lr=1e-3, weight_decay=0.01, adam_epsilon=1e-9,
        use_wandb=False, model_code='slsrec', dataset_code='ml-1m',
    )
    for k, v in over.items():
        setattr(args, k, v)
    return args


def _loaders(args, batch_size=4):
    L = args.bert_max_len
    n = BATCHES_PER_EPOCH * batch_size
    g = torch.Generator().manual_seed(0)
    seqs = torch.randint(1, args.num_items + 1, (n, L), generator=g)
    labels = torch.randint(1, args.num_items + 1, (n, L), generator=g)
    train = data_utils.DataLoader(data_utils.TensorDataset(seqs, labels),
                                  batch_size=batch_size, shuffle=False)
    eval_seqs = seqs[:batch_size]
    eval_labels = labels[:batch_size, -1:].contiguous()     # one held-out item
    ev = data_utils.DataLoader(data_utils.TensorDataset(eval_seqs, eval_labels),
                               batch_size=batch_size, shuffle=False)
    return train, ev, ev


def _run(args):
    """Train and return the (epoch, accum_iter) of every validation."""
    seed_all()
    train, val, test = _loaders(args)
    model = SLSRec(args)
    with tempfile.TemporaryDirectory() as tmp:
        trainer = SLSRecTrainer(args, model, train, val, test, tmp, use_wandb=False)
        calls = []
        original = trainer.validate

        def spy(epoch, accum_iter):
            calls.append((epoch, accum_iter))
            return original(epoch, accum_iter)

        trainer.validate = spy
        trainer.train()
        assert os.path.isfile(os.path.join(tmp, 'val_metrics.jsonl'))
    return calls


def test_epoch_strategy_validates_every_val_epochs():
    """--val_strategy=epoch --val_epochs=5 over 11 epochs: initial + after 5 and 10."""
    calls = _run(_harness_args(val_strategy='epoch', val_epochs=5, num_epochs=11))
    # BaseTrainer.train() always validates once before the first epoch
    assert calls[0] == (0, 0), calls
    epochs = [e + 1 for e, _ in calls[1:]]        # 0-indexed epoch -> 1-indexed
    assert epochs == [5, 10], f'expected validations after epochs 5 and 10, got {epochs}'


def test_epoch_strategy_default_validates_every_epoch():
    calls = _run(_harness_args(val_strategy='epoch', val_epochs=1, num_epochs=4))
    epochs = [e + 1 for e, _ in calls[1:]]
    assert epochs == [1, 2, 3, 4], epochs


def test_iteration_strategy_validates_every_val_iterations():
    """The LRURec default path must be untouched by the val_epochs addition."""
    calls = _run(_harness_args(val_strategy='iteration', val_iterations=3, num_epochs=5))
    iters = [i for _, i in calls[1:]]
    assert iters == [3, 6, 9], f'2 batches/epoch x 5 epochs = 10 steps -> {iters}'


def test_val_epochs_is_ignored_under_iteration_strategy():
    a = _run(_harness_args(val_strategy='iteration', val_iterations=3, val_epochs=5, num_epochs=5))
    b = _run(_harness_args(val_strategy='iteration', val_iterations=3, val_epochs=1, num_epochs=5))
    assert [i for _, i in a] == [i for _, i in b]


@pytest.mark.parametrize('cadence,patience,expected_epochs', [
    (5, 10, 50),     # the paper: every 5 epochs, patience 10 -> 50 epochs
    (1, 10, 10),
])
def test_patience_translates_to_the_paper_epoch_span(cadence, patience, expected_epochs):
    """Sanity on the arithmetic quoted in reproduction_report.md sec. 8.1."""
    assert cadence * patience == expected_epochs


def test_paper_schedule_flags_resolve():
    """The exact flag combination documented for the paper-literal schedule."""
    import sys
    saved = sys.argv
    try:
        sys.argv = ['x']
        import config as cfg
        a = cfg.parser.parse_args([
            '--dataset_code=ml-1m', '--model_code=slsrec',
            '--num_epochs=1000', '--val_strategy=epoch', '--val_epochs=5',
        ])
        cfg.set_template(a)
    finally:
        sys.argv = saved
    assert a.num_epochs == 1000
    assert a.val_strategy == 'epoch'
    assert a.val_epochs == 5
    assert a.early_stopping_patience == 10
    assert a.bert_hidden_units == 64
    assert a.train_batch_size == 128
    assert a.best_metric == 'Recall@10'
