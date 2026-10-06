"""Equation (20): dropout includes the uncertainty estimator's residual."""

import pytest
import torch
import torch.nn.functional as F

from conftest import make_args, make_batch
from model.slsrec import PFFNBlock, SLSRec, UncertaintyEstimator


@pytest.mark.parametrize('dropout', [0.0, 0.2, 0.75])
@pytest.mark.parametrize('training', [False, True])
def test_uncertainty_matches_printed_equation_20(dropout, training):
    args = make_args(bert_dropout=dropout)
    estimator = SLSRec(args).model.uncertainty
    estimator.train(training)
    # Isolate equation (20) from the embedding dropout in equation (1).
    estimator.embedding.dropout.p = 0.0
    seqs, labels = make_batch(args)
    xc, _ = estimator.embedding(seqs)
    block = estimator.pffn_block

    torch.manual_seed(123)
    expected_hidden = F.dropout(block.pffn(xc) + xc, p=dropout, training=training)
    expected_hidden = F.layer_norm(
        expected_hidden, block.layer_norm.normalized_shape,
        block.layer_norm.weight, block.layer_norm.bias, block.layer_norm.eps,
    )
    expected = (estimator.embedding.token(labels) * expected_hidden).sum(-1)

    torch.manual_seed(123)
    actual = estimator(seqs, labels)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_full_dropout_removes_uncertainty_residual():
    args = make_args(bert_dropout=1.0)
    estimator = SLSRec(args).model.uncertainty.train()
    estimator.embedding.dropout.p = 0.0
    seqs, labels = make_batch(args)
    with torch.no_grad():
        for parameter in estimator.pffn_block.pffn.parameters():
            parameter.zero_()
    # With PFFN=0, equation (20) still drops xc. The old placement retained xc.
    actual = estimator(seqs, labels)
    torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)


def test_legacy_uncertainty_weights_load_and_keep_eval_outputs():
    args = make_args(bert_dropout=0.75)
    legacy = UncertaintyEstimator(
        args.num_items, args.bert_hidden_units, args.bert_max_len, args.bert_dropout,
    )
    # Reconstruct the former module layout and dropout placement.
    legacy.pffn_block = PFFNBlock(
        args.bert_hidden_units, 4 * args.bert_hidden_units, args.bert_dropout,
    )
    current = UncertaintyEstimator(
        args.num_items, args.bert_hidden_units, args.bert_max_len, args.bert_dropout,
    )
    current.load_state_dict(legacy.state_dict(), strict=True)
    legacy.eval()
    current.eval()
    seqs, labels = make_batch(args)
    with torch.no_grad():
        torch.testing.assert_close(current(seqs, labels), legacy(seqs, labels), rtol=0, atol=0)
