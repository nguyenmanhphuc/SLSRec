"""Uncertainty estimator: where the dropout sits around the residual.

The paper draws and prints this block two different ways:

    Fig. 2      PFFN -> Dropout -> Add & Norm, residual enters Add & Norm
                directly:              LN(Dropout(PFFN(xc)) + xc)     'fig2'
    eq. (20)    as printed:            LN(Dropout(PFFN(xc) + xc))     'eq20'

Fig. 2 agrees with every other block of the model (Fig. 1, eq. 7, eq. 14), so
it is the default; ``--unc_dropout_form=eq20`` selects the printed equation.
The two forms are identical at eval time and differ only in training.
"""

import pytest
import torch
import torch.nn.functional as F

from conftest import make_args, make_batch
from model.slsrec import PFFNBlock, SLSRec, UncertaintyEstimator, UncertaintyPFFNBlock


def _expected(block, xc, dropout, training, form):
    if form == 'fig2':
        h = F.dropout(block.pffn(xc), p=dropout, training=training) + xc
    else:
        h = F.dropout(block.pffn(xc) + xc, p=dropout, training=training)
    return F.layer_norm(h, block.layer_norm.normalized_shape,
                        block.layer_norm.weight, block.layer_norm.bias, block.layer_norm.eps)


def test_default_follows_fig2():
    model = SLSRec(make_args())
    est = model.model.uncertainty
    assert est.dropout_form == 'fig2'
    assert type(est.pffn_block) is PFFNBlock


def test_eq20_flag_selects_the_printed_form():
    est = SLSRec(make_args(unc_dropout_form='eq20')).model.uncertainty
    assert est.dropout_form == 'eq20'
    assert type(est.pffn_block) is UncertaintyPFFNBlock


@pytest.mark.parametrize('form', ['fig2', 'eq20'])
@pytest.mark.parametrize('dropout', [0.0, 0.2, 0.75])
@pytest.mark.parametrize('training', [False, True])
def test_estimator_matches_its_formula(form, dropout, training):
    args = make_args(bert_dropout=dropout, unc_dropout_form=form)
    est = SLSRec(args).model.uncertainty
    est.train(training)
    est.embedding.dropout.p = 0.0          # isolate the block from eq. (1)'s dropout
    seqs, labels = make_batch(args)
    xc, _ = est.embedding(seqs)

    torch.manual_seed(123)
    hidden = _expected(est.pffn_block, xc, dropout, training, form)
    expected = (est.embedding.token(labels) * hidden).sum(-1)
    torch.manual_seed(123)
    actual = est(seqs, labels)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize('form,keeps_residual', [('fig2', True), ('eq20', False)])
def test_full_dropout_tells_the_forms_apart(form, keeps_residual):
    """With PFFN zeroed and dropout=1 in training, Fig. 2 keeps xc (residual
    bypasses the dropout) while eq. (20) drops it too."""
    args = make_args(bert_dropout=1.0, unc_dropout_form=form)
    est = SLSRec(args).model.uncertainty.train()
    est.embedding.dropout.p = 0.0
    seqs, labels = make_batch(args)
    with torch.no_grad():
        for p in est.pffn_block.pffn.parameters():
            p.zero_()
        xc, _ = est.embedding(seqs)
        out = est(seqs, labels)
    if keeps_residual:
        ln = F.layer_norm(xc, est.pffn_block.layer_norm.normalized_shape,
                          est.pffn_block.layer_norm.weight, est.pffn_block.layer_norm.bias,
                          est.pffn_block.layer_norm.eps)
        torch.testing.assert_close(out, (est.embedding.token(labels) * ln).sum(-1))
        assert out.abs().max() > 0
    else:
        torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)


def test_forms_are_identical_at_eval_and_share_checkpoints():
    """Same state dict, same eval output -- old eq20 checkpoints load into the
    fig2 model and vice versa."""
    args = make_args(bert_dropout=0.75)
    kw = (args.num_items, args.bert_hidden_units, args.bert_max_len, args.bert_dropout)
    fig2 = UncertaintyEstimator(*kw, dropout_form='fig2')
    eq20 = UncertaintyEstimator(*kw, dropout_form='eq20')
    eq20.load_state_dict(fig2.state_dict(), strict=True)
    fig2.eval(); eq20.eval()
    seqs, labels = make_batch(args)
    with torch.no_grad():
        torch.testing.assert_close(fig2(seqs, labels), eq20(seqs, labels), rtol=0, atol=0)


def test_forms_differ_in_training():
    args = make_args(bert_dropout=0.5)
    kw = (args.num_items, args.bert_hidden_units, args.bert_max_len, args.bert_dropout)
    fig2 = UncertaintyEstimator(*kw, dropout_form='fig2')
    eq20 = UncertaintyEstimator(*kw, dropout_form='eq20')
    eq20.load_state_dict(fig2.state_dict(), strict=True)
    fig2.train(); eq20.train()
    seqs, labels = make_batch(args)
    torch.manual_seed(0); a = fig2(seqs, labels)
    torch.manual_seed(0); b = eq20(seqs, labels)
    assert not torch.allclose(a, b), 'the two placements should differ while training'


def test_unknown_form_is_rejected():
    with pytest.raises(ValueError):
        UncertaintyEstimator(10, 8, 4, 0.1, dropout_form='nope')
