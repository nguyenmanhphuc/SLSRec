"""Recurrence tests: sequential vs. parallel scan, finiteness and gradient flow."""

import pytest
import torch

from conftest import make_args, make_batch, seed_all
from model.bdlru import BDLRU, hillis_steele_scan, sequential_scan
from model.slsrec import SLSRec


@pytest.mark.parametrize('length', [1, 2, 3, 8, 17, 64, 200])
def test_parallel_scan_matches_sequential(length):
    seed_all()
    a = torch.rand(3, length, 7) * 0.999
    b = torch.randn(3, length, 7)
    ref = sequential_scan(a, b)
    fast = hillis_steele_scan(a, b)
    err = (ref - fast).abs().max().item()
    assert err < 1e-4, f'max abs error {err} at L={length}'


def test_bdlru_sequential_vs_parallel_module():
    """The BD-LRU module itself must agree across scan implementations."""
    seed_all()
    args = make_args(bert_max_len=32)
    seqs, _ = make_batch(args)
    mask = seqs > 0
    x = torch.randn(seqs.size(0), seqs.size(1), 8)

    torch_mod = BDLRU(8, scan_impl='torch')
    seq_mod = BDLRU(8, scan_impl='sequential')
    seq_mod.load_state_dict(torch_mod.state_dict())

    err = (torch_mod(x, mask) - seq_mod(x, mask)).abs().max().item()
    assert err < 1e-4, f'max abs error {err}'


def test_forward_backward_finite():
    seed_all()
    args = make_args()
    model = SLSRec(args)
    seqs, labels = make_batch(args)

    logits = model(seqs)[0]
    assert logits.shape == (seqs.size(0), args.bert_max_len, args.num_items + 1)
    assert torch.isfinite(logits).all()

    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=0)
    assert torch.isfinite(loss)
    loss.backward()
    for n, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f'non-finite grad in {n}'


def test_gradients_reach_every_component():
    """Every named sub-module of the paper's architecture must receive gradient."""
    seed_all()
    args = make_args()
    model = SLSRec(args)
    seqs, labels = make_batch(args)

    logits = model(seqs, perturb=False)[0]
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=0)
    logvar = model.log_variance(seqs, labels)
    (loss + logvar.mean()).backward()

    def grad_norm(name):
        p = dict(model.named_parameters())[name]
        assert p.grad is not None, f'{name} has no grad'
        return p.grad.abs().sum().item()

    required = {
        'item embeddings':     'model.embedding.token.weight',
        'position embeddings': 'model.embedding.position.weight',
        'recurrence lambda':   'model.blocks.0.sg_lru.bdlru.Lambda',
        'input+recurrence gates': 'model.blocks.0.sg_lru.bdlru.gates.weight',
        'bd-lru convolution':  'model.blocks.0.sg_lru.conv2.conv.weight',
        'gate convolution':    'model.blocks.0.sg_lru.conv1.conv.weight',
        'SAM query':           'model.blocks.0.sg_lru.sam.w_q.weight',
        'SAM key':             'model.blocks.0.sg_lru.sam.w_k.weight',
        'SAM value':           'model.blocks.0.sg_lru.sam.w_v.weight',
        'output projection':   'model.blocks.0.sg_lru.w3.weight',
        'PFFN':                'model.blocks.0.pffn_block.pffn.w8.weight',
        'uncertainty PFFN':    'model.uncertainty.pffn_block.pffn.w8.weight',
        'uncertainty embed':   'model.uncertainty.embedding.token.weight',
    }
    for label, name in required.items():
        assert grad_norm(name) > 0, f'zero gradient for {label} ({name})'


def test_gate_and_conv_switches_change_output():
    """Toggling the SAM branch / the conv / the activation changes the output."""
    seed_all()
    base_args = make_args()
    seqs, _ = make_batch(base_args)

    def out(**over):
        seed_all()
        m = SLSRec(make_args(**over)).eval()
        with torch.no_grad():
            return m(seqs)[0]

    ref = out()
    assert not torch.allclose(ref, out(use_sam_gate=False), atol=1e-6)
    assert not torch.allclose(ref, out(sam_conv=False), atol=1e-6)
    assert not torch.allclose(ref, out(sam_conv_activation='silu'), atol=1e-6)
    assert not torch.allclose(ref, out(bdlru_conv_activation='none'), atol=1e-6)
    assert not torch.allclose(ref, out(use_bdlru=False), atol=1e-6)


def test_sam_disabled_reduces_to_bdlru_checkpoint():
    """Ablation A: with no SAM and no gate-conv the model is the BD-LRU (RecBLR-style)
    checkpoint -- the SAM parameters must then not exist at all."""
    args = make_args(use_sam_gate=False, sam_conv=False)
    model = SLSRec(args)
    names = [n for n, _ in model.named_parameters()]
    assert not any('sam' in n for n in names)
    assert not any('conv1' in n for n in names)
    assert any('bdlru' in n for n in names)
    assert any('conv2' in n for n in names)


@pytest.mark.parametrize('use_uncertainty', [False, True])
def test_pffn_bias_switch(use_uncertainty):
    """--pffn_bias adds exactly the W6/W7/W8 biases of every PFFN (eq. 13 prints
    none); the default stays bias-free."""
    def pffns(model):
        return [m for m in model.modules() if type(m).__name__ == 'PFFN']

    off = SLSRec(make_args(use_uncertainty=use_uncertainty))
    on = SLSRec(make_args(use_uncertainty=use_uncertainty, pffn_bias=True))
    assert all(l.bias is None for p in pffns(off) for l in (p.w6, p.w7, p.w8))
    assert all(l.bias is not None for p in pffns(on) for l in (p.w6, p.w7, p.w8))

    D = make_args().bert_hidden_units
    n_pffn = len(pffns(on))
    assert n_pffn == make_args().bert_num_blocks + (1 if use_uncertainty else 0)
    extra = sum(p.numel() for p in on.parameters()) - sum(p.numel() for p in off.parameters())
    assert extra == n_pffn * (4 * D + 4 * D + D)


def test_sglru_dropout_switch():
    """--sglru_dropout changes the eq. (7) dropout only; every other dropout keeps the
    shared rate, and leaving it unset reproduces the shared rate everywhere."""
    def rates(model):
        sg = {blk.sg_lru.dropout.p for blk in model.model.blocks}
        rest = {m.p for n, m in model.named_modules()
                if isinstance(m, torch.nn.Dropout) and '.sg_lru.dropout' not in n and '.sam.' not in n}
        return sg, rest

    sg, rest = rates(SLSRec(make_args(bert_dropout=0.75)))
    assert sg == {0.75} and rest == {0.75}
    sg, rest = rates(SLSRec(make_args(bert_dropout=0.75, sglru_dropout=0.2)))
    assert sg == {0.2} and rest == {0.75}


def test_recurrence_is_stable():
    """a_t must live in (0, 1] so the recurrence cannot blow up."""
    seed_all()
    mod = BDLRU(16)
    x = torch.randn(4, 64, 16) * 5.0
    mask = torch.ones(4, 64, dtype=torch.bool)
    r, i = mod.gates(x).chunk(2, dim=-1)
    a = torch.exp(mod.c * torch.nn.functional.logsigmoid(mod.Lambda) * torch.sigmoid(r))
    assert (a > 0).all() and (a <= 1.0).all()
    h = mod(x, mask)
    assert torch.isfinite(h).all()

    # initialisation: with r_t == 1, a spans [r_min, r_max]
    a_init = torch.exp(mod.c * torch.nn.functional.logsigmoid(mod.Lambda))
    assert abs(a_init.min().item() - 0.9) < 1e-4
    assert abs(a_init.max().item() - 0.999) < 1e-4
