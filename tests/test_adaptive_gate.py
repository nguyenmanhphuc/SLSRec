"""Adaptive gate (research extension) -- correctness and experimental validity.

The headline test is ``test_neutral_init_is_bit_identical_to_original``: with
neutral initialisation every gate variant must reproduce the original SLSRec
layer exactly.  That is what makes a gate A/B comparison a controlled
experiment rather than a comparison of two unrelated models.
"""

import pytest
import torch

from conftest import make_args, make_batch, seed_all
from model.adaptive_gate import AdaptiveGate, FusionGate, gate_statistics
from model.slsrec import SLSRec

GATE_MODES = ['scalar', 'channel', 'token', 'context']


def _aligned_pair(**over):
    """Build a `sum` model and a gated model that share every non-gate weight.

    Adding gate parameters shifts the RNG stream used by truncated_normal_init,
    so the two models are aligned by copying the shared state dict rather than
    by re-seeding.
    """
    seed_all()
    base = SLSRec(make_args(gate_mode='sum', fusion_mode='mul')).eval()
    seed_all()
    gated = SLSRec(make_args(**over)).eval()
    missing, unexpected = gated.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected, unexpected
    assert all('gate' in m for m in missing), missing   # only gate params are extra
    return base, gated


@pytest.mark.parametrize('mode', GATE_MODES)
def test_neutral_init_is_bit_identical_to_original(mode):
    base, gated = _aligned_pair(gate_mode=mode)
    seqs, _ = make_batch(make_args())
    with torch.no_grad():
        a, b = base(seqs)[0], gated(seqs)[0]
    err = (a - b).abs().max().item()
    assert err == 0.0, f'gate_mode={mode} drifted from eq.(3) at neutral init: {err}'


def test_adaptive_fusion_neutral_init_is_bit_identical():
    base, gated = _aligned_pair(fusion_mode='adaptive')
    seqs, _ = make_batch(make_args())
    with torch.no_grad():
        a, b = base(seqs)[0], gated(seqs)[0]
    assert (a - b).abs().max().item() == 0.0


def test_both_axes_together_are_neutral():
    base, gated = _aligned_pair(gate_mode='context', fusion_mode='adaptive',
                                fusion_source='context')
    seqs, _ = make_batch(make_args())
    with torch.no_grad():
        assert (base(seqs)[0] - gated(seqs)[0]).abs().max().item() == 0.0


@pytest.mark.parametrize('mode', GATE_MODES)
def test_gate_changes_output_once_it_moves_off_neutral(mode):
    """Sanity: the gate is actually wired in, not a no-op."""
    _, gated = _aligned_pair(gate_mode=mode)
    seqs, _ = make_batch(make_args())
    with torch.no_grad():
        before = gated(seqs)[0].clone()
        for m in gated.modules():                       # push every gate off 0.5
            if isinstance(m, AdaptiveGate):
                if m.mode in ('scalar', 'channel'):
                    m.gate_logit.fill_(2.0)
                else:
                    lin = m.proj[-1] if isinstance(m.proj, torch.nn.Sequential) else m.proj
                    lin.bias.fill_(2.0)
        after = gated(seqs)[0]
    assert not torch.allclose(before, after, atol=1e-6), f'{mode} gate has no effect'


@pytest.mark.parametrize('mode', GATE_MODES)
def test_gate_coefficient_is_bounded(mode):
    seed_all()
    model = SLSRec(make_args(gate_mode=mode)).eval()
    seqs, _ = make_batch(make_args())
    with torch.no_grad():
        model(seqs)
    stats = gate_statistics(model)
    assert stats, 'no gate statistics recorded'
    for name, s in stats.items():
        assert 0.0 <= s['min'] <= s['max'] <= 1.0, f'{name}: g outside [0,1] -> {s}'


def test_fusion_coefficient_is_bounded():
    seed_all()
    model = SLSRec(make_args(fusion_mode='adaptive')).eval()
    seqs, _ = make_batch(make_args())
    with torch.no_grad():
        model(seqs)
    for name, s in gate_statistics(model).items():
        assert 0.0 <= s['min'] <= s['max'] <= 2.0, f'{name}: a outside [0,2] -> {s}'


@pytest.mark.parametrize('mode', GATE_MODES)
def test_gate_receives_gradient(mode):
    seed_all()
    args = make_args(gate_mode=mode, fusion_mode='adaptive')
    model = SLSRec(args)
    seqs, labels = make_batch(args)
    logits = model(seqs)[0]
    torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=0).backward()

    gate_params = [(n, p) for n, p in model.named_parameters()
                   if 'adaptive_gate' in n or 'fusion_gate' in n]
    assert gate_params, 'no gate parameters found'
    for n, p in gate_params:
        assert p.grad is not None, f'{n} has no grad'
        if n.endswith('weight') or 'gate_logit' in n:
            assert p.grad.abs().sum().item() > 0, f'{n} received zero gradient'


@pytest.mark.parametrize('mode', GATE_MODES)
def test_gate_preserves_causality(mode):
    seed_all()
    args = make_args(gate_mode=mode, fusion_mode='adaptive', bert_max_len=12)
    model = SLSRec(args).eval()
    seqs, _ = make_batch(args, batch_size=2, real_len=12)
    t = 9
    alt = seqs.clone()
    alt[:, t] = (alt[:, t] % args.num_items) + 1
    with torch.no_grad():
        ha, _ = model.model.encode(seqs)
        hb, _ = model.model.encode(alt)
    err = (ha[:, :t] - hb[:, :t]).abs().max().item()
    assert err < 1e-6, f'{mode}: future leakage {err}'


@pytest.mark.parametrize('mode', GATE_MODES)
def test_gate_preserves_padding_invariance(mode):
    seed_all()
    args = make_args(gate_mode=mode, fusion_mode='adaptive', bert_max_len=24)
    model = SLSRec(args).eval()
    real_len = 9
    body = torch.randint(1, args.num_items + 1, (2, real_len))

    def hidden(total_len):
        s = torch.zeros(2, total_len, dtype=torch.long)
        s[:, -real_len:] = body
        with torch.no_grad():
            h, _ = model.model.encode(s)
        return h[:, -real_len:]

    err = (hidden(12) - hidden(24)).abs().max().item()
    assert err < 1e-5, f'{mode}: padding drift {err}'


def test_parameter_overhead_is_small():
    """The adaptive gate must not win simply by being bigger.

    Measured at the real ML-1M dimensions (|I|=3416, D=64, d=128) -- the tiny
    dimensions used elsewhere in this file would exaggerate the overhead,
    because the gate scales with d while the embeddings scale with |I|.
    """
    real = dict(num_items=3416, bert_hidden_units=64, bert_max_len=200)
    seed_all()
    base = sum(p.numel() for p in SLSRec(make_args(gate_mode='sum', **real)).parameters())
    counts = {}
    for mode in GATE_MODES:
        seed_all()
        counts[mode] = sum(p.numel() for p in SLSRec(make_args(gate_mode=mode, **real)).parameters())
    for mode, n in counts.items():
        overhead = (n - base) / base * 100
        assert overhead < 10.0, f'{mode} adds {overhead:.2f}% parameters -- too much to attribute'
    # the two controls must be essentially free
    assert counts['scalar'] - base <= 2 * make_args().bert_num_blocks   # ~1 param per layer
    assert (counts['channel'] - base) / base * 100 < 0.1


def test_scalar_mode_is_a_capacity_control():
    """`scalar` adds ~nothing, so if it matches `token` the gain is capacity,
    not adaptivity.  Check it really is static across positions."""
    seed_all()
    model = SLSRec(make_args(gate_mode='scalar')).eval()
    seqs, _ = make_batch(make_args())
    with torch.no_grad():
        model(seqs)
    for name, s in gate_statistics(model).items():
        assert s['std_over_positions'] == 0.0, f'{name} is not static: {s}'
