"""--fusion_mode kan / kan_res: KAN fusion of G and H (extension, not in the SLSRec paper)."""

import pytest
import torch

from conftest import make_args, make_batch, seed_all
from model.kan import KANFusion, KANLinear
from model.slsrec import SLSRec
from trainer.slsrec import slsrec_losses

MODES = ['kan', 'kan_res']


def test_bsplines_are_a_partition_of_unity_on_the_grid():
    layer = KANLinear(3, 2, grid_size=5, spline_order=3, grid_range=(-2.0, 2.0))
    x = torch.linspace(-1.99, 1.99, 101).unsqueeze(-1).expand(-1, 3)
    bases = layer.b_splines(x)
    assert bases.shape == (101, 3, 8)
    assert torch.allclose(bases.sum(-1), torch.ones(101, 3), atol=1e-5)
    assert (bases >= 0).all()


def test_kan_res_starts_exactly_as_the_product():
    """Same weights everywhere else: kan_res at init equals eq. (6)."""
    seed_all()
    base = SLSRec(make_args(bert_dropout=0.0)).eval()
    seed_all()
    kan = SLSRec(make_args(fusion_mode='kan_res', bert_dropout=0.0)).eval()
    missing, unexpected = kan.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all('fusion_gate' in k for k in missing)
    seqs, _ = make_batch(make_args())
    with torch.no_grad():
        assert torch.allclose(base(seqs)[0], kan(seqs)[0], atol=1e-6)


def test_kan_init_survives_the_global_init():
    """The model's truncated-normal init must not overwrite the KAN layers."""
    seed_all()
    m = SLSRec(make_args(fusion_mode='kan'))
    fusion = m.model.blocks[0].sg_lru.fusion_gate
    assert isinstance(fusion, KANFusion)
    assert torch.allclose(fusion.layer_norm.weight, torch.ones_like(fusion.layer_norm.weight))
    assert fusion.layers[0].base_weight.abs().max() > 0.04      # kaiming, not trunc-normal(0.02)
    seed_all()
    r = SLSRec(make_args(fusion_mode='kan_res'))
    last = r.model.blocks[0].sg_lru.fusion_gate.layers[-1]
    assert last.base_weight.abs().max() == 0 and last.spline_weight.abs().max() == 0


@pytest.mark.parametrize('mode', MODES)
def test_kan_fusion_is_causal(mode):
    seed_all()
    args = make_args(fusion_mode=mode)
    model = SLSRec(args).eval()
    seqs, _ = make_batch(args)
    alt = seqs.clone()
    t = args.bert_max_len - 3
    alt[:, t] = (alt[:, t] % args.num_items) + 1
    with torch.no_grad():
        a, _ = model.model.encode(seqs)
        b, _ = model.model.encode(alt)
    assert torch.allclose(a[:, :t], b[:, :t], atol=1e-6)


@pytest.mark.parametrize('mode', MODES)
def test_kan_fusion_ignores_padding_amount(mode):
    seed_all()
    args = make_args(fusion_mode=mode, bert_max_len=24)
    model = SLSRec(args).eval()
    body = torch.randint(1, args.num_items + 1, (3, 10))

    def hidden(total):
        s = torch.zeros(3, total, dtype=torch.long)
        s[:, -10:] = body
        with torch.no_grad():
            return model.model.encode(s)[0][:, -10:]

    assert torch.allclose(hidden(10), hidden(24), atol=1e-5)


@pytest.mark.parametrize('mode', MODES)
def test_kan_fusion_trains(mode):
    seed_all()
    args = make_args(fusion_mode=mode, use_uncertainty=False)
    model = SLSRec(args).train()
    seqs, labels = make_batch(args, batch_size=8, real_len=args.bert_max_len - 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    first = None
    for _ in range(30):
        opt.zero_grad()
        l = slsrec_losses(args, model, seqs, labels, training=True)
        assert torch.isfinite(l['L_total'])
        l['L_total'].backward()
        opt.step()
        first = first if first is not None else l['L_rec'].item()
    assert l['L_rec'].item() < first
