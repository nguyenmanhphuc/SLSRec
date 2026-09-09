"""Padding invariance: extra left padding must not change anything real."""

import torch

from conftest import make_args, make_batch, seed_all
from model.bdlru import BDLRU
from model.slsrec import SLSRec


def test_extra_left_padding_does_not_change_representations():
    """Right-align the same items in tensors of different total length; the
    hidden states at the real positions must be bit-for-bit comparable."""
    seed_all()
    args = make_args(bert_max_len=24)
    model = SLSRec(args).eval()

    real_len = 10
    body = torch.randint(1, args.num_items + 1, (3, real_len))

    def hidden(total_len, pad_fill=0):
        s = torch.full((3, total_len), pad_fill, dtype=torch.long)
        s[:, -real_len:] = body
        with torch.no_grad():
            h, _ = model.model.encode(s)
        return h[:, -real_len:]

    h10 = hidden(10)     # no padding at all
    h15 = hidden(15)     # 5 pad tokens in front
    h24 = hidden(24)     # 14 pad tokens in front

    assert torch.allclose(h10, h15, atol=1e-6), (h10 - h15).abs().max().item()
    assert torch.allclose(h10, h24, atol=1e-6), (h10 - h24).abs().max().item()
    assert h10.abs().max() > 0  # sanity: the representations are not all zero


def test_padding_amount_does_not_change_scores():
    """A sequence padded to two different total lengths gives identical scores
    at the corresponding real positions."""
    seed_all()
    args = make_args(bert_max_len=32)
    model = SLSRec(args).eval()

    real_len = 9
    body = torch.randint(1, args.num_items + 1, (2, real_len))

    def scores_for(total_len):
        s = torch.zeros(2, total_len, dtype=torch.long)
        s[:, -real_len:] = body
        with torch.no_grad():
            return model(s)[0]

    a = scores_for(20)
    b = scores_for(32)
    err = (a[:, -real_len:] - b[:, -real_len:]).abs().max().item()
    assert err < 1e-5, f'score drift from padding: {err}'


def test_padded_positions_do_not_affect_bdlru_state():
    seed_all()
    mod = BDLRU(8).eval()
    B, L, D = 2, 16, 8
    x = torch.randn(B, L, D)
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[:, 6:] = True                       # 6 leading pad steps

    h_ref = mod(x, mask)
    x2 = x.clone()
    x2[:, :6] = torch.randn(B, 6, D) * 100   # garbage in the padded region
    h_new = mod(x2, mask)
    assert torch.allclose(h_ref[:, 6:], h_new[:, 6:], atol=1e-6)
    assert (h_ref[:, :6] == 0).all()


def test_attention_ignores_padded_keys():
    seed_all()
    args = make_args(bert_max_len=16)
    model = SLSRec(args).eval()
    sam = model.model.blocks[0].sg_lru.sam

    B, L, d = 2, 16, args.bert_hidden_units * args.slsrec_expand
    x = torch.randn(B, L, d)
    mask = torch.zeros(B, L, dtype=torch.bool)
    mask[:, 5:] = True

    ref = sam(x * mask.unsqueeze(-1), mask)
    x2 = x.clone()
    x2[:, :5] = torch.randn(B, 5, d) * 50
    new = sam(x2 * mask.unsqueeze(-1), mask)
    assert torch.allclose(ref[:, 5:], new[:, 5:], atol=1e-6)
    assert (ref[:, :5] == 0).all()
