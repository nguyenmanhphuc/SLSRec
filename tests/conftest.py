"""Shared fixtures / helpers for the SLSRec test-suite.

Builds argument namespaces directly (instead of going through ``config.py``'s
module-level ``parse_known_args``) so that the tests are independent of the CLI.
"""

import os
import random
import sys
from argparse import Namespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = 42


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_args(**overrides):
    """SLSRec defaults, shrunk so the tests run fast on CPU."""
    args = Namespace(
        # harness
        model_code='slsrec', dataset_code='ml-1m', device='cpu', seed=SEED,
        num_items=40, bert_max_len=16, bert_hidden_units=16, bert_num_blocks=2,
        bert_dropout=0.0, bert_attn_dropout=0.0,
        metric_ks=[1, 5, 10], best_metric='Recall@10',
        # architecture
        slsrec_expand=2, slsrec_d_conv=4, scan_impl='torch', use_item_bias=False,
        use_sam_gate=True, sam_conv=True, sam_conv_activation='none',
        sam_num_heads=1, sam_attn_dropout=0.0,
        use_bdlru=True, bdlru_c=8.0, bdlru_r_min=0.9, bdlru_r_max=0.999,
        bdlru_lambda_init='linspace', bdlru_conv_activation='silu',
        # perturbation
        use_perturbation=True, perturbation_probability=0.1,
        perturbation_low=0.0, perturbation_high=1.0, perturbation_weight=1.0,
        # uncertainty
        use_uncertainty=True, uncertainty_alpha=0.3, uncertainty_beta=0.0,
        uncertainty_loss='approx', uncertainty_logvar_clamp=8.0,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def make_batch(args, batch_size=4, real_len=None, generator=None):
    """Left-padded item-id sequences + per-position next-item labels."""
    L = args.bert_max_len
    real_len = real_len if real_len is not None else L - 3
    g = generator or torch.Generator().manual_seed(SEED)
    seqs = torch.zeros(batch_size, L, dtype=torch.long)
    labels = torch.zeros(batch_size, L, dtype=torch.long)
    full = torch.randint(1, args.num_items + 1, (batch_size, real_len + 1), generator=g)
    seqs[:, L - real_len:] = full[:, :-1]
    labels[:, L - real_len:] = full[:, 1:]
    return seqs, labels


@pytest.fixture
def args():
    return make_args()


@pytest.fixture(autouse=True)
def _deterministic():
    seed_all()
    yield
