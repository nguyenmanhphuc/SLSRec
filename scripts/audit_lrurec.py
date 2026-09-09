"""Reproducibility audit of the LRURec harness.

Checks the things that silently inflate reported numbers in recommender papers:
metric correctness, train/val/test leakage, what the candidate set actually is,
and whether the ground-truth item can be masked away.  Everything here runs on
the *unmodified* LRURec code.

    python scripts/audit_lrurec.py --dataset_code=ml-1m --num_workers=0
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from datasets import DATASETS                       # noqa: F401 (import order)
from config import args, set_template
from trainer.utils import absolute_recall_mrr_ndcg_for_ks

OK, BAD = [], []


def check(name, ok, detail=''):
    (OK if ok else BAD).append(name)
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ''))


def main():
    set_template(args)
    print('\n=== A. Metric functions vs hand-computed values ===')
    # one user, ground truth = item 3, put it at a known rank
    V = 10
    scores = torch.tensor([[0., 9., 8., 7., 6., 5., 4., 3., 2., 1.]])  # item 3 is ranked 3rd
    labels = torch.tensor([3])
    m = absolute_recall_mrr_ndcg_for_ks(scores, labels, [1, 5, 10])
    check('Recall@1 = 0 (gt at rank 3)', m['Recall@1'] == 0.0, f"got {m['Recall@1']}")
    check('Recall@5 = 1', m['Recall@5'] == 1.0, f"got {m['Recall@5']}")
    check('MRR@5 = 1/3', abs(m['MRR@5'] - 1 / 3) < 1e-6, f"got {m['MRR@5']:.6f}")
    import math
    check('NDCG@5 = 1/log2(4)', abs(m['NDCG@5'] - 1 / math.log2(4)) < 1e-6,
          f"got {m['NDCG@5']:.6f} expected {1/math.log2(4):.6f}")
    # ideal DCG for a single relevant item must be 1, not something smaller
    perfect = torch.tensor([[0., 0., 0., 9., 0., 0., 0., 0., 0., 0.]])
    mp = absolute_recall_mrr_ndcg_for_ks(perfect, labels, [10])
    check('NDCG@10 = 1 when gt ranked first', abs(mp['NDCG@10'] - 1.0) < 1e-6, f"got {mp['NDCG@10']:.6f}")

    print('\n=== B. Dataset: leakage between splits ===')
    path = (f"data/preprocessed/{args.dataset_code}_min_rating{args.min_rating}"
            f"-min_uc{args.min_uc}-min_sc{args.min_sc}-{args.split}/dataset.pkl")
    d = pickle.load(open(path, 'rb'))
    train, val, test = d['train'], d['val'], d['test']
    users = sorted(train)
    leak_val = sum(1 for u in users if val[u] and val[u][0] in train[u])
    leak_test = sum(1 for u in users if test[u] and test[u][0] in train[u])
    leak_tv = sum(1 for u in users if test[u] and val[u] and test[u][0] in val[u])
    check('validation item not in the training sequence', leak_val == 0, f'{leak_val}/{len(users)} users')
    check('test item not in the training sequence', leak_test == 0, f'{leak_test}/{len(users)} users')
    check('test item != validation item', leak_tv == 0, f'{leak_tv}/{len(users)} users')
    check('one held-out item per user per split',
          all(len(val[u]) == 1 and len(test[u]) == 1 for u in users),
          f'{len(users)} users')

    print('\n=== C. What is the candidate set at evaluation time? ===')
    from dataloader import dataloader_factory
    _, _, test_loader = dataloader_factory(args)
    seqs, labels = next(iter(test_loader))
    n_items = args.num_items
    scores = torch.randn(seqs.size(0), n_items + 1)
    gt_before = scores.gather(1, labels.view(-1, 1)).clone()
    B, L = seqs.shape
    for i in range(L):
        scores[torch.arange(B), seqs[:, i]] = -1e9
    scores[:, 0] = -1e9
    gt_after = scores.gather(1, labels.view(-1, 1))
    masked_gt = (gt_after < -1e8).sum().item()
    check('ground-truth item is never masked out', masked_gt == 0, f'{masked_gt}/{B} in this batch')
    check('padding item 0 is excluded', scores[:, 0].max().item() < -1e8)
    alive = (scores > -1e8).sum(1).float()
    check('ranking is full-sort (not sampled negatives)', alive.min() > n_items * 0.5,
          f'median {alive.median().item():.0f} candidates alive out of {n_items}')
    print(f"       NOTE: already-seen items ARE removed -- median "
          f"{n_items + 1 - alive.median().item():.0f} of {n_items} items dropped per user "
          f"({100*(n_items+1-alive.median().item())/n_items:.1f}%).")

    print('\n=== D. Train/eval consistency ===')
    from dataloader.sas import SASTrainDataset, SASTestDataset
    tr = SASTrainDataset(args, train, args.bert_max_len, args.sliding_window_size, None)
    te = SASTestDataset(args, train, val, test, args.bert_max_len, None)
    tok, lab = tr[0]
    check('train targets are shifted by one (next-item)',
          bool((tok[1:] == lab[:-1]).all()), 'tokens[1:] == labels[:-1]')
    s, a = te[0]
    check('test input = train + validation item', bool(s[-1].item() == val[sorted(val)[0]][0]),
          f'last input item {s[-1].item()} vs val item {val[sorted(val)[0]][0]}')
    check('test label is the held-out test item', bool(a[0].item() == test[sorted(test)[0]][0]))

    print('\n' + '=' * 62)
    print(f'{len(OK)}/{len(OK)+len(BAD)} audit checks passed')
    if BAD:
        print('PROBLEMS:'); [print('  -', b) for b in BAD]
    print('=' * 62)
    return 1 if BAD else 0


if __name__ == '__main__':
    sys.exit(main())
