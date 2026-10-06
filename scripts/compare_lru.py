"""Compare an LRURec baseline run against upstream and against the paper.

Three columns: the unmodified `_ref_LRURec/` source, this repository's `lru`
path, and the published LRURec numbers.  The first two share a byte-identical
`dataset.pkl`, `model/lru.py` and `trainer/lru.py`, so they should agree to
within GPU nondeterminism; a larger gap means the harness drifted and SLSRec
results built on it would not be trustworthy.

    python scripts/compare_lru.py \
        --ref  _ref_LRURec/experiments/lru/ml-1m_0.01_0.2_0.2/test_metrics.json \
        --ours experiments/lru/lru_baseline_ml-1m_seed42/test_metrics.json
"""

import argparse
import json
import os
import sys

# LRURec, WSDM 2024 -- the "LRURec" column of _ref_LRURec/media/performance.png.
PAPER = {
    'ml-1m':  {'NDCG@10': 0.19298, 'NDCG@20': 0.21990, 'Recall@10': 0.32636, 'Recall@20': 0.43313},
    'beauty': {'NDCG@10': 0.03088, 'NDCG@20': 0.03560, 'Recall@10': 0.05284, 'Recall@20': 0.07161},
    'video':  {'NDCG@10': 0.06198, 'NDCG@20': 0.07547, 'Recall@10': 0.11337, 'Recall@20': 0.16705},
    'sports': {'NDCG@10': 0.01815, 'NDCG@20': 0.02149, 'Recall@10': 0.03148, 'Recall@20': 0.04477},
    'steam':  {'NDCG@10': 0.06934, 'NDCG@20': 0.08508, 'Recall@10': 0.12831, 'Recall@20': 0.19082},
}

METRICS = ['Recall@10', 'Recall@20', 'NDCG@10', 'NDCG@20']


def load(path, label):
    if not os.path.isfile(path):
        print('%s: missing %s' % (label, path), file=sys.stderr)
        return None
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', default='_ref_LRURec/experiments/lru/ml-1m_0.01_0.2_0.2/test_metrics.json')
    parser.add_argument('--ours', default='experiments/lru/lru_baseline_ml-1m_seed42/test_metrics.json')
    parser.add_argument('--dataset', default='ml-1m', choices=sorted(PAPER))
    parser.add_argument('--tol', type=float, default=0.005,
                        help='max |upstream - ours| accepted as GPU nondeterminism')
    args = parser.parse_args()

    ref = load(args.ref, 'upstream')
    ours = load(args.ours, 'ours')
    paper = PAPER[args.dataset]
    if ref is None and ours is None:
        return 2

    header = ['metric', 'upstream', 'ours', 'ours-upstream', 'paper', 'ours vs paper']
    rows = []
    for m in METRICS:
        r = ref.get(m) if ref else None
        o = ours.get(m) if ours else None
        p = paper.get(m)
        rows.append([
            m,
            '%.5f' % r if r is not None else '-',
            '%.5f' % o if o is not None else '-',
            '%+.5f' % (o - r) if (r is not None and o is not None) else '-',
            '%.5f' % p,
            '%+.1f%%' % (100.0 * (o - p) / p) if o is not None else '-',
        ])

    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(header)]
    print('| ' + ' | '.join(h.ljust(w) for h, w in zip(header, widths)) + ' |')
    print('|' + '|'.join('-' * (w + 2) for w in widths) + '|')
    for row in rows:
        print('| ' + ' | '.join(c.ljust(w) for c, w in zip(row, widths)) + ' |')
    print()

    status = 0
    if ref is not None and ours is not None:
        worst = max(abs(ours[m] - ref[m]) for m in METRICS if m in ref and m in ours)
        if worst <= args.tol:
            print('PASS  upstream vs ours: max |delta| = %.5f <= tol %.5f' % (worst, args.tol))
        else:
            print('FAIL  upstream vs ours: max |delta| = %.5f > tol %.5f' % (worst, args.tol))
            print('      the harness is NOT equivalent to upstream -- fix before running SLSRec')
            status = 1
    else:
        print('SKIP  upstream vs ours: only one of the two runs is present')

    if ours is not None:
        gaps = [100.0 * (ours[m] - paper[m]) / paper[m] for m in METRICS if m in ours]
        print('INFO  ours vs published LRURec: %+.1f%% to %+.1f%% (single seed, different hardware)'
              % (min(gaps), max(gaps)))
    return status


if __name__ == '__main__':
    sys.exit(main())
