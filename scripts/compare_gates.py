"""Compare the original SLSRec gate against the adaptive gate variants.

Reads the runs produced by scripts/run_gate_sweep.sh and reports, per variant:

  * test metrics, mean +/- std over seeds
  * the *validation plateau* mean -- the average of the last k validation
    rounds rather than the single best draw.  On ML-1M the plateau band is
    ~1.6% wide, which is the same order as the effects being measured, so
    comparing single best-checkpoint numbers is not reliable on its own.
  * the delta against the `sum` baseline, with an explicit warning whenever
    that delta is inside the noise band.

``--gates`` additionally loads each run's checkpoint and reports what the gate
actually learned: a near-zero spread across positions means the adaptive gate
collapsed to a static value and is not using its extra capacity.

    python scripts/compare_gates.py
    python scripts/compare_gates.py --gates --dataset ml-1m
"""

import argparse
import json
import os
import re
import statistics
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

def run_re(prefix):
    return re.compile(rf'^{re.escape(prefix)}_(?P<variant>[a-z]+)_(?P<dataset>.+)_seed(?P<seed>\d+)$')


METRICS = ['Recall@10', 'Recall@20', 'NDCG@10', 'NDCG@20', 'MRR@10']
BASELINE = 'sum'
ORDER = ['sum', 'scalar', 'channel', 'token', 'context', 'fusion', 'both']

# variant -> the flags run_gate_sweep.sh used, needed to rebuild the model
FLAGS = {
    'sum':     dict(gate_mode='sum'),
    'scalar':  dict(gate_mode='scalar'),
    'channel': dict(gate_mode='channel'),
    'token':   dict(gate_mode='token'),
    'context': dict(gate_mode='context'),
    'fusion':  dict(fusion_mode='adaptive', fusion_source='token'),
    'both':    dict(gate_mode='token', fusion_mode='adaptive'),
}


def discover(experiments_root, dataset, prefix='gate'):
    RUN_RE = run_re(prefix)
    runs = defaultdict(list)
    root = os.path.join(experiments_root, 'slsrec')
    if not os.path.isdir(root):
        return runs
    for name in sorted(os.listdir(root)):
        m = RUN_RE.match(name)
        if not m or (dataset and m.group('dataset') != dataset):
            continue
        path = os.path.join(root, name)
        test_path = os.path.join(path, 'test_metrics.json')
        if not os.path.isfile(test_path):
            print(f'[skip] {name} -- no test_metrics.json (still running?)')
            continue
        entry = {'name': name, 'path': path, 'seed': int(m.group('seed')),
                 'test': json.load(open(test_path))}
        val_path = os.path.join(path, 'val_metrics.jsonl')
        if os.path.isfile(val_path):
            rows = [json.loads(l) for l in open(val_path) if l.strip()]
            entry['val'] = [r for r in rows if r.get('accum_iter', 0) > 0]
        runs[m.group('variant')].append(entry)
    return runs


def plateau(rows, metric, last_k):
    vals = [r[metric] for r in rows[-last_k:] if metric in r]
    if not vals:
        return None, None
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def agg(values):
    if not values:
        return None, None
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiments', default=os.path.join(ROOT, 'experiments'))
    ap.add_argument('--dataset', default='ml-1m')
    ap.add_argument('--metrics', nargs='+', default=METRICS)
    ap.add_argument('--last-k', type=int, default=10,
                    help='validation rounds averaged for the plateau estimate')
    ap.add_argument('--prefix', default='gate', help='experiment-name prefix used by the sweep')
    ap.add_argument('--gates', action='store_true', help='also inspect what the gates learned')
    ap.add_argument('--csv', default=None)
    args = ap.parse_args()

    runs = discover(args.experiments, args.dataset, args.prefix)
    if not runs:
        print(f'No gate runs found under {args.experiments}/slsrec for dataset {args.dataset}.')
        print('Run:  bash scripts/run_gate_sweep.sh   (or scripts/run_gate_sweep.ps1 on Windows)')
        return 1
    variants = [v for v in ORDER if v in runs] + [v for v in runs if v not in ORDER]

    # ---------------- test metrics ---------------- #
    print(f'\n=== Test metrics, mean +/- std over seeds ({args.dataset}) ===\n')
    header = f"{'variant':<9} {'seeds':>5}  " + '  '.join(f'{m:>17}' for m in args.metrics)
    print(header); print('-' * len(header))
    table = {}
    for v in variants:
        row, cells = {}, []
        for m in args.metrics:
            mu, sd = agg([r['test'][m] for r in runs[v] if m in r['test']])
            row[m] = mu
            cells.append(f'{mu:.5f}+-{sd:.5f}' if mu is not None else f"{'--':>17}")
        table[v] = row
        print(f'{v:<9} {len(runs[v]):>5}  ' + '  '.join(f'{c:>17}' for c in cells))

    # ---------------- delta vs the original gate ---------------- #
    if BASELINE in table:
        print(f'\n=== Relative change vs `{BASELINE}` (original SLSRec eq. 3) ===\n')
        noise = {}
        for m in args.metrics:
            spreads = []
            for v in variants:
                if 'val' in runs[v][0]:
                    for r in runs[v]:
                        _, sd = plateau(r.get('val', []), m, args.last_k)
                        if sd:
                            spreads.append(sd / max(table[v][m] or 1e-9, 1e-9) * 100)
            noise[m] = statistics.mean(spreads) if spreads else None

        hdr = f"{'variant':<9}  " + '  '.join(f'{m:>12}' for m in args.metrics)
        print(hdr); print('-' * len(hdr))
        for v in variants:
            if v == BASELINE:
                continue
            cells = []
            for m in args.metrics:
                b, x = table[BASELINE][m], table[v][m]
                if b is None or x is None:
                    cells.append(f"{'--':>12}"); continue
                d = (x - b) / b * 100
                nb = noise.get(m)
                mark = '' if nb is None else ('' if abs(d) > 2 * nb else ' ~')
                cells.append(f'{d:>+10.2f}%{mark:<2}')
            print(f'{v:<9}  ' + '  '.join(f'{c:>12}' for c in cells))
        if any(noise.values()):
            band = ', '.join(f'{m} +-{noise[m]:.2f}%' for m in args.metrics if noise.get(m))
            print(f'\n  "~" marks a change smaller than 2x the within-run validation noise.')
            print(f'  Noise band (std of the last {args.last_k} validation rounds): {band}')
            print('  A "~" result is not evidence of a difference -- add seeds before concluding.')

    # ---------------- validation plateau ---------------- #
    if any('val' in r for rs in runs.values() for r in rs):
        print(f'\n=== Validation plateau (mean of the last {args.last_k} rounds, over seeds) ===\n')
        hdr = f"{'variant':<9}  " + '  '.join(f'{m:>14}' for m in args.metrics[:3])
        print(hdr); print('-' * len(hdr))
        for v in variants:
            cells = []
            for m in args.metrics[:3]:
                mus = [plateau(r.get('val', []), m, args.last_k)[0] for r in runs[v]]
                mus = [x for x in mus if x is not None]
                mu, sd = agg(mus)
                cells.append(f'{mu:.5f}+-{sd:.4f}' if mu is not None else f"{'--':>14}")
            print(f'{v:<9}  ' + '  '.join(f'{c:>14}' for c in cells))

    # ---------------- what did the gates learn? ---------------- #
    if args.gates:
        inspect_gates(runs, variants, args.dataset)

    if args.csv:
        import csv
        with open(args.csv, 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['variant'] + args.metrics)
            for v in variants:
                w.writerow([v] + [table[v][m] for m in args.metrics])
        print(f'\nwrote {args.csv}')
    return 0


def inspect_gates(runs, variants, dataset):
    """Load one checkpoint per variant and report the learned coefficients."""
    import copy
    import torch

    sys.argv = ['inspect', f'--dataset_code={dataset}', '--model_code=slsrec', '--num_workers=0']
    from datasets import DATASETS                      # noqa: F401  (import order)
    from config import args as cfg, set_template
    set_template(cfg)
    from dataloader import dataloader_factory
    from model import model_factory
    from model.adaptive_gate import gate_statistics

    _, val_loader, _ = dataloader_factory(cfg)
    seqs, _ = next(iter(val_loader))

    print('\n=== What the gates learned (first seed of each variant) ===')
    print('  g -> 1 means the SAM (global) branch dominates, g -> 0 the conv (local) branch.')
    print('  std_pos ~ 0 means the gate is effectively static: the adaptivity is unused.\n')
    hdr = f"{'variant':<9} {'layer':<8} {'mean':>7} {'std_pos':>9} {'std_chan':>9} {'min':>7} {'max':>7}"
    print(hdr); print('-' * len(hdr))
    for v in variants:
        if v == 'sum' or v not in FLAGS:
            continue
        run = sorted(runs[v], key=lambda r: r['seed'])[0]
        ckpt = os.path.join(run['path'], 'models', 'best_acc_model.pth')
        if not os.path.isfile(ckpt):
            print(f'{v:<9} (no checkpoint)'); continue
        a = copy.deepcopy(cfg)
        for k, val in FLAGS[v].items():
            setattr(a, k, val)
        model = model_factory(a)
        model.load_state_dict(torch.load(ckpt, map_location='cpu')['model_state_dict'])
        model.eval()
        with torch.no_grad():
            model(seqs)
        for name, s in gate_statistics(model).items():
            layer = name.split('.')[2] if len(name.split('.')) > 2 else name
            print(f"{v:<9} {layer:<8} {s['mean']:>7.3f} {s['std_over_positions']:>9.4f} "
                  f"{s['std_over_channels']:>9.4f} {s['min']:>7.3f} {s['max']:>7.3f}")


if __name__ == '__main__':
    sys.exit(main())
