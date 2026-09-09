"""Aggregate a finished sweep into a mean +/- std table.

Reads every ``experiments/<model>/<exp_name>/test_metrics.json`` whose folder is
named ``<ablation>_<dataset>_seed<N>`` (the layout scripts/run_sweep.sh
produces) and prints one markdown row per ablation x dataset, averaged over
seeds -- step 2 of reproduction_report.md section 10.

    python scripts/collect_results.py
    python scripts/collect_results.py --dataset ml-1m --csv results.csv
"""

import argparse
import csv
import json
import os
import re
import statistics
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS = os.path.join(ROOT, 'experiments')

# scripts/run_sweep.sh: <ablation>_<dataset>_seed<N>
RUN_RE = re.compile(r'^(?P<ablation>[A-D]_[a-z_]+)_(?P<dataset>.+)_seed(?P<seed>\d+)$')

DEFAULT_METRICS = ['Recall@10', 'Recall@20', 'NDCG@10', 'NDCG@20', 'MRR@10', 'MRR@20']


def load_runs(experiments_root):
    """-> {(ablation, dataset): {seed: metrics_dict}}, plus the unparsed folders."""
    runs, skipped = defaultdict(dict), []
    if not os.path.isdir(experiments_root):
        return runs, skipped
    for model_code in sorted(os.listdir(experiments_root)):
        model_dir = os.path.join(experiments_root, model_code)
        if not os.path.isdir(model_dir):
            continue
        for exp_name in sorted(os.listdir(model_dir)):
            metrics_path = os.path.join(model_dir, exp_name, 'test_metrics.json')
            if not os.path.isfile(metrics_path):
                continue
            m = RUN_RE.match(exp_name)
            if m is None:
                skipped.append(os.path.join(model_code, exp_name))
                continue
            with open(metrics_path) as f:
                runs[(m.group('ablation'), m.group('dataset'))][int(m.group('seed'))] = json.load(f)
    return runs, skipped


def summarise(values):
    """mean +/- sample std; std is None for a single seed."""
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else None
    return mean, std


def format_cell(mean, std):
    return '%.4f' % mean if std is None else '%.4f +/- %.4f' % (mean, std)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', default=EXPERIMENTS)
    parser.add_argument('--dataset', default=None, help='only report this dataset')
    parser.add_argument('--metrics', nargs='+', default=DEFAULT_METRICS)
    parser.add_argument('--csv', default=None, help='also write the table here')
    args = parser.parse_args()

    runs, skipped = load_runs(args.experiments)
    if args.dataset:
        runs = {k: v for k, v in runs.items() if k[1] == args.dataset}
    if not runs:
        print('no sweep runs found under %s' % args.experiments, file=sys.stderr)
        print('(folders must be named <ablation>_<dataset>_seed<N>)', file=sys.stderr)
        return 1

    rows = []
    for (ablation, dataset), by_seed in sorted(runs.items()):
        seeds = sorted(by_seed)
        row = {'ablation': ablation, 'dataset': dataset,
               'seeds': ','.join(str(s) for s in seeds), 'n': len(seeds)}
        for metric in args.metrics:
            values = [by_seed[s][metric] for s in seeds if metric in by_seed[s]]
            if not values:
                row[metric] = ''
                continue
            mean, std = summarise(values)
            row[metric] = format_cell(mean, std)
            row[metric + '_mean'] = mean
            row[metric + '_std'] = std if std is not None else ''
        rows.append(row)

    header = ['ablation', 'dataset', 'n'] + args.metrics
    widths = [max(len(h), *(len(str(r.get(h, ''))) for r in rows)) for h in header]
    print('| ' + ' | '.join(h.ljust(w) for h, w in zip(header, widths)) + ' |')
    print('|' + '|'.join('-' * (w + 2) for w in widths) + '|')
    for r in rows:
        print('| ' + ' | '.join(str(r.get(h, '')).ljust(w) for h, w in zip(header, widths)) + ' |')

    incomplete = [r for r in rows if r['n'] < 5]
    if incomplete:
        print()
        for r in incomplete:
            print('note: %s/%s has %d seed(s): %s'
                  % (r['ablation'], r['dataset'], r['n'], r['seeds']))
    if skipped:
        print()
        print('ignored (name is not <ablation>_<dataset>_seed<N>): ' + ', '.join(skipped))

    if args.csv:
        fields = ['ablation', 'dataset', 'seeds', 'n']
        for metric in args.metrics:
            fields += [metric + '_mean', metric + '_std']
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        print()
        print('wrote %s' % args.csv)
    return 0


if __name__ == '__main__':
    sys.exit(main())
