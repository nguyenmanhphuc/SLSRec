"""Join every run's hyperparameters to its test metrics, and to the papers.

`trainer/base.py` writes `run_config.json` (the full resolved namespace, after
set_template) and `run_summary.json` (cost, early stopping, best validation)
next to `test_metrics.json`.  This walks the experiment folders, prints one row
per run with the headline hyperparameters plus **whatever actually differs
between the runs** -- so a dropout sweep grows a dropout column by itself -- and
puts the published number for the same model/dataset beside it.  `--columns all`
widens the table to the whole namespace, `--columns varying` narrows it to just
what changed.

    python scripts/collect_hparams.py                          # everything found
    python scripts/collect_hparams.py --model lru --dataset ml-1m
    python scripts/collect_hparams.py --include-ref            # + unmodified _ref_LRURec runs
    python scripts/collect_hparams.py --csv results.csv        # full namespace, every key

Runs produced before run_config.json existed still appear: their hyperparameters
are read back from the upstream folder name (`<dataset>_<wd>_<dropout>_<attn>`)
and marked `*`.
"""

import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METRICS = ['Recall@10', 'Recall@20', 'NDCG@10', 'NDCG@20']

# SLSRec, IEEE TCSS 2026, Table II -- the LRURec / RecBLR / SLSRec columns.
# These are the numbers a reproduction has to land on.
SLSREC_PAPER = {
    'ml-1m':  {'lru':    {'Recall@10': 0.32619, 'Recall@20': 0.43313, 'NDCG@10': 0.19292, 'NDCG@20': 0.21988},
               'recblr': {'Recall@10': 0.34626, 'Recall@20': 0.44979, 'NDCG@10': 0.20715, 'NDCG@20': 0.23385},
               'slsrec': {'Recall@10': 0.35595, 'Recall@20': 0.46732, 'NDCG@10': 0.21332, 'NDCG@20': 0.24141}},
    'beauty': {'lru':    {'Recall@10': 0.05249, 'Recall@20': 0.07202, 'NDCG@10': 0.03058, 'NDCG@20': 0.03551},
               'recblr': {'Recall@10': 0.05345, 'Recall@20': 0.07329, 'NDCG@10': 0.03105, 'NDCG@20': 0.03606},
               'slsrec': {'Recall@10': 0.05601, 'Recall@20': 0.07727, 'NDCG@10': 0.03252, 'NDCG@20': 0.03790}},
    'video':  {'lru':    {'Recall@10': 0.11271, 'Recall@20': 0.16638, 'NDCG@10': 0.06061, 'NDCG@20': 0.07409},
               'recblr': {'Recall@10': 0.11725, 'Recall@20': 0.17202, 'NDCG@10': 0.06367, 'NDCG@20': 0.07748},
               'slsrec': {'Recall@10': 0.11748, 'Recall@20': 0.17353, 'NDCG@10': 0.06370, 'NDCG@20': 0.07783}},
    'sports': {'lru':    {'Recall@10': 0.03142, 'Recall@20': 0.04484, 'NDCG@10': 0.01810, 'NDCG@20': 0.02147},
               'recblr': {'Recall@10': 0.03067, 'Recall@20': 0.04368, 'NDCG@10': 0.01762, 'NDCG@20': 0.02088},
               'slsrec': {'Recall@10': 0.03356, 'Recall@20': 0.04768, 'NDCG@10': 0.01918, 'NDCG@20': 0.02273}},
    'steam':  {'lru':    {'Recall@10': 0.12743, 'Recall@20': 0.18905, 'NDCG@10': 0.06850, 'NDCG@20': 0.08401},
               'recblr': {'Recall@10': 0.13040, 'Recall@20': 0.19305, 'NDCG@10': 0.07046, 'NDCG@20': 0.08623},
               'slsrec': {'Recall@10': 0.14089, 'Recall@20': 0.20697, 'NDCG@10': 0.07640, 'NDCG@20': 0.09245}},
}

# LRURec, WSDM 2024, its own table.  Close to but not identical with the row the
# SLSRec paper reports for it -- worth seeing both before calling a gap a failure.
LRUREC_PAPER = {
    'ml-1m':  {'Recall@10': 0.32636, 'Recall@20': 0.43313, 'NDCG@10': 0.19298, 'NDCG@20': 0.21990},
    'beauty': {'Recall@10': 0.05284, 'Recall@20': 0.07161, 'NDCG@10': 0.03088, 'NDCG@20': 0.03560},
    'video':  {'Recall@10': 0.11337, 'Recall@20': 0.16705, 'NDCG@10': 0.06198, 'NDCG@20': 0.07547},
    'sports': {'Recall@10': 0.03148, 'Recall@20': 0.04477, 'NDCG@10': 0.01815, 'NDCG@20': 0.02149},
    'steam':  {'Recall@10': 0.12831, 'Recall@20': 0.19082, 'NDCG@10': 0.06934, 'NDCG@20': 0.08508},
}

# Never worth a column: fixed by the protocol, or an artefact of the folder layout.
NEVER_VARY = {'exp_name', 'export_root', 'min_uc', 'min_sc', 'min_rating', 'split',
              'metric_ks', 'best_metric'}

# Always shown, even when constant across the table: without these a single-run
# table would report a number with no configuration attached to it.
CORE = ['seed', 'bert_dropout', 'bert_attn_dropout', 'weight_decay', 'lr',
        'num_epochs', 'val_strategy', 'val_iterations']
CORE_SLSREC = ['perturbation_probability', 'uncertainty_alpha',
               'use_sam_gate', 'use_perturbation', 'use_uncertainty']


def infer_from_folder(model_code, exp_name):
    """Upstream LRURec names its folder <dataset>_<wd>_<dropout>_<attn_dropout>."""
    parts = exp_name.rsplit('_', 3)
    if len(parts) != 4:
        return None
    dataset, wd, dropout, attn = parts
    try:
        return {'dataset_code': dataset, 'model_code': model_code,
                'weight_decay': float(wd), 'bert_dropout': float(dropout),
                'bert_attn_dropout': float(attn)}
    except ValueError:
        return None


def load_runs(roots):
    runs = []
    for root, label in roots:
        experiments = os.path.join(root, 'experiments')
        if not os.path.isdir(experiments):
            continue
        for model_code in sorted(os.listdir(experiments)):
            model_dir = os.path.join(experiments, model_code)
            if not os.path.isdir(model_dir):
                continue
            for exp_name in sorted(os.listdir(model_dir)):
                run_dir = os.path.join(model_dir, exp_name)
                metrics_path = os.path.join(run_dir, 'test_metrics.json')
                if not os.path.isfile(metrics_path):
                    continue
                with open(metrics_path) as f:
                    metrics = json.load(f)
                config, summary = {}, {}
                config_path = os.path.join(run_dir, 'run_config.json')
                summary_path = os.path.join(run_dir, 'run_summary.json')
                if os.path.isfile(config_path):
                    with open(config_path) as f:
                        config = json.load(f)
                if os.path.isfile(summary_path):
                    with open(summary_path) as f:
                        summary = json.load(f)
                inferred = False
                if config:
                    hparams = config.get('hparams', {})
                else:
                    hparams = infer_from_folder(model_code, exp_name) or {}
                    inferred = bool(hparams)
                runs.append({
                    'run': (label + ':' if label else '') + exp_name,
                    'model': hparams.get('model_code', model_code),
                    'dataset': hparams.get('dataset_code', '?'),
                    'hparams': hparams,
                    'metrics': metrics,
                    'summary': summary,
                    'config': config,
                    'inferred': inferred,
                })
    return runs


def varying_keys(runs):
    """Hyperparameters that are not the same across every run in the table."""
    keys = set()
    for r in runs:
        keys.update(r['hparams'])
    varying = []
    for k in sorted(keys - NEVER_VARY):
        # Only over the runs that actually carry the key: a legacy run whose
        # hyperparameters were recovered from its folder name is missing most of
        # them, and counting 'missing' as a distinct value would make every
        # single key look like it varies.
        values = {json.dumps(r['hparams'][k], default=str) for r in runs if k in r['hparams']}
        if len(values) > 1:
            varying.append(k)
    return varying


def columns_for(runs, model, mode):
    """Which hyperparameters get a column: the headline ones plus whatever varies."""
    present = set()
    for r in runs:
        present.update(r['hparams'])
    varying = varying_keys(runs)
    if mode == 'varying':
        return varying
    if mode == 'all':
        return sorted(present - NEVER_VARY)
    core = [k for k in CORE + (CORE_SLSREC if model == 'slsrec' else []) if k in present]
    return core + [k for k in varying if k not in core]


def reference(dataset, model, which):
    if which == 'lrurec' and model == 'lru':
        return LRUREC_PAPER.get(dataset), 'LRURec paper'
    return SLSREC_PAPER.get(dataset, {}).get(model), 'SLSRec paper Table II'


def render(rows, header):
    widths = [max([len(h)] + [len(str(r[i])) for r in rows]) for i, h in enumerate(header)]
    out = ['| ' + ' | '.join(h.ljust(w) for h, w in zip(header, widths)) + ' |',
           '|' + '|'.join('-' * (w + 2) for w in widths) + '|']
    for r in rows:
        out.append('| ' + ' | '.join(str(c).ljust(w) for c, w in zip(r, widths)) + ' |')
    return '\n'.join(out)


def fmt(v):
    if v is None:
        return '-'
    if isinstance(v, float):
        return '%.6g' % v
    return str(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments', default=os.path.join(ROOT, 'experiments'))
    parser.add_argument('--include-ref', action='store_true',
                        help='also scan _ref_LRURec/experiments (unmodified upstream runs)')
    parser.add_argument('--model', default=None, help='lru | slsrec')
    parser.add_argument('--dataset', default=None)
    parser.add_argument('--sort', default='Recall@10', help='metric to rank by')
    parser.add_argument('--reference', default='slsrec', choices=['slsrec', 'lrurec'],
                        help='which published table to compare against')
    parser.add_argument('--metrics', nargs='+', default=METRICS)
    parser.add_argument('--columns', default='core', choices=['core', 'varying', 'all'],
                        help='core: headline hyperparameters + whatever differs (default); '
                             'varying: only what differs; all: every key in the namespace')
    parser.add_argument('--csv', default=None,
                        help='write every run with its COMPLETE hyperparameter namespace')
    args = parser.parse_args()

    roots = [(os.path.dirname(os.path.abspath(args.experiments)), '')]
    if args.include_ref:
        roots.append((os.path.join(ROOT, '_ref_LRURec'), 'upstream'))

    runs = load_runs(roots)
    if args.model:
        runs = [r for r in runs if r['model'] == args.model]
    if args.dataset:
        runs = [r for r in runs if r['dataset'] == args.dataset]
    if not runs:
        print('no finished runs found (a run needs test_metrics.json)', file=sys.stderr)
        return 1

    groups = {}
    for r in runs:
        groups.setdefault((r['model'], r['dataset']), []).append(r)

    for (model, dataset), group in sorted(groups.items()):
        group.sort(key=lambda r: r['metrics'].get(args.sort, float('-inf')), reverse=True)
        varying = columns_for(group, model, args.columns)
        ref, ref_label = reference(dataset, model, args.reference)

        header = ['run'] + varying + args.metrics
        if ref:
            header += ['vs paper']
        header += ['epochs', 'vals', 'stop', 'min']
        rows = []
        for r in group:
            summary = r['summary']
            row = [r['run'] + (' *' if r['inferred'] else '')]
            row += [fmt(r['hparams'].get(k)) for k in varying]
            row += ['%.5f' % r['metrics'][m] if m in r['metrics'] else '-' for m in args.metrics]
            if ref:
                got = r['metrics'].get(args.sort)
                row.append('%+.1f%%' % (100.0 * (got - ref[args.sort]) / ref[args.sort])
                           if got is not None else '-')
            row += [
                fmt(summary.get('epochs_completed')),
                fmt(summary.get('validations')),
                'early' if summary.get('early_stopped') else fmt(summary.get('early_stopped')),
                '%.0f' % (summary['elapsed_seconds'] / 60.0) if summary.get('elapsed_seconds') else '-',
            ]
            rows.append(row)

        print('\n### %s on %s -- %d run(s), ranked by %s\n' % (model, dataset, len(group), args.sort))
        print(render(rows, header))
        if len(group) > 1 and not varying_keys(group):
            print('\n(no hyperparameter differs between these runs)')
        if ref:
            print('\npublished (%s): ' % ref_label
                  + '  '.join('%s %.5f' % (m, ref[m]) for m in args.metrics if m in ref))
            best = group[0]
            got = best['metrics'].get(args.sort)
            if got is not None:
                gap = 100.0 * (got - ref[args.sort]) / ref[args.sort]
                verdict = ('reproduced' if abs(gap) <= 3 else
                           'close' if abs(gap) <= 5 else 'NOT reproduced')
                print('best run %s: %s %.5f vs %.5f -> %+.1f%%  [%s]'
                      % (best['run'], args.sort, got, ref[args.sort], gap, verdict))
        if any(r['inferred'] for r in group):
            print('\n* hyperparameters read from the folder name: the run predates run_config.json')

    if args.csv:
        keys = set()
        best_metric = 'Recall@10'
        for r in runs:
            keys.update(r['hparams'])
            best_metric = r['summary'].get('best_metric') or best_metric
        best_val_field = 'best_val_' + best_metric
        fields = (['run', 'model', 'dataset'] + sorted(keys) + args.metrics
                  + ['epochs_completed', 'validations', 'early_stopped', 'elapsed_seconds',
                     best_val_field, 'git_commit', 'gpu', 'torch', 'started_at'])
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            for r in runs:
                summary, env = r['summary'], r['config'].get('env', {})
                row = {'run': r['run'], 'model': r['model'], 'dataset': r['dataset']}
                row.update({k: r['hparams'].get(k) for k in keys})
                row.update({m: r['metrics'].get(m) for m in args.metrics})
                row.update({
                    'epochs_completed': summary.get('epochs_completed'),
                    'validations': summary.get('validations'),
                    'early_stopped': summary.get('early_stopped'),
                    'elapsed_seconds': summary.get('elapsed_seconds'),
                    best_val_field: (summary.get('best_val_metrics') or {}).get(
                        summary.get('best_metric')),
                    'git_commit': env.get('git_commit'),
                    'gpu': env.get('gpu'),
                    'torch': env.get('torch'),
                    'started_at': r['config'].get('started_at'),
                })
                writer.writerow(row)
        print('\nwrote %s (%d run(s), %d hyperparameter column(s))'
              % (args.csv, len(runs), len(keys)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
