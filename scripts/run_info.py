"""What did a finished run actually do?  Read straight from its artifacts.

The same command line can resolve to different configurations when defaults
change between commits, so this reports what each run *recorded*: its
schedule (from run_config.json, and independently from the spacing of its
validations), where it peaked and stopped, and its test metrics.

It also detects the append-mode trap: train_losses.jsonl and val_metrics.jsonl
are opened with 'a', so two runs sharing one --exp_name end up interleaved in
the same files.  Each run found in the logs is reported separately.

    python scripts/run_info.py experiments/slsrec/D_paperfig_ml-1m_seed42
    python scripts/run_info.py experiments/slsrec/D_*
"""

import glob
import json
import os
import statistics
import sys

BATCHES_PER_EPOCH = {'ml-1m': 56}


def split_runs(rows):
    """Split appended records where accum_iter falls back (a new run started)."""
    runs, cur, last = [], [], -1
    for r in rows:
        it = r.get('accum_iter', 0)
        if cur and it < last:
            runs.append(cur)
            cur = []
        cur.append(r)
        last = it
    if cur:
        runs.append(cur)
    return runs


def cadence(rows, per_epoch):
    its = [r['accum_iter'] for r in rows]
    gaps = [b - a for a, b in zip(its, its[1:]) if b > a]
    if not gaps:
        return 'only the initial validation'
    g = statistics.median(gaps)
    txt = f'every {g:.0f} iters'
    if per_epoch:
        ep = g / per_epoch
        txt += f' (= {ep:.2f} epochs)'
        if abs(ep - 5) < 1e-6:
            txt += '  <- paper schedule'
        elif abs(g - 500) < 1e-6:
            txt += '  <- LRURec schedule'
    return txt


def main(paths):
    for path in paths:
        if not os.path.isdir(path):
            continue
        name = os.path.basename(os.path.normpath(path))
        print(f'\n=== {name} ===')

        cfg_path = os.path.join(path, 'run_config.json')
        dataset = 'ml-1m'
        if os.path.isfile(cfg_path):
            c = json.load(open(cfg_path))
            h = c.get('hparams', {})
            dataset = h.get('dataset_code', dataset)
            print(f"  run_config  started {c.get('started_at')} | git {c.get('env', {}).get('git_commit')} "
                  f"| torch {c.get('env', {}).get('torch')} | device {c.get('env', {}).get('device')}")
            print(f"              num_epochs={h.get('num_epochs')} val_strategy={h.get('val_strategy')} "
                  f"val_iterations={h.get('val_iterations')} val_epochs={h.get('val_epochs')} "
                  f"patience={h.get('early_stopping_patience')}")
            print(f"              uncertainty_loss={h.get('uncertainty_loss')} "
                  f"unc_dropout_form={h.get('unc_dropout_form', '(not recorded: predates the flag)')} "
                  f"num_workers={h.get('num_workers')} seed={h.get('seed')}")
        else:
            print('  run_config  (none -- run predates run_config.json)')
        per_epoch = BATCHES_PER_EPOCH.get(dataset)

        val_path = os.path.join(path, 'val_metrics.jsonl')
        if os.path.isfile(val_path):
            rows = [json.loads(l) for l in open(val_path) if l.strip()]
            runs = split_runs(rows)
            if len(runs) > 1:
                print(f'  !!! val_metrics.jsonl holds {len(runs)} runs appended into one folder -- '
                      f'run_config.json and the checkpoints belong to the LAST one only')
            for i, r in enumerate(runs, 1):
                tag = f'run {i}/{len(runs)}' if len(runs) > 1 else 'validation'
                scored = [x for x in r if x.get('accum_iter', 0) > 0 and 'Recall@10' in x]
                line = f'  {tag:<11} {len(r)} validations, cadence {cadence(r, per_epoch)}'
                if scored:
                    best = max(scored, key=lambda x: x['Recall@10'])
                    line += (f"\n              best val R@10 {best['Recall@10']:.5f} at epoch {best['epoch']}"
                             f" | last validation at epoch {r[-1]['epoch']}")
                print(line)

        test_path = os.path.join(path, 'test_metrics.json')
        if os.path.isfile(test_path):
            t = json.load(open(test_path))
            print('  test        ' + '  '.join(f'{k} {t[k]:.5f}' for k in
                                             ('Recall@10', 'Recall@20', 'NDCG@10', 'NDCG@20') if k in t))
        else:
            print('  test        (no test_metrics.json -- unfinished)')


if __name__ == '__main__':
    args = sys.argv[1:] or sorted(glob.glob('experiments/slsrec/*'))
    expanded = []
    for a in args:
        expanded.extend(sorted(glob.glob(a)) or [a])
    main(expanded)
