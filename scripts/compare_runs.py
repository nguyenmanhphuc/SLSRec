"""Side-by-side comparison of finished runs: config, test metrics, validation plateau.

    python scripts/compare_runs.py noP_beauty_seed42 noP_d06_beauty_seed42 noP_d075_sg02_beauty_seed42 \
        --base noP_beauty_seed42 --ref 0.07345 0.03589

Run names are looked up under experiments/slsrec/ and experiments/lru/.
--base: run the others are compared against (relative %, test and plateau).
--ref R20 N20: a paper reference for Recall@20 / NDCG@20 (relative %).
Plateau = mean validation metric over epochs >= --plateau_from (default 50),
less sensitive than the single best checkpoint.  A run folder that holds
several appended runs is flagged; only its first run is used for the plateau.
"""

import argparse
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_KEYS = ['Recall@1', 'Recall@10', 'NDCG@10', 'Recall@20', 'NDCG@20', 'Recall@50']


def find(name):
    for sub in ('slsrec', 'lru'):
        d = os.path.join(ROOT, 'experiments', sub, name)
        if os.path.isdir(d):
            return d
    # fall back to any copy nested deeper (e.g. experiments/slsrec/slsrec/<name>); newest wins
    hits = []
    for base, dirs, _ in os.walk(os.path.join(ROOT, 'experiments')):
        if base.count(os.sep) - ROOT.count(os.sep) > 4:
            dirs[:] = []
            continue
        if name in dirs and os.path.exists(os.path.join(base, name, 'run_config.json')):
            hits.append(os.path.join(base, name))
    return max(hits, key=os.path.getmtime) if hits else None


def load(name, plateau_from):
    d = find(name)
    if d is None:
        return {'name': name, 'missing': True}
    hp = json.load(open(os.path.join(d, 'run_config.json')))['hparams']
    tp = os.path.join(d, 'test_metrics.json')
    test = json.load(open(tp)) if os.path.exists(tp) else None
    rows, appended = [], False
    vp = os.path.join(d, 'val_metrics.jsonl')
    if os.path.exists(vp):
        seen = set()
        for line in open(vp):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get('accum_iter', 1) <= 0:
                continue
            if r['epoch'] in seen:          # a second run appended into the same folder
                appended = True
                break
            seen.add(r['epoch'])
            rows.append(r)
    plat = [r for r in rows if r['epoch'] >= plateau_from]
    best = max(rows, key=lambda r: r['Recall@10']) if rows else None
    return {
        'name': name, 'missing': False, 'test': test, 'appended': appended,
        'cfg': f"P={hp.get('use_perturbation')}/{hp.get('perturbation_probability')} "
               f"U={hp.get('use_uncertainty')}/{hp.get('uncertainty_loss')} "
               f"drop={hp.get('bert_dropout')} sglru={hp.get('sglru_dropout')} seed={hp.get('seed')}",
        'plat_r10': statistics.mean(r['Recall@10'] for r in plat) if plat else None,
        'plat_n10': statistics.mean(r['NDCG@10'] for r in plat) if plat else None,
        'best_ep': best['epoch'] if best else None,
        'last_ep': rows[-1]['epoch'] if rows else None,
    }


def pct(a, b):
    return '' if a is None or b in (None, 0) else f'{100 * (a / b - 1):+.1f}%'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--base', default=None)
    ap.add_argument('--ref', nargs=2, type=float, metavar=('R20', 'N20'), default=None)
    ap.add_argument('--plateau_from', type=int, default=50)
    a = ap.parse_args()

    runs = [load(n, a.plateau_from) for n in a.runs]
    base = next((r for r in runs if r['name'] == a.base and not r['missing']), None)

    for r in runs:
        if r['missing']:
            print(f"{r['name']}: (not found)")
            continue
        flag = '  [!] several runs appended -- plateau uses the first' if r['appended'] else ''
        print(f"{r['name']}\n  {r['cfg']}  best ep {r['best_ep']}  last ep {r['last_ep']}{flag}")
    print()

    w = max(len(r['name']) for r in runs) + 2
    print(f"{'run':<{w}}" + ''.join(f'{k:>11}' for k in TEST_KEYS) + f"{'CE':>8}{'plat R10':>10}{'plat N10':>10}")
    for r in runs:
        if r['missing']:
            continue
        t = r['test']
        cells = ''.join(f'{t[k]:>11.5f}' for k in TEST_KEYS) + f"{t.get('CE', float('nan')):>8.2f}" if t \
            else f"{'(no test yet)':>{11 * len(TEST_KEYS) + 8}}"
        pl = f"{r['plat_r10']:>10.5f}{r['plat_n10']:>10.5f}" if r['plat_r10'] is not None else ''
        print(f"{r['name']:<{w}}{cells}{pl}")

    if base or a.ref:
        print()
        hdr = f"{'run':<{w}}"
        if base:
            hdr += f"{'vs base: R@20':>15}{'N@20':>8}{'R@1':>8}{'plat R10':>10}{'plat N10':>10}"
        if a.ref:
            hdr += f"{'vs paper: R@20':>16}{'N@20':>8}"
        print(hdr)
        for r in runs:
            if r['missing'] or r['test'] is None:
                continue
            t, line = r['test'], f"{r['name']:<{w}}"
            if base:
                bt = base['test'] or {}
                line += (f"{pct(t['Recall@20'], bt.get('Recall@20')):>15}{pct(t['NDCG@20'], bt.get('NDCG@20')):>8}"
                         f"{pct(t['Recall@1'], bt.get('Recall@1')):>8}"
                         f"{pct(r['plat_r10'], base['plat_r10']):>10}{pct(r['plat_n10'], base['plat_n10']):>10}")
            if a.ref:
                line += f"{pct(t['Recall@20'], a.ref[0]):>16}{pct(t['NDCG@20'], a.ref[1]):>8}"
            print(line)


if __name__ == '__main__':
    main()
