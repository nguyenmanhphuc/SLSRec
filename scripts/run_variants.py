"""Train every model variant for the given seed(s), then print one comparison table.

One command per seed gives RecBLR, the SLSRec ablations of Table III and full
SLSRec side by side, each next to the number the SLSRec paper reports for it:

    python scripts/run_variants.py                          # default set, seed 42, ml-1m
    python scripts/run_variants.py --variants recblr        # RecBLR only
    python scripts/run_variants.py --seeds 42 43 44
    python scripts/run_variants.py --quick                  # 10-epoch trial of the whole pipeline
    python scripts/run_variants.py --dry_run                # print the commands only
    python scripts/run_variants.py --list                   # every known variant
    python scripts/run_variants.py --collect_only --include_legacy   # table from runs already on disk
    python scripts/run_variants.py --variants wo_pu p_layer --reuse_legacy  # train only what no old run covers
    python scripts/run_variants.py --variants recblr slsrec -- --num_workers=2   # flags after -- go to train.py
    python scripts/run_variants.py --variants @p            # alternative readings of P (eq. 17)
    python scripts/run_variants.py --variants @u            # alternative readings of U (eq. 19-24)

@p trains each reading of the perturbation with U off, next to w/o P&U (no P) and
w/o U (the literal P); @u each reading of the uncertainty with P off, next to
w/o P&U and w/o P (the literal U).  Relative columns are then against wo_pu.

Runs land in experiments/<model_code>/<prefix>_<variant>_<dataset>_seed<seed>,
logs in runlogs/<prefix>_<dataset>/.  A run whose folder already has
test_metrics.json is skipped, so an interrupted sweep restarts where it stopped;
an incomplete folder is renamed aside (never deleted) before retraining, because
the trainer appends to its jsonl logs.

A run is attributed to a variant by its run_config.json, not by its folder name:
the variant's flags are resolved through config.set_template exactly as
train.py does, and every hyperparameter that defines the variant must match.
--include_legacy uses the same check to pick up older runs (any folder name);
a key an old run predates counts as that flag's default (what the old code did),
so the run can only match a variant that asks for the default; it is marked '*'.

Tables are also written to experiments/compare/<prefix>_<dataset>.{md,csv,json}.
"""

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from collect_hparams import SLSREC_PAPER  # noqa: E402  (Table II)

METRICS = ['Recall@10', 'Recall@20', 'NDCG@10', 'NDCG@20']

OFF_PU = ['--use_perturbation=false', '--use_uncertainty=false']
OFF_P = ['--use_perturbation=false']
OFF_U = ['--use_uncertainty=false']

# name -> (model_code, flags, paper reference key, description)
VARIANTS = OrderedDict([
    ('lru',          ('lru', [], 'lru',
                      'LRURec baseline, LRURec protocol (500 ep, val every 500 it)')),
    ('recblr',       ('recblr', [], 'recblr',
                      'official RecBLR (model/recblr.py), paper protocol')),
    ('bdlru',        ('slsrec', ['--use_sam_gate=false', '--sam_conv=false'] + OFF_PU, None,
                      'SLSRec block with BD-LRU only, gate SiLU(L1) -- RecBLR-like, not RecBLR')),
    ('wo_sam',       ('slsrec', ['--use_sam_gate=false'], 'w/o SAM',
                      'G = Conv1-D1(L1)')),
    ('wo_bdlru',     ('slsrec', ['--use_bdlru=false'], 'w/o BD-LRU',
                      'H = Conv1-D2(L2); P lives in the BD-LRU, so L_per only sees new dropout')),
    ('wo_p',         ('slsrec', ['--use_perturbation=false'], 'w/o P', '')),
    ('wo_u',         ('slsrec', ['--use_uncertainty=false'], 'w/o U', '')),
    ('wo_pu',        ('slsrec', OFF_PU, 'w/o P&U', 'SAM + BD-LRU, plain CE')),
    ('slsrec',       ('slsrec', [], 'slsrec', 'full SLSRec, eq. (24) approx L_unc')),
    ('slsrec_exact', ('slsrec', ['--uncertainty_loss=exact'], None,
                      'full SLSRec, exact eq. (19)/(21) L_unc')),

    # --- perturbation readings, U off: compare with wo_pu (no P) and wo_u (literal P) ---
    ('p_bdlru_conv', ('slsrec', OFF_U + ['--perturbation_site=bdlru_conv'], None,
                      'P on L2, before Conv1-D2')),
    ('p_layer',      ('slsrec', OFF_U + ['--perturbation_site=layer'], None,
                      'P on each layer input X^(l-1): gate branch, BD-LRU and residual')),
    ('p_embedding',  ('slsrec', OFF_U + ['--perturbation_site=embedding'], None,
                      'P on X^(0) once, before the first layer')),
    ('p_shuffle',    ('slsrec', OFF_U + ['--perturbation_noise=shuffle'], None,
                      'P at the BD-LRU input, vector borrowed from another sequence')),
    ('p_layer_shuffle', ('slsrec', OFF_U + ['--perturbation_site=layer', '--perturbation_noise=shuffle'],
                         None, 'P on each layer input, vector borrowed from another sequence')),
    ('p_element',    ('slsrec', OFF_U + ['--perturbation_granularity=element'], None,
                      'P at the BD-LRU input, each component replaced independently')),
    ('p_clean_gates', ('slsrec', OFF_U + ['--perturbation_gates=clean'], None,
                       'P only in the input term of eq. (12); r_t, i_t from the clean x_t')),

    # --- uncertainty readings, P off: compare with wo_pu (no U) and wo_p (literal U) ---
    ('u_exact',      ('slsrec', OFF_P + ['--uncertainty_loss=exact'], None,
                      'exact eq. (19)/(21): sigma^2 as a softmax temperature')),
    ('u_beta',       ('slsrec', OFF_P + ['--uncertainty_beta=1'], None,
                      'eq. (22) with the beta Log(sigma) penalty kept (beta=1, Kendall et al.)')),
    ('u_sequence',   ('slsrec', OFF_P + ['--uncertainty_granularity=sequence'], None,
                      'one sigma_u per sequence from its last row, as Sec. III-C2 words it')),
    ('u_sequence_beta', ('slsrec', OFF_P + ['--uncertainty_granularity=sequence', '--uncertainty_beta=1'],
                         None, 'per-sequence sigma_u with the beta=1 penalty')),
    ('u_global_beta', ('slsrec', OFF_P + ['--uncertainty_granularity=global', '--uncertainty_beta=1'],
                       None, 'one global sigma ([45]) with beta=1 -- capacity control')),
    ('u_norm',       ('slsrec', OFF_P + ['--uncertainty_weight_norm=batch'], None,
                      '1/sigma^2 normalised to batch mean 1: U cannot switch off (control)')),
])
DEFAULT_VARIANTS = ['recblr', 'wo_sam', 'wo_bdlru', 'wo_p', 'wo_u', 'wo_pu', 'slsrec']
GROUPS = OrderedDict([
    ('table3', DEFAULT_VARIANTS),
    ('p', ['wo_pu', 'wo_u', 'p_clean_gates', 'p_bdlru_conv', 'p_layer', 'p_embedding', 'p_shuffle',
           'p_layer_shuffle', 'p_element']),
    ('u', ['wo_pu', 'wo_p', 'u_exact', 'u_beta', 'u_sequence', 'u_sequence_beta',
           'u_global_beta', 'u_norm']),
    ('all', list(VARIANTS)),
])
QUICK_FLAGS = ['--num_epochs=10', '--val_strategy=epoch', '--val_epochs=5']

# SLSRec Table III -- Recall@20 / NDCG@20 of each ablation.
TABLE3 = {
    'ml-1m':  {'w/o SAM': (0.46123, 0.23863), 'w/o BD-LRU': (0.45815, 0.23601),
               'w/o P': (0.45803, 0.23503), 'w/o U': (0.45987, 0.23698), 'w/o P&U': (0.44852, 0.22980)},
    'beauty': {'w/o SAM': (0.07248, 0.03601), 'w/o BD-LRU': (0.07084, 0.03518),
               'w/o P': (0.07362, 0.03627), 'w/o U': (0.07584, 0.03682), 'w/o P&U': (0.07345, 0.03589)},
    'video':  {'w/o SAM': (0.17197, 0.07663), 'w/o BD-LRU': (0.17139, 0.07887),
               'w/o P': (0.17261, 0.07746), 'w/o U': (0.17306, 0.07662), 'w/o P&U': (0.17217, 0.07551)},
    'sports': {'w/o SAM': (0.04731, 0.02214), 'w/o BD-LRU': (0.04564, 0.02144),
               'w/o P': (0.04694, 0.02185), 'w/o U': (0.04309, 0.01994), 'w/o P&U': (0.04436, 0.02085)},
    'steam':  {'w/o SAM': (0.20427, 0.09134), 'w/o BD-LRU': (0.19924, 0.08800),
               'w/o P': (0.20351, 0.09096), 'w/o U': (0.20595, 0.09209), 'w/o P&U': (0.18744, 0.08142)},
}

# Hyperparameters that define a variant.  A run belongs to a variant only if all
# of the relevant ones match what train.py would resolve for that variant.
COMMON_KEYS = ['model_code', 'dataset_code', 'bert_hidden_units', 'bert_num_blocks', 'bert_max_len',
               'bert_dropout', 'optimizer', 'lr', 'weight_decay', 'train_batch_size',
               'max_grad_norm', 'num_epochs', 'val_strategy', 'early_stopping_patience', 'best_metric']
MODEL_KEYS = {
    'lru': ['bert_attn_dropout'],
    'recblr': ['slsrec_expand', 'slsrec_d_conv', 'exclude_lambda_from_decay'],
    'slsrec': ['slsrec_expand', 'slsrec_d_conv', 'exclude_lambda_from_decay', 'use_item_bias',
               'use_sam_gate', 'sam_conv', 'sam_conv_activation', 'sam_num_heads', 'sam_attn_dropout',
               'use_bdlru', 'bdlru_c', 'bdlru_r_min', 'bdlru_r_max', 'bdlru_lambda_init',
               'bdlru_conv_activation', 'use_perturbation', 'use_uncertainty',
               'gate_mode', 'fusion_mode'],
}
PERTURB_KEYS = ['perturbation_probability', 'perturbation_low', 'perturbation_high', 'perturbation_weight',
                'perturbation_site', 'perturbation_noise', 'perturbation_granularity',
                'perturbation_gates']
UNC_KEYS = ['uncertainty_alpha', 'uncertainty_beta', 'uncertainty_loss', 'unc_dropout_form',
            'uncertainty_logvar_clamp', 'uncertainty_granularity', 'uncertainty_weight_norm']


# --------------------------------------------------------------------------- #
# Resolving what a variant means
# --------------------------------------------------------------------------- #
def _config_module():
    """Import config.py without letting it parse this script's own argv."""
    saved = sys.argv
    try:
        sys.argv = [saved[0]]
        import config
    finally:
        sys.argv = saved
    return config


def train_argv(variant, dataset, seed, prefix, extra):
    model_code, flags, _, _ = VARIANTS[variant]
    return (['--dataset_code=' + dataset, '--model_code=' + model_code, '--seed=%d' % seed]
            + flags + list(extra) + ['--exp_name=' + exp_name(prefix, variant, dataset, seed)])


def exp_name(prefix, variant, dataset, seed):
    return '%s_%s_%s_seed%d' % (prefix, variant, dataset, seed)


def resolve(argv):
    """The namespace train.py would train with for this argv."""
    cfg = _config_module()
    args = cfg.parser.parse_args(argv)
    cfg.set_template(args)
    return vars(args)


def identity(hp):
    model = hp.get('model_code')
    keys = COMMON_KEYS + MODEL_KEYS.get(model, [])
    if hp.get('val_strategy') == 'iteration':
        keys = keys + ['val_iterations']
    else:
        keys = keys + ['val_epochs']
    if model == 'slsrec':
        if hp.get('use_perturbation'):
            keys = keys + PERTURB_KEYS
        if hp.get('use_uncertainty'):
            keys = keys + UNC_KEYS
    return keys


def _same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) < 1e-12
        except (TypeError, ValueError):
            return False
    return a == b


def match(expected, actual):
    """(matches, keys the run predates).  A key absent from an old run stands for that
    flag's default -- what the code did before the flag existed."""
    parser = _config_module().parser
    missing = []
    for k in identity(expected):
        if k not in actual or (actual[k] is None and expected.get(k) is not None):
            if not _same(expected.get(k), parser.get_default(k)):
                return False, []
            missing.append(k)
            continue
        if not _same(expected.get(k), actual[k]):
            return False, []
    return True, missing


# --------------------------------------------------------------------------- #
# Reading runs back
# --------------------------------------------------------------------------- #
def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _last_jsonl(path):
    last = None
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    last = json.loads(line)
    return last


def read_run(folder):
    """Metrics + training diagnostics of one finished run, or None."""
    tm = os.path.join(folder, 'test_metrics.json')
    rc = os.path.join(folder, 'run_config.json')
    if not (os.path.isfile(tm) and os.path.isfile(rc)):
        return None
    config = _load_json(rc)
    hp = config.get('hparams', {})
    summary_path = os.path.join(folder, 'run_summary.json')
    summary = _load_json(summary_path) if os.path.isfile(summary_path) else {}
    best_val = summary.get('best_val_metrics') or {}
    last = _last_jsonl(os.path.join(folder, 'train_losses.jsonl')) or {}

    l_rec = last.get('L_rec', last.get('loss'))
    diag = OrderedDict()
    diag['best_epoch'] = (summary.get('best_val_at') or {}).get('epoch')
    diag['epochs'] = summary.get('epochs_completed')
    diag['minutes'] = summary['elapsed_seconds'] / 60.0 if summary.get('elapsed_seconds') else None
    diag['val_R@20'] = best_val.get('Recall@20')
    diag['val_CE'] = best_val.get('CE')
    diag['train_L_rec'] = l_rec
    # how much the perturbed pass costs over the clean one -- ~0 means P is inert
    diag['L_per-L_rec'] = (last['L_perturb'] - l_rec
                           if hp.get('model_code') == 'slsrec' and hp.get('use_perturbation')
                           and l_rec is not None and 'L_perturb' in last else None)
    # alpha * L_unc / L_rec: with the approx loss this is the extra weight eq. (24)
    # puts on L_rec; ~0 means sigma^2 ran away to the clamp and U switched off
    diag['a*L_unc/L_rec'] = (hp.get('uncertainty_alpha') * last['L_uncertainty'] / l_rec
                             if hp.get('model_code') == 'slsrec' and hp.get('use_uncertainty')
                             and l_rec and 'L_uncertainty' in last else None)
    return {
        'folder': folder,
        'hparams': hp,
        'metrics': _load_json(tm),
        'diag': diag,
        'commit': config.get('env', {}).get('git_commit'),
        'gpu': config.get('env', {}).get('gpu'),
        'started_at': config.get('started_at') or '',
    }


def all_run_folders():
    root = os.path.join(ROOT, 'experiments')
    for model in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        mdir = os.path.join(root, model)
        if not os.path.isdir(mdir) or model == 'compare':
            continue
        for name in sorted(os.listdir(mdir)):
            folder = os.path.join(mdir, name)
            if os.path.isfile(os.path.join(folder, 'run_config.json')):
                yield folder


def collect(variants, dataset, seeds, prefix, extra, include_legacy):
    """variant -> list of runs (one per seed; legacy runs may add more seeds)."""
    found = defaultdict(list)
    expected = {v: resolve(train_argv(v, dataset, seeds[0], prefix, extra)) for v in variants}
    own = {}
    for v in variants:
        model_code = VARIANTS[v][0]
        for s in seeds:
            own[os.path.normcase(os.path.join(ROOT, 'experiments', model_code,
                                              exp_name(prefix, v, dataset, s)))] = (v, s)

    for folder in all_run_folders():
        key = os.path.normcase(folder)
        if key not in own and not include_legacy:
            continue
        run = read_run(folder)
        if run is None:
            continue
        hp = run['hparams']
        candidates = [own[key][0]] if key in own else variants
        for v in candidates:
            ok, missing = match(expected[v], hp)
            if ok:
                if key in own and own[key][1] != hp.get('seed'):
                    continue
                run['missing'] = missing
                run['legacy'] = key not in own
                found[v].append(run)
                break
            elif key in own:
                print('[warn] %s does not match variant %r -- left out of the table'
                      % (os.path.relpath(folder, ROOT), v))
    for v in found:
        # one run per (variant, seed): prefer this sweep's own run, then the run that
        # knows the most flags, then the newest
        found[v].sort(key=lambda r: r['started_at'], reverse=True)
        found[v].sort(key=lambda r: (r['hparams'].get('seed'), r['legacy'], len(r['missing'])))
        dedup = OrderedDict()
        for r in found[v]:
            dedup.setdefault(r['hparams'].get('seed'), r)
        found[v] = list(dedup.values())
    return found


# --------------------------------------------------------------------------- #
# Table
# --------------------------------------------------------------------------- #
def paper_value(variant, dataset, metric):
    ref = VARIANTS[variant][2]
    if ref is None:
        return None
    if ref in ('lru', 'recblr', 'slsrec'):
        return SLSREC_PAPER.get(dataset, {}).get(ref, {}).get(metric)
    t3 = TABLE3.get(dataset, {}).get(ref)
    if t3 is None:
        return None
    return {'Recall@20': t3[0], 'NDCG@20': t3[1]}.get(metric)


def _mean_std(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else None)


def _fmt(x, nd=4, pct=False, sign=False):
    if x is None:
        return '-'
    if pct:
        return ('%+.2f%%' if sign else '%.2f%%') % (100 * x)
    return ('%+.' + str(nd) + 'f' if sign else '%.' + str(nd) + 'f') % x


def build_rows(found, variants, dataset, reference):
    rows = []
    agg = {}
    for v in variants:
        runs = found.get(v, [])
        stats = {m: _mean_std([r['metrics'].get(m) for r in runs]) for m in METRICS}
        dstats = OrderedDict()
        for k in (runs[0]['diag'].keys() if runs else []):
            dstats[k] = _mean_std([r['diag'].get(k) for r in runs])[0]
        agg[v] = (runs, stats, dstats)
    ref_stats = agg.get(reference, (None, {}, None))[1]
    for v in variants:
        runs, stats, dstats = agg[v]
        row = OrderedDict()
        row['variant'] = v
        row['seeds'] = ','.join(str(r['hparams'].get('seed')) for r in runs) or '-'
        row['legacy'] = any(r['legacy'] for r in runs)
        row['predates'] = sorted({k for r in runs for k in r.get('missing', [])})
        for m in METRICS:
            mean, std = stats[m]
            row[m] = mean
            row[m + '_std'] = std
            pv = paper_value(v, dataset, m)
            row[m + '_paper'] = pv
            row[m + '_gap'] = (mean / pv - 1) if (mean is not None and pv) else None
            ref_mean = ref_stats.get(m, (None, None))[0] if ref_stats else None
            row[m + '_vs_ref'] = (mean / ref_mean - 1) if (mean is not None and ref_mean) else None
            ref_pv = paper_value(reference, dataset, m) if reference else None
            row[m + '_paper_vs_ref'] = (pv / ref_pv - 1) if (pv and ref_pv) else None
        row.update(dstats or {})
        row['description'] = VARIANTS[v][3]
        rows.append(row)
    return rows


def render(rows, dataset, reference):
    out = []
    out.append('### %s -- test metrics (mean over seeds; std in parentheses)\n' % dataset)
    # the relative columns read like Table III: "drop percentage compared to SLSRec"
    rel = ['R@20 vs %s' % reference, 'paper', 'N@20 vs %s' % reference, 'paper'] if reference else []
    header = ['variant', 'seeds', 'R@10', 'R@20', 'N@10', 'N@20', 'paper R@20', 'gap vs paper'] + rel
    out.append('| ' + ' | '.join(header) + ' |')
    out.append('|' + '---|' * len(header))
    for r in rows:
        def cell(m):
            if r[m] is None:
                return '-'
            s = _fmt(r[m])
            return s + (' (%s)' % _fmt(r[m + '_std']) if r[m + '_std'] is not None else '')
        cells = [r['variant'] + ('*' if r['predates'] else ''), r['seeds'],
                 cell('Recall@10'), cell('Recall@20'), cell('NDCG@10'), cell('NDCG@20'),
                 _fmt(r['Recall@20_paper']), _fmt(r['Recall@20_gap'], pct=True, sign=True)]
        if reference:
            cells += [_fmt(r[k], pct=True, sign=True) for k in
                      ('Recall@20_vs_ref', 'Recall@20_paper_vs_ref', 'NDCG@20_vs_ref', 'NDCG@20_paper_vs_ref')]
        out.append('| ' + ' | '.join(cells) + ' |')

    out.append('\n### %s -- training diagnostics (mean over seeds)\n' % dataset)
    out.append('| variant | best ep | epochs | minutes | val R@20 | val-test R@20 | val CE '
               '| train L_rec | L_per-L_rec | a*L_unc/L_rec | note |')
    out.append('|' + '---|' * 11)
    for r in rows:
        vt = (r.get('val_R@20') - r['Recall@20']) if (r.get('val_R@20') is not None
                                                      and r['Recall@20'] is not None) else None
        out.append('| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            r['variant'], _fmt(r.get('best_epoch'), 0), _fmt(r.get('epochs'), 0),
            _fmt(r.get('minutes'), 1), _fmt(r.get('val_R@20')), _fmt(vt, sign=True),
            _fmt(r.get('val_CE'), 3), _fmt(r.get('train_L_rec'), 3),
            _fmt(r.get('L_per-L_rec'), 4, sign=True), _fmt(r.get('a*L_unc/L_rec'), 5),
            r['description']))
    if any(r['predates'] for r in rows):
        out.append('\n`*` includes a run older than some flag(s) (%s); taken as the default.'
                   % ', '.join(sorted({k for r in rows for k in r['predates']})))
    if any(r['legacy'] for r in rows):
        out.append('Includes runs from outside this sweep (--include_legacy), matched by run_config.json.')
    out.append('\n`L_per-L_rec`: last-epoch cost of the perturbed pass; ~0 = perturbation is inert. '
               '`a*L_unc/L_rec`: last-epoch extra weight of the uncertainty term; ~0 = sigma^2 sits '
               'at the clamp and U is off.  Seed-to-seed std of test R@20 on ML-1M measured here is '
               '0.002 (w/o P&U) to 0.006 (LRURec), so a single-seed difference below ~1% is not a '
               'finding.')
    return '\n'.join(out)


def write_outputs(rows, text, prefix, dataset):
    out_dir = os.path.join(ROOT, 'experiments', 'compare')
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, '%s_%s' % (prefix, dataset))
    with open(base + '.md', 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    with open(base + '.json', 'w') as f:
        json.dump(rows, f, indent=2, default=str)
    fields = list(rows[0].keys()) if rows else []
    with open(base + '.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (';'.join(v) if isinstance(v, list) else v) for k, v in r.items()})
    return base


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #
def run_one(job, log_dir, stream):
    variant, seed, argv, folder = job
    name = os.path.basename(folder)
    if os.path.isdir(folder) and not os.path.isfile(os.path.join(folder, 'test_metrics.json')):
        stale = folder + '.incomplete-' + time.strftime('%Y%m%d-%H%M%S')
        os.rename(folder, stale)
        print('[run] %s: incomplete folder moved to %s' % (name, os.path.relpath(stale, ROOT)))
    cmd = [sys.executable, 'train.py'] + argv
    log_path = os.path.join(log_dir, name + '.log')
    print('[run] %s  started %s  (log: %s)' % (name, time.strftime('%H:%M:%S'),
                                                  os.path.relpath(log_path, ROOT)), flush=True)
    t0 = time.time()
    if stream:
        rc = subprocess.call(cmd, cwd=ROOT)
    else:
        with open(log_path, 'w', encoding='utf-8', errors='replace') as log:
            log.write(' '.join(cmd) + '\n')
            log.flush()
            rc = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    minutes = (time.time() - t0) / 60
    tm = os.path.join(folder, 'test_metrics.json')
    if rc == 0 and os.path.isfile(tm):
        m = _load_json(tm)
        print('[done] %s  %.1f min  R@10 %.4f  R@20 %.4f  N@10 %.4f  N@20 %.4f' % (
            name, minutes, m['Recall@10'], m['Recall@20'], m['NDCG@10'], m['NDCG@20']), flush=True)
    else:
        print('[FAIL] %s  exit code %s after %.1f min -- see %s' % (
            name, rc, minutes, os.path.relpath(log_path, ROOT)), flush=True)
    return rc


def main():
    argv = sys.argv[1:]
    passthrough = []
    if '--' in argv:
        i = argv.index('--')
        argv, passthrough = argv[:i], argv[i + 1:]

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--variants', nargs='+', default=None,
                   help='variant names and/or groups (%s); default: @table3 = %s'
                        % (' '.join('@' + g for g in GROUPS), ' '.join(DEFAULT_VARIANTS)))
    p.add_argument('--seeds', nargs='+', type=int, default=[42])
    p.add_argument('--dataset', default='ml-1m')
    p.add_argument('--prefix', default=None, help="folder prefix (default 'cmp', or 'quick' with --quick)")
    p.add_argument('--reference', default=None,
                   help='variant the relative columns are computed against; default slsrec '
                        '(as paper Table III), else wo_pu when slsrec is not in the run')
    p.add_argument('--quick', action='store_true',
                   help='trial run: ' + ' '.join(QUICK_FLAGS) + ' for every variant')
    p.add_argument('--jobs', type=int, default=1, help='runs in parallel (one GPU: keep 1)')
    p.add_argument('--stream', action='store_true',
                   help='show training output in this console instead of a log file (jobs=1)')
    p.add_argument('--force', action='store_true', help='retrain runs that already finished')
    p.add_argument('--dry_run', action='store_true')
    p.add_argument('--collect_only', action='store_true', help='skip training, only build the table')
    p.add_argument('--include_legacy', action='store_true',
                   help='also tabulate older runs whose run_config.json matches a variant '
                        '(they are still retrained under --prefix; see --reuse_legacy)')
    p.add_argument('--reuse_legacy', action='store_true',
                   help='--include_legacy, and do not train a (variant, seed) that an older run '
                        'with matching run_config.json already covers')
    p.add_argument('--list', action='store_true', help='list the known variants and exit')
    a = p.parse_args(argv)

    if a.list:
        for v, (model, flags, ref, desc) in VARIANTS.items():
            print('%-16s %-7s %-62s %s' % (v, model, ' '.join(flags) or '(defaults)',
                                           desc + ('' if ref is None else '  [paper: %s]' % ref)))
        print()
        for g, members in GROUPS.items():
            print('@%-15s %s' % (g, ' '.join(members)))
        return 0

    variants = []
    for name in a.variants or ['@table3']:
        if name.startswith('@') and name[1:] not in GROUPS:
            p.error('unknown group %s (groups: %s)' % (name, ' '.join('@' + g for g in GROUPS)))
        for v in (GROUPS[name[1:]] if name.startswith('@') else [name]):
            if v not in variants:
                variants.append(v)
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        p.error('unknown variant(s): %s (see --list)' % ', '.join(unknown))
    if a.reference is not None:
        reference = a.reference if a.reference in variants else None
    else:
        reference = next((r for r in ('slsrec', 'wo_pu') if r in variants), None)
    prefix = a.prefix or ('quick' if a.quick else 'cmp')
    extra = (QUICK_FLAGS if a.quick else []) + passthrough
    a.include_legacy = a.include_legacy or a.reuse_legacy
    covered = set()
    if a.reuse_legacy and not a.collect_only:
        for v, runs in collect(variants, a.dataset, a.seeds, prefix, extra, True).items():
            covered |= {(v, r['hparams'].get('seed')) for r in runs if r['legacy']}

    if not a.collect_only:
        log_dir = os.path.join(ROOT, 'runlogs', '%s_%s' % (prefix, a.dataset))
        jobs = []
        for s in a.seeds:                        # seed-major: a finished seed is a complete table
            for v in variants:
                targv = train_argv(v, a.dataset, s, prefix, extra)
                folder = os.path.join(ROOT, 'experiments', VARIANTS[v][0], exp_name(prefix, v, a.dataset, s))
                done = os.path.isfile(os.path.join(folder, 'test_metrics.json'))
                if (v, s) in covered and not done:
                    print('[reuse] %s seed %d: an older matching run is used instead' % (v, s))
                    continue
                if done and not a.force:
                    run = read_run(folder)
                    if run is not None and not match(resolve(targv), run['hparams'])[0]:
                        print('[conflict] %s finished with different hyperparameters than this '
                              'command asks for -- use another --prefix, or --force to retrain'
                              % os.path.basename(folder))
                    else:
                        print('[skip] %s already finished' % os.path.basename(folder))
                    continue
                if done and a.force and not a.dry_run:
                    old = folder + '.old-' + time.strftime('%Y%m%d-%H%M%S')
                    os.rename(folder, old)
                    print('[run] %s: previous result moved to %s' % (os.path.basename(folder),
                                                                     os.path.relpath(old, ROOT)))
                jobs.append((v, s, targv, folder))
        if a.dry_run:
            for _, _, targv, _ in jobs:
                print(' '.join(['python', 'train.py'] + targv))
            return 0
        if jobs:
            os.makedirs(log_dir, exist_ok=True)
            stream = a.stream and a.jobs == 1
            if a.jobs > 1:
                with ThreadPoolExecutor(max_workers=a.jobs) as pool:
                    codes = list(pool.map(lambda j: run_one(j, log_dir, False), jobs))
            else:
                codes = [run_one(j, log_dir, stream) for j in jobs]
            failed = sum(1 for c in codes if c != 0)
            if failed:
                print('[warn] %d of %d run(s) failed' % (failed, len(jobs)))

    found = collect(variants, a.dataset, a.seeds, prefix, extra, a.include_legacy)
    rows = build_rows(found, variants, a.dataset, reference)
    text = render(rows, a.dataset, reference)
    print()
    print(text)
    base = write_outputs(rows, text, prefix, a.dataset)
    print('\nwrote %s.{md,csv,json}' % os.path.relpath(base, ROOT))
    return 0


if __name__ == '__main__':
    sys.exit(main())
