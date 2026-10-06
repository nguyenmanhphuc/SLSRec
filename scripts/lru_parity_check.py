"""Sub-minute parity check: unmodified LRURec vs this repo's `lru` path.

Builds the ML-1M dataloaders and the LRU model on both sides with the same
seed, then compares the first shuffled training batch and its loss.  Because
`model/lru.py`, `trainer/lru.py`, every `datasets/` module and the preprocessed
`dataset.pkl` are byte-identical between the two trees, a matching batch and
loss means the random stream and the data pipeline agree -- i.e. the harness
SLSRec is built on has not drifted from upstream.

Run this before the full baseline: it costs seconds instead of GPU hours.

    python scripts/lru_parity_check.py
"""

import argparse
import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = os.path.join(ROOT, '_ref_LRURec')
SHIMS = os.path.join(ROOT, 'scripts', '_ref_shims')

PROBE = '''
import json, sys, tempfile
sys.argv = ['train.py', '--dataset_code=%(dataset)s', %(model_flag)s
            '--seed=%(seed)d', '--num_workers=0']
%(seed_import)s
from datasets import DATASETS                    # noqa: F401  (import order)
from config import *                             # noqa: F401,F403
from model import *                              # noqa: F401,F403
from dataloader import *                         # noqa: F401,F403
from trainer import *                            # noqa: F401,F403

set_template(args)
seed_everything(args.seed)
train_loader, val_loader, test_loader = dataloader_factory(args)
model = %(build_model)s
with tempfile.TemporaryDirectory() as tmp:
    trainer = %(build_trainer)s
    batch = next(iter(train_loader))
    loss = trainer.calculate_loss([x.to(args.device) for x in batch])

print('@@PARITY@@' + json.dumps({
    'max_len': args.bert_max_len,
    'val_iterations': args.val_iterations,
    'train_batch_size': args.train_batch_size,
    'num_items': args.num_items,
    'train_batches': len(train_loader),
    'val_batches': len(val_loader),
    'params': sum(p.numel() for p in model.parameters()),
    'encoder_params': sum(p.numel() for n, p in model.named_parameters() if 'embedding' not in n),
    'batch_seq_sum': int(batch[0].sum()),
    'batch_label_sum': int(batch[1].sum()),
    'loss': float(loss.item()),
}))
'''

UPSTREAM = {
    'model_flag': '',
    'seed_import': 'import wandb  # noqa: F401\nfrom pytorch_lightning import seed_everything',
    'build_model': 'LRU(args)',
    'build_trainer': 'LRUTrainer(args, model, train_loader, val_loader, test_loader, tmp, False)',
}
OURS = {
    'model_flag': "'--model_code=lru',",
    'seed_import': 'from trainer.utils import seed_everything',
    'build_model': 'model_factory(args)',
    'build_trainer': 'trainer_factory(args, model, train_loader, val_loader, test_loader, tmp, False)',
}


def shim_env():
    """PYTHONPATH carrying stubs only for packages that are genuinely missing."""
    env = dict(os.environ)
    missing = [m for m in ('pytorch_lightning', 'wandb')
               if subprocess.call([sys.executable, '-c', 'import ' + m],
                                  stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL) != 0]
    if not missing:
        return env, []
    tmp = tempfile.mkdtemp(prefix='lru_shims_')
    atexit.register(shutil.rmtree, tmp, True)
    for mod in missing:
        with open(os.path.join(SHIMS, mod + '.py')) as src, \
             open(os.path.join(tmp, mod + '.py'), 'w') as dst:
            dst.write(src.read())
    env['PYTHONPATH'] = tmp + os.pathsep + env.get('PYTHONPATH', '')
    return env, missing


def probe(cwd, subst, env=None):
    """Run PROBE with cwd on sys.path[0] (stdin form) and parse its JSON line."""
    source = PROBE % subst
    out = subprocess.run([sys.executable, '-'], input=source, cwd=cwd, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in out.stdout.splitlines():
        if line.startswith('@@PARITY@@'):
            return json.loads(line[len('@@PARITY@@'):]), out
    return None, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ml-1m')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--loss-tol', type=float, default=1e-6)
    args = parser.parse_args()

    if not os.path.isdir(REF):
        print('_ref_LRURec/ is missing -- see scripts/verify_lrurec.sh for how to fetch it',
              file=sys.stderr)
        return 2

    common = {'dataset': args.dataset, 'seed': args.seed}
    env, missing = shim_env()
    if missing:
        print('[parity] stubbing %s for the upstream run' % ', '.join(missing))

    print('[parity] upstream  _ref_LRURec/ ...')
    ref, ref_out = probe(REF, dict(UPSTREAM, **common), env)
    print('[parity] this repo  --model_code=lru ...')
    ours, ours_out = probe(ROOT, dict(OURS, **common))

    for label, got, out in (('upstream', ref, ref_out), ('ours', ours, ours_out)):
        if got is None:
            print('\n[parity] the %s probe produced no result:' % label, file=sys.stderr)
            print(out.stderr.strip()[-2000:], file=sys.stderr)
            return 2

    keys = ['max_len', 'val_iterations', 'train_batch_size', 'num_items', 'train_batches',
            'val_batches', 'params', 'encoder_params', 'batch_seq_sum', 'batch_label_sum']
    rows = [(k, str(ref[k]), str(ours[k]), 'ok' if ref[k] == ours[k] else 'MISMATCH') for k in keys]
    delta = abs(ref['loss'] - ours['loss'])
    rows.append(('loss', '%.8f' % ref['loss'], '%.8f' % ours['loss'],
                 'ok' if delta <= args.loss_tol else 'MISMATCH (delta %.2e)' % delta))

    header = ('field', 'upstream', 'ours', '')
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(4)]
    print()
    print('| ' + ' | '.join(h.ljust(w) for h, w in zip(header, widths)) + ' |')
    print('|' + '|'.join('-' * (w + 2) for w in widths) + '|')
    for r in rows:
        print('| ' + ' | '.join(c.ljust(w) for c, w in zip(r, widths)) + ' |')
    print()

    bad = [r[0] for r in rows if r[3] != 'ok']
    if bad:
        print('FAIL  %s differ -- the harness is not equivalent to upstream' % ', '.join(bad))
        return 1
    print('PASS  identical batch and identical loss: same RNG stream, same data pipeline.')
    print('      Safe to spend GPU time on the full baseline (scripts/verify_lrurec.sh).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
