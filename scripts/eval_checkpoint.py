"""Score a run's saved best checkpoint on the test set, without retraining.

Rebuilds the run's configuration from its run_config.json, loads
models/best_acc_model.pth and evaluates it exactly as Trainer.test() does
(same calculate_metrics, same per-batch averaging).  Useful when a run was
killed or its test step was lost after training had already finished.

    python scripts/eval_checkpoint.py experiments/slsrec/full_exact_d075_beauty_seed42
    python scripts/eval_checkpoint.py experiments/slsrec/<run> --write   # also save test_metrics.json

--write refuses to overwrite an existing test_metrics.json.
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def build_args(run_dir):
    hp = json.load(open(os.path.join(run_dir, 'run_config.json')))['hparams']
    sys.argv = ['eval_checkpoint']                  # config.py parses sys.argv on import
    import config as cfg
    args = cfg.parser.parse_args([f'--dataset_code={hp["dataset_code"]}', f'--model_code={hp["model_code"]}'])
    cfg.set_template(args)
    for k, v in hp.items():
        if hasattr(args, k) and k not in ('device', 'num_workers'):
            setattr(args, k, v)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.num_workers = 0
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--ckpt', default='best_acc_model.pth')
    ap.add_argument('--write', action='store_true', help='save test_metrics.json (never overwrites)')
    a = ap.parse_args()

    run_dir = os.path.abspath(a.run_dir)
    os.chdir(ROOT)
    args = build_args(run_dir)
    from dataloader import dataloader_factory
    from model import model_factory
    from trainer import TRAINERS
    from trainer.utils import AverageMeterSet

    _, _, test_loader = dataloader_factory(args)     # also sets args.num_items / num_users
    model = model_factory(args)
    state = torch.load(os.path.join(run_dir, 'models', a.ckpt), map_location='cpu')
    model.load_state_dict(state['model_state_dict'])
    model.to(args.device).eval()

    # Trainer.calculate_metrics only touches self.model / self.metric_ks / self.args
    fake = SimpleNamespace(model=model, metric_ks=args.metric_ks, args=args)
    calc = TRAINERS[args.model_code].calculate_metrics
    meters = AverageMeterSet()
    with torch.no_grad():
        for batch in test_loader:
            batch = [x.to(args.device) for x in batch]
            for k, v in calc(fake, batch).items():
                meters.update(k, v)
    metrics = meters.averages()

    print(f'run {os.path.basename(run_dir)}  ckpt={a.ckpt}  saved at epoch {state.get("epoch")}')
    print(json.dumps(metrics, indent=1))
    if a.write:
        out = os.path.join(run_dir, 'test_metrics.json')
        if os.path.exists(out):
            print(f'[not written] {out} already exists')
        else:
            with open(out, 'w') as f:
                json.dump(metrics, f, indent=4)
            print(f'written {out}')


if __name__ == '__main__':
    main()
