"""Short real-data smoke test for all four SLSRec ablations.

Builds the real ML-1M dataloaders (LRURec protocol) and, for each ablation,
runs a handful of optimisation steps plus one full-sort validation pass.
Reports the four loss terms and the validation metrics so that every ablation
is shown to be selectable purely through configuration.

    python scripts/smoke_ablations.py --dataset_code=ml-1m --num_workers=0 \
        --smoke_iters 10 --smoke_val_batches 4
"""

import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from datasets import DATASETS                     # noqa: F401  (import order matters)
from config import args, set_template
from dataloader import dataloader_factory
from model import model_factory
from trainer.slsrec import slsrec_losses
from trainer.utils import absolute_recall_mrr_ndcg_for_ks, seed_everything

ABLATIONS = {
    'A_bdlru_ce':     dict(use_sam_gate=False, sam_conv=False,
                           use_perturbation=False, use_uncertainty=False),
    'B_sam_gate':     dict(use_perturbation=False, use_uncertainty=False),
    'C_perturbation': dict(use_perturbation=True, use_uncertainty=False),
    'D_full_slsrec':  dict(use_perturbation=True, use_uncertainty=True),
}


def evaluate(model, loader, device, metric_ks, max_batches):
    model.eval()
    sums, n = {}, 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            seqs, labels = [t.to(device) for t in batch]
            scores = model(seqs)[0][:, -1, :]
            for j in range(seqs.size(1)):
                scores[torch.arange(scores.size(0)), seqs[:, j]] = -1e9
            scores[:, 0] = -1e9
            m = absolute_recall_mrr_ndcg_for_ks(scores, labels.view(-1), metric_ks)
            for k, v in m.items():
                sums[k] = sums.get(k, 0.0) + v
            n += 1
    return {k: v / max(n, 1) for k, v in sums.items()}


def main():
    args.model_code = 'slsrec'   # before set_template: it fills the SLSRec defaults
    set_template(args)
    iters = getattr(args, 'smoke_iters', 10)
    val_batches = getattr(args, 'smoke_val_batches', 4)

    seed_everything(args.seed)
    train_loader, val_loader, _ = dataloader_factory(args)
    device = args.device
    print(f'device={device}  items={args.num_items}  users={args.num_users}  '
          f'max_len={args.bert_max_len}  batches/epoch={len(train_loader)}')

    results = {}
    for name, overrides in ABLATIONS.items():
        a = copy.deepcopy(args)
        for k, v in overrides.items():
            setattr(a, k, v)
        seed_everything(a.seed)
        model = model_factory(a).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay,
                                eps=a.adam_epsilon)

        model.train()
        history, t0 = [], time.time()
        for i, batch in enumerate(train_loader):
            if i >= iters:
                break
            seqs, labels = [t.to(device) for t in batch]
            opt.zero_grad()
            losses = slsrec_losses(a, model, seqs, labels, training=True)
            losses['L_total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.max_grad_norm)
            opt.step()
            history.append({k: round(v.item(), 4) for k, v in losses.items()})
        train_time = time.time() - t0

        metrics = evaluate(model, val_loader, device, a.metric_ks, val_batches)
        results[name] = {
            'params': n_params,
            'first_step': history[0],
            'last_step': history[-1],
            'sec_per_iter': round(train_time / max(len(history), 1), 2),
            'val': {k: round(v, 5) for k, v in metrics.items()
                    if k in ('Recall@10', 'Recall@20', 'NDCG@10', 'NDCG@20', 'MRR@10')},
        }
        print(f'\n=== {name} ===')
        print(json.dumps(results[name], indent=2))
        assert all(torch.isfinite(torch.tensor(v)) for v in history[-1].values()), 'non-finite loss'

    with open('smoke_ablations.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nwrote smoke_ablations.json')


if __name__ == '__main__':
    main()
