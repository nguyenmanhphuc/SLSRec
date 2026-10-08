"""Is a finished run over-fitting or under-fitting?  Measured on its checkpoint.

For a sample of users the checkpoint (eval mode, no dropout) scores two targets
with the same history masking and the same unmasked-logit CE as validation:

  seen    the last *training* item, from the items before it -- a target the
          model was trained on (per-position CE covers every training position)
  unseen  the validation item, from the whole training sequence -- never trained on

A large seen/unseen gap means memorisation (over-fitting).  A small gap with
both low means the model cannot even fit its own training targets, i.e. it is
under-fitting / over-regularised.  The seen targets are also scored in train
mode (dropout on, averaged over a few draws): that is the loss the optimiser
actually sees, so a big train-mode vs eval-mode CE difference on the *same*
targets points at the dropout train/eval mismatch rather than at the data.

Finally the train-mode L_rec curve from train_losses.jsonl is printed next to
the validation CE curve from val_metrics.jsonl.

    python scripts/fit_check.py experiments/slsrec/noP_beauty_seed42
    python scripts/fit_check.py experiments/slsrec/noP_beauty_seed42 --ckpt recent --users 20000
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def build_args(run_dir):
    hp = json.load(open(os.path.join(run_dir, 'run_config.json')))['hparams']
    sys.argv = ['fit_check']                    # config.py parses sys.argv on import
    import config as cfg
    args = cfg.parser.parse_args([f'--dataset_code={hp["dataset_code"]}', f'--model_code={hp["model_code"]}'])
    cfg.set_template(args)
    for k, v in hp.items():
        if hasattr(args, k) and k not in ('device', 'num_workers'):
            setattr(args, k, v)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.num_workers = 0
    return args


def pad_left(seq, L):
    seq = seq[-L:]
    return [0] * (L - len(seq)) + seq


@torch.no_grad()
def score(model, inputs, targets, device, bs, train_mode=False, draws=1):
    """CE on unmasked logits, then Recall/NDCG@10/20 with the input items masked."""
    from trainer.utils import absolute_recall_mrr_ndcg_for_ks
    model.train(train_mode)
    tot, n = {}, 0
    for i in range(0, len(inputs), bs):
        x = torch.LongTensor(inputs[i:i + bs]).to(device)
        y = torch.LongTensor(targets[i:i + bs]).to(device)
        ce = np.mean([F.cross_entropy(model(x)[0][:, -1, :], y).item() for _ in range(draws)])
        logits = model(x)[0][:, -1, :]
        logits.scatter_(1, x, -1e9)
        logits[:, 0] = -1e9
        m = absolute_recall_mrr_ndcg_for_ks(logits, y, [10, 20])
        m['CE'] = ce
        for k in ('CE', 'Recall@10', 'NDCG@10', 'Recall@20', 'NDCG@20'):
            tot[k] = tot.get(k, 0.0) + m[k] * len(x)
        n += len(x)
    model.eval()
    return {k: v / n for k, v in tot.items()}


def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run_dir')
    ap.add_argument('--ckpt', choices=['best', 'recent'], default='best')
    ap.add_argument('--users', type=int, default=10000, help='users to sample (0 = all)')
    ap.add_argument('--batch', type=int, default=256)
    a = ap.parse_args()

    run_dir = os.path.abspath(a.run_dir)
    os.chdir(ROOT)
    args = build_args(run_dir)
    from datasets import dataset_factory
    from model import model_factory

    data = dataset_factory(args).load_dataset()
    train, val = data['train'], data['val']
    args.num_items, args.num_users = len(data['smap']), len(data['umap'])   # normally set by the dataloader
    L = args.bert_max_len
    users = [u for u in sorted(train) if len(train[u]) >= 2 and len(val.get(u, [])) > 0]
    # a target already inside its own input window is masked out by evaluation -> drop those users
    users = [u for u in users if train[u][-1] not in train[u][:-1][-L:] and val[u][0] not in train[u][-L:]]
    rng = np.random.RandomState(0)
    if a.users and a.users < len(users):
        users = sorted(rng.choice(users, a.users, replace=False).tolist())

    seen_x = [pad_left(train[u][:-1], L) for u in users]
    seen_y = [train[u][-1] for u in users]
    unseen_x = [pad_left(train[u], L) for u in users]
    unseen_y = [val[u][0] for u in users]

    model = model_factory(args)
    f = 'best_acc_model.pth' if a.ckpt == 'best' else 'checkpoint-recent.pth'
    state = torch.load(os.path.join(run_dir, 'models', f), map_location='cpu')
    model.load_state_dict(state['model_state_dict'])
    model.to(args.device).eval()

    seen = score(model, seen_x, seen_y, args.device, a.batch)
    unseen = score(model, unseen_x, unseen_y, args.device, a.batch)
    seen_tr = score(model, seen_x, seen_y, args.device, a.batch, train_mode=True, draws=3)

    print(f'run {os.path.basename(run_dir)}  ckpt={a.ckpt}  users={len(users)}  '
          f'dropout={args.bert_dropout}  uniform CE=ln({args.num_items})={np.log(args.num_items):.2f}')
    print(f'{"":<34}{"CE":>8}{"R@10":>9}{"N@10":>9}{"R@20":>9}{"N@20":>9}')
    for name, m in (('seen   (last train item), eval', seen), ('unseen (val item),        eval', unseen),
                    ('seen   (last train item), train', seen_tr)):
        print(f'{name:<34}{m["CE"]:>8.3f}{m["Recall@10"]:>9.4f}{m["NDCG@10"]:>9.4f}{m["Recall@20"]:>9.4f}{m["NDCG@20"]:>9.4f}')
    r = seen['Recall@20'] / max(unseen['Recall@20'], 1e-12)
    print(f'seen/unseen R@20 = {r:.2f}x   seen-unseen CE = {seen["CE"] - unseen["CE"]:+.2f}   '
          f'train-mode minus eval-mode CE on seen = {seen_tr["CE"] - seen["CE"]:+.2f}')

    tl = read_jsonl(os.path.join(run_dir, 'train_losses.jsonl'))
    vl = [r for r in read_jsonl(os.path.join(run_dir, 'val_metrics.jsonl')) if r.get('accum_iter', 1) > 0]
    if tl and vl:
        tl_by_ep = {r['epoch']: r for r in tl}
        print('\nepoch   train L_rec (dropout on)   val CE   val R@10')
        for r in vl:
            t = tl_by_ep.get(r['epoch'], {})
            print(f'{r["epoch"]:>5}   {t.get("L_rec", float("nan")):>22.3f}   {r.get("CE", float("nan")):>6.3f}   {r["Recall@10"]:.5f}')


if __name__ == '__main__':
    main()
