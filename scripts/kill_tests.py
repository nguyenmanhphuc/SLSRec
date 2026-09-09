"""Kill-tests for SLSRec: adversarial probes that try to *falsify* the paper's claims.

Each probe states a claim from the paper and a falsifier -- a measurement whose
outcome, if observed, means the claim cannot hold.  Nothing here is a unit test
of the reproduction; the reproduction is assumed correct (see `tests/`).  These
probes attack the *model*.

    python scripts/kill_tests.py --probe gap
    python scripts/kill_tests.py --probe transparency --dataset_code=ml-1m --num_workers=0
    python scripts/kill_tests.py --probe routing      --dataset_code=ml-1m --num_workers=0
    python scripts/kill_tests.py --probe gates        --dataset_code=ml-1m --num_workers=0
    python scripts/kill_tests.py --probe overfit
    python scripts/kill_tests.py --probe all          --dataset_code=ml-1m --num_workers=0

Probes
------
gap           C1: L_Per is a distinct training signal from L_Rec (eq. 18 vs 16).
              Falsifier: |L_Per - L_Rec| stays inside the epoch-to-epoch noise of
              L_Rec itself -> the perturbed pass carries no extra information and
              L_total ~ 2*L_Rec, i.e. a loss rescale, which AdamW is invariant to.

transparency  C2: replacing an item representation with U(0,1) inside the
              recurrence is a meaningful corruption (Sec. III-D).
              Falsifier: sweeping theta from 0.1 to 1.0 (every position, every
              layer) leaves CE / Recall@10 / the top-10 list essentially
              unchanged -> nothing was corrupted, so nothing was regularised.

routing       C3: the BD-LRU recurrence carries the sequence signal (Sec. III-C).
              Falsifier: zeroing H, or removing the recurrence and keeping only
              Conv1-D2, costs little Recall@10 -> the perturbation is applied to a
              branch the model does not rely on.

gates         C4 (mechanism): if perturbation *is* felt, the model should not be
              able to trivially detect and gate it away.
              Falsifier: the input gate i_t is systematically smaller at perturbed
              positions than at clean ones -> the model learned a U(0,1) detector,
              and the noise never enters the state.

overfit       C5: perturbation injection combats overfitting (Sec. III-D).
              Falsifier: a model trained with theta=1.0 and w_per=5 still memorises
              8 synthetic sequences to >90% train accuracy -> the perturbed branch
              imposes no capacity cost, so it cannot be an anti-overfitting term.
"""

import argparse
import copy
import json
import math
import os
import random
import sys
import time
from argparse import Namespace
from contextlib import contextmanager

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from datasets import DATASETS                      # noqa: F401  (import order matters)
from config import args as cfg_args, set_template
from dataloader import dataloader_factory
from model import model_factory
from model.bdlru import BDLRU, parallel_scan
from model.slsrec import SLSRec
from trainer.slsrec import slsrec_losses
from trainer.utils import absolute_recall_mrr_ndcg_for_ks, seed_everything

DEFAULT_CKPT = os.path.join('experiments', 'slsrec', 'smoke3ep_D_full_slsrec',
                            'models', 'best_acc_model.pth')


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def blocks_of(model):
    return model.model.blocks


def set_perturbation(model, theta, low=0.0, high=1.0):
    for b in blocks_of(model):
        b.sg_lru.bdlru.perturb_prob = theta
        b.sg_lru.bdlru.perturb_low = low
        b.sg_lru.bdlru.perturb_high = high


def rank_scores(logits, seqs):
    """LRURec full-sort protocol: drop already-seen items and the padding id."""
    scores = logits.clone()
    for i in range(seqs.size(1)):
        scores[torch.arange(scores.size(0)), seqs[:, i]] = -1e9
    scores[:, 0] = -1e9
    return scores


def fmt(v, n=4):
    return f'{v:.{n}f}' if isinstance(v, float) else str(v)


def table(rows, headers):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = '  '.join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    out = [line, '  '.join('-' * w for w in widths)]
    for r in rows:
        out.append('  '.join(str(c).ljust(widths[i]) for i, c in enumerate(r)))
    return chr(10).join(out)


def load_model(args, ckpt=None, train_steps=0, train_loader=None):
    """Build a SLSRec model; load `ckpt` if given, else train `train_steps` steps."""
    seed_everything(args.seed)
    model = model_factory(args).to(args.device)
    if ckpt and os.path.exists(ckpt):
        sd = torch.load(ckpt, map_location=args.device, weights_only=False)
        sd = sd.get('model_state_dict', sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f'[load] {ckpt}  missing={len(missing)} unexpected={len(unexpected)}')
    elif train_steps > 0:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay, eps=args.adam_epsilon)
        model.train()
        t0 = time.time()
        for i, batch in enumerate(train_loader):
            if i >= train_steps:
                break
            seqs, labels = [t.to(args.device) for t in batch]
            opt.zero_grad()
            slsrec_losses(args, model, seqs, labels, training=True)['L_total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
        print(f'[warmup] {train_steps} steps in {time.time() - t0:.1f}s')
    else:
        print('[load] WARNING: untrained model -- probe results are meaningless')
    return model


# --------------------------------------------------------------------------- #
# probe 1 -- L_Per vs L_Rec gap (no compute, reads existing logs)
# --------------------------------------------------------------------------- #
def probe_gap(opts):
    print(chr(10) + '=' * 78)
    print('PROBE gap -- is L_Per a distinct signal from L_Rec?')
    print('=' * 78)
    paths = opts.losses or []
    if not paths:
        root = os.path.join('experiments', 'slsrec')
        for d, _, files in os.walk(root):
            if 'train_losses.jsonl' in files:
                paths.append(os.path.join(d, 'train_losses.jsonl'))
    if not paths:
        print('no train_losses.jsonl found under experiments/slsrec/ -- run a training job first')
        return {}

    out = {}
    for p in paths:
        recs = [json.loads(l) for l in open(p) if l.strip()]
        recs = [r for r in recs if 'L_perturb' in r and r.get('L_perturb', 0) != 0]
        if not recs:
            continue
        rows, gaps = [], []
        for r in recs:
            gap = r['L_perturb'] - r['L_rec']
            gaps.append(gap)
            rows.append([r['epoch'], fmt(r['L_rec']), fmt(r['L_perturb']),
                         f'{gap:+.5f}', f'{100 * gap / r["L_rec"]:+.3f}%'])
        drift = [abs(recs[i + 1]['L_rec'] - recs[i]['L_rec']) for i in range(len(recs) - 1)]
        print(chr(10) + p)
        print(table(rows, ['epoch', 'L_rec', 'L_perturb', 'gap', 'gap/L_rec']))
        mg = float(np.mean(np.abs(gaps)))
        md = float(np.mean(drift)) if drift else float('nan')
        print(f'mean |gap| = {mg:.5f}   mean |epoch-to-epoch move of L_rec| = {md:.5f}'
              f'   ratio = {mg / md:.4f}' if drift else f'mean |gap| = {mg:.5f}')
        print('VERDICT: ' + ('FALSIFIED -- the perturbed pass reproduces L_rec to within '
                             'a fraction of the training step size; L_total is ~2*L_rec.'
                             if drift and mg < 0.05 * md else 'not falsified here'))
        out[p] = {'mean_abs_gap': mg, 'mean_lrec_step': md, 'gaps': gaps}
    return out


# --------------------------------------------------------------------------- #
# probe 2 -- perturbation transparency
# --------------------------------------------------------------------------- #
@torch.no_grad()
def measure(model, loader, args, max_batches, theta, low, high, ks=(10, 20), seed=0):
    set_perturbation(model, theta, low, high)
    model.eval()
    torch.manual_seed(seed)
    agg = dict(ce=0.0, kl=0.0, drift=0.0, overlap=0.0, n=0)
    met = {}
    nb = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        seqs, labels = [t.to(args.device) for t in batch]
        hc, _ = model.model.encode(seqs, perturb=False)
        lc = model.model.logits_from_hidden(hc)[:, -1, :]
        if theta > 0:
            hp, _ = model.model.encode(seqs, perturb=True)
            lp = model.model.logits_from_hidden(hp)[:, -1, :]
        else:
            hp, lp = hc, lc

        agg['ce'] += F.cross_entropy(lp, labels.view(-1)).item()
        agg['kl'] += 0.5 * (F.kl_div(F.log_softmax(lp, -1), F.log_softmax(lc, -1),
                                     log_target=True, reduction='batchmean').item()
                            + F.kl_div(F.log_softmax(lc, -1), F.log_softmax(lp, -1),
                                       log_target=True, reduction='batchmean').item())
        d = (hp[:, -1] - hc[:, -1]).norm(dim=-1) / hc[:, -1].norm(dim=-1).clamp_min(1e-8)
        agg['drift'] += d.mean().item()
        tc = rank_scores(lc, seqs).topk(10, dim=-1).indices
        tp = rank_scores(lp, seqs).topk(10, dim=-1).indices
        ov = torch.stack([torch.isin(tp[b], tc[b]).float().mean() for b in range(tp.size(0))])
        agg['overlap'] += ov.mean().item()

        m = absolute_recall_mrr_ndcg_for_ks(rank_scores(lp, seqs), labels.view(-1), list(ks))
        for k, v in m.items():
            met[k] = met.get(k, 0.0) + v
        nb += 1
    nb = max(nb, 1)
    res = {k: v / nb for k, v in agg.items() if k != 'n'}
    res.update({k: v / nb for k, v in met.items()})
    return res


def probe_transparency(opts, args, model, val_loader):
    print(chr(10) + '=' * 78)
    print('PROBE transparency -- does U(0,1) replacement actually corrupt anything?')
    print('=' * 78)
    grid = [(t, 0.0, 1.0) for t in opts.thetas]
    grid += [(0.1, lo, hi) for lo, hi in opts.ranges]
    rows, out = [], {}
    for theta, lo, hi in grid:
        r = measure(model, val_loader, args, opts.n_batches, theta, lo, hi, seed=opts.seed)
        key = f'theta={theta} U({lo},{hi})'
        out[key] = r
        rows.append([theta, f'U({lo},{hi})', fmt(r['ce']), fmt(r['Recall@10']),
                     fmt(r['NDCG@10']), fmt(r['overlap'], 3), fmt(r['kl'], 4), fmt(r['drift'], 4)])
    print(table(rows, ['theta', 'range', 'CE', 'Recall@10', 'NDCG@10',
                       'top10 overlap', 'sym-KL', 'rel drift']))
    base = out[f'theta=0.0 U(0.0,1.0)'] if f'theta=0.0 U(0.0,1.0)' in out else None
    worst = out.get('theta=1.0 U(0.0,1.0)')
    if base and worst:
        dr = 100 * (base['Recall@10'] - worst['Recall@10']) / max(base['Recall@10'], 1e-9)
        print(f'{chr(10)}theta=0 -> theta=1.0 (every position, every layer replaced by noise): '
              f'Recall@10 falls {dr:.1f}%, top-10 list keeps '
              f'{100 * worst["overlap"]:.0f}% of its items.')
        print('VERDICT: ' + ('FALSIFIED -- total corruption of the BD-LRU input is nearly '
                             'invisible at the output, so theta=0.1 corrupts nothing.'
                             if dr < 25 else 'not falsified -- the perturbation does bite.'))
    set_perturbation(model, args.perturbation_probability)
    return out


# --------------------------------------------------------------------------- #
# probe 3 -- where does the signal actually flow?
# --------------------------------------------------------------------------- #
@torch.no_grad()
def probe_routing(opts, args, model, val_loader):
    print(chr(10) + '=' * 78)
    print('PROBE routing -- which branch carries the sequence signal?')
    print('=' * 78)

    def hooks_for(name):
        hs = []
        for b in blocks_of(model):
            sg = b.sg_lru
            if name == 'full':
                pass
            elif name == 'H := 0 (BD-LRU output zeroed)':
                hs.append(sg.bdlru.register_forward_hook(lambda m, i, o: torch.zeros_like(o)))
            elif name == 'H := Conv1-D2 (recurrence removed)':
                hs.append(sg.bdlru.register_forward_hook(lambda m, i, o: i[0]))
            elif name == 'G := 1 (gate removed)':
                if getattr(sg, 'use_sam_gate', False):
                    hs.append(sg.sam.register_forward_hook(lambda m, i, o: torch.zeros_like(o)))
                if getattr(sg, 'sam_conv', False):
                    hs.append(sg.conv1.register_forward_hook(lambda m, i, o: torch.ones_like(o)))
            elif name == 'SAM := 0 (conv gate only)':
                hs.append(sg.sam.register_forward_hook(lambda m, i, o: torch.zeros_like(o)))
            elif name == 'Conv1-D1 := 0 (SAM gate only)':
                hs.append(sg.conv1.register_forward_hook(lambda m, i, o: torch.zeros_like(o)))
        return hs

    variants = ['full', 'H := 0 (BD-LRU output zeroed)', 'H := Conv1-D2 (recurrence removed)',
                'G := 1 (gate removed)', 'SAM := 0 (conv gate only)',
                'Conv1-D1 := 0 (SAM gate only)']
    rows, out, base = [], {}, None
    for name in variants:
        hs = hooks_for(name)
        try:
            r = measure(model, val_loader, args, opts.n_batches, 0.0, 0.0, 1.0, seed=opts.seed)
        finally:
            for h in hs:
                h.remove()
        out[name] = r
        base = base if base is not None else r['Recall@10']
        rows.append([name, fmt(r['ce']), fmt(r['Recall@10']), fmt(r['NDCG@10']),
                     f'{100 * (r["Recall@10"] - base) / max(base, 1e-9):+.1f}%'])
    print(table(rows, ['variant', 'CE', 'Recall@10', 'NDCG@10', 'vs full']))
    h0 = out['H := 0 (BD-LRU output zeroed)']['Recall@10']
    print(f'{chr(10)}VERDICT: ' + (
        'FALSIFIED -- the model survives losing the whole BD-LRU output, so the branch '
        'the perturbation attacks is not the one doing the work.'
        if h0 > 0.5 * base else 'not falsified -- BD-LRU carries real signal.'))
    return out


# --------------------------------------------------------------------------- #
# probe 4 -- has the model learned a U(0,1) detector?
# --------------------------------------------------------------------------- #
@contextmanager
def instrumented_bdlru():
    orig = BDLRU.forward

    def fwd(self, x, mask=None, perturb=False):
        if perturb and self.perturb_prob > 0:
            x, self.last_perturb_mask = self._perturb(x, mask)
        else:
            self.last_perturb_mask = None
        recurrence, inp = self.gates(x).chunk(2, dim=-1)
        r_t, i_t = torch.sigmoid(recurrence), torch.sigmoid(inp)
        a_t = torch.exp(self.c * F.logsigmoid(self.Lambda) * r_t)
        b_t = torch.sqrt((1.0 - a_t.pow(2)).clamp_min(1e-8)) * (i_t * x)
        if mask is not None:
            b_t = b_t * mask.unsqueeze(-1).to(b_t.dtype)
        h = parallel_scan(a_t, b_t, impl=self.scan_impl)
        if mask is not None:
            h = h * mask.unsqueeze(-1).to(h.dtype)
        self.last_i_t, self.last_a_t, self.last_b_t = i_t, a_t, b_t
        return h

    BDLRU.forward = fwd
    try:
        yield
    finally:
        BDLRU.forward = orig


@torch.no_grad()
def probe_gates(opts, args, model, val_loader):
    print(chr(10) + '=' * 78)
    print('PROBE gates -- does the model detect and gate away the U(0,1) noise?')
    print('=' * 78)
    set_perturbation(model, opts.gate_theta)
    model.eval()
    rows, out = [], {}
    with instrumented_bdlru():
        torch.manual_seed(opts.seed)
        for i, batch in enumerate(val_loader):
            if i >= max(1, opts.n_batches // 2):
                break
            seqs, _ = [t.to(args.device) for t in batch]
            model.model.encode(seqs, perturb=True)
            for li, b in enumerate(blocks_of(model)):
                bd = b.sg_lru.bdlru
                pm = bd.last_perturb_mask
                real = (seqs > 0)
                pert = pm & real
                clean = (~pm) & real
                if pert.sum() == 0 or clean.sum() == 0:
                    continue
                st = out.setdefault(li, dict(i_p=[], i_c=[], b_p=[], b_c=[], a_p=[], a_c=[]))
                st['i_p'].append(bd.last_i_t[pert].mean().item())
                st['i_c'].append(bd.last_i_t[clean].mean().item())
                st['b_p'].append(bd.last_b_t[pert].norm(dim=-1).mean().item())
                st['b_c'].append(bd.last_b_t[clean].norm(dim=-1).mean().item())
                st['a_p'].append(bd.last_a_t[pert].mean().item())
                st['a_c'].append(bd.last_a_t[clean].mean().item())
    res = {}
    for li, st in sorted(out.items()):
        m = {k: float(np.mean(v)) for k, v in st.items()}
        res[f'layer{li}'] = m
        rows.append([li, fmt(m['i_c']), fmt(m['i_p']), f'{m["i_p"] / m["i_c"]:.3f}',
                     fmt(m['b_c']), fmt(m['b_p']), f'{m["b_p"] / m["b_c"]:.3f}',
                     fmt(m['a_c']), fmt(m['a_p'])])
    print(table(rows, ['layer', 'i_t clean', 'i_t pert', 'ratio',
                       '||b_t|| clean', '||b_t|| pert', 'ratio', 'a_t clean', 'a_t pert']))
    ratios = [res[k]['i_p'] / res[k]['i_c'] for k in res]
    print(f'{chr(10)}VERDICT: ' + (
        'the input gate is suppressed at perturbed positions -> a learned noise detector; '
        'the perturbation never reaches the state.'
        if ratios and min(ratios) < 0.8 else
        'no gate suppression -- the noise DOES enter the state, so if the output still '
        'does not move (see `transparency`), the state itself is not what the head reads.'))
    set_perturbation(model, args.perturbation_probability)
    return res


# --------------------------------------------------------------------------- #
# probe 5 -- can perturbation stop memorisation at all?
# --------------------------------------------------------------------------- #
def tiny_args(**over):
    a = Namespace(
        model_code='slsrec', dataset_code='ml-1m', device='cpu', seed=42,
        num_items=40, bert_max_len=16, bert_hidden_units=16, bert_num_blocks=2,
        bert_dropout=0.0, bert_attn_dropout=0.0, metric_ks=[1, 5, 10], best_metric='Recall@10',
        slsrec_expand=2, slsrec_d_conv=4, scan_impl='torch', use_item_bias=False,
        use_sam_gate=True, sam_conv=True, sam_conv_activation='none',
        sam_num_heads=1, sam_attn_dropout=0.0,
        use_bdlru=True, bdlru_c=8.0, bdlru_r_min=0.9, bdlru_r_max=0.999,
        bdlru_lambda_init='linspace', bdlru_conv_activation='silu',
        use_perturbation=True, perturbation_probability=0.1,
        perturbation_low=0.0, perturbation_high=1.0, perturbation_weight=1.0,
        use_uncertainty=False, uncertainty_alpha=0.3, uncertainty_beta=0.0,
        uncertainty_loss='approx', uncertainty_logvar_clamp=8.0,
    )
    for k, v in over.items():
        setattr(a, k, v)
    return a


def tiny_batch(a, batch_size=8, real_len=None):
    L = a.bert_max_len
    real_len = real_len if real_len is not None else L - 2
    g = torch.Generator().manual_seed(42)
    seqs = torch.zeros(batch_size, L, dtype=torch.long)
    labels = torch.zeros(batch_size, L, dtype=torch.long)
    full = torch.randint(1, a.num_items + 1, (batch_size, real_len + 1), generator=g)
    seqs[:, L - real_len:] = full[:, :-1]
    labels[:, L - real_len:] = full[:, 1:]
    return seqs, labels


def probe_overfit(opts):
    print(chr(10) + '=' * 78)
    print('PROBE overfit -- can perturbation injection stop memorisation of 8 sequences?')
    print('=' * 78)
    configs = [('B  no perturbation', dict(use_perturbation=False))]
    for th in opts.thetas:
        if th == 0.0:
            continue
        configs.append((f'C  theta={th}, w=1', dict(perturbation_probability=th)))
    configs += [
        ('C  theta=1.0, w=5', dict(perturbation_probability=1.0, perturbation_weight=5.0)),
        ('C  theta=1.0, U(-5,5), w=5', dict(perturbation_probability=1.0, perturbation_weight=5.0,
                                            perturbation_low=-5.0, perturbation_high=5.0)),
    ]
    rows, out = [], {}
    for name, over in configs:
        a = tiny_args(**over)
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        model = SLSRec(a)
        model.train()
        seqs, labels = tiny_batch(a)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
        first = None
        for _ in range(opts.steps):
            opt.zero_grad()
            l = slsrec_losses(a, model, seqs, labels, training=True)
            first = first or {k: v.item() for k, v in l.items()}
            l['L_total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(seqs)[0]
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                                 ignore_index=0).item()
            valid = labels != 0
            acc = (logits.argmax(-1)[valid] == labels[valid]).float().mean().item()
        out[name] = dict(final_ce=ce, train_acc=acc, first=first)
        rows.append([name, fmt(first['L_rec'], 3), fmt(ce, 4), fmt(acc, 3),
                     'MEMORISED' if acc > 0.9 else 'blocked'])
    print(table(rows, ['config', 'init L_rec', 'final CE', 'train acc', 'verdict']))
    hard = out.get('C  theta=1.0, w=5', {}).get('train_acc', 0)
    print(f'{chr(10)}VERDICT: ' + (
        'FALSIFIED -- even theta=1.0 with w_per=5 leaves memorisation untouched, so '
        'L_Per imposes no capacity cost and cannot be an anti-overfitting term.'
        if hard > 0.9 else 'not falsified -- extreme perturbation does block memorisation.'))
    return out


# --------------------------------------------------------------------------- #
# probe 6 -- compare finished runs (generalisation gap + best metric)
# --------------------------------------------------------------------------- #
def probe_compare(opts):
    """Read experiments/slsrec/<exp>/{train_losses,val_metrics}.jsonl for each run.

    The claim under test is that the perturbation configurations generalise better.
    The evidence would be a smaller train-vs-val CE gap and a higher best Recall@10;
    the falsifier is that B, C(theta=0) and C(theta=0.1) land inside seed noise.
    """
    print(chr(10) + '=' * 78)
    print('PROBE compare -- generalisation gap across finished runs')
    print('=' * 78)
    root = os.path.join('experiments', 'slsrec')
    exps = opts.exps or sorted(d for d in os.listdir(root)
                               if os.path.isdir(os.path.join(root, d))) if os.path.isdir(root) else []
    rows, out = [], {}
    for e in exps:
        d = os.path.join(root, e)
        tp, vp = os.path.join(d, 'train_losses.jsonl'), os.path.join(d, 'val_metrics.jsonl')
        if not (os.path.exists(tp) and os.path.exists(vp)):
            continue
        tr = [json.loads(l) for l in open(tp) if l.strip()]
        va = [json.loads(l) for l in open(vp) if l.strip()]
        va_ce = [v for v in va if 'CE' in v]
        if not tr or not va_ce:
            continue
        best = max(va_ce, key=lambda v: v.get('Recall@10', 0))
        # last validation at or before the end of training
        train_ce = tr[-1]['L_rec']
        gap = best['CE'] - train_ce
        out[e] = dict(train_L_rec=train_ce, val_CE=best['CE'], gap=gap,
                      best_recall10=best.get('Recall@10'), best_ndcg10=best.get('NDCG@10'),
                      at_iter=best['accum_iter'], epochs=tr[-1]['epoch'])
        rows.append([e, tr[-1]['epoch'], best['accum_iter'], fmt(train_ce), fmt(best['CE']),
                     f'{gap:+.4f}', fmt(best.get('Recall@10', float('nan'))),
                     fmt(best.get('NDCG@10', float('nan')))])
    if not rows:
        print('no run has both train_losses.jsonl and val_metrics.jsonl yet '
              '(val_metrics.jsonl is written by the patched trainer/base.py)')
        return {}
    print(table(rows, ['exp_name', 'epochs', 'best@iter', 'train L_rec', 'val CE',
                       'gap', 'Recall@10', 'NDCG@10']))
    print(chr(10) + 'Read this as: if the perturbation runs do not show a smaller gap AND a '
                    'higher Recall@10 than their matched no-perturbation control, eq. (17)/(18) '
                    'is not buying generalisation.')
    return out


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--probe', default='all',
                   choices=['all', 'gap', 'transparency', 'routing', 'gates', 'overfit',
                            'compare'])
    p.add_argument('--exps', type=str, nargs='*', default=None,
                   help='compare probe: experiment folder names under experiments/slsrec/')
    p.add_argument('--ckpt', default=DEFAULT_CKPT)
    p.add_argument('--warmup_steps_probe', type=int, default=0,
                   help='if no checkpoint: train this many steps first')
    p.add_argument('--n_batches', type=int, default=6, help='validation batches per measurement')
    p.add_argument('--thetas', type=float, nargs='+', default=[0.0, 0.05, 0.1, 0.25, 0.5, 1.0])
    p.add_argument('--ranges', type=str, default='-5:5,0:10',
                   help='extra U(low,high) ranges at theta=0.1, e.g. "-5:5,0:10"')
    p.add_argument('--gate_theta', type=float, default=0.5)
    p.add_argument('--steps', type=int, default=220, help='tiny-overfit steps')
    p.add_argument('--losses', type=str, nargs='*', default=None)
    p.add_argument('--probe_seed', type=int, default=42, dest='seed',
                   help='seed for the perturbation draws inside a probe')
    p.add_argument('--out', default='kill_tests.json')
    opts, _ = p.parse_known_args()
    opts.ranges = [tuple(float(x) for x in r.split(':')) for r in opts.ranges.split(',') if r]

    results = {}
    if opts.probe in ('all', 'gap'):
        results['gap'] = probe_gap(opts)
    if opts.probe in ('all', 'overfit'):
        results['overfit'] = probe_overfit(opts)
    if opts.probe in ('all', 'compare'):
        results['compare'] = probe_compare(opts)

    if opts.probe in ('all', 'transparency', 'routing', 'gates'):
        cfg_args.model_code = 'slsrec'
        set_template(cfg_args)
        args = cfg_args
        seed_everything(args.seed)
        train_loader, val_loader, _ = dataloader_factory(args)
        print(f'{chr(10)}device={args.device} items={args.num_items} max_len={args.bert_max_len} '
              f'val_batches={len(val_loader)}')
        model = load_model(args, opts.ckpt, opts.warmup_steps_probe, train_loader)
        if opts.probe in ('all', 'transparency'):
            results['transparency'] = probe_transparency(opts, args, model, val_loader)
        if opts.probe in ('all', 'routing'):
            results['routing'] = probe_routing(opts, args, model, val_loader)
        if opts.probe in ('all', 'gates'):
            results['gates'] = probe_gates(opts, args, model, val_loader)

    with open(opts.out, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f'{chr(10)}wrote {opts.out}')


if __name__ == '__main__':
    main()
