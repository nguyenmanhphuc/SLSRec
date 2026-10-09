"""Measure, on trained checkpoints, *why* perturbation (P) and uncertainty (U) act or not.

    python scripts/diagnose_pu.py experiments/slsrec/C_perturbation_ml-1m_seed42
    python scripts/diagnose_pu.py RUN_DIR [RUN_DIR ...] --batches 8 --json out.json
    python scripts/diagnose_pu.py experiments/slsrec/B_sam_gate_ml-1m_seed42 --force_p   # P probes on a model trained without P

Every probe runs on real training windows of the run's own dataset, with the
run's own resolved hyperparameters (run_config.json) and best checkpoint.

P probes (eval mode unless stated -- isolates the perturbation from dropout)
  input stats    the BD-LRU input x_t vs the U(0,1) replacement: can the
                 network tell them apart by range alone?
  gates          input gate i_t / recurrence gate r_t on replaced vs clean
                 steps: did the model learn to shut the noise out?
  dCE            per-position CE with perturbation minus without, at theta and
                 at theta = 0.5 / 1.0; plus the last position (what is evaluated)
  routing        dCE when the BD-LRU output H is zeroed in every layer: how much
                 the trained model depends on the branch that P corrupts
  grad cos       train mode: cos(grad L_rec, grad L_per) on the main model, next
                 to cos(grad L_rec, grad L_rec') for two dropout draws.  If the
                 two are equal, L_per is only another dropout sample of L_rec and
                 L_total ~ 2 L_rec -- a rescale AdamW is invariant to.

Item-noise probe (--item_noise, any model incl. recblr/lru; the paper's Fig. 4 test)
  replace each real input item of the validation sequences, with the given
  probability, by a uniformly random item and report Recall@10/@20.  P is meant
  to buy exactly this robustness, so a P-trained model should degrade less.

U probes
  log sigma^2    distribution, fraction pinned at the clamp (no gradient there)
  weight         alpha/sigma^2: mean, coefficient of variation, Spearman
                 correlation with the per-position CE (does U find hard samples?)
  grad           train mode: cos(grad L_rec, grad alpha*L_unc) and the norm
                 ratio on the main model; gradient norm reaching the U branch
"""

import argparse
import json
import math
import os
import sys
from argparse import Namespace
from contextlib import contextmanager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def _import_harness():
    saved = sys.argv
    try:
        sys.argv = [saved[0]]
        import config  # noqa: F401  (circular-import order of the harness)
        from dataloader import dataloader_factory
        from model import model_factory
        from trainer.slsrec import slsrec_losses, _ce_per_position
        from trainer.utils import absolute_recall_mrr_ndcg_for_ks
    finally:
        sys.argv = saved
    return config, dataloader_factory, model_factory, slsrec_losses, _ce_per_position, \
        absolute_recall_mrr_ndcg_for_ks


(config, dataloader_factory, model_factory, slsrec_losses, _ce_per_position,
 absolute_recall_mrr_ndcg_for_ks) = _import_harness()


# --------------------------------------------------------------------------- #
def load_run(run_dir, device):
    with open(os.path.join(run_dir, 'run_config.json')) as f:
        hp = json.load(f)['hparams']
    args = Namespace(**hp)
    args.num_workers, args.device = 0, device
    model = model_factory(args)
    ckpt = torch.load(os.path.join(run_dir, 'models', 'best_acc_model.pth'), map_location='cpu')
    model.load_state_dict(ckpt[config.STATE_DICT_KEY], strict=True)
    return args, model.to(device)


def loaders(args, seed):
    torch.manual_seed(seed)
    return dataloader_factory(args)


def train_batches(loader, device, n):
    out = []
    for seqs, labels in loader:
        out.append((seqs.to(device), labels.to(device)))
        if len(out) == n:
            break
    return out


def probe_item_noise(args, model, val_loader, ratios, seed=0):
    """Fig. 4 of the paper: random-item noise in the input sequences at evaluation."""
    model.eval()
    gen = torch.Generator(device='cpu').manual_seed(seed)
    res = {}
    with torch.no_grad():
        for rho in ratios:
            tot, n = {}, 0
            for seqs, labels in val_loader:
                noise = torch.randint(1, args.num_items + 1, seqs.shape, generator=gen)
                flip = (torch.rand(seqs.shape, generator=gen) < rho) & (seqs > 0)
                s = torch.where(flip, noise, seqs).to(args.device)
                scores = model(s)[0][:, -1, :]
                scores.scatter_(1, s, -1e9)
                scores[:, 0] = -1e9
                m = absolute_recall_mrr_ndcg_for_ks(scores, labels.view(-1).to(args.device), [10, 20])
                for k in ('Recall@10', 'Recall@20'):
                    tot[k] = tot.get(k, 0.0) + m[k] * seqs.size(0)
                n += seqs.size(0)
            res[rho] = {k: v / n for k, v in tot.items()}
    return res


def main_params(model):
    return [(n, p) for n, p in model.named_parameters() if 'uncertainty' not in n and p.requires_grad]


def flat_grad(loss, params):
    gs = torch.autograd.grad(loss, [p for _, p in params], retain_graph=True, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                      for g, (_, p) in zip(gs, params)])


def cos(a, b):
    return (a @ b / (a.norm() * b.norm()).clamp_min(1e-30)).item()


def masked_mean(x, valid):
    return ((x * valid).sum() / valid.sum().clamp_min(1)).item()


def spearman(a, b):
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return (ra @ rb / (ra.norm() * rb.norm()).clamp_min(1e-30)).item()


def bdlrus(model):
    return [b.sg_lru.bdlru for b in model.model.blocks if getattr(b.sg_lru, 'use_bdlru', False)]


@contextmanager
def theta(model, value):
    """Set the perturbation rate of every perturbation module, whatever the site."""
    mods = [(m, 'perturb_prob') for m in bdlrus(model)]
    mods += [(m, 'prob') for m in model.modules() if type(m).__name__ == 'TokenPerturbation']
    saved = [getattr(m, attr) for m, attr in mods]
    for m, attr in mods:
        setattr(m, attr, value)
    try:
        yield
    finally:
        for (m, attr), v in zip(mods, saved):
            setattr(m, attr, v)


@contextmanager
def record_bdlru(model):
    """Capture, per BD-LRU, its (clean) input, mask, and the perturbed input + replace mask."""
    rec = {i: {} for i in range(len(bdlrus(model)))}
    hooks, originals = [], []
    for i, m in enumerate(bdlrus(model)):
        def pre(mod, inp, i=i):
            rec[i]['x'], rec[i]['mask'] = inp[0].detach(), inp[1]
        hooks.append(m.register_forward_pre_hook(pre))
        orig = m._perturb

        def wrapped(x, mask, orig=orig, i=i):
            xb, rep = orig(x, mask)
            rec[i]['xbar'], rec[i]['replace'] = xb.detach(), rep
            return xb, rep
        originals.append((m, orig))
        m._perturb = wrapped
    try:
        yield rec
    finally:
        for h in hooks:
            h.remove()
        for m, orig in originals:
            m._perturb = orig


@contextmanager
def zero_h(model):
    hooks = [m.register_forward_hook(lambda mod, i, o: torch.zeros_like(o)) for m in bdlrus(model)]
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


def ce_pos(model, seqs, labels, perturb=False):
    return _ce_per_position(model(seqs, perturb=perturb)[0], labels)


# --------------------------------------------------------------------------- #
def probe_p(args, model, batches, draws=3):
    res = {}
    model.eval()
    trained_theta = args.perturbation_probability if args.use_perturbation else 0.0
    th = trained_theta if trained_theta > 0 else 0.1      # probe a P-free model at the paper's theta
    with torch.no_grad(), theta(model, th):
        # input statistics and gates, layer by layer (only where the BD-LRU itself perturbs)
        layers = []
        site = getattr(args, 'perturbation_site', 'bdlru')
        for seqs, labels in (batches[:2] if site == 'bdlru' else []):
            with record_bdlru(model) as rec:
                torch.manual_seed(0)
                model(seqs, perturb=True)
            for i, r in rec.items():
                m = bdlrus(model)[i]
                real = r['mask'].bool()
                x = r['x'][real]
                gate = lambda v: torch.sigmoid(m.gates(v)).chunk(2, dim=-1)  # noqa: E731
                rep = r['replace'] & real
                r_bar, i_bar = gate(r['xbar'][rep])
                r_cln, i_cln = gate(r['x'][rep])
                layers.append(dict(layer=i, x_mean=x.mean().item(), x_std=x.std().item(),
                                   x_chan_std=x.std(0).mean().item(),
                                   x_in_0_1=((x >= 0) & (x <= 1)).float().mean().item(),
                                   x_norm=x.norm(dim=-1).mean().item(),
                                   noise_norm=math.sqrt(m.d_model / 3.0),
                                   i_clean=i_cln.mean().item(), i_noise=i_bar.mean().item(),
                                   r_clean=r_cln.mean().item(), r_noise=r_bar.mean().item()))
        agg = {}
        for row in layers:
            agg.setdefault(row['layer'], []).append(row)
        res['layers'] = [{k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
                         for _, rows in sorted(agg.items())]
        res['site'] = site
        res['gates_see'] = getattr(args, 'perturbation_gates', 'perturbed')

        # dCE at theta, 0.5, 1.0; routing
        def dce(fn):
            tot = last = 0.0
            for seqs, labels in batches:
                valid = (labels != 0).float()
                clean = ce_pos(model, seqs, labels)
                d = fn(seqs, labels) - clean
                tot += masked_mean(d, valid)
                last += d[:, -1].mean().item()
            return tot / len(batches), last / len(batches)

        def perturbed(value):
            def fn(seqs, labels):
                with theta(model, value):
                    acc = 0
                    for k in range(draws):
                        torch.manual_seed(100 + k)
                        acc = acc + ce_pos(model, seqs, labels, perturb=True)
                    return acc / draws
            return fn

        res['clean_ce'] = sum(masked_mean(ce_pos(model, s, l), (l != 0).float()) for s, l in batches) / len(batches)
        for value in sorted({th, 0.5, 1.0}):
            res['dce_theta_%g' % value] = dce(perturbed(value))

        def without_h(seqs, labels):
            with zero_h(model):
                return ce_pos(model, seqs, labels)
        res['dce_zero_H'] = dce(without_h)

        # last-position KL and top-10 overlap at theta
        kl = overlap = 0.0
        for seqs, labels in batches:
            clean = model(seqs)[0][:, -1]
            torch.manual_seed(7)
            dirty = model(seqs, perturb=True)[0][:, -1]
            kl += F.kl_div(F.log_softmax(dirty, -1), F.log_softmax(clean, -1),
                           log_target=True, reduction='batchmean').item()
            a, b = clean.topk(10).indices, dirty.topk(10).indices
            overlap += (a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).float().mean().item()
        res['last_kl'] = kl / len(batches)
        res['last_top10_overlap'] = overlap / len(batches)

    # gradient geometry, train mode (what the optimiser sees)
    model.train()
    params = main_params(model)
    c_per, c_drop, ratio = [], [], []
    with theta(model, th):
        for seqs, labels in batches[:4]:
            valid = (labels != 0).float()

            def l_rec(seed, perturb=False):
                torch.manual_seed(seed)
                ce = ce_pos(model, seqs, labels, perturb=perturb)
                return (ce * valid).sum() / valid.sum()
            g_rec = flat_grad(l_rec(1), params)
            g_rec2 = flat_grad(l_rec(2), params)
            g_per = flat_grad(l_rec(3, perturb=True), params)
            c_per.append(cos(g_rec, g_per))
            c_drop.append(cos(g_rec, g_rec2))
            ratio.append((g_per - g_rec).norm().item() / g_rec.norm().item())
    res['grad_cos_rec_per'] = sum(c_per) / len(c_per)
    res['grad_cos_rec_rec_other_dropout'] = sum(c_drop) / len(c_drop)
    res['grad_rel_diff_per'] = sum(ratio) / len(ratio)
    model.eval()
    return res


def probe_u(args, model, batches):
    res = {}
    clamp = args.uncertainty_logvar_clamp
    lvs, ws, ces = [], [], []
    model.eval()
    with torch.no_grad():
        for seqs, labels in batches:
            valid = labels != 0
            lv = model.log_variance(seqs, labels)[valid]
            ce = ce_pos(model, seqs, labels)[valid]
            lvs.append(lv)
            ces.append(ce)
            ws.append(args.uncertainty_alpha * torch.exp(-lv))
    lv, w, ce = torch.cat(lvs), torch.cat(ws), torch.cat(ces)
    q = torch.quantile(lv.float()[torch.randperm(lv.numel(), device=lv.device)[:100000]],
                       torch.tensor([0.05, 0.5, 0.95], device=lv.device))
    res['logvar_mean'], res['logvar_std'] = lv.mean().item(), lv.std().item()
    res['logvar_p05_p50_p95'] = [round(v, 4) for v in q.tolist()]
    res['frac_at_clamp'] = ((lv.abs() >= clamp - 1e-3).float().mean().item()
                            if clamp and clamp > 0 else 0.0)
    res['weight_mean'] = w.mean().item()
    res['weight_cv'] = (w.std() / w.mean()).item()
    res['spearman_weight_vs_ce'] = spearman(w, ce)

    model.train()
    params = main_params(model)
    unc_params = [(n, p) for n, p in model.named_parameters() if 'uncertainty' in n or 'global_logvar' in n]
    cs, ratios, unc_norm = [], [], []
    for seqs, labels in batches[:4]:
        torch.manual_seed(1)
        l = slsrec_losses(args, model, seqs, labels, training=True)
        g_rec = flat_grad(l['L_rec'], params)
        g_unc = flat_grad(args.uncertainty_alpha * l['L_uncertainty'], params)
        cs.append(cos(g_rec, g_unc))
        ratios.append(g_unc.norm().item() / g_rec.norm().item())
        unc_norm.append(flat_grad(args.uncertainty_alpha * l['L_uncertainty'], unc_params).norm().item())
    res['grad_cos_rec_unc'] = sum(cs) / len(cs)
    res['grad_norm_ratio_unc_over_rec'] = sum(ratios) / len(ratios)
    res['grad_norm_into_U_branch'] = sum(unc_norm) / len(unc_norm)
    model.eval()
    return res


def fmt(v):
    if isinstance(v, (list, tuple)):
        return '(' + ', '.join(fmt(x) for x in v) + ')'
    if isinstance(v, float):
        return '%.4g' % v
    return str(v)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='+')
    p.add_argument('--batches', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--force_p', action='store_true', help='run the P probes even if the run had P off')
    p.add_argument('--item_noise', nargs='*', type=float, default=None,
                   help='validation item-noise ratios for the Fig. 4 probe, e.g. 0 0.1 0.2')
    p.add_argument('--json', default=None)
    a = p.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    report = {}
    data_cache = {}
    for run in a.runs:
        args, model = load_run(run, device)
        key = (args.dataset_code, args.bert_max_len)
        if key not in data_cache:
            train_loader, val_loader, _ = loaders(args, a.seed)
            data_cache[key] = (train_batches(train_loader, device, a.batches), val_loader)
        batches, val_loader = data_cache[key]
        name = os.path.basename(os.path.normpath(run))
        is_slsrec = args.model_code == 'slsrec'
        out = {'P': None, 'U': None, 'item_noise': None}
        if is_slsrec and (args.use_perturbation or a.force_p) and bdlrus(model):
            out['P'] = probe_p(args, model, batches)
        if is_slsrec and args.use_uncertainty:
            out['U'] = probe_u(args, model, batches)
        if a.item_noise:
            out['item_noise'] = probe_item_noise(args, model, val_loader, a.item_noise)
        report[name] = out

        print('\n=== %s  (%s, P=%s, U=%s%s)' % (
            name, args.model_code, getattr(args, 'use_perturbation', None) if is_slsrec else '-',
            args.use_uncertainty if is_slsrec else '-',
            ', ' + args.uncertainty_loss if is_slsrec and args.use_uncertainty else ''))
        if out['item_noise']:
            base = out['item_noise'][min(out['item_noise'])]
            print('  item noise (val): ' + '   '.join(
                'rho %.2f R@10 %.4f R@20 %.4f (%+.1f%%)' % (
                    rho, m['Recall@10'], m['Recall@20'], 100 * (m['Recall@20'] / base['Recall@20'] - 1))
                for rho, m in sorted(out['item_noise'].items())))
        if out['P']:
            r = out['P']
            print('  P  site %s, gates see %s, clean CE %.4f' % (r['site'], r['gates_see'], r['clean_ce']))
            if r['gates_see'] == 'clean':
                print('     (gates use the clean x_t: the "noise" gate values below are what they would be, not what is used)')
            for L in r['layers']:
                print('     layer %d: x mean %.3f std %.3f (per-channel std %.3f), |x| %.2f vs |U(0,1)| %.2f, '
                      'x in [0,1] %.0f%% | i_t clean %.3f -> noise %.3f, r_t clean %.3f -> noise %.3f'
                      % (L['layer'], L['x_mean'], L['x_std'], L['x_chan_std'], L['x_norm'], L['noise_norm'],
                         100 * L['x_in_0_1'], L['i_clean'], L['i_noise'], L['r_clean'], L['r_noise']))
            for k in sorted(k for k in r if k.startswith('dce_')):
                print('     %-22s all positions %+.4f   last position %+.4f' % (k, r[k][0], r[k][1]))
            print('     last position: KL(clean||perturbed) %.4f, top-10 overlap %.1f%%'
                  % (r['last_kl'], 100 * r['last_top10_overlap']))
            print('     grad cos(L_rec, L_per) %.4f   vs cos(L_rec, L_rec other dropout) %.4f   '
                  '|g_per - g_rec| / |g_rec| %.3f'
                  % (r['grad_cos_rec_per'], r['grad_cos_rec_rec_other_dropout'], r['grad_rel_diff_per']))
        if out['U']:
            r = out['U']
            print('  U  log sigma^2 mean %.3f std %.3f p05/p50/p95 %s, at clamp %.1f%%'
                  % (r['logvar_mean'], r['logvar_std'], fmt(r['logvar_p05_p50_p95']), 100 * r['frac_at_clamp']))
            print('     alpha/sigma^2 mean %.5f, CV %.3f, Spearman(weight, CE) %+.3f'
                  % (r['weight_mean'], r['weight_cv'], r['spearman_weight_vs_ce']))
            print('     grad cos(L_rec, a*L_unc) %.4f, |g_unc|/|g_rec| %.4f, grad into U branch %.3g'
                  % (r['grad_cos_rec_unc'], r['grad_norm_ratio_unc_over_rec'], r['grad_norm_into_U_branch']))
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()

    if a.json:
        with open(a.json, 'w') as f:
            json.dump(report, f, indent=2)
        print('\nwrote', a.json)
    return 0


if __name__ == '__main__':
    sys.exit(main())
