"""SLSRec trainer.

Implements the SLSRec objective on top of the unchanged LRURec harness
(per-position next-item cross entropy, full-sort evaluation, Recall@10 model
selection).

    L_Rec  = -Log(y_ui)                                             eq. (16)
    L_Per  = -Log(ybar_ui)      (perturbed recurrence)              eq. (18)
    L_Unc  = (1/sigma_u^2) L_Rec + beta Log(sigma_u)                eq. (22)
    L_tot  = L_Rec + w_per * L_Per + alpha * L_Unc
           = (1 + alpha/sigma_u^2) L_Rec + L_Per                    eq. (23)/(24)

With the paper's defaults (beta = 0, w_per = 1) the two forms coincide, i.e.
``L_tot`` is exactly eq. (24).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseTrainer
from .utils import absolute_recall_mrr_ndcg_for_ks

_CE_NONE = nn.CrossEntropyLoss(ignore_index=0, reduction='none')


def _masked_mean(per_pos, valid):
    return (per_pos * valid).sum() / valid.sum().clamp_min(1)


def _ce_per_position(logits, labels):
    return _CE_NONE(logits.reshape(-1, logits.size(-1)), labels.reshape(-1)).view_as(labels)


def slsrec_losses(args, model, seqs, labels, training=True):
    """Compute L_rec / L_perturb / L_uncertainty / L_total for one batch.

    Exposed as a free function so it can be unit-tested without building a
    full trainer (dataloaders, loggers, checkpoint directory).
    """
    valid = (labels != 0).float()

    logits = model(seqs)[0]                                              # eq. (15)
    clean = model.model.clean_pass_state() if getattr(args, 'kap_perturb', False) else None
    rec_per_pos = _ce_per_position(logits, labels)                       # eq. (16)
    l_rec = _masked_mean(rec_per_pos, valid)

    # ---- perturbation injection, eq. (17)/(18) --------------------------- #
    if args.use_perturbation and training:
        logits_p = model(seqs, perturb=True)[0]
        l_per = _masked_mean(_ce_per_position(logits_p, labels), valid)
    else:
        l_per = torch.zeros((), device=logits.device)

    # ---- Gibbs uncertainty scaling, eq. (19)-(22) ------------------------ #
    if args.use_uncertainty:
        logvar = model.log_variance(seqs, labels)                        # eq. (20)
        inv_var = torch.exp(-logvar)                                     # 1 / sigma_u^2
        if getattr(args, 'uncertainty_weight_norm', 'none') == 'batch':
            # mean weight pinned to 1: sigma can only redistribute, never vanish
            inv_var = inv_var / _masked_mean(inv_var, valid).clamp_min(1e-12)
        if args.uncertainty_loss == 'approx':                            # eq. (22)
            unc_per_pos = inv_var * rec_per_pos
        elif args.uncertainty_loss == 'exact':                           # eq. (19)/(21)
            unc_per_pos = _ce_per_position(logits * inv_var.unsqueeze(-1), labels)
        else:
            raise ValueError(f'unknown uncertainty_loss: {args.uncertainty_loss}')
        # beta * Log(sigma_u) = beta/2 * Log(sigma_u^2); the paper sets beta = 0
        unc_per_pos = unc_per_pos + args.uncertainty_beta * 0.5 * logvar
        l_unc = _masked_mean(unc_per_pos, valid)
    else:
        l_unc = torch.zeros((), device=logits.device)

    l_total = l_rec + args.perturbation_weight * l_per + args.uncertainty_alpha * l_unc  # eq. (23)
    out = {'L_rec': l_rec, 'L_perturb': l_per, 'L_uncertainty': l_unc}
    if training and getattr(args, 'kap_perturb', False):
        kap_losses = _kaptrec_losses(args, model, seqs, labels, logits, rec_per_pos, valid, clean)
        l_total = l_total + kap_losses.pop('kap_total')
        out.update(kap_losses)
    out['L_total'] = l_total
    return out


def _kaptrec_losses(args, model, seqs, labels, logits, rec_per_pos, valid, clean):
    """KAPTRec C3 (extension):  lambda_p L_rec^pert + lambda_c omega_u JS(p_clean || p_pert)
    + alpha_u L_unc + budget.  ``clean`` is the state of the clean pass slsrec_losses ran."""
    inner = model.model
    mask = seqs > 0
    z, g, h = clean['z'], clean['g'], clean['h']
    label_emb = inner.embedding.token(labels)
    delta, pi = inner.kap(clean['x0'], z, g, h, label_emb, mask)

    logits_p = inner.logits_from_hidden(inner.represent(seqs, emb_delta=delta))
    l_pert = _masked_mean(_ce_per_position(logits_p, labels), valid)

    # per-sequence weights omega_u from the history-aware uncertainty (1 if off)
    B = seqs.size(0)
    omega = torch.ones(B, device=seqs.device)
    l_unc = torch.zeros((), device=seqs.device)
    if inner.kap_uncertainty is not None:
        e_y = inner.embedding.token(labels[:, -1])          # target of the last (real) position
        sigma = inner.kap_uncertainty(z, g, h, e_y, mask)
        ce_last = rec_per_pos[:, -1]
        l_unc = (ce_last / (2 * sigma ** 2) + torch.log(sigma)).mean()
        w = 1.0 / sigma.detach() ** 2
        omega = w / w.mean()

    # Jensen-Shannon between the clean and perturbed next-item distributions, valid rows only
    rows = valid.bool()
    lp = F.log_softmax(logits[rows], dim=-1)
    lq = F.log_softmax(logits_p[rows], dim=-1)
    lm = torch.logsumexp(torch.stack([lp, lq]), dim=0) - torch.log(torch.tensor(2.0, device=lp.device))
    js = 0.5 * ((lp.exp() * (lp - lm)).sum(-1) + (lq.exp() * (lq - lm)).sum(-1))
    w_rows = omega.unsqueeze(1).expand_as(valid)[rows]
    l_js = (w_rows * js).mean()

    pi_mean = (pi.squeeze(-1) * mask).sum() / mask.sum().clamp_min(1)
    l_budget = (pi_mean - getattr(args, 'kap_pi_target', 0.1)) ** 2
    total = (getattr(args, 'kap_lambda_p', 1.0) * l_pert + getattr(args, 'kap_lambda_c', 1.0) * l_js
             + getattr(args, 'kap_unc_alpha', 0.1) * l_unc + getattr(args, 'kap_budget_weight', 10.0) * l_budget)
    return {'L_kap_pert': l_pert, 'L_kap_js': l_js, 'L_kap_unc': l_unc, 'L_kap_budget': l_budget,
            'kap_pi_mean': pi_mean.detach(), 'kap_total': total}


class SLSRecTrainer(BaseTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root, use_wandb):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root, use_wandb)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.loss_parts = {}

    def calculate_loss(self, batch):
        seqs, labels = batch
        losses = slsrec_losses(self.args, self.model, seqs, labels, training=self.model.training)
        self.loss_parts = {k: v.item() for k, v in losses.items()}
        return losses['L_total']

    def calculate_metrics(self, batch):
        seqs, labels = batch
        scores = self.model(seqs)[0][:, -1, :]                # last position -> eq. (15)
        # Held-out cross entropy on the *unmasked* logits, directly comparable to the
        # training L_rec: train L_rec - val CE is the generalisation gap that the
        # perturbation term (eq. 17/18) is supposed to close.
        val_ce = F.cross_entropy(scores, labels.view(-1)).item()
        # drop already-seen items: one scatter instead of a 200-step loop of
        # indexed writes (each of which also copied a CPU arange to the GPU)
        scores.scatter_(1, seqs, -1e9)
        scores[:, 0] = -1e9                                            # drop the padding item
        metrics = absolute_recall_mrr_ndcg_for_ks(scores, labels.view(-1), self.metric_ks)
        metrics['CE'] = val_ce
        return metrics
