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
    return {'L_rec': l_rec, 'L_perturb': l_per, 'L_uncertainty': l_unc, 'L_total': l_total}


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
        B, L = seqs.shape
        for i in range(L):
            scores[torch.arange(scores.size(0)), seqs[:, i]] = -1e9   # drop already-seen items
        scores[:, 0] = -1e9                                            # drop the padding item
        metrics = absolute_recall_mrr_ndcg_for_ks(scores, labels.view(-1), self.metric_ks)
        metrics['CE'] = val_ce
        return metrics
