"""KAPTRec C3: context-guided adaptive perturbation + history-aware uncertainty.

Research extension on top of SLSRec (not in the SLSRec paper).  Training only:
inference always uses one clean encoder pass.

Adaptive perturbation (slides 10-12).  From the clean pass, per position t:

    f_t   = stopgrad([e_y(t), Z_t, 1 - cos(G_t, H_t), 1 - cos(Z_{t-1}, Z_t), t / L])
    pi_t  = pi_max * sigmoid(MLP_pi(f_t))                 where to perturb
    m_t   ~ Bernoulli(pi_t)          (straight-through Gumbel-sigmoid)
    eps_t = eps_min + (eps_max - eps_min) * sigmoid(MLP_eps(f_t))   how much
    d_ctx = (a_t - x_t) / ||a_t - x_t||,   a_t = W_c stopgrad(Z_t)
    d_rnd = sign(x_t) (*) xi / ||xi||,     xi ~ N(0, I)
    d_t   = normalize((1 - rho) d_ctx + rho d_rnd)        which direction
    x~_t  = x_t + m_t * eps_t * ||x_t|| * d_t             on the input embeddings

eps is relative to ||x_t|| so the bound does not depend on the embedding scale.

Two points the slides leave open, and how they are resolved here:

* What trains the controller.  If pi, eps and W_c simply minimise the same loss as
  the encoder, the cheapest solution is a harmless perturbation (pi -> 0, eps ->
  eps_min, d -> a direction the encoder ignores), the same collapse that makes
  SLSRec's U inert.  --kap_controller=adv (default) trains them through a
  gradient-reversal layer, so they *maximise* the perturbed loss and the encoder
  minimises it (adversarial augmentation); a budget term keeps mean(pi) at
  --kap_pi_target.  --kap_controller=min is the naive reading, kept for comparison.
* The uncertainty loss.  L_unc = CE_u / (2 sigma_u^2) + log sigma_u on the last
  position, the heteroscedastic form with the log term (beta > 0), which unlike
  SLSRec eq. (24) has a finite optimum.  omega_u = stopgrad(sigma_u^-2), normalised
  to mean 1 over the batch, down-weights the consistency term of uncertain users.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -g


def grad_reverse(x):
    return _GradReverse.apply(x)


def _cos_dissimilarity(a, b):
    return 1.0 - F.cosine_similarity(a, b, dim=-1, eps=1e-8)


def position_features(z, g, h, label_emb, mask):
    """f_t = [e_y(t), Z_t, 1-cos(G_t,H_t), 1-cos(Z_{t-1},Z_t), t/L], detached.  [B, L, 2D+3]."""
    B, L, _ = z.shape
    m = mask.unsqueeze(-1).to(z.dtype)
    zp = F.pad((z * m)[:, :-1], (0, 0, 1, 0))
    trans = _cos_dissimilarity(z, zp).unsqueeze(-1) * m
    disagree = _cos_dissimilarity(g, h).unsqueeze(-1) * m
    # position counted from the first real item, normalised by the window length
    pos = (torch.cumsum(mask.long(), dim=1).to(z.dtype) / L).unsqueeze(-1) * m
    return torch.cat([label_emb, z, disagree, trans, pos], dim=-1).detach()


class AdaptivePerturbation(nn.Module):
    def __init__(self, D, pi_max=0.3, eps_min=0.05, eps_max=0.5, rho=0.5, tau=0.5, controller='adv'):
        super().__init__()
        if controller not in ('adv', 'min'):
            raise ValueError(f'unknown kap_controller: {controller}')
        fdim = 2 * D + 3
        self.pi_mlp = nn.Sequential(nn.Linear(fdim, D), nn.GELU(), nn.Linear(D, 1))
        self.eps_mlp = nn.Sequential(nn.Linear(fdim, D), nn.GELU(), nn.Linear(D, 1))
        self.context = nn.Linear(D, D)                      # W_c
        self.pi_max, self.eps_min, self.eps_max = pi_max, eps_min, eps_max
        self.rho, self.tau, self.controller = rho, tau, controller

    def _st_bernoulli(self, pi):
        """Straight-through Gumbel-sigmoid: hard 0/1 forward, soft gradient backward."""
        pi = pi.clamp(1e-6, 1 - 1e-6)
        u1, u2 = torch.rand_like(pi).clamp_min(1e-10), torch.rand_like(pi).clamp_min(1e-10)
        gumbel = -torch.log(-torch.log(u1)) + torch.log(-torch.log(u2))
        soft = torch.sigmoid((torch.log(pi) - torch.log1p(-pi) + gumbel) / self.tau)
        hard = (soft > 0.5).to(pi.dtype)
        return hard + (soft - soft.detach())     # (s - s) == 0 exactly, so m is 0/1 forward

    def forward(self, x, z, g, h, label_emb, mask):
        """x: clean input embeddings [B, L, D].  Returns (delta, pi) with delta to add to x."""
        m = mask.unsqueeze(-1).to(x.dtype)
        f = position_features(z, g, h, label_emb, mask)
        pi = self.pi_max * torch.sigmoid(self.pi_mlp(f))                       # [B, L, 1]
        eps = self.eps_min + (self.eps_max - self.eps_min) * torch.sigmoid(self.eps_mlp(f))
        a = self.context(z.detach())
        if self.controller == 'adv':
            pi_used, eps_used, a_used = grad_reverse(pi), grad_reverse(eps), grad_reverse(a)
        else:
            pi_used, eps_used, a_used = pi, eps, a
        xd = x.detach()
        d_ctx = (a_used - xd) / ((a_used - xd).norm(dim=-1, keepdim=True) + 1e-8)
        xi = torch.randn_like(xd)
        d_rnd = torch.sign(xd) * xi / (xi.norm(dim=-1, keepdim=True) + 1e-8)
        d = (1 - self.rho) * d_ctx + self.rho * d_rnd
        d = d / (d.norm(dim=-1, keepdim=True) + 1e-8)
        mt = self._st_bernoulli(pi_used)
        delta = mt * eps_used * xd.norm(dim=-1, keepdim=True) * d * m
        return delta, pi


class HistoryUncertainty(nn.Module):
    """sigma_u = softplus(MLP[c_u || Z_L || e_y || dbar_u || rbar_u]) (slide 7), with
    alpha_t = softmax_t(v' tanh(W_z Z_t + W_y e_y)) and c_u = sum_t alpha_t Z_t.
    Features are detached: this head reads the encoder, it does not reshape it."""

    def __init__(self, D, d):
        super().__init__()
        self.w_z, self.w_y = nn.Linear(D, D), nn.Linear(D, D, bias=False)
        self.v = nn.Linear(D, 1, bias=False)
        self.mlp = nn.Sequential(nn.Linear(3 * D + 2, D), nn.GELU(), nn.Linear(D, 1))

    def forward(self, z, g, h, e_y, mask):
        z, g, h, e_y = z.detach(), g.detach(), h.detach(), e_y.detach()
        score = self.v(torch.tanh(self.w_z(z) + self.w_y(e_y).unsqueeze(1))).squeeze(-1)
        # rows with no real item (a one-item user predicting its first item) get alpha = 0
        # everywhere instead of softmax(-inf, ..., -inf) = NaN
        has = mask.bool().any(dim=1, keepdim=True)
        score = score.masked_fill(~mask.bool(), float('-inf')).masked_fill(~has, 0.0)
        alpha = torch.softmax(score, dim=-1) * has.to(z.dtype)                           # [B, L]
        c = (alpha.unsqueeze(-1) * z).sum(1)
        dbar = (alpha * _cos_dissimilarity(g, h)).sum(1, keepdim=True)
        zp = F.pad(z[:, :-1], (0, 0, 1, 0))
        has_prev = F.pad(mask[:, :-1], (1, 0)).bool() & mask.bool()
        rbar = (alpha * _cos_dissimilarity(z, zp) * has_prev.to(z.dtype)).sum(1, keepdim=True)
        return F.softplus(self.mlp(torch.cat([c, z[:, -1], e_y, dbar, rbar], dim=-1))).squeeze(-1) + 1e-3
