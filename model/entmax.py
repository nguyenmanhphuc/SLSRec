"""alpha-entmax (Peters, Niculae & Martins, 2019) for the SAM -- a research extension.

alpha-entmax(z) = argmax_{p in simplex} p.z + H_alpha(p), with Tsallis entropy H_alpha.
alpha = 1 is softmax, alpha = 2 is sparsemax; for alpha > 1 small scores get exactly
zero weight.  SAFA (Cao et al.) uses it with a per-sequence learned alpha in (1, 2).

The solution is p_i = [(alpha-1) z_i - tau]_+ ^ (1/(alpha-1)) with tau set so that
sum p = 1; tau is found by bisection.  The backward pass is the closed form of
Peters et al. (2019), including the gradient with respect to alpha, so alpha can be
learned.  Masked entries may be -inf: they get p = 0 and no gradient.
"""

import torch


def _p(x, alpha):
    return torch.clamp(x, min=0) ** (1.0 / (alpha - 1.0))


class EntmaxBisectFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, alpha, n_iter=50):
        # alpha: tensor broadcastable to X with size 1 along the last dim
        alpha = alpha.to(X.dtype)
        d = X.shape[-1]
        Xs = X * (alpha - 1.0)
        max_val = Xs.max(dim=-1, keepdim=True).values
        tau_lo = max_val - 1.0                                  # sum p >= 1 here
        tau_hi = max_val - (1.0 / d) ** (alpha - 1.0)           # sum p <= 1 here
        f_lo = _p(Xs - tau_lo, alpha).sum(-1, keepdim=True) - 1.0
        dm = tau_hi - tau_lo
        p_m = None
        for _ in range(n_iter):
            dm = dm / 2
            tau_m = tau_lo + dm
            p_m = _p(Xs - tau_m, alpha)
            f_m = p_m.sum(-1, keepdim=True) - 1.0
            tau_lo = torch.where(f_m * f_lo >= 0, tau_m, tau_lo)
        p_m = p_m / p_m.sum(-1, keepdim=True)
        ctx.save_for_backward(p_m, alpha)
        return p_m

    @staticmethod
    def backward(ctx, dY):
        Y, alpha = ctx.saved_tensors
        pos = Y > 0
        gppr = torch.where(pos, Y.clamp_min(1e-30) ** (2.0 - alpha), torch.zeros_like(Y))
        dX = dY * gppr
        q = dX.sum(-1, keepdim=True) / gppr.sum(-1, keepdim=True)
        dX = dX - q * gppr

        d_alpha = None
        if ctx.needs_input_grad[1]:
            S = torch.where(pos, Y * torch.log(Y.clamp_min(1e-30)), torch.zeros_like(Y))
            ent = S.sum(-1, keepdim=True)
            Y_sk = gppr / gppr.sum(-1, keepdim=True)
            d_alpha = dY * (Y - Y_sk) / ((alpha - 1.0) ** 2)
            d_alpha = d_alpha - dY * (S - Y_sk * ent) / (alpha - 1.0)
            d_alpha = d_alpha.sum(-1, keepdim=True)
            # reduce to alpha's own shape (it was broadcast against X)
            while d_alpha.dim() > alpha.dim():
                d_alpha = d_alpha.sum(0)
            for i, s in enumerate(alpha.shape):
                if s == 1 and d_alpha.shape[i] != 1:
                    d_alpha = d_alpha.sum(i, keepdim=True)
        return dX, d_alpha, None


def entmax_bisect(X, alpha, n_iter=50):
    """alpha-entmax over the last dim.  alpha: float or tensor broadcastable to X
    with size 1 in the last dim; values must be in (1, 2]."""
    if not torch.is_tensor(alpha):
        alpha = torch.full((1,) * X.dim(), float(alpha), device=X.device, dtype=X.dtype)
    if alpha.dim() != X.dim() or alpha.shape[-1] != 1:
        raise ValueError('alpha must have X.dim() dims and size 1 in the last one')
    return EntmaxBisectFunction.apply(X, alpha, n_iter)
