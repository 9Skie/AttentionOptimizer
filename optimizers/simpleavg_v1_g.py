#
# SimpleAvg V1-G: g_t INCLUDED in averaging window + EMA on both moments.
#
# g_t is part of the averaging window. Average is computed over all gradients.
# EMA is applied to both m and v. No separate mix parameter since all window
# members get equal weight.
#
#   window   = [g_t, g_t-1, g_t-2, ..., g_t-L]
#   attended = mean(window)
#   m_t = beta1 * m_t-1 + (1 - beta1) * attended
#   v_t = beta2 * v_t-1 + (1 - beta2) * attended^2
#

import torch
from torch.optim import Optimizer


class SimpleAvgV1G(Optimizer):
    """
    SimpleAvg V1-G: g_t in averaging window, EMA on both moments.

    Direct counterpart to AttnRaw V1-G: same window structure, but plain
    uniform average instead of cosine-similarity attention.

    Args:
        params:         model parameters
        lr:             learning rate
        betas:          (beta1, beta2) — EMA decay for first and second moment
        eps:            numerical stability term
        weight_decay:   decoupled weight decay
        context_length: number of past gradients to store (K)
                        Window includes g_t, total K+1 gradients
    """

    def __init__(
        self,
        params,
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        context_length=8,
    ):
        if context_length < 1:
            raise ValueError("context_length must be >= 1")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            context_length=context_length,
        )
        super().__init__(params, defaults)

    def _init_param_state(self, p: torch.Tensor):
        state = self.state[p]
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
        state["grad_history"] = []

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            K = group["context_length"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    self._init_param_state(p)

                g = p.grad.float()
                g_flat = g.flatten()

                state["step"] += 1
                t = state["step"]

                past = state["grad_history"][:K]
                window = torch.stack([g_flat] + past, dim=0)
                attended = window.mean(dim=0)
                attended_t = attended.reshape_as(p)

                m = state["exp_avg"]
                m.mul_(beta1).add_(attended_t, alpha=1.0 - beta1)
                m_hat = m / (1.0 - beta1**t)

                v = state["exp_avg_sq"]
                v.mul_(beta2).addcmul_(attended_t, attended_t, value=1.0 - beta2)
                v_hat = v / (1.0 - beta2**t)

                state["grad_history"] = (
                    [g_flat.detach().clone()] + state["grad_history"]
                )[:K]

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                p.addcdiv_(
                    m_hat.to(p.dtype),
                    v_hat.sqrt().add_(eps),
                    value=-lr,
                )

        return loss
