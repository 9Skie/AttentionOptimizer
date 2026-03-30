#
# AttnRaw V3: Past-only attention + current residual + EMA on top.
#
# Current gradient g_t is the query only. Attention retrieves from past
# gradients, then a residual current-gradient path is added before the Adam-style
# EMA:
#
#   a_t = cosine-attention(query=g_t, values=[g_{t-1}, ..., g_{t-L}])
#   u_t = (1 - lambda) * g_t + lambda * a_t
#   m_t = beta1 * m_{t-1} + (1 - beta1) * u_t
#   v_t = beta2 * v_{t-1} + (1 - beta2) * u_t^2

import torch
from torch.optim import Optimizer


class AttnRawV3(Optimizer):
    """
    AttnRaw V3: past-only cosine attention with a current-gradient residual,
    followed by Adam-style EMA.

    Args:
        params:         model parameters
        lr:             learning rate
        betas:          (beta1, beta2) — EMA decay for first and second moment
        eps:            numerical stability term
        weight_decay:   decoupled weight decay
        context_length: number of past gradients to attend over
        mix_beta:       lambda weight on attended past gradients
    """

    def __init__(
        self,
        params,
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        context_length=8,
        mix_beta=0.9,
    ):
        if context_length < 1:
            raise ValueError("context_length must be >= 1")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            context_length=context_length,
            mix_beta=mix_beta,
        )
        super().__init__(params, defaults)

    def _init_param_state(self, p: torch.Tensor):
        state = self.state[p]
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
        state["grad_history"] = []

    def _compute_past_mix(self, g_flat: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """Cosine-attention weighted sum over past-only history."""
        norms = history.norm(dim=1).clamp(min=1e-8)
        query_norm = g_flat.norm().clamp(min=1e-8)
        scores = history @ g_flat / (norms * query_norm)
        alpha = torch.softmax(scores, dim=0)
        return alpha @ history

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
            mix_beta = group["mix_beta"]

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
                if past:
                    history = torch.stack(past, dim=0)
                    a_t_flat = self._compute_past_mix(g_flat, history)
                    u_t_flat = (1.0 - mix_beta) * g_flat + mix_beta * a_t_flat
                else:
                    u_t_flat = g_flat

                u_t = u_t_flat.reshape_as(p)

                m = state["exp_avg"]
                m.mul_(beta1).add_(u_t, alpha=1.0 - beta1)
                m_hat = m / (1.0 - beta1 ** t)

                v = state["exp_avg_sq"]
                v.mul_(beta2).addcmul_(u_t, u_t, value=1.0 - beta2)
                v_hat = v / (1.0 - beta2 ** t)

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
