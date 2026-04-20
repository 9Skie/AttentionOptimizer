#
# AttnRaw V1: Past-only attention + EMA on both moments.
#
# g_t is NOT in the attention window. Attention is computed over past gradients.
# EMA is applied to both m and v.
#
#   attended = attention(g_t, [g_t-1, g_t-2, ..., g_t-L])
#   blended = mix * g_t + (1 - mix) * attended
#   m_t = beta1 * m_t-1 + (1 - beta1) * blended
#   v_t = beta2 * v_t-1 + (1 - beta2) * blended^2
#

import torch
from torch.optim import Optimizer


class AttnRawV1(Optimizer):
    """
    AttnRaw V1: past-only cosine attention, EMA on both moments.

    Args:
        params:         model parameters
        lr:             learning rate
        betas:          (beta1, beta2) — EMA decay for first and second moment
        eps:            numerical stability term
        weight_decay:   decoupled weight decay
        context_length: number of past gradients to store (K)
        temperature:    softmax temperature for attention scores
        mix:            blend g_t with attended (0.0 = all past, 1.0 = all current)
                       e.g., mix=0.1 means 10% g_t, 90% attended
    """

    def __init__(
        self,
        params,
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        context_length=8,
        temperature=1.0,
        mix=0.9,
    ):
        if context_length < 1:
            raise ValueError("context_length must be >= 1")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= mix <= 1.0:
            raise ValueError("mix must be in [0, 1]")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            context_length=context_length,
            temperature=temperature,
            mix=mix,
        )
        super().__init__(params, defaults)

    def _init_param_state(self, p: torch.Tensor):
        state = self.state[p]
        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
        state["grad_history"] = []

    def _compute_attention(
        self,
        g_flat: torch.Tensor,
        history: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Cosine attention over past gradients (g_t NOT included)."""
        g_norm = g_flat.norm().clamp(min=1e-8)
        history_norms = history.norm(dim=1).clamp(min=1e-8)
        scores = history @ g_flat
        scores = scores / (history_norms * g_norm)
        alpha = torch.softmax(scores / temperature, dim=0)
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
            temperature = group["temperature"]

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

                mix = group["mix"]

                past = state["grad_history"][:K]
                if past:
                    history = torch.stack(past, dim=0)
                    attended = self._compute_attention(g_flat, history, temperature)
                    blended = mix * g_flat + (1.0 - mix) * attended
                else:
                    attended = g_flat
                    blended = g_flat

                blended_t = blended.reshape_as(p)

                m = state["exp_avg"]
                m.mul_(beta1).add_(blended_t, alpha=1.0 - beta1)
                m_hat = m / (1.0 - beta1**t)

                v = state["exp_avg_sq"]
                v.mul_(beta2).addcmul_(blended_t, blended_t, value=1.0 - beta2)
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
