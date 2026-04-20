#
# AttnRaw V3: Fresh m_t and v_t.
#
# attended = attention(g_t, [g_t-1, g_t-2, ..., g_t-L])
# m_tilde = mix * g_t + (1 - mix) * attended
# m_t = m_tilde (fresh, no EMA on m)
# v_t = beta2 * g_t^2 + (1 - beta2) * attended^2 (fresh, no EMA on v)
# theta -= lr * m_t / (sqrt(v_t) + eps)
#
# State kept: Neither m nor v
#

import torch
from torch.optim import Optimizer


class AttnRawV3(Optimizer):
    """
    AttnRaw V3: past-only attention, fresh m_t and v_t (no EMA on either).

    Args:
        params:         model parameters
        lr:             learning rate
        betas:          (beta2,) — decay factor for v_t computation
        eps:            numerical stability term
        weight_decay:   decoupled weight decay
        context_length: number of past gradients to store (K)
        temperature:    softmax temperature for attention scores
        mix:            blend g_t with attended (0.0 = all past, 1.0 = all current)
    """

    def __init__(
        self,
        params,
        lr=3e-4,
        betas=(0.999,),
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
            beta2 = group["betas"][0]
            eps = group["eps"]
            wd = group["weight_decay"]
            K = group["context_length"]
            temperature = group["temperature"]
            mix = group["mix"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    self._init_param_state(p)

                g = p.grad.float()
                g_flat = g.flatten()

                state["step"] += 1

                past = state["grad_history"][:K]
                if past:
                    history = torch.stack(past, dim=0)
                    attended = self._compute_attention(g_flat, history, temperature)
                    m_tilde = mix * g_flat + (1.0 - mix) * attended
                    v_t = beta2 * g_flat.pow(2) + (1.0 - beta2) * attended.pow(2)
                else:
                    attended = g_flat
                    m_tilde = g_flat
                    v_t = g_flat.pow(2)

                m_t = m_tilde.reshape_as(p)
                v_t = v_t.reshape_as(p)

                state["grad_history"] = (
                    [g_flat.detach().clone()] + state["grad_history"]
                )[:K]

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                p.addcdiv_(
                    m_t.to(p.dtype),
                    v_t.sqrt().add_(eps),
                    value=-lr,
                )

        return loss
