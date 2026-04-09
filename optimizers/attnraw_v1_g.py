#
# AttnRaw V1-G: Cosine attention with g_t INCLUDED in window + mix_beta.
#
# g_t is part of the attention window, softmax is computed over K+1 gradients,
# then the result is blended with g_t via mix_beta:
#
#   window = [g_t, g_{t-1}, ..., g_{t-K}]
#   scores = cos(g_t, each window item)
#   attended = softmax(scores/T) @ window
#   g_bar = mix_beta * g_t + (1 - mix_beta) * attended
#
# Temperature controls attention sharpness.
# No EMA on numerator — g_bar is used directly.
# Second moment uses EMA for normalization.
#

import torch
from torch.optim import Optimizer


class AttnRawV1G(Optimizer):
    """
    AttnRaw V1-G: g_t in attention window + mix_beta blend.

    Args:
        params:         model parameters
        lr:             learning rate
        betas:          (beta2,) — EMA decay for second moment
        eps:            numerical stability term
        weight_decay:   decoupled weight decay
        context_length: number of PAST gradients to store (K)
                        Window is K+1 items total (including g_t)
        temperature:    softmax temperature for attention scores
        mix_beta:       weight on current gradient in final blend
                        g_bar = mix_beta * g_t + (1 - mix_beta) * attended
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
        mix_beta=0.9,
    ):
        if context_length < 1:
            raise ValueError("context_length must be >= 1")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 < mix_beta <= 1.0:
            raise ValueError("mix_beta must be in (0, 1]")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            context_length=context_length,
            temperature=temperature,
            mix_beta=mix_beta,
        )
        super().__init__(params, defaults)

    def _init_param_state(self, p: torch.Tensor):
        state = self.state[p]
        state["step"] = 0
        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
        state["grad_history"] = []

    def _compute_attention(
        self,
        g_flat: torch.Tensor,
        window: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Cosine attention over window (including g_t)."""
        g_norm = g_flat.norm().clamp(min=1e-8)
        window_norms = window.norm(dim=1).clamp(min=1e-8)
        scores = window @ g_flat
        scores = scores / (window_norms * g_norm)
        alpha = torch.softmax(scores / temperature, dim=0)
        return alpha @ window

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
                window = torch.stack([g_flat] + past, dim=0)
                attended = self._compute_attention(g_flat, window, temperature)
                g_bar_flat = mix_beta * g_flat + (1.0 - mix_beta) * attended

                g_bar = g_bar_flat.reshape_as(p)

                v = state["exp_avg_sq"]
                v.mul_(beta2).addcmul_(g_bar, g_bar, value=1.0 - beta2)
                v_hat = v / (1.0 - beta2**t)

                state["grad_history"] = (
                    [g_flat.detach().clone()] + state["grad_history"]
                )[:K]

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                p.addcdiv_(
                    g_bar.to(p.dtype),
                    v_hat.sqrt().add_(eps),
                    value=-lr,
                )

        return loss
