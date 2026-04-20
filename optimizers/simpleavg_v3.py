#
# SimpleAvg V3: Past-only uniform average + explicit mix + fresh m and v.
#
# g_t is NOT in the averaged window. Average is computed over past gradients.
# g_t is blended back in via explicit mix, matching AttnRaw V3 structure.
# Both m_t and v_t are fresh (no EMA).
#
#   attended = mean([g_t-1, g_t-2, ..., g_t-L])
#   blended  = mix * g_t + (1 - mix) * attended
#   m_t = blended (fresh)
#   v_t = beta2 * g_t^2 + (1 - beta2) * attended^2 (fresh)
#

import torch
from torch.optim import Optimizer


class SimpleAvgV3(Optimizer):
    """
    SimpleAvg V3: past-only uniform average, explicit mix with g_t, fresh m and v.

    Args:
        params:         model parameters
        lr:             learning rate
        betas:          (beta2,) — decay factor for v computation
        eps:            numerical stability term
        weight_decay:   decoupled weight decay
        context_length: number of past gradients to store (K)
        mix:            blend g_t with averaged past (0.0 = all past, 1.0 = all current)
    """

    def __init__(
        self,
        params,
        lr=3e-4,
        betas=(0.999,),
        eps=1e-8,
        weight_decay=0.0,
        context_length=8,
        mix=0.9,
    ):
        if context_length < 1:
            raise ValueError("context_length must be >= 1")
        if not 0.0 <= mix <= 1.0:
            raise ValueError("mix must be in [0, 1]")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            context_length=context_length,
            mix=mix,
        )
        super().__init__(params, defaults)

    def _init_param_state(self, p: torch.Tensor):
        state = self.state[p]
        state["step"] = 0
        state["grad_history"] = []

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
                    attended = history.mean(dim=0)
                    blended = mix * g_flat + (1.0 - mix) * attended
                    v_t = beta2 * g_flat.pow(2) + (1.0 - beta2) * attended.pow(2)
                else:
                    blended = g_flat
                    v_t = g_flat.pow(2)

                m_t = blended.reshape_as(p)
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
