#
# AttnOptA: Learned W_Q/W_K with differentiable step + val split.
#
# W_Q/W_K are trained by: take a virtual differentiable step on the train
# batch using the current attention-weighted m_tilde, then measure val loss
# on the updated model. Backprop val loss through the virtual step to get
# gradients on W_Q/W_K.
#
# The optimizer exposes two methods:
#   .step(closure)       — normal model update (no_grad)
#   .meta_step(model, val_x, val_y, lr)  — updates W_Q/W_K via val loss
#
# train.py calls meta_step every N steps after accumulating gradients.
#
# Architecture: same as AttnOptB (per-tensor W_Q/W_K, p = numel//8, d_attn=64)

import math
import torch
import torch.nn.functional as F
from torch.optim import Optimizer


class AttnOptA(Optimizer):
    """
    Per-tensor learned attention optimizer trained via differentiable val step.

    Args:
        params:         model parameters
        lr:             learning rate for model parameter updates
        lr_meta:        learning rate for W_Q/W_K meta-updates
        betas:          (beta1, beta2) for Adam-style first/second moments
        eps:            numerical stability
        weight_decay:   decoupled weight decay
        context_length: gradient history window K
        d_attn:         attention projection dimension
        meta_every:     update W_Q/W_K every N optimizer steps
    """

    def __init__(
        self,
        params,
        lr=3e-4,
        lr_meta=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        context_length=8,
        d_attn=64,
        meta_every=10,
    ):
        defaults = dict(
            lr=lr,
            lr_meta=lr_meta,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            context_length=context_length,
            d_attn=d_attn,
            meta_every=meta_every,
        )
        super().__init__(params, defaults)

    def _init_param_state(self, p: torch.Tensor, d_attn: int):
        state = self.state[p]
        d = p.numel()
        p_dim = max(1, d // 8)

        state["step"] = 0
        state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
        state["grad_history"] = []

        state["W_Q"] = torch.nn.init.orthogonal_(
            torch.empty(d, p_dim, dtype=torch.float32, device=p.device)
        ).requires_grad_(True)
        state["W_K"] = torch.nn.init.orthogonal_(
            torch.empty(d, p_dim, dtype=torch.float32, device=p.device)
        ).requires_grad_(True)

        state["wq_exp_avg"] = torch.zeros(d, p_dim, dtype=torch.float32, device=p.device)
        state["wq_exp_avg_sq"] = torch.zeros(d, p_dim, dtype=torch.float32, device=p.device)
        state["wk_exp_avg"] = torch.zeros(d, p_dim, dtype=torch.float32, device=p.device)
        state["wk_exp_avg_sq"] = torch.zeros(d, p_dim, dtype=torch.float32, device=p.device)

        # Snapshot of last computed m_tilde and v_hat for virtual step
        state["last_m_tilde"] = None
        state["last_v_hat"] = None

    def _attend(self, g_flat, history, W_Q, W_K, d_attn):
        q = g_flat @ W_Q                        # (p_dim,)
        ks = history @ W_K                      # (K, p_dim)
        scores = ks @ q / math.sqrt(d_attn)     # (K,)
        alpha = torch.softmax(scores, dim=0)
        return alpha @ history                  # (d,)

    def _adam_update(self, param, grad, exp_avg, exp_avg_sq, t, lr_meta, beta1=0.9, beta2=0.999, eps=1e-8):
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        m_hat = exp_avg / (1.0 - beta1 ** t)
        v_hat = exp_avg_sq / (1.0 - beta2 ** t)
        param.data.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr_meta)

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
            d_attn = group["d_attn"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    self._init_param_state(p, d_attn)

                g = p.grad.float()
                g_flat = g.flatten()

                state["step"] += 1
                t = state["step"]

                W_Q = state["W_Q"]
                W_K = state["W_K"]

                state["grad_history"] = (
                    [g_flat.detach().clone()] + state["grad_history"]
                )[:K]
                history = torch.stack(state["grad_history"], dim=0)

                # Compute m_tilde (detached for model update, but stored with
                # grad for meta_step to use later)
                with torch.enable_grad():
                    m_tilde_flat = self._attend(g_flat, history, W_Q, W_K, d_attn)

                m_tilde = m_tilde_flat.detach().reshape_as(p)

                # Store for virtual step in meta_step
                m = state["exp_avg"]
                m.mul_(beta1).add_(m_tilde, alpha=1.0 - beta1)
                m_hat = m / (1.0 - beta1 ** t)

                v = state["exp_avg_sq"]
                v.mul_(beta2).addcmul_(m_tilde, m_tilde, value=1.0 - beta2)
                v_hat = v / (1.0 - beta2 ** t)

                state["last_m_tilde"] = m_tilde_flat          # has grad via W_Q/W_K
                state["last_v_hat"] = v_hat.detach().clone()  # no grad needed

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                p.addcdiv_(
                    m_hat.to(p.dtype),
                    v_hat.sqrt().add_(eps),
                    value=-lr,
                )

        return loss

    def meta_step(self, model, val_x, val_y):
        """
        Update W_Q/W_K using val loss after a virtual differentiable step.

        Virtual step: theta' = theta - lr * m_hat / (sqrt(v_hat) + eps)
        where m_hat is derived from m_tilde which flows back to W_Q/W_K.

        Call this after .step() every meta_every steps.
        """
        # Collect all params and build virtual theta'
        # We re-derive m_hat from last_m_tilde (which has grad through W_Q/W_K)
        virtual_params = {}

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]

            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0 or state["last_m_tilde"] is None:
                    continue

                t = state["step"]
                m_tilde_flat = state["last_m_tilde"]    # has grad through W_Q/W_K
                v_hat = state["last_v_hat"]

                m_tilde = m_tilde_flat.reshape_as(p)

                # Recompute m_hat differentiably
                # Use current exp_avg state — detach the EMA accumulation,
                # only let gradient flow through the current m_tilde term
                m_ema_prev = state["exp_avg"].detach().clone()
                m_new = beta1 * m_ema_prev + (1.0 - beta1) * m_tilde
                m_hat = m_new / (1.0 - beta1 ** t)

                # Virtual differentiable step (no in-place)
                update = m_hat / (v_hat.sqrt().add(eps))
                p_prime = p.detach() - lr * update.to(p.dtype)
                virtual_params[p] = p_prime

        if not virtual_params:
            return

        # Temporarily swap model params with virtual ones
        original_data = {}
        for p, p_prime in virtual_params.items():
            original_data[p] = p.data
            p.data = p_prime.data  # swap in virtual params

        # Forward on val batch with virtual params
        with torch.enable_grad():
            _, val_loss = model(val_x, val_y)

        # Backward to get gradients on W_Q/W_K
        # Need to manually associate the loss with virtual_params' graph
        # Restore original params first so model is intact
        for p, orig in original_data.items():
            p.data = orig

        # Collect all W_Q/W_K
        meta_params = []
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    continue
                meta_params.extend([state["W_Q"], state["W_K"]])

        # Recompute val loss through the virtual step properly
        # (swap data again, run forward with autograd enabled)
        for p, p_prime in virtual_params.items():
            p.data = p_prime.data

        val_loss.backward()

        for p, orig in original_data.items():
            p.data = orig

        # Update W_Q/W_K
        for group in self.param_groups:
            lr_meta = group["lr_meta"]
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    continue
                t = state["step"]
                W_Q, W_K = state["W_Q"], state["W_K"]
                if W_Q.grad is not None:
                    self._adam_update(
                        W_Q, W_Q.grad,
                        state["wq_exp_avg"], state["wq_exp_avg_sq"],
                        t, lr_meta,
                    )
                    W_Q.grad.zero_()
                if W_K.grad is not None:
                    self._adam_update(
                        W_K, W_K.grad,
                        state["wk_exp_avg"], state["wk_exp_avg_sq"],
                        t, lr_meta,
                    )
                    W_K.grad.zero_()
