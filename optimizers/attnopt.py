# optimizers/attnopt.py
#
# Attention optimizer variants for studying whether attention can replace
# or augment Adam's first-moment EMA over a short full-gradient history.

import torch
import torch.nn.functional as F
from torch.optim import Optimizer


def rms_norm_tensor(t, eps=1e-8):
    rms = t.pow(2).mean().sqrt().clamp(min=eps)
    return t / rms


def grad_stats(g_normed):
    return torch.stack([
        g_normed.mean(),
        g_normed.pow(2).mean(),
        g_normed.abs().mean(),
    ])


def fixed_recency_embeddings(context_length: int, d_pos: int):
    positions = torch.arange(context_length, dtype=torch.float32).unsqueeze(1)
    scales = torch.arange(d_pos, dtype=torch.float32).unsqueeze(0) + 1.0
    return torch.cos(positions / scales)


class AttnOpt(Optimizer):
    def __init__(
        self,
        params,
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        context_length=8,
        n_heads=1,
        d_attn=8,
        d_pos=4,
        moment_mode="pure",
        gate_value=0.5,
        trainable_attn=False,
        aux_lr=1e-3,
        seed=42,
    ):
        if moment_mode not in {"pure", "gated"}:
            raise ValueError(f"Unknown moment_mode: {moment_mode}")
        if d_attn % n_heads != 0:
            raise ValueError("d_attn must be divisible by n_heads")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            context_length=context_length,
            n_heads=n_heads,
            d_attn=d_attn,
            d_pos=d_pos,
            moment_mode=moment_mode,
            gate_value=gate_value,
            trainable_attn=trainable_attn,
            aux_lr=aux_lr,
        )
        super().__init__(params, defaults)

        self.context_length = context_length
        self.n_heads = n_heads
        self.d_attn = d_attn
        self.d_pos = d_pos
        self.d_in = 3 + d_pos
        self.head_dim = d_attn // n_heads
        self.scale = self.head_dim ** -0.5
        self.trainable_attn = trainable_attn

        rng = torch.Generator()
        rng.manual_seed(seed)

        self.W_q = torch.randn(self.d_in, d_attn, generator=rng) * (self.d_in ** -0.5)
        self.W_k = torch.randn(self.d_in, d_attn, generator=rng) * (self.d_in ** -0.5)
        self.pos_emb = fixed_recency_embeddings(context_length, d_pos)
        self._attn_device = None
        self._last_alpha = None
        self._last_aux_loss = None

    def _ensure_attn_device(self, device):
        if self._attn_device == device:
            return
        self.W_q = self.W_q.detach().to(device).requires_grad_(self.trainable_attn)
        self.W_k = self.W_k.detach().to(device).requires_grad_(self.trainable_attn)
        self.pos_emb = self.pos_emb.detach().to(device).requires_grad_(self.trainable_attn)
        self._attn_device = device

    def _trainable_tensors(self):
        return [self.W_q, self.W_k, self.pos_emb]

    def _compute_attn_weights(self, query_stats, slot_stats, n_heads):
        slot_count = slot_stats.shape[0]
        q_input = torch.cat([query_stats, self.pos_emb[0]], dim=0)
        q_proj = q_input @ self.W_q

        k_inputs = []
        for idx in range(slot_count):
            pos_idx = min(idx, self.context_length - 1)
            k_inputs.append(torch.cat([slot_stats[idx], self.pos_emb[pos_idx]], dim=0))
        k_proj = torch.stack(k_inputs, dim=0) @ self.W_k

        q_heads = q_proj.view(n_heads, self.head_dim)
        k_heads = k_proj.view(slot_count, n_heads, self.head_dim).permute(1, 0, 2)
        scores = (q_heads.unsqueeze(1) * k_heads).sum(-1) * self.scale
        alpha = torch.softmax(scores, dim=-1).mean(dim=0)
        self._last_alpha = alpha.detach().cpu()
        return alpha

    def _weighted_history(self, alpha, slot_values):
        shape = [slot_values.shape[0]] + [1] * (slot_values.ndim - 1)
        return (alpha.view(shape) * slot_values).sum(0)

    def _train_attention_params(self, state, target_grad, n_heads, aux_lr):
        if not self.trainable_attn:
            return False
        prev_pack = state.get("prev_pack")
        if prev_pack is None:
            return False

        with torch.enable_grad():
            alpha = self._compute_attn_weights(prev_pack["query_stats"], prev_pack["slot_stats"], n_heads)
            pred = self._weighted_history(alpha, prev_pack["slot_values"])
            target = rms_norm_tensor(target_grad.float()).detach()
            aux_loss = 1.0 - F.cosine_similarity(
                pred.reshape(1, -1),
                target.reshape(1, -1),
            ).mean()
            grads = torch.autograd.grad(aux_loss, self._trainable_tensors(), allow_unused=True)

        with torch.no_grad():
            for tensor, grad in zip(self._trainable_tensors(), grads):
                if grad is not None:
                    tensor.add_(grad, alpha=-aux_lr)
        self._last_aux_loss = float(aux_loss.detach().cpu())
        return True

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
            context_length = group["context_length"]
            n_heads = group["n_heads"]
            moment_mode = group["moment_mode"]
            gate_value = group["gate_value"]
            aux_lr = group["aux_lr"]

            aux_updated = False

            for p in group["params"]:
                if p.grad is None:
                    continue

                self._ensure_attn_device(p.device)

                g = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["grad_history"] = []
                    state["stat_history"] = []
                    state["prev_pack"] = None

                if self.trainable_attn and not aux_updated:
                    aux_updated = self._train_attention_params(state, g, n_heads, aux_lr)

                state["step"] += 1
                t = state["step"]

                g_normed = rms_norm_tensor(g.float())
                stats_t = grad_stats(g_normed)

                history_vals = [g_normed] + state["grad_history"][: context_length - 1]
                history_stats = [stats_t] + state["stat_history"][: context_length - 1]
                slot_values = torch.stack(history_vals, dim=0)
                slot_stats = torch.stack(history_stats, dim=0)

                if slot_values.shape[0] == 1:
                    m_attn = g_normed
                    self._last_alpha = torch.tensor([1.0])
                else:
                    alpha = self._compute_attn_weights(stats_t, slot_stats, n_heads)
                    m_attn = self._weighted_history(alpha, slot_values)

                m_ema = state["exp_avg"]
                m_ema.mul_(beta1).add_(g_normed, alpha=1 - beta1)

                v = state["exp_avg_sq"]
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                v_hat = v / (1 - beta2 ** t)

                if moment_mode == "pure":
                    m_tilde = m_attn
                else:
                    m_tilde = (1.0 - gate_value) * m_ema + gate_value * m_attn

                state["grad_history"] = [g_normed.detach().clone()] + state["grad_history"][: context_length - 1]
                state["stat_history"] = [stats_t.detach().clone()] + state["stat_history"][: context_length - 1]
                state["prev_pack"] = {
                    "query_stats": stats_t.detach().clone(),
                    "slot_stats": slot_stats.detach().clone(),
                    "slot_values": slot_values.detach().clone(),
                }

                if wd != 0:
                    p.mul_(1 - lr * wd)

                p.addcdiv_(m_tilde.to(p.dtype), v_hat.sqrt().add_(eps), value=-lr)

        return loss
