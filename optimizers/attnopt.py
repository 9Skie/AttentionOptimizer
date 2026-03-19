# optimizers/attnopt.py
#
# Per-element attention optimizer.
#
# Each scalar element of each parameter tensor attends over its own
# K-step gradient history independently. W_Q, W_K, and pos_emb are
# per-parameter and always trained online via an aux loss.

import torch
import torch.nn.functional as F
from torch.optim import Optimizer


def rms_norm_tensor(t, eps=1e-8):
    rms = t.pow(2).mean().sqrt().clamp(min=eps)
    return t / rms


def positional_embedding(context_length: int, d_pos: int) -> torch.Tensor:
    # TODO: decide on positional encoding scheme for recency
    raise NotImplementedError


class AttnOpt(Optimizer):
    """
    Per-element attention optimizer.

    Replaces (or blends with) Adam's first-moment EMA by computing, for each
    scalar element independently, a softmax-weighted sum of its own K-step
    gradient history. W_Q, W_K, and pos_emb are per-parameter and always
    trained online via next-gradient prediction.

    Args:
        params:         model parameters
        lr:             learning rate
        betas:          (beta1, beta2) for EMA / second moment
        eps:            numerical stability term
        weight_decay:   decoupled weight decay
        context_length: gradient history window K
        d_attn:         attention projection dimension
        d_pos:          positional embedding dimension
        moment_mode:    "pure"  — attention fully replaces EMA
                        "gated" — (1-gate)*EMA + gate*attention
        gate_value:     λ in gated mode
        aux_lr:         learning rate for W_Q, W_K, pos_emb updates
        chunk_size:     elements processed at once (memory vs speed)
        seed:           RNG seed for W_Q/W_K initialisation
    """

    def __init__(
        self,
        params,
        lr=3e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0,
        context_length=8,
        d_attn=8,
        d_pos=4,
        moment_mode="pure",
        gate_value=0.5,
        aux_lr=1e-3,
        chunk_size=65536,
        seed=42,
    ):
        if moment_mode not in {"pure", "gated"}:
            raise ValueError(f"Unknown moment_mode: {moment_mode}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            context_length=context_length,
            d_attn=d_attn,
            d_pos=d_pos,
            moment_mode=moment_mode,
            gate_value=gate_value,
            aux_lr=aux_lr,
            chunk_size=chunk_size,
        )
        super().__init__(params, defaults)

        self.d_pos = d_pos
        self.d_in = 1 + d_pos
        self.d_attn = d_attn
        self.scale = d_attn ** -0.5
        self.context_length = context_length

        self._rng = torch.Generator()
        self._rng.manual_seed(seed)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _init_param_state(self, p: torch.Tensor):
        state = self.state[p]
        state["step"] = 0
        state["exp_avg"]      = torch.zeros_like(p, dtype=torch.float32)
        state["exp_avg_sq"]   = torch.zeros_like(p, dtype=torch.float32)
        state["grad_history"] = []   # list of (numel,) float32 tensors

        # Per-parameter W_Q, W_K: (d_in, d_attn) — always learnable
        W_q = torch.randn(self.d_in, self.d_attn, generator=self._rng) * (self.d_in ** -0.5)
        W_k = torch.randn(self.d_in, self.d_attn, generator=self._rng) * (self.d_in ** -0.5)
        state["W_q"] = W_q.to(p.device).requires_grad_(True)
        state["W_k"] = W_k.to(p.device).requires_grad_(True)

        # Per-parameter positional embedding: (context_length, d_pos)
        # TODO: initialise once encoding scheme is decided
        state["pos_emb"] = None

    def _attn_chunk(
        self,
        g_chunk:    torch.Tensor,   # (C,)
        hist_chunk: torch.Tensor,   # (K, C)
        W_q:        torch.Tensor,   # (d_in, d_attn)
        W_k:        torch.Tensor,   # (d_in, d_attn)
        pos_emb:    torch.Tensor,   # (context_length, d_pos)
    ) -> torch.Tensor:
        """Per-element attended first moment for one chunk of elements."""
        K, C = hist_chunk.shape

        # Query: [grad_value, pos_emb[0]] per element → (C, d_in)
        pos_q = pos_emb[0].unsqueeze(0).expand(C, -1)
        q_in  = torch.cat([g_chunk.unsqueeze(-1), pos_q], dim=-1)

        # Keys: [grad_value, pos_emb[j]] per element per slot → (K, C, d_in)
        pos_k = pos_emb[:K].unsqueeze(1).expand(-1, C, -1)
        k_in  = torch.cat([hist_chunk.unsqueeze(-1), pos_k], dim=-1)

        q      = q_in @ W_q                                        # (C, d_attn)
        k      = k_in.reshape(-1, self.d_in) @ W_k                # (K*C, d_attn)
        k      = k.reshape(K, C, self.d_attn)

        scores = torch.einsum("cd,kcd->ck", q, k) * self.scale    # (C, K)
        alpha  = torch.softmax(scores, dim=-1)                     # (C, K)
        m_attn = torch.einsum("ck,kc->c", alpha, hist_chunk)      # (C,)
        return m_attn

    def _compute_attn(
        self,
        g_flat:     torch.Tensor,
        history:    torch.Tensor,
        W_q:        torch.Tensor,
        W_k:        torch.Tensor,
        pos_emb:    torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        n = g_flat.shape[0]
        parts = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            parts.append(
                self._attn_chunk(
                    g_flat[start:end],
                    history[:, start:end],
                    W_q, W_k, pos_emb,
                )
            )
        return torch.cat(parts, dim=0)

    def _update_attn_params(
        self,
        state:      dict,
        g_flat:     torch.Tensor,
        aux_lr:     float,
        chunk_size: int,
    ):
        """Update W_Q, W_K, pos_emb via next-gradient prediction aux loss."""
        if len(state["grad_history"]) < 1 or state["pos_emb"] is None:
            return

        W_q     = state["W_q"]
        W_k     = state["W_k"]
        pos_emb = state["pos_emb"]
        history = torch.stack(state["grad_history"], dim=0)
        n       = g_flat.shape[0]

        with torch.enable_grad():
            parts = []
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                parts.append(
                    self._attn_chunk(
                        history[0, start:end],
                        history[:, start:end],
                        W_q, W_k, pos_emb,
                    )
                )
            pred = torch.cat(parts, dim=0)
            aux_loss = 1.0 - F.cosine_similarity(
                pred.reshape(1, -1),
                g_flat.detach().reshape(1, -1),
            ).mean()
            grads = torch.autograd.grad(
                aux_loss, [W_q, W_k, pos_emb], allow_unused=True
            )

        with torch.no_grad():
            for tensor, grad in zip([W_q, W_k, pos_emb], grads):
                if grad is not None:
                    tensor.add_(grad, alpha=-aux_lr)

    # ------------------------------------------------------------------ #
    # Optimiser step                                                       #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr           = group["lr"]
            beta1, beta2 = group["betas"]
            eps          = group["eps"]
            wd           = group["weight_decay"]
            K            = group["context_length"]
            moment_mode  = group["moment_mode"]
            gate_value   = group["gate_value"]
            aux_lr       = group["aux_lr"]
            chunk_size   = group["chunk_size"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    self._init_param_state(p)

                g        = p.grad.float()
                g_normed = rms_norm_tensor(g)
                g_flat   = g_normed.flatten()

                # Update W_Q, W_K, pos_emb before consuming the new gradient
                # (skipped until pos_emb encoding is implemented)
                if state["pos_emb"] is not None:
                    self._update_attn_params(state, g_flat, aux_lr, chunk_size)

                state["step"] += 1
                t = state["step"]

                # Build history window: [g_t, g_{t-1}, ..., g_{t-K+1}]
                history_list = [g_flat] + state["grad_history"][: K - 1]

                if len(history_list) == 1 or state["pos_emb"] is None:
                    m_attn = g_flat
                else:
                    history = torch.stack(history_list, dim=0)
                    m_attn  = self._compute_attn(
                        g_flat, history,
                        state["W_q"], state["W_k"],
                        state["pos_emb"], chunk_size,
                    )

                # EMA first moment (kept for gated mode)
                m_ema = state["exp_avg"].flatten()
                m_ema.mul_(beta1).add_(g_flat, alpha=1 - beta1)
                state["exp_avg"].copy_(m_ema.reshape_as(p))

                # Second moment (unchanged from Adam)
                v = state["exp_avg_sq"]
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                v_hat = v / (1 - beta2 ** t)

                if moment_mode == "pure":
                    m_tilde = m_attn.reshape_as(p)
                else:
                    m_tilde = (
                        (1.0 - gate_value) * state["exp_avg"]
                        + gate_value * m_attn.reshape_as(p)
                    )

                state["grad_history"] = (
                    [g_flat.detach().clone()] + state["grad_history"][: K - 1]
                )

                if wd != 0.0:
                    p.mul_(1 - lr * wd)

                p.addcdiv_(
                    m_tilde.to(p.dtype),
                    v_hat.sqrt().add_(eps),
                    value=-lr,
                )

        return loss
