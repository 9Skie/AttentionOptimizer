# AttnOpt Type 2: Learning W_Q and W_K

## The Core Problem

AttnRaw (Type 1) uses cosine similarity to compute attention scores over gradient history:

```
s_i = cos(g_t, g_{t-i})
alpha = softmax(s)
m_tilde = mix_beta * (alpha @ history) + (1 - mix_beta) * g_t
```

This is parameter-free. The question is: can learned projections W_Q and W_K do better?

```
q_t = g_t @ W_Q          # (d_param, d_attn)
k_i = g_{t-i} @ W_K      # (d_param, d_attn)
s_i = q_t @ k_i^T / sqrt(d_attn)
alpha = softmax(s)
m_tilde = mix_beta * (alpha @ history) + (1 - mix_beta) * g_t
```

W_Q and W_K learn to project gradient vectors into a space where similarity is more meaningful
than raw cosine in the original gradient space.

## The Self-Referential Problem

W_Q and W_K influence how m_tilde is computed, which influences the optimizer step on theta,
which influences the next gradient, which is used to update W_Q and W_K.

Naively training W_Q/W_K with the same gradient signal they help process creates a circular
dependency. The optimizer is partially optimizing itself using its own output.

This is not necessarily fatal — Adam's beta hyperparameters are also fixed and "self-referential"
in a sense — but it means naive gradient updates on W_Q/W_K may be unstable or degenerate
(e.g. W_Q/W_K collapsing to zero, or learning to ignore history entirely).

---

## Option A: Differentiable Optimizer Step (MAML-style)

### Idea
Split each batch into a train half and a val half.

1. Run forward/backward on train half → get g_t
2. Compute m_tilde using current W_Q, W_K
3. Take a differentiable optimizer step on theta (no @no_grad, no in-place ops)
4. Run forward on val half using the *updated* theta → get loss_val
5. Backprop loss_val through the optimizer step → get dL/dW_Q, dL/dW_K
6. Update W_Q, W_K via Muon using those gradients

### Why This Works
W_Q/W_K are updated based on how well the optimizer step improved val loss — not the
same batch. The gradient signal answers: "given this gradient, did your attention
weighting produce a good update?"

### Key Implementation Requirements
- `differentiable_step()`: a version of the optimizer step without `@no_grad` and
  without in-place operations (use `p - lr * update` instead of `p.add_(...)`)
- The computation graph must be kept alive through the step so autograd can trace back
  to W_Q/W_K
- W_Q/W_K must be `nn.Parameter` objects (or tracked tensors), not plain tensors

### Memory Cost
Keeping the computation graph through the optimizer step means storing activations
for all parameters across the step. Roughly 2-3x normal training memory.
At 85M params on a 32GB 5090, this may be tight. Gradient checkpointing could help.

### Related Work
- **MAML** (Finn et al. 2017): learns initialization by differentiating through
  gradient steps. Exact same computational structure.
- **Learned Optimizers** (Andrychowicz et al. 2016, Metz et al. 2022): learn optimizer
  parameters by differentiating through training trajectories. Option A is a single-step
  version of this.
- **DARTS** (Liu et al. 2019): architecture search via differentiating through one
  gradient step on a validation set. Direct inspiration for the train/val split.

### Failure Modes
- Memory OOM from keeping the computation graph
- Gradient explosion through the optimizer step (need careful clipping)
- W_Q/W_K converging to degenerate solutions (all-zero, identity-like)

---

## Option B: Frozen Meta-Update

### Idea
Periodically freeze theta, run a meta-update loop on W_Q/W_K only.

Every N steps:
1. Save a snapshot of theta
2. For M iterations:
   a. Sample a val batch
   b. Compute loss with current theta and current W_Q/W_K
   c. Backprop to get gradient signal for W_Q/W_K (treating theta as fixed)
   d. Update W_Q/W_K via Muon
3. Restore theta, continue normal training

### Why This Is Simpler
No differentiable optimizer step needed. W_Q/W_K are updated by directly measuring
how well they help predict good updates for the fixed theta snapshot. No computation
graph through the optimizer step required.

### Approximation Made
This does NOT differentiate through the optimizer step — it treats "does this attention
weighting produce a low-loss gradient direction" as a direct signal, without asking
"does this attention weighting produce a good *parameter update*." This is a looser
signal but much cheaper.

### Related Work
- **Hypernetworks** (Ha et al. 2016): a separate network generates weights for the
  main network. W_Q/W_K behave like a tiny hypernetwork for the optimizer.
- **Meta-SGD** (Li et al. 2017): learns per-parameter learning rates alongside
  initialization — a simpler version of the same meta-learning idea.
- **ProxyNAS / SMASH**: treat architecture parameters as separate from model parameters
  and alternate their updates. Same alternating structure as Option B.

### Failure Modes
- W_Q/W_K update signal is stale (theta has moved since the snapshot)
- N too small → W_Q/W_K updated too frequently, unstable
- N too large → W_Q/W_K rarely updated, underfits

---

## Open Questions

1. **What should d_attn be?** Smaller = cheaper, but less expressive.
   d_attn = 16 or 32 is probably a reasonable start for an 85M model.

2. **Shared vs per-layer W_Q/W_K?** Per-layer is more expressive but K * n_layer
   times more parameters. Shared W_Q/W_K across all layers is a reasonable first test.

3. **Per-tensor vs per-row attention?** Current Type 1 uses one attention distribution
   per parameter tensor. Type 2 could use one per row (each row of a weight matrix
   gets its own query). Much more expressive, much more memory.

4. **Does mix_beta stay fixed or become learned?** Could be a simple scalar parameter
   updated alongside W_Q/W_K.

5. **Is the val split worth the 2x data cost?** If W_Q/W_K are small and converge
   quickly, the overhead might be negligible. If they need lots of signal, 50% of
   tokens going to meta-updates is expensive.

---

## Variant: EMA on Top of Attention (Hybrid)

Instead of replacing Adam's EMA entirely, use attention to compute a smarter
instantaneous signal, then run EMA on top:

```
a_t = Attention(g_t, g_{t-1}, ..., g_{t-L})   # attention over full window incl. current
m_t = beta1 * m_{t-1} + (1 - beta1) * a_t     # EMA smooths over all history
v_t = beta2 * v_{t-1} + (1 - beta2) * a_t^2   # second moment as usual
theta_t = theta_{t-1} - lr * m_t / (sqrt(v_t) + eps)
```

**Why this is better than pure AttnRaw:**
- Attention handles short-term selective weighting (which of the last L gradients matter)
- EMA handles long-term momentum (memory beyond the K-step window)
- mix_beta disappears — beta1 does the smoothing job naturally
- Degrades gracefully: if attention learns nothing (uniform weights), a_t ≈ mean of
  recent gradients, EMA on top ≈ Adam behavior

**Compared to current AttnRaw:**
- AttnRaw: one-shot mix, forgets everything beyond K steps
- Hybrid: infinite memory via EMA, attention just sharpens the instantaneous signal

This is a more natural decomposition — matches how attention works in sequence models
(attention selects relevant context, sequential processing happens on top).

Run ID candidate: `ATTNEMA-8` — same K=8 window, beta1=0.9, attention replaces the
raw g_t input to the EMA rather than replacing the EMA entirely.

---

## Recommendation

Start with **Option B** (frozen meta-update) to validate that W_Q/W_K can learn
anything useful without the memory and implementation complexity of Option A.
If W_Q/W_K improve over AttnRaw in Option B, then invest in Option A's
differentiable step for a cleaner, more principled training signal.
