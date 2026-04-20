# Attention Optimizer Math

## Core Attention

All AttnRaw variants use the same attention pattern:

```text
query = g_t
keys/values = gradient history

score_i = cos(g_t, g_{t-i})
        = (g_t · g_{t-i}) / (||g_t|| ||g_{t-i}||)

alpha_i = softmax(score_i / T)

attended = sum_i alpha_i * g_{t-i}
```

Notes:

- The implementation uses cosine similarity, not raw dot product.
- `T` is only used where a temperature sweep is actually defined.
- `SimpleAvg` replaces the softmax weights with uniform averaging.

## Adam Baseline

```text
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
theta_t = theta_{t-1} - lr * m_t / (sqrt(v_t) + eps)
```

## Experiment 1 Variants

Experiment 1 is the state-retention sweep:

- Keep both `m` and `v`: `V1`
- Keep only `v`: `V2`
- Keep neither: `V3`

### AttnRaw-V1

Past-only attention plus an explicit current/past blend.

Window:

```text
[g_{t-1}, g_{t-2}, ..., g_{t-L}]
```

Formula:

```text
attended = attention(g_t, [g_{t-1}, ..., g_{t-L}])
blended = mix * g_t + (1 - mix) * attended

m_t = beta1 * m_{t-1} + (1 - beta1) * blended
v_t = beta2 * v_{t-1} + (1 - beta2) * blended^2
```

Default mix:

```text
mix = 0.9
```

So by default V1 means:

```text
90% current gradient + 10% attended past gradients
```

This is why the explicit MIX sweep exists in Experiment 2.

### AttnRaw-V2

Past-only attention, explicit current/past blend, fresh `m`, EMA `v`.

Window:

```text
[g_{t-1}, g_{t-2}, ..., g_{t-L}]
```

Formula:

```text
attended = attention(g_t, [g_{t-1}, ..., g_{t-L}])
blended = mix * g_t + (1 - mix) * attended

m_t = blended
v_t = beta2 * v_{t-1} + (1 - beta2) * blended^2
```

Default mix:

```text
mix = 0.9
```

So by default V2 also means:

```text
90% current gradient + 10% attended past gradients
```

Temperature:

```text
alpha_i = softmax(score_i / T)
```

So the `-T0.5 / -T1.0 / -T2.0` runs are real temperature variants, not just naming.

### AttnRaw-V3

Past-only attention, explicit current/past blend, fresh `m`, fresh `v`.

Window:

```text
[g_{t-1}, g_{t-2}, ..., g_{t-L}]
```

Formula:

```text
attended = attention(g_t, [g_{t-1}, ..., g_{t-L}])
blended = mix * g_t + (1 - mix) * attended

m_t = blended
v_t = beta2 * g_t^2 + (1 - beta2) * attended^2
```

Default mix:

```text
mix = 0.9
```

So by default V3 also means:

```text
90% current gradient + 10% attended past gradients for m_t
```

Temperature:

```text
alpha_i = softmax(score_i / T)
```

So the `-T0.5 / -T1.0 / -T2.0` runs are real temperature variants here too.

### SimpleAvg-V1

Past-only uniform-average baseline for V1. Mirrors AttnRaw-V1 exactly — same window,
same mix — with averaging replacing attention.

Window:

```text
[g_{t-1}, g_{t-2}, ..., g_{t-L}]
```

Formula:

```text
attended = mean([g_{t-1}, ..., g_{t-L}])
blended  = mix * g_t + (1 - mix) * attended

m_t = beta1 * m_{t-1} + (1 - beta1) * blended
v_t = beta2 * v_{t-1} + (1 - beta2) * blended^2
```

Default mix:

```text
mix = 0.9
```

### SimpleAvg-V2

Past-only uniform-average baseline for V2.

Window:

```text
[g_{t-1}, g_{t-2}, ..., g_{t-L}]
```

Formula:

```text
attended = mean([g_{t-1}, ..., g_{t-L}])
blended  = mix * g_t + (1 - mix) * attended

m_t = blended
v_t = beta2 * v_{t-1} + (1 - beta2) * blended^2
```

Default mix:

```text
mix = 0.9
```

### SimpleAvg-V3

Past-only uniform-average baseline for V3.

Window:

```text
[g_{t-1}, g_{t-2}, ..., g_{t-L}]
```

Formula:

```text
attended = mean([g_{t-1}, ..., g_{t-L}])
blended  = mix * g_t + (1 - mix) * attended

m_t = blended
v_t = beta2 * g_t^2 + (1 - beta2) * attended^2
```

Default mix:

```text
mix = 0.9
```

## Experiment 2 Variants

Experiment 2 explores the V1 family.

### SimpleAvg-V1-G

The "g_t included in the window" counterpart to AttnRaw-V1-G. Direct apples-to-apples
comparison: same full window, plain mean instead of cosine attention.

Window:

```text
[g_t, g_{t-1}, g_{t-2}, ..., g_{t-L}]
```

Formula:

```text
attended = mean([g_t, g_{t-1}, ..., g_{t-L}])

m_t = beta1 * m_{t-1} + (1 - beta1) * attended
v_t = beta2 * v_{t-1} + (1 - beta2) * attended^2
```

No mix parameter — all window members get equal weight (1 / (L+1)).

### AttnRaw-V1-G

This is the "g_t included in the attention window" variant.

Window:

```text
[g_t, g_{t-1}, g_{t-2}, ..., g_{t-L}]
```

Formula:

```text
attended = attention(g_t, [g_t, g_{t-1}, ..., g_{t-L}])

m_t = beta1 * m_{t-1} + (1 - beta1) * attended
v_t = beta2 * v_{t-1} + (1 - beta2) * attended^2
```

Important:

- `V1-G` has **no separate mix parameter**.
- Since `g_t` is already inside the attention window, the softmax weights decide how much `g_t` contributes.

### AttnRaw V1 MIX Sweep

This reuses `AttnRaw-V1` and changes only `mix`.

Examples:

```text
MIX10 -> mix = 0.10
MIX25 -> mix = 0.25
MIX50 -> mix = 0.50
MIX75 -> mix = 0.75
MIX90 -> mix = 0.90
```

Interpretation:

```text
mix = 0.10 -> 10% current gradient, 90% attended past
mix = 0.25 -> 25% current gradient, 75% attended past
mix = 0.50 -> 50% current gradient, 50% attended past
mix = 0.75 -> 75% current gradient, 25% attended past
mix = 0.90 -> 90% current gradient, 10% attended past
```

## Summary

| Variant | Window | m state | v state | Extra knob |
| --- | --- | --- | --- | --- |
| `ATTNRAW-V1` | past only | keep `m` | keep `v` | `mix`, `temperature` |
| `ATTNRAW-V1-G` | full incl. `g_t` | keep `m` | keep `v` | `temperature` |
| `ATTNRAW-V2` | past only | no `m` state | keep `v` | `mix`, `temperature` |
| `ATTNRAW-V3` | past only | no `m` state | no `v` state | `mix`, `temperature` |
| `SIMPLEAVG-V1` | past only | keep `m` | keep `v` | `mix` |
| `SIMPLEAVG-V1-G` | full incl. `g_t` | keep `m` | keep `v` | none |
| `SIMPLEAVG-V2` | past only | no `m` state | keep `v` | `mix` |
| `SIMPLEAVG-V3` | past only | no `m` state | no `v` state | `mix` |
