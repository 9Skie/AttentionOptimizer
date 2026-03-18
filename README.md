## Motivation

This project came from two intersecting ideas.

First, Attention Residuals￼ showed that replacing fixed residual connections with attention-based ones can improve performance.

- <img src="assets/residuals.png" width="200"/>


- Second, Andrej Karpathy's question whether stochastic gradient descent could also use attention in it:

<img src="assets/kaparthy.png" width="400"/>


That made me look at Adam’s first-moment EMA differently: it compresses gradient history into a single exponentially decayed running average, much like a hidden state bottleneck in sequential models.

So the question becomes: instead of forcing optimization history through one EMA, can an optimizer use attention to attend over recent gradients and decide what matters?

---

## AttnOpt: Attention as a First Moment

### The Idea

Adam's update rule uses an EMA of gradients as its first moment:

$$m_t = \beta_1 \, m_{t-1} + (1 - \beta_1) \, g_t$$


AttnOpt replaces that fixed decay with a learned, selective attention over a sliding window:

$$
\theta_{t+1}

\theta_t

\eta ,
\frac{
\sum_{i=0}^{K-1} \alpha_i \hat{g}_{t-i}
}{
\sqrt{\hat{v}_t} + \varepsilon
}
$$




First normalize the current gradient:

$$
\hat{g}_t = \frac{g_t}{\mathrm{RMS}(g_t)}
$$

where

$$
\mathrm{RMS}(g_t) = \sqrt{\frac{1}{d}\sum_{j=1}^d g_{t,j}^2}
$$

Then compute a compact statistic vector for each step:

$$
s_t =
\begin{bmatrix}
\mathrm{mean}(\hat{g}_t) \
\mathbb{E}[\hat{g}_t^2] \
\mathbb{E}[|\hat{g}_t|]
\end{bmatrix}
\in \mathbb{R}^3
$$

Concatenate this with a positional embedding:

$$
x_t = [, s_t ,|, p_t ,] \in \mathbb{R}^{3 + d_{\mathrm{pos}}}
$$

Over a window of length K, form the query and keys as

$$
q_t = x_t W_Q
$$

$$
k_{t-i} = x_{t-i} W_K, \qquad i=0,1,\dots,K-1
$$

Then compute attention weights:

$$
\alpha_i

\frac{
\exp!\left(\frac{q_t k_{t-i}^{\top}}{\sqrt{d_{\mathrm{head}}}}\right)
}{
\sum_{j=0}^{K-1}
\exp!\left(\frac{q_t k_{t-j}^{\top}}{\sqrt{d_{\mathrm{head}}}}\right)
}
\qquad i=0,1,\dots,K-1
$$

Finally, define the attended first moment as a weighted sum of recent normalized gradients:

$$
m_{\mathrm{attn},t}

\sum_{i=0}^{K-1} \alpha_i \hat{g}_{t-i}
$$

The second moment is kept the same as Adam:

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

with the usual bias-corrected version

$$
\hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

## Test Bed

The model under test is **Karpathy's nanoGPT** (GPT-2), extended with the incremental architecture and training improvements documented in the [nanoGPT community discussion #481](https://github.com/karpathy/nanochat/discussions/481). Pre-training runs on HuggingFace's **[FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)** dataset.

The goal is to see whether AttnOpt can match or beat Adam/AdamW/Muon on validation loss at a fixed token budget.

---

## Run Matrix

Training budget: ~`1.07B` tokens per run (`4,096` steps × `262,144` tokens/step).

| ID | Optimizer |
|---|---|
| `BASE-SGD` | SGD + momentum |
| `BASE-ADAM` | Adam |
| `BASE-ADAMW` | AdamW |
| `BASE-MUON` | Muon |
| `ATTN-PURE-8-TRAIN` | attention replaces EMA, context window 8 |
| `ATTN-PURE-16-TRAIN` | attention replaces EMA, context window 16 |
| `ATTN-GATED-8-TRAIN` | `0.5 × EMA + 0.5 × attention`, context window 8 |
| `ATTN-GATED-16-TRAIN` | `0.5 × EMA + 0.5 × attention`, context window 16 |

---

## Results

Live runs tracked on Weights & Biases (`attn-optimizer`). Loss curves compared in `analysis/results.ipynb`.
