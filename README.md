# Attention Optimizer

Experiments focused on one question first:

Can attention over recent full-gradient history replace or augment Adam's first-moment EMA on GPT pretraining?

## Step 1 Scope

This repo now targets the optimizer study only. The model architecture is the standard GPT baseline from `model/gpt.py`.
The current default training budget is about `1.07B` tokens per run (`4,096` steps at `262,144` tokens/step).

The run matrix is:

| ID | Optimizer |
|---|---|
| `BASE-SGD` | SGD + momentum |
| `BASE-ADAM` | Adam |
| `BASE-ADAMW` | AdamW |
| `BASE-MUON` | Muon |
| `ATTN-PURE-8-TRAIN` | attention replaces EMA, context 8 |
| `ATTN-PURE-16-TRAIN` | attention replaces EMA, context 16 |
| `ATTN-GATED-8-TRAIN` | `0.5 * EMA + 0.5 * attention`, context 8 |
| `ATTN-GATED-16-TRAIN` | `0.5 * EMA + 0.5 * attention`, context 16 |

## AttnOpt Design

- Values are the normalized raw gradients themselves.
- The history window is the last `8` or `16` optimizer steps.
- Keys and queries come from low-dimensional gradient statistics plus recency embeddings.
- `W_q`, `W_k`, and recency embeddings are updated online in the trainable variants using a next-gradient prediction surrogate.
- The Adam second moment is kept.

This is not full meta-learning. It is a cheaper online learned-attention approximation suitable for the first experiment pass.

## Project Structure

```text
attn-optimizer/
├── model/
│   └── gpt.py
├── optimizers/
│   ├── attnopt.py
│   └── muon.py
├── data/
│   └── fineweb.py
├── configs/
│   └── runs.py
├── train.py
├── preflight.py
├── launchers/
│   ├── launch_local.sh
│   ├── launch_single.sh
│   └── launch_vast.sh
└── analysis/
    ├── attnopt_toy.py
    ├── attnopt_compare.py
    └── results.ipynb
```

## Setup

```bash
pip install -r requirements.txt
wandb login
```

Run the synthetic smoke test before any paid job:

```bash
python preflight.py --tiny
```

## Local Reproduction

Clone the repo, create an environment, install dependencies, and set the W&B/FineWeb env vars you want to use:

```bash
git clone <your-repo-url> attn-optimizer
cd attn-optimizer

python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

wandb login
export WANDB_ENTITY="your_team_name"
export WANDB_PROJECT="attn-optimizer"
export FINEWEB_MAX_SHARDS=10   # or smaller for a cheaper test
```

Then verify the repo and run either a single experiment or the full sweep:

```bash
python preflight.py --tiny
bash launchers/launch_single.sh BASE-ADAMW
# or
bash launchers/launch_local.sh
```

To inspect results locally after runs finish:

```bash
jupyter notebook analysis/results.ipynb
```

## Running

Single run:

```bash
bash launchers/launch_single.sh BASE-ADAMW
```

Sequential sweep on one GPU:

```bash
bash launchers/launch_local.sh
```

Vast.ai:

```bash
ssh root@<vast-instance>
cd /workspace/attn-optimizer
bash launchers/launch_vast.sh
```

Optional FineWeb cap:

```bash
FINEWEB_MAX_SHARDS=2 bash launchers/launch_single.sh BASE-ADAMW
```

## Toy CPU Diagnostics

Inspect the attention optimizer mechanism:

```bash
python analysis/attnopt_toy.py --steps 30
python analysis/attnopt_compare.py --steps 120
```

## Results

Track live runs on Weights & Biases with project `attn-optimizer`, then use `analysis/results.ipynb` to compare loss curves.
