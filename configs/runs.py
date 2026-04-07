# configs/runs.py
# Simple experiment: Muon + SimpleAvg variants for quick comparison

import itertools

HISTORY_LENGTHS = [4, 8, 16]

RUNS = {}

# --- Muon: Newton-based optimizer (embeddings use Adam) ---
RUNS["MUON"] = {
    "optimizer": "muon",
    "lr": 3e-4,
}

# --- SimpleAvg-v1: keep both m_{t-1} and v_{t-1} ---
for L in HISTORY_LENGTHS:
    key = f"AVG-V1-L{L}"
    RUNS[key] = {
        "optimizer": "simpleavg_v1",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "avg_config": {
            "context_length": L,
            "mix_beta": 0.9,
        },
    }

# --- SimpleAvg-v2: remove m_{t-1}, keep v_{t-1} ---
for L in HISTORY_LENGTHS:
    key = f"AVG-V2-L{L}"
    RUNS[key] = {
        "optimizer": "simpleavg_v2",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "avg_config": {
            "context_length": L,
            "mix_beta": 0.9,
        },
    }

# --- SimpleAvg-v3: remove both m_{t-1} and v_{t-1} ---
for L in HISTORY_LENGTHS:
    key = f"AVG-V3-L{L}"
    RUNS[key] = {
        "optimizer": "simpleavg_v3",
        "lr": 3e-4,
        "weight_decay": 0.0,
        "avg_config": {
            "context_length": L,
            "mix_beta": 0.9,
        },
    }

TRAIN_CONFIG = {
    "max_steps": 16_000,
    "warmup_steps": 500,
    "min_lr_ratio": 0.1,
    "micro_batch_size": 16,
    "grad_accum_steps": 16,
    "seq_len": 1024,
    "grad_clip": 1.0,
    "log_interval": 100,
    "seed": 42,
}

MODEL_CONFIG = {
    "n_layer": 6,
    "n_head": 8,
    "n_embd": 512,
    "vocab_size": 50304,
    "block_size": 1024,
}
