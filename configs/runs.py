# configs/runs.py
#
# Experiment organization:
# - Experiment 1: state-retention sweep
#   AttnRaw/Avg variants compare keeping both moments vs only v vs neither.
# - Experiment 2: V1-family variations
#   V1-G temperature sweep + V1 mix sweep.
#
# Notes:
# - AttnRaw-V1 uses past-only attention plus an explicit `mix` between current
#   gradient and attended past gradients.
# - AttnRaw-V1-G includes g_t in the attention window, so it has no separate mix.
# - AttnRaw-V1/V2/V3 all use temperature in the softmax over cosine scores.

RUNS = {
    # ==========================================================================
    # EXPERIMENT 1: State-Retention Sweep
    # ==========================================================================
    # --- Baselines ---
    "ADAMW": {
        "optimizer": "adamw",
        "lr": 3e-4,
    },
    "MUON": {
        "optimizer": "muon",
        "lr": 3e-4,
    },
    "SGD": {
        "optimizer": "sgd",
        "lr": 0.1,
    },
    # --- SimpleAvg: keep both m and v ---
    "SIMPLEAVG-V1-L4": {
        "optimizer": "simpleavg_v1",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 4},
    },
    "SIMPLEAVG-V1-L8": {
        "optimizer": "simpleavg_v1",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 8},
    },
    "SIMPLEAVG-V1-L16": {
        "optimizer": "simpleavg_v1",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 16},
    },
    # --- SimpleAvg: keep only v ---
    "SIMPLEAVG-V2-L4": {
        "optimizer": "simpleavg_v2",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 4},
    },
    "SIMPLEAVG-V2-L8": {
        "optimizer": "simpleavg_v2",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 8},
    },
    "SIMPLEAVG-V2-L16": {
        "optimizer": "simpleavg_v2",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 16},
    },
    # --- SimpleAvg: keep neither m nor v ---
    "SIMPLEAVG-V3-L4": {
        "optimizer": "simpleavg_v3",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 4},
    },
    "SIMPLEAVG-V3-L8": {
        "optimizer": "simpleavg_v3",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 8},
    },
    "SIMPLEAVG-V3-L16": {
        "optimizer": "simpleavg_v3",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 16},
    },
    # --- AttnRaw V1: keep both m and v, past-only attention + explicit mix ---
    "ATTNRAW-V1-L4-T0.5": {
        "optimizer": "attnraw_v1",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 4, "temperature": 0.5, "mix": 0.9},
    },
    "ATTNRAW-V1-L4-T1.0": {
        "optimizer": "attnraw_v1",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 4, "temperature": 1.0, "mix": 0.9},
    },
    "ATTNRAW-V1-L4-T2.0": {
        "optimizer": "attnraw_v1",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 4, "temperature": 2.0, "mix": 0.9},
    },
    "ATTNRAW-V1-L8-T0.5": {
        "optimizer": "attnraw_v1",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 8, "temperature": 0.5, "mix": 0.9},
    },
    "ATTNRAW-V1-L8-T1.0": {
        "optimizer": "attnraw_v1",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 8, "temperature": 1.0, "mix": 0.9},
    },
    "ATTNRAW-V1-L8-T2.0": {
        "optimizer": "attnraw_v1",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 8, "temperature": 2.0, "mix": 0.9},
    },
    "ATTNRAW-V1-L16-T0.5": {
        "optimizer": "attnraw_v1",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 16, "temperature": 0.5, "mix": 0.9},
    },
    "ATTNRAW-V1-L16-T1.0": {
        "optimizer": "attnraw_v1",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 16, "temperature": 1.0, "mix": 0.9},
    },
    "ATTNRAW-V1-L16-T2.0": {
        "optimizer": "attnraw_v1",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 16, "temperature": 2.0, "mix": 0.9},
    },
    # --- AttnRaw V2: keep only v ---
    "ATTNRAW-V2-L4-T0.5": {
        "optimizer": "attnraw_v2",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 4, "temperature": 0.5, "mix": 0.9},
    },
    "ATTNRAW-V2-L4-T1.0": {
        "optimizer": "attnraw_v2",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 4, "temperature": 1.0, "mix": 0.9},
    },
    "ATTNRAW-V2-L4-T2.0": {
        "optimizer": "attnraw_v2",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 4, "temperature": 2.0, "mix": 0.9},
    },
    "ATTNRAW-V2-L8-T0.5": {
        "optimizer": "attnraw_v2",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 8, "temperature": 0.5, "mix": 0.9},
    },
    "ATTNRAW-V2-L8-T1.0": {
        "optimizer": "attnraw_v2",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 8, "temperature": 1.0, "mix": 0.9},
    },
    "ATTNRAW-V2-L8-T2.0": {
        "optimizer": "attnraw_v2",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 8, "temperature": 2.0, "mix": 0.9},
    },
    "ATTNRAW-V2-L16-T0.5": {
        "optimizer": "attnraw_v2",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 16, "temperature": 0.5, "mix": 0.9},
    },
    "ATTNRAW-V2-L16-T1.0": {
        "optimizer": "attnraw_v2",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 16, "temperature": 1.0, "mix": 0.9},
    },
    "ATTNRAW-V2-L16-T2.0": {
        "optimizer": "attnraw_v2",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 16, "temperature": 2.0, "mix": 0.9},
    },
    # --- AttnRaw V3: keep neither m nor v ---
    "ATTNRAW-V3-L4-T0.5": {
        "optimizer": "attnraw_v3",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 4, "temperature": 0.5, "mix": 0.9},
    },
    "ATTNRAW-V3-L4-T1.0": {
        "optimizer": "attnraw_v3",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 4, "temperature": 1.0, "mix": 0.9},
    },
    "ATTNRAW-V3-L4-T2.0": {
        "optimizer": "attnraw_v3",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 4, "temperature": 2.0, "mix": 0.9},
    },
    "ATTNRAW-V3-L8-T0.5": {
        "optimizer": "attnraw_v3",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 8, "temperature": 0.5, "mix": 0.9},
    },
    "ATTNRAW-V3-L8-T1.0": {
        "optimizer": "attnraw_v3",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 8, "temperature": 1.0, "mix": 0.9},
    },
    "ATTNRAW-V3-L8-T2.0": {
        "optimizer": "attnraw_v3",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 8, "temperature": 2.0, "mix": 0.9},
    },
    "ATTNRAW-V3-L16-T0.5": {
        "optimizer": "attnraw_v3",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 16, "temperature": 0.5, "mix": 0.9},
    },
    "ATTNRAW-V3-L16-T1.0": {
        "optimizer": "attnraw_v3",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 16, "temperature": 1.0, "mix": 0.9},
    },
    "ATTNRAW-V3-L16-T2.0": {
        "optimizer": "attnraw_v3",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 16, "temperature": 2.0, "mix": 0.9},
    },
    # ==========================================================================
    # EXPERIMENT 2: V1 Variations
    # ==========================================================================
    # --- V1-G temperature sweep ---
    "ATTNRAW-V1-G-L4-T0.5": {
        "optimizer": "attnraw_v1_g",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 4, "temperature": 0.5},
    },
    "ATTNRAW-V1-G-L4-T1.0": {
        "optimizer": "attnraw_v1_g",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 4, "temperature": 1.0},
    },
    "ATTNRAW-V1-G-L4-T2.0": {
        "optimizer": "attnraw_v1_g",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 4, "temperature": 2.0},
    },
    # --- SimpleAvg-G baseline (g_t in window, no temperature) ---
    "SIMPLEAVG-G-V1-L4": {
        "optimizer": "simpleavg_v1_g",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 4},
    },
    # --- V1 mix sweep ---
    "ATTNRAW-MIX10-L4-T1.0": {
        "optimizer": "attnraw_v1",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 4, "temperature": 1.0, "mix": 0.1},
    },
    "ATTNRAW-MIX25-L4-T1.0": {
        "optimizer": "attnraw_v1",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 4, "temperature": 1.0, "mix": 0.25},
    },
    "ATTNRAW-MIX50-L4-T1.0": {
        "optimizer": "attnraw_v1",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 4, "temperature": 1.0, "mix": 0.5},
    },
    "ATTNRAW-MIX75-L4-T1.0": {
        "optimizer": "attnraw_v1",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 4, "temperature": 1.0, "mix": 0.75},
    },
    "ATTNRAW-MIX90-L4-T1.0": {
        "optimizer": "attnraw_v1",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 4, "temperature": 1.0, "mix": 0.9},
    },
    "SIMPLEAVG-MIX10-L4-T1.0": {
        "optimizer": "simpleavg_v1",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 4, "mix": 0.1},
    },
    "SIMPLEAVG-MIX25-L4-T1.0": {
        "optimizer": "simpleavg_v1",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 4, "mix": 0.25},
    },
    "SIMPLEAVG-MIX50-L4-T1.0": {
        "optimizer": "simpleavg_v1",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 4, "mix": 0.5},
    },
    "SIMPLEAVG-MIX75-L4-T1.0": {
        "optimizer": "simpleavg_v1",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 4, "mix": 0.75},
    },
    "SIMPLEAVG-MIX90-L4-T1.0": {
        "optimizer": "simpleavg_v1",
        "lr": 3e-4,
        "grad_opt_config": {"context_length": 4, "mix": 0.9},
    },
}


TRAIN_CONFIG = {
    "max_steps": 7_500,
    "warmup_steps": 500,
    "min_lr_ratio": 0.1,
    "micro_batch_size": 32,
    "grad_accum_steps": 8,
    "seq_len": 1024,
    "grad_clip": 1.0,
    "log_interval": 25,
    "seed": 42,
}

MODEL_CONFIG = {
    "n_layer": 6,
    "n_head": 8,
    "n_embd": 512,
    "vocab_size": 50304,
    "block_size": 1024,
}
