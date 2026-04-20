#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_SELECTOR="all"
EXPERIMENT_1_SEED=42
SEEDS_CSV="42,67,69"
FAILED_RUNS=()

EXPERIMENT_1_RUNS=(
    "ADAMW"
    "MUON"
    "SGD"
    "SIMPLEAVG-V1-L4"
    "SIMPLEAVG-V1-L8"
    "SIMPLEAVG-V1-L16"
    "SIMPLEAVG-V2-L4"
    "SIMPLEAVG-V2-L8"
    "SIMPLEAVG-V2-L16"
    "SIMPLEAVG-V3-L4"
    "SIMPLEAVG-V3-L8"
    "SIMPLEAVG-V3-L16"
    "ATTNRAW-V1-L4-T0.5"
    "ATTNRAW-V1-L4-T1.0"
    "ATTNRAW-V1-L4-T2.0"
    "ATTNRAW-V1-L8-T0.5"
    "ATTNRAW-V1-L8-T1.0"
    "ATTNRAW-V1-L8-T2.0"
    "ATTNRAW-V1-L16-T0.5"
    "ATTNRAW-V1-L16-T1.0"
    "ATTNRAW-V1-L16-T2.0"
    "ATTNRAW-V2-L4-T0.5"
    "ATTNRAW-V2-L4-T1.0"
    "ATTNRAW-V2-L4-T2.0"
    "ATTNRAW-V2-L8-T0.5"
    "ATTNRAW-V2-L8-T1.0"
    "ATTNRAW-V2-L8-T2.0"
    "ATTNRAW-V2-L16-T0.5"
    "ATTNRAW-V2-L16-T1.0"
    "ATTNRAW-V2-L16-T2.0"
    "ATTNRAW-V3-L4-T0.5"
    "ATTNRAW-V3-L4-T1.0"
    "ATTNRAW-V3-L4-T2.0"
    "ATTNRAW-V3-L8-T0.5"
    "ATTNRAW-V3-L8-T1.0"
    "ATTNRAW-V3-L8-T2.0"
    "ATTNRAW-V3-L16-T0.5"
    "ATTNRAW-V3-L16-T1.0"
    "ATTNRAW-V3-L16-T2.0"
)

EXPERIMENT_2_RUNS=(
    "SIMPLEAVG-G-V1-L4"
    "ATTNRAW-V1-G-L4-T0.5"
    "ATTNRAW-V1-G-L4-T1.0"
    "ATTNRAW-V1-G-L4-T2.0"
    "ATTNRAW-MIX10-L4-T1.0"
    "ATTNRAW-MIX25-L4-T1.0"
    "ATTNRAW-MIX50-L4-T1.0"
    "ATTNRAW-MIX75-L4-T1.0"
    "ATTNRAW-MIX90-L4-T1.0"
    "SIMPLEAVG-MIX10-L4-T1.0"
    "SIMPLEAVG-MIX25-L4-T1.0"
    "SIMPLEAVG-MIX50-L4-T1.0"
    "SIMPLEAVG-MIX75-L4-T1.0"
    "SIMPLEAVG-MIX90-L4-T1.0"
)

usage() {
    echo "Usage: bash run_experiments.sh [--experiment 1|2|all] [--seeds 42,67,69]"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment)
            EXPERIMENT_SELECTOR="$2"
            shift 2
            ;;
        --seeds)
            SEEDS_CSV="$2"
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            usage
            exit 1
            ;;
    esac
done

if [[ "$EXPERIMENT_SELECTOR" != "1" && "$EXPERIMENT_SELECTOR" != "2" && "$EXPERIMENT_SELECTOR" != "all" ]]; then
    echo "Invalid --experiment value: $EXPERIMENT_SELECTOR"
    usage
    exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set. FineWeb requires a HuggingFace token."
    echo "  export HF_TOKEN=hf_..."
    exit 1
fi

IFS=',' read -r -a EXPERIMENT_2_SEEDS <<< "$SEEDS_CSV"

run_group() {
    local group_name="$1"
    local log_dir="$2"
    local ckpt_dir="$3"
    local seed="$4"
    shift 4
    local run_ids=("$@")
    local group_failed=0

    mkdir -p "$log_dir" "$ckpt_dir"

    echo ""
    echo "============================================"
    echo "${group_name}"
    echo "log dir : ${log_dir}"
    echo "ckpt dir: ${ckpt_dir}"
    echo "seed    : ${seed}"
    echo "runs    : ${#run_ids[@]}"
    echo "============================================"

    for run_id in "${run_ids[@]}"; do
        echo ""
        echo "--------------------------------------------"
        echo "Starting ${run_id}"
        echo "--------------------------------------------"
        if LOG_DIR="$log_dir" \
            CKPT_DIR="$ckpt_dir" \
            SEED="$seed" \
            python "$ROOT_DIR/train.py" --run_id "$run_id"; then
            echo "Finished ${run_id}"
        else
            local exit_code=$?
            echo "FAILED ${run_id} (seed=${seed}, exit=${exit_code})"
            FAILED_RUNS+=("${group_name}"$'\t'"${seed}"$'\t'"${run_id}"$'\t'"${exit_code}")
            group_failed=1
        fi
    done

    return "$group_failed"
}

had_failure=0

if [[ "$EXPERIMENT_SELECTOR" == "1" || "$EXPERIMENT_SELECTOR" == "all" ]]; then
    if ! run_group \
        "Experiment 1" \
        "$ROOT_DIR/logs/experiment_1" \
        "$ROOT_DIR/checkpoints/experiment_1" \
        "$EXPERIMENT_1_SEED" \
        "${EXPERIMENT_1_RUNS[@]}"; then
        had_failure=1
    fi
fi

if [[ "$EXPERIMENT_SELECTOR" == "2" || "$EXPERIMENT_SELECTOR" == "all" ]]; then
    for seed in "${EXPERIMENT_2_SEEDS[@]}"; do
        if ! run_group \
            "Experiment 2 (seed ${seed})" \
            "$ROOT_DIR/logs/experiment_2/seed_${seed}" \
            "$ROOT_DIR/checkpoints/experiment_2/seed_${seed}" \
            "$seed" \
            "${EXPERIMENT_2_RUNS[@]}"; then
            had_failure=1
        fi
    done
fi

if [[ "$had_failure" -ne 0 ]]; then
    echo ""
    echo "Some experiment jobs failed:"
    for entry in "${FAILED_RUNS[@]}"; do
        IFS=$'\t' read -r group_name seed run_id exit_code <<< "$entry"
        echo "  [${group_name}] ${run_id} (seed=${seed}, exit=${exit_code})"
    done
    exit 1
fi

echo "All experiment jobs completed successfully."
