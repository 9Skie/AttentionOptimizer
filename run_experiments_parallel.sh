#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_SELECTOR="all"
EXPERIMENT_1_SEED=42
SEEDS_CSV="42,67,69"

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
    echo "Usage: bash run_experiments_parallel.sh [NUM_GPUS] [--experiment 1|2|all] [--seeds 42,67,69]"
}

detect_gpus() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '
        return
    fi
    echo 1
}

N_GPUS=""
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
            if [[ -z "$N_GPUS" && "$1" =~ ^[0-9]+$ ]]; then
                N_GPUS="$1"
                shift
            else
                usage
                exit 1
            fi
            ;;
    esac
done

if [[ -z "$N_GPUS" ]]; then
    N_GPUS="$(detect_gpus)"
fi
if ! [[ "$N_GPUS" =~ ^[0-9]+$ ]] || [[ "$N_GPUS" -lt 1 ]]; then
    usage
    exit 1
fi
if [[ "$EXPERIMENT_SELECTOR" != "1" && "$EXPERIMENT_SELECTOR" != "2" && "$EXPERIMENT_SELECTOR" != "all" ]]; then
    echo "Invalid --experiment value: $EXPERIMENT_SELECTOR"
    usage
    exit 1
fi

IFS=',' read -r -a EXPERIMENT_2_SEEDS <<< "$SEEDS_CSV"

STATE_DIR="$ROOT_DIR/.run_experiments_parallel"
JOB_FILE="$STATE_DIR/jobs.tsv"
LOCK_FILE="$STATE_DIR/jobs.lock"
FAILED_FILE="$STATE_DIR/failed.tsv"
mkdir -p "$STATE_DIR"

cleanup() {
    rm -rf "$STATE_DIR"
}
trap cleanup EXIT

append_job() {
    local log_dir="$1"
    local ckpt_dir="$2"
    local seed="$3"
    local run_id="$4"
    printf '%s\t%s\t%s\t%s\n' "$log_dir" "$ckpt_dir" "$seed" "$run_id" >> "$JOB_FILE"
}

: > "$JOB_FILE"
: > "$FAILED_FILE"

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set. FineWeb requires a HuggingFace token."
    echo "  export HF_TOKEN=hf_..."
    exit 1
fi

echo "Ensuring FineWeb data is cached before launching workers..."
python "$ROOT_DIR/data/fineweb.py" --max-shards 20
echo "Data ready."

if [[ "$EXPERIMENT_SELECTOR" == "1" || "$EXPERIMENT_SELECTOR" == "all" ]]; then
    for run_id in "${EXPERIMENT_1_RUNS[@]}"; do
        append_job \
            "$ROOT_DIR/logs/experiment_1" \
            "$ROOT_DIR/checkpoints/experiment_1" \
            "$EXPERIMENT_1_SEED" \
            "$run_id"
    done
fi

if [[ "$EXPERIMENT_SELECTOR" == "2" || "$EXPERIMENT_SELECTOR" == "all" ]]; then
    for seed in "${EXPERIMENT_2_SEEDS[@]}"; do
        for run_id in "${EXPERIMENT_2_RUNS[@]}"; do
            append_job \
                "$ROOT_DIR/logs/experiment_2/seed_${seed}" \
                "$ROOT_DIR/checkpoints/experiment_2/seed_${seed}" \
                "$seed" \
                "$run_id"
        done
    done
fi

TOTAL_JOBS=$(wc -l < "$JOB_FILE" | tr -d ' ')
echo "Using $N_GPUS GPU(s)"
echo "Total jobs: $TOTAL_JOBS"

claim_job() {
    exec 9>"$LOCK_FILE"
    flock 9

    if [[ ! -s "$JOB_FILE" ]]; then
        flock -u 9
        exec 9>&-
        return 1
    fi

    IFS=$'\t' read -r log_dir ckpt_dir seed run_id < "$JOB_FILE"
    tail -n +2 "$JOB_FILE" > "$JOB_FILE.tmp"
    mv "$JOB_FILE.tmp" "$JOB_FILE"

    flock -u 9
    exec 9>&-

    printf '%s\t%s\t%s\t%s\n' "$log_dir" "$ckpt_dir" "$seed" "$run_id"
    return 0
}

record_failure() {
    local gpu_name="$1"
    local run_id="$2"
    local seed="$3"
    local exit_code="$4"

    exec 8>"$LOCK_FILE"
    flock 8
    printf '%s\t%s\t%s\t%s\n' "$gpu_name" "$run_id" "$seed" "$exit_code" >> "$FAILED_FILE"
    flock -u 8
    exec 8>&-
}

worker() {
    local gpu_id="$1"
    local gpu_name="GPU${gpu_id}"

    while true; do
        local job
        if ! job=$(claim_job); then
            echo "[$gpu_name] No jobs remaining"
            break
        fi

        local log_dir ckpt_dir seed run_id
        IFS=$'\t' read -r log_dir ckpt_dir seed run_id <<< "$job"
        mkdir -p "$log_dir" "$ckpt_dir"

        echo "[$gpu_name] Starting $run_id (seed=$seed)"
        if LOG_DIR="$log_dir" \
            CKPT_DIR="$ckpt_dir" \
            SEED="$seed" \
            CUDA_VISIBLE_DEVICES="$gpu_id" \
            python "$ROOT_DIR/train.py" --run_id "$run_id"; then
            echo "[$gpu_name] Finished $run_id (seed=$seed)"
        else
            exit_code=$?
            echo "[$gpu_name] FAILED $run_id (seed=$seed, exit=$exit_code)"
            record_failure "$gpu_name" "$run_id" "$seed" "$exit_code"
        fi
    done
}

for ((gpu = 0; gpu < N_GPUS; gpu++)); do
    worker "$gpu" &
done

wait

if [[ -s "$FAILED_FILE" ]]; then
    echo "Some experiment jobs failed:"
    while IFS=$'\t' read -r gpu_name run_id seed exit_code; do
        echo "  [$gpu_name] $run_id (seed=$seed, exit=$exit_code)"
    done < "$FAILED_FILE"
    exit 1
fi

echo "All experiment jobs completed successfully."
