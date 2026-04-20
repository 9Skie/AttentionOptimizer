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
JOB_CATALOG_FILE="$STATE_DIR/jobs_catalog.tsv"
LOCK_FILE="$STATE_DIR/jobs.lock"
DONE_FILE="$STATE_DIR/done.tsv"
FAILED_FILE="$STATE_DIR/failed.tsv"
PROGRESS_FILE="${PROGRESS_FILE:-$ROOT_DIR/logs/experiment_progress_parallel.json}"
mkdir -p "$STATE_DIR"
mkdir -p "$(dirname "$PROGRESS_FILE")"

cleanup() {
    rm -rf "$STATE_DIR"
}
trap cleanup EXIT

append_job() {
    local exp_label="$1"
    local log_dir="$2"
    local ckpt_dir="$3"
    local seed="$4"
    local run_id="$5"
    printf '%s\t%s\t%s\t%s\t%s\n' "$exp_label" "$log_dir" "$ckpt_dir" "$seed" "$run_id" >> "$JOB_FILE"
}

: > "$JOB_FILE"
: > "$JOB_CATALOG_FILE"
: > "$DONE_FILE"
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
            "experiment_1" \
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
                "experiment_2_seed_${seed}" \
                "$ROOT_DIR/logs/experiment_2/seed_${seed}" \
                "$ROOT_DIR/checkpoints/experiment_2/seed_${seed}" \
                "$seed" \
                "$run_id"
        done
    done
fi

cp "$JOB_FILE" "$JOB_CATALOG_FILE"

TOTAL_JOBS=$(wc -l < "$JOB_FILE" | tr -d ' ')
TOTAL_EXPERIMENT_GROUPS=$(awk -F'\t' '{labels[$1]=1} END {print length(labels)}' "$JOB_CATALOG_FILE")
echo "Using $N_GPUS GPU(s)"
echo "Total jobs: $TOTAL_JOBS"
echo "Experiment groups: $TOTAL_EXPERIMENT_GROUPS"
echo "Progress file: $PROGRESS_FILE"

write_progress_json() {
    python - "$JOB_CATALOG_FILE" "$DONE_FILE" "$PROGRESS_FILE" "$TOTAL_JOBS" "$TOTAL_EXPERIMENT_GROUPS" <<'PY'
import json
import os
import sys
import time

job_catalog_path, done_path, progress_path, total_jobs_raw, total_groups_raw = sys.argv[1:6]
total_jobs = int(total_jobs_raw)
total_groups = int(total_groups_raw)

jobs = []
experiment_totals = {}

with open(job_catalog_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        exp_label, _log_dir, _ckpt_dir, seed, run_id = line.split("\t")
        key = f"{exp_label}|{seed}|{run_id}"
        jobs.append((key, exp_label, seed, run_id))
        experiment_totals[exp_label] = experiment_totals.get(exp_label, 0) + 1

done_map = {}
if os.path.exists(done_path):
    with open(done_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            exp_label, gpu_name, run_id, seed, status, exit_code = line.split("\t")
            key = f"{exp_label}|{seed}|{run_id}"
            done_map[key] = {
                "experiment_group": exp_label,
                "seed": seed,
                "run_id": run_id,
                "gpu": gpu_name,
                "status": status,
                "exit_code": int(exit_code),
            }

completed = len(done_map)
failed = sum(1 for item in done_map.values() if item["status"] != "ok")
remaining = max(total_jobs - completed, 0)

experiment_done = {}
for item in done_map.values():
    exp_label = item["experiment_group"]
    experiment_done[exp_label] = experiment_done.get(exp_label, 0) + 1

experiment_progress = []
for exp_label in sorted(experiment_totals):
    done_count = experiment_done.get(exp_label, 0)
    total_count = experiment_totals[exp_label]
    experiment_progress.append(
        {
            "experiment_group": exp_label,
            "done": done_count,
            "total": total_count,
            "remaining": max(total_count - done_count, 0),
        }
    )

job_status = []
for key, exp_label, seed, run_id in jobs:
    item = done_map.get(key)
    if item is None:
        job_status.append(
            {
                "experiment_group": exp_label,
                "seed": seed,
                "run_id": run_id,
                "status": "pending",
                "exit_code": None,
                "gpu": None,
            }
        )
    else:
        job_status.append(
            {
                "experiment_group": exp_label,
                "seed": seed,
                "run_id": run_id,
                "status": item["status"],
                "exit_code": item["exit_code"],
                "gpu": item["gpu"],
            }
        )

payload = {
    "updated_at_unix": time.time(),
    "total_jobs": total_jobs,
    "completed_jobs": completed,
    "remaining_jobs": remaining,
    "failed_jobs": failed,
    "total_experiment_groups": total_groups,
    "remaining_experiment_groups": sum(1 for e in experiment_progress if e["remaining"] > 0),
    "experiment_progress": experiment_progress,
    "jobs": job_status,
}

with open(progress_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
PY
}

write_progress_json

claim_job() {
    local exp_label log_dir ckpt_dir seed run_id
    exec 9>"$LOCK_FILE"
    flock 9

    if [[ ! -s "$JOB_FILE" ]]; then
        flock -u 9
        exec 9>&-
        return 1
    fi

    IFS=$'\t' read -r exp_label log_dir ckpt_dir seed run_id < "$JOB_FILE"
    tail -n +2 "$JOB_FILE" > "$JOB_FILE.tmp"
    mv "$JOB_FILE.tmp" "$JOB_FILE"

    flock -u 9
    exec 9>&-

    printf '%s\t%s\t%s\t%s\t%s\n' "$exp_label" "$log_dir" "$ckpt_dir" "$seed" "$run_id"
    return 0
}

record_completion() {
    local exp_label="$1"
    local gpu_name="$2"
    local run_id="$3"
    local seed="$4"
    local status="$5"
    local exit_code="$6"
    local completed remaining failed exp_done exp_total exp_remaining experiments_left

    exec 8>"$LOCK_FILE"
    flock 8
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$exp_label" "$gpu_name" "$run_id" "$seed" "$status" "$exit_code" >> "$DONE_FILE"
    if [[ "$status" != "ok" ]]; then
        printf '%s\t%s\t%s\t%s\t%s\n' "$exp_label" "$gpu_name" "$run_id" "$seed" "$exit_code" >> "$FAILED_FILE"
    fi

    completed=$(wc -l < "$DONE_FILE" | tr -d ' ')
    failed=$(wc -l < "$FAILED_FILE" | tr -d ' ')
    remaining=$((TOTAL_JOBS - completed))
    exp_done=$(awk -F'\t' -v exp="$exp_label" '$1==exp {c++} END {print c+0}' "$DONE_FILE")
    exp_total=$(awk -F'\t' -v exp="$exp_label" '$1==exp {c++} END {print c+0}' "$JOB_CATALOG_FILE")
    exp_remaining=$((exp_total - exp_done))
    experiments_left=$(awk -F'\t' 'NR==FNR {total[$1]++; next} {done[$1]++} END {for (k in total) if (done[k] < total[k]) c++; print c+0}' "$JOB_CATALOG_FILE" "$DONE_FILE")

    write_progress_json

    echo "[Progress] ${completed}/${TOTAL_JOBS} runs done, ${remaining} left, failed=${failed} | ${exp_label}: ${exp_done}/${exp_total} done, ${exp_remaining} left | experiment groups left=${experiments_left}/${TOTAL_EXPERIMENT_GROUPS}"

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

        local exp_label log_dir ckpt_dir seed run_id
        IFS=$'\t' read -r exp_label log_dir ckpt_dir seed run_id <<< "$job"
        mkdir -p "$log_dir" "$ckpt_dir"

        echo "[$gpu_name] Starting $run_id (seed=$seed, group=$exp_label)"
        if LOG_DIR="$log_dir" \
            CKPT_DIR="$ckpt_dir" \
            SEED="$seed" \
            CUDA_VISIBLE_DEVICES="$gpu_id" \
            python "$ROOT_DIR/train.py" --run_id "$run_id"; then
            echo "[$gpu_name] Finished $run_id (seed=$seed)"
            record_completion "$exp_label" "$gpu_name" "$run_id" "$seed" "ok" 0
        else
            exit_code=$?
            echo "[$gpu_name] FAILED $run_id (seed=$seed, exit=$exit_code)"
            record_completion "$exp_label" "$gpu_name" "$run_id" "$seed" "failed" "$exit_code"
        fi
    done
}

for ((gpu = 0; gpu < N_GPUS; gpu++)); do
    worker "$gpu" &
done

wait

if [[ -s "$FAILED_FILE" ]]; then
    echo "Some experiment jobs failed:"
    while IFS=$'\t' read -r exp_label gpu_name run_id seed exit_code; do
        echo "  [${exp_label}] [$gpu_name] $run_id (seed=$seed, exit=$exit_code)"
    done < "$FAILED_FILE"
    exit 1
fi

echo "All experiment jobs completed successfully."
