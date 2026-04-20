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

TOTAL_RUNS=0
TOTAL_EXPERIMENT_GROUPS=0
if [[ "$EXPERIMENT_SELECTOR" == "1" || "$EXPERIMENT_SELECTOR" == "all" ]]; then
    TOTAL_RUNS=$((TOTAL_RUNS + ${#EXPERIMENT_1_RUNS[@]}))
    TOTAL_EXPERIMENT_GROUPS=$((TOTAL_EXPERIMENT_GROUPS + 1))
fi
if [[ "$EXPERIMENT_SELECTOR" == "2" || "$EXPERIMENT_SELECTOR" == "all" ]]; then
    TOTAL_RUNS=$((TOTAL_RUNS + ${#EXPERIMENT_2_RUNS[@]} * ${#EXPERIMENT_2_SEEDS[@]}))
    TOTAL_EXPERIMENT_GROUPS=$((TOTAL_EXPERIMENT_GROUPS + ${#EXPERIMENT_2_SEEDS[@]}))
fi

RUNS_COMPLETED=0
FAILED_COUNT=0
EXPERIMENT_GROUPS_COMPLETED=0
PROGRESS_FILE="${PROGRESS_FILE:-$ROOT_DIR/logs/experiment_progress_sequential.json}"
STATE_DIR="$ROOT_DIR/.run_experiments_state"
JOB_CATALOG_FILE="$STATE_DIR/jobs_catalog_sequential.tsv"
DONE_FILE="$STATE_DIR/done_sequential.tsv"

mkdir -p "$STATE_DIR"
mkdir -p "$(dirname "$PROGRESS_FILE")"

: > "$JOB_CATALOG_FILE"
: > "$DONE_FILE"

echo "Total runs queued: $TOTAL_RUNS"
echo "Experiment groups queued: $TOTAL_EXPERIMENT_GROUPS"
echo "Progress file: $PROGRESS_FILE"

append_catalog_job() {
    local exp_label="$1"
    local seed="$2"
    local run_id="$3"
    printf '%s\t%s\t%s\n' "$exp_label" "$seed" "$run_id" >> "$JOB_CATALOG_FILE"
}

if [[ "$EXPERIMENT_SELECTOR" == "1" || "$EXPERIMENT_SELECTOR" == "all" ]]; then
    for run_id in "${EXPERIMENT_1_RUNS[@]}"; do
        append_catalog_job "experiment_1" "$EXPERIMENT_1_SEED" "$run_id"
    done
fi
if [[ "$EXPERIMENT_SELECTOR" == "2" || "$EXPERIMENT_SELECTOR" == "all" ]]; then
    for seed in "${EXPERIMENT_2_SEEDS[@]}"; do
        for run_id in "${EXPERIMENT_2_RUNS[@]}"; do
            append_catalog_job "experiment_2_seed_${seed}" "$seed" "$run_id"
        done
    done
fi

write_progress_json() {
    python - "$JOB_CATALOG_FILE" "$DONE_FILE" "$PROGRESS_FILE" "$TOTAL_RUNS" "$TOTAL_EXPERIMENT_GROUPS" <<'PY'
import json
import os
import sys
import time

catalog_path, done_path, progress_path, total_runs_raw, total_groups_raw = sys.argv[1:6]
total_runs = int(total_runs_raw)
total_groups = int(total_groups_raw)

jobs = []
exp_totals = {}
with open(catalog_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        exp_label, seed, run_id = line.split("\t")
        key = f"{exp_label}|{seed}|{run_id}"
        jobs.append((key, exp_label, seed, run_id))
        exp_totals[exp_label] = exp_totals.get(exp_label, 0) + 1

done_map = {}
if os.path.exists(done_path):
    with open(done_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            exp_label, seed, run_id, status, exit_code = line.split("\t")
            key = f"{exp_label}|{seed}|{run_id}"
            done_map[key] = {
                "experiment_group": exp_label,
                "seed": seed,
                "run_id": run_id,
                "status": status,
                "exit_code": int(exit_code),
            }

completed = len(done_map)
failed = sum(1 for item in done_map.values() if item["status"] != "ok")
remaining = max(total_runs - completed, 0)

exp_done = {}
for item in done_map.values():
    exp_label = item["experiment_group"]
    exp_done[exp_label] = exp_done.get(exp_label, 0) + 1

experiment_progress = []
for exp_label in sorted(exp_totals):
    done_count = exp_done.get(exp_label, 0)
    total_count = exp_totals[exp_label]
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
            }
        )

payload = {
    "updated_at_unix": time.time(),
    "total_jobs": total_runs,
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

run_group() {
    local exp_label="$1"
    local group_name="$2"
    local log_dir="$3"
    local ckpt_dir="$4"
    local seed="$5"
    shift 5
    local run_ids=("$@")
    local group_total="${#run_ids[@]}"
    local group_done=0
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
        local status="ok"
        local exit_code=0
        if LOG_DIR="$log_dir" \
            CKPT_DIR="$ckpt_dir" \
            SEED="$seed" \
            python "$ROOT_DIR/train.py" --run_id "$run_id"; then
            echo "Finished ${run_id}"
        else
            exit_code=$?
            status="failed"
            echo "FAILED ${run_id} (seed=${seed}, exit=${exit_code})"
            FAILED_RUNS+=("${group_name}"$'\t'"${seed}"$'\t'"${run_id}"$'\t'"${exit_code}")
            group_failed=1
            FAILED_COUNT=$((FAILED_COUNT + 1))
        fi

        printf '%s\t%s\t%s\t%s\t%s\n' "$exp_label" "$seed" "$run_id" "$status" "$exit_code" >> "$DONE_FILE"
        write_progress_json

        group_done=$((group_done + 1))
        RUNS_COMPLETED=$((RUNS_COMPLETED + 1))

        local runs_left group_left groups_left
        runs_left=$((TOTAL_RUNS - RUNS_COMPLETED))
        group_left=$((group_total - group_done))
        groups_left=$((TOTAL_EXPERIMENT_GROUPS - EXPERIMENT_GROUPS_COMPLETED - 1))
        if [[ "$group_left" -gt 0 ]]; then
            groups_left=$((groups_left + 1))
        fi
        echo "[Progress] ${RUNS_COMPLETED}/${TOTAL_RUNS} runs done, ${runs_left} left, failed=${FAILED_COUNT} | ${group_name}: ${group_done}/${group_total} done, ${group_left} left | experiment groups left=${groups_left}/${TOTAL_EXPERIMENT_GROUPS}"
    done

    EXPERIMENT_GROUPS_COMPLETED=$((EXPERIMENT_GROUPS_COMPLETED + 1))
    local groups_left
    groups_left=$((TOTAL_EXPERIMENT_GROUPS - EXPERIMENT_GROUPS_COMPLETED))
    echo "[Experiment Progress] ${EXPERIMENT_GROUPS_COMPLETED}/${TOTAL_EXPERIMENT_GROUPS} groups completed, ${groups_left} groups left"

    return "$group_failed"
}

had_failure=0

if [[ "$EXPERIMENT_SELECTOR" == "1" || "$EXPERIMENT_SELECTOR" == "all" ]]; then
    if ! run_group \
        "experiment_1" \
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
            "experiment_2_seed_${seed}" \
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
