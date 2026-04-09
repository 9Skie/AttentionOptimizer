#!/usr/bin/env bash
# launch_local.sh
#
# Runs the optimizer-study experiment matrix sequentially on one GPU.
# Good fit for a single RunPod instance or local machine.

set -e

RUN_IDS=(
    SIMPLEAVG-L4
    ATTNRAW-V1-L4
    ATTNRAW-V1-L4-MIX10
    ATTNRAW-V1-G-L4
    ATTNRAW-V1-G-L4-T0.5
    ATTNRAW-V2-L4
    ATTNRAW-V3-L4
    ATTNRAW-V3-L4-MIX10
    MUON
)

echo "Starting sequential training of ${#RUN_IDS[@]} runs..."
echo ""

for RUN_ID in "${RUN_IDS[@]}"; do
    echo "============================================"
    echo " Starting: $RUN_ID"
    echo "============================================"
    python train.py --run_id "$RUN_ID"
    echo ""
    echo " Done: $RUN_ID"
    echo ""
done

echo "All runs complete."