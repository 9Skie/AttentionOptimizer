#!/usr/bin/env bash
# launchers/launch_single.sh
#
# Run a single optimizer-study experiment by run ID.

set -e

if [ -z "$1" ]; then
    echo "Usage: bash launchers/launch_single.sh <RUN_ID>"
    echo ""
    echo "Available run IDs:"
    echo "  Baselines: BASE-SGD BASE-ADAM BASE-ADAMW BASE-MUON"
    echo "  AttnOpt:   ATTN-PURE-8-TRAIN ATTN-PURE-16-TRAIN"
    echo "             ATTN-GATED-8-TRAIN ATTN-GATED-16-TRAIN"
    exit 1
fi

RUN_ID="$1"
echo "Running: $RUN_ID"
python train.py --run_id "$RUN_ID"
