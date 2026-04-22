#!/usr/bin/env bash
set -euo pipefail

export STAGE_LABEL="${STAGE_LABEL:-A1}"
export STAGE_DIR="${STAGE_DIR:-A1_single_u10_upstream_tgt15}"
export BENCHMARK_KEY="${BENCHMARK_KEY:-single_u10_upstream_tgt15}"
export FLOW_PATH="${FLOW_PATH:-wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy}"
export TASK_GEOMETRY="${TASK_GEOMETRY:-upstream}"
export TARGET_SPEED="${TARGET_SPEED:-1.5}"
export HISTORY_LENGTH="${HISTORY_LENGTH:-4}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_protocol_stage_common.sh"
