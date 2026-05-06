#!/usr/bin/env bash
# DEPRECATED 2026-05-06: A1 stage 已撤销（thesis 矩阵下调，详见 docs/online_rl_line_summary.md）。
# 此脚本保留作为历史接口；不要再用它汇总新实验。
set -euo pipefail

export STAGE_LABEL="${STAGE_LABEL:-A1}"
export STAGE_DIR="${STAGE_DIR:-A1_single_u10_upstream_tgt15}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/summarize_protocol_stage_common.sh"
