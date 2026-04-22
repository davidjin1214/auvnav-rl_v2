#!/usr/bin/env bash
set -euo pipefail

export STAGE_LABEL="${STAGE_LABEL:-A1}"
export STAGE_DIR="${STAGE_DIR:-A1_single_u10_upstream_tgt15}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/summarize_protocol_stage_common.sh"
