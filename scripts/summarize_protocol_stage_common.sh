#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_PREFIX="${PYTHON_PREFIX:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_CMD=()
if [[ -n "$PYTHON_PREFIX" ]]; then
  read -r -a PYTHON_PREFIX_ARR <<< "$PYTHON_PREFIX"
  PYTHON_CMD+=("${PYTHON_PREFIX_ARR[@]}")
fi
PYTHON_CMD+=("$PYTHON_BIN")

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
mkdir -p "$MPLCONFIGDIR"
export MPLCONFIGDIR

STUDY_ROOT="${STUDY_ROOT:-protocol_screen_v2}"
if [[ -z "${STAGE_DIR:-}" ]]; then
  echo "summarize_protocol_stage_common.sh requires STAGE_DIR" >&2
  exit 1
fi

OBJECTIVE="${OBJECTIVE:-efficiency_v2}"
ALGO_TAG="${ALGO_TAG:-sac_vanilla}"
DEFAULT_SUITE_ROOT="experiments/${STUDY_ROOT}/${STAGE_DIR}/${OBJECTIVE}/${ALGO_TAG}"
LEGACY_SUITE_ROOT="experiments/${STUDY_ROOT}/${STAGE_DIR}/${OBJECTIVE}"
if [[ -n "${SUITE_ROOT:-}" ]]; then
  SUITE_ROOT="${SUITE_ROOT}"
elif [[ -d "$DEFAULT_SUITE_ROOT" ]]; then
  SUITE_ROOT="$DEFAULT_SUITE_ROOT"
elif [[ -d "$LEGACY_SUITE_ROOT" ]]; then
  SUITE_ROOT="$LEGACY_SUITE_ROOT"
else
  SUITE_ROOT="$DEFAULT_SUITE_ROOT"
fi
SUMMARY_DIR="${SUMMARY_DIR:-${SUITE_ROOT}/summary}"
PLOT_PREFIX="${PLOT_PREFIX:-${SUMMARY_DIR}/ablation_overview}"

echo "[summary] stage=${STAGE_LABEL:-stage} suite_root=${SUITE_ROOT}"
echo "[summary] output_dir=${SUMMARY_DIR}"
"${PYTHON_CMD[@]}" -m scripts.summarize_suite \
  --suite-root "$SUITE_ROOT" \
  --output-dir "$SUMMARY_DIR"

echo "[plot] output_prefix=${PLOT_PREFIX}"
"${PYTHON_CMD[@]}" -m scripts.plot_suite \
  --suite-root "$SUMMARY_DIR" \
  --output-prefix "$PLOT_PREFIX"

echo "[summary] done"
