#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASE_RUNNER="${BASE_RUNNER:-scripts/run_offline_td3bc_phase0.sh}"
MODE="${MODE:-all}"

PYTHON_PREFIX="${PYTHON_PREFIX:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_CMD=()
if [[ -n "$PYTHON_PREFIX" ]]; then
  read -r -a PYTHON_PREFIX_ARR <<< "$PYTHON_PREFIX"
  PYTHON_CMD+=("${PYTHON_PREFIX_ARR[@]}")
fi
PYTHON_CMD+=("$PYTHON_BIN")
DEVICE="${DEVICE:-cuda}"

BENCHMARK_KEY="${BENCHMARK_KEY:-single_u10_cross_tgt15}"
FLOW_PATH="${FLOW_PATH:-wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy}"
TASK_GEOMETRY="${TASK_GEOMETRY:-cross_stream}"
TARGET_SPEED="${TARGET_SPEED:-1.5}"
OBJECTIVE="${OBJECTIVE:-efficiency_v2}"
PROBE_LAYOUT="${PROBE_LAYOUT:-s0}"
HISTORY_LENGTH="${HISTORY_LENGTH:-4}"

DATASET_POLICY="${DATASET_POLICY:-crosscomp}"
DATASET_POLICY_MIXTURE="${DATASET_POLICY_MIXTURE:-}"
ACTION_NOISE_STD="${ACTION_NOISE_STD:-0.0}"
ACTION_NOISE_CLIP="${ACTION_NOISE_CLIP:-0.5}"
COLLECT_WORKERS="${COLLECT_WORKERS:-8}"

BASE_DATASET_NAME="${BASE_DATASET_NAME:-${DATASET_POLICY}_${PROBE_LAYOUT}_h${HISTORY_LENGTH}_${OBJECTIVE}_re150_u10cross_fixdone}"
BASE_DATASET_EPISODES="${BASE_DATASET_EPISODES:-500}"
BASE_DATASET_SEED="${BASE_DATASET_SEED:-0}"

SEED_EXTENSION_ALPHAS="${SEED_EXTENSION_ALPHAS:-0.0 0.05 0.1 0.25}"
SEED_EXTENSION_SEEDS="${SEED_EXTENSION_SEEDS:-42 43 44 45 46}"

SIZE_ABLATION_EPISODES="${SIZE_ABLATION_EPISODES:-500 1000 2000}"
SIZE_ABLATION_ALPHAS="${SIZE_ABLATION_ALPHAS:-0.0 0.05 0.1 0.25}"
SIZE_ABLATION_SEEDS="${SIZE_ABLATION_SEEDS:-42 43 44 45 46}"

TOTAL_STEPS="${TOTAL_STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
GAMMA="${GAMMA:-0.99}"
EVAL_EVERY="${EVAL_EVERY:-0}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-0}"
TRAIN_SKIP_FINAL_EVAL="${TRAIN_SKIP_FINAL_EVAL:-1}"
RUN_FINAL_EVAL="${RUN_FINAL_EVAL:-1}"
RUN_BASELINE_EVAL="${RUN_BASELINE_EVAL:-1}"
EVAL_WORKERS="${EVAL_WORKERS:-8}"
EVAL_WORKER_DEVICE="${EVAL_WORKER_DEVICE:-cpu}"
TRAIN_EVAL_WORKERS="${TRAIN_EVAL_WORKERS:-$EVAL_WORKERS}"
TRAIN_EVAL_WORKER_DEVICE="${TRAIN_EVAL_WORKER_DEVICE:-$EVAL_WORKER_DEVICE}"

SUMMARY_OUTPUT_DIR="${SUMMARY_OUTPUT_DIR:-results/offline/td3bc/phase0/summaries}"

run_cmd() {
  echo
  echo "[cmd] $*"
  "$@"
}

dataset_name_for_episodes() {
  local episodes="$1"
  if [[ "$episodes" == "$BASE_DATASET_EPISODES" ]]; then
    echo "$BASE_DATASET_NAME"
  else
    echo "${BASE_DATASET_NAME}_ep${episodes}"
  fi
}

run_phase0() {
  local mode="$1"
  local dataset_name="$2"
  local dataset_episodes="$3"
  local alphas="$4"
  local seeds="$5"

  run_cmd env \
    PYTHON_PREFIX="$PYTHON_PREFIX" \
    PYTHON_BIN="$PYTHON_BIN" \
    MODE="$mode" \
    DEVICE="$DEVICE" \
    BENCHMARK_KEY="$BENCHMARK_KEY" \
    FLOW_PATH="$FLOW_PATH" \
    TASK_GEOMETRY="$TASK_GEOMETRY" \
    TARGET_SPEED="$TARGET_SPEED" \
    OBJECTIVE="$OBJECTIVE" \
    PROBE_LAYOUT="$PROBE_LAYOUT" \
    HISTORY_LENGTH="$HISTORY_LENGTH" \
    DATASET_POLICY="$DATASET_POLICY" \
    DATASET_POLICY_MIXTURE="$DATASET_POLICY_MIXTURE" \
    DATASET_EPISODES="$dataset_episodes" \
    DATASET_SEED="$BASE_DATASET_SEED" \
    COLLECT_WORKERS="$COLLECT_WORKERS" \
    ACTION_NOISE_STD="$ACTION_NOISE_STD" \
    ACTION_NOISE_CLIP="$ACTION_NOISE_CLIP" \
    DATASET_NAME="$dataset_name" \
    ALPHAS="$alphas" \
    SEEDS="$seeds" \
    TOTAL_STEPS="$TOTAL_STEPS" \
    BATCH_SIZE="$BATCH_SIZE" \
    GAMMA="$GAMMA" \
    EVAL_EVERY="$EVAL_EVERY" \
    CHECKPOINT_EVERY="$CHECKPOINT_EVERY" \
    TRAIN_SKIP_FINAL_EVAL="$TRAIN_SKIP_FINAL_EVAL" \
    RUN_FINAL_EVAL="$RUN_FINAL_EVAL" \
    RUN_BASELINE_EVAL="$RUN_BASELINE_EVAL" \
    EVAL_WORKERS="$EVAL_WORKERS" \
    EVAL_WORKER_DEVICE="$EVAL_WORKER_DEVICE" \
    TRAIN_EVAL_WORKERS="$TRAIN_EVAL_WORKERS" \
    TRAIN_EVAL_WORKER_DEVICE="$TRAIN_EVAL_WORKER_DEVICE" \
    bash "$BASE_RUNNER"
}

write_summary() {
  local dataset_filter="$1"
  mkdir -p "$SUMMARY_OUTPUT_DIR"
  run_cmd "${PYTHON_CMD[@]}" -m scripts.summarize_offline_phase0 \
    --dataset-filter "$dataset_filter" \
    --output-json "$SUMMARY_OUTPUT_DIR/${dataset_filter}_summary.json" \
    --output-csv-prefix "$SUMMARY_OUTPUT_DIR/${dataset_filter}_summary"
}

run_seed_extension() {
  run_phase0 \
    train \
    "$BASE_DATASET_NAME" \
    "$BASE_DATASET_EPISODES" \
    "$SEED_EXTENSION_ALPHAS" \
    "$SEED_EXTENSION_SEEDS"
  write_summary "$BASE_DATASET_NAME"
}

run_size_ablation() {
  local episodes dataset_name mode
  for episodes in $SIZE_ABLATION_EPISODES; do
    dataset_name="$(dataset_name_for_episodes "$episodes")"
    mode="all"
    if [[ "$episodes" == "$BASE_DATASET_EPISODES" ]]; then
      mode="train"
    fi
    run_phase0 \
      "$mode" \
      "$dataset_name" \
      "$episodes" \
      "$SIZE_ABLATION_ALPHAS" \
      "$SIZE_ABLATION_SEEDS"
    if [[ "$mode" != "all" && "$RUN_BASELINE_EVAL" == "1" ]]; then
      run_phase0 \
        baseline \
        "$dataset_name" \
        "$episodes" \
        "$SIZE_ABLATION_ALPHAS" \
        "$SIZE_ABLATION_SEEDS"
    fi
    write_summary "$dataset_name"
  done
}

case "$MODE" in
  all)
    run_seed_extension
    run_size_ablation
    ;;
  seed_extension)
    run_seed_extension
    ;;
  size_ablation)
    run_size_ablation
    ;;
  summarize)
    write_summary "$BASE_DATASET_NAME"
    ;;
  *)
    echo "Unsupported MODE: ${MODE}"
    echo "Supported MODE values: all, seed_extension, size_ablation, summarize"
    exit 1
    ;;
esac
