#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

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
BASE_DATASET_NAME="${BASE_DATASET_NAME:-${DATASET_POLICY}_${PROBE_LAYOUT}_h${HISTORY_LENGTH}_${OBJECTIVE}_re150_u10cross_fixdone}"
BASE_DATASET_EPISODES="${BASE_DATASET_EPISODES:-500}"
BASE_DATASET_SEED="${BASE_DATASET_SEED:-0}"

NOISY_ACTION_NOISE_STD="${NOISY_ACTION_NOISE_STD:-0.05}"
NOISY_ACTION_NOISE_CLIP="${NOISY_ACTION_NOISE_CLIP:-0.15}"
COLLECT_WORKERS="${COLLECT_WORKERS:-8}"

NOISY_SUPPORT_EPISODES="${NOISY_SUPPORT_EPISODES:-1000 2000}"
NOISY_SUPPORT_SEEDS="${NOISY_SUPPORT_SEEDS:-42 43}"
NOISY_SUPPORT_TRAIN_EPOCHS="${NOISY_SUPPORT_TRAIN_EPOCHS:-64}"
NOISY_SUPPORT_CHECKPOINT_EVERY_EPOCHS="${NOISY_SUPPORT_CHECKPOINT_EVERY_EPOCHS:-8}"
NOISY_SUPPORT_VAL_MANIFEST_EPISODES="${NOISY_SUPPORT_VAL_MANIFEST_EPISODES:-40}"
NOISY_SUPPORT_TEST_MANIFEST_EPISODES="${NOISY_SUPPORT_TEST_MANIFEST_EPISODES:-40}"
NOISY_SUPPORT_ALPHAS_1000="${NOISY_SUPPORT_ALPHAS_1000:-0.0 0.25}"
NOISY_SUPPORT_ALPHAS_2000="${NOISY_SUPPORT_ALPHAS_2000:-0.0 0.15 0.2}"

BATCH_SIZE="${BATCH_SIZE:-256}"
DROP_LAST_BATCH="${DROP_LAST_BATCH:-0}"
SAMPLING_MODE="${SAMPLING_MODE:-shuffle_no_replacement}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
ACTOR_LR="${ACTOR_LR:-3e-4}"
CRITIC_LR="${CRITIC_LR:-3e-4}"
GAMMA="${GAMMA:-0.99}"
TAU="${TAU:-0.005}"
POLICY_NOISE="${POLICY_NOISE:-0.2}"
NOISE_CLIP="${NOISE_CLIP:-0.5}"
POLICY_FREQ="${POLICY_FREQ:-2}"
NORMALIZER_EPS="${NORMALIZER_EPS:-1e-3}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-10.0}"
LOG_EVERY="${LOG_EVERY:-1000}"
USE_LAYERNORM="${USE_LAYERNORM:-0}"
USE_ASYMMETRIC_CRITIC="${USE_ASYMMETRIC_CRITIC:-0}"
PRIVILEGED_ACTOR_UPDATE_MODE="${PRIVILEGED_ACTOR_UPDATE_MODE:-zeros}"
RUN_BASELINE_EVAL="${RUN_BASELINE_EVAL:-1}"
EVAL_WORKERS="${EVAL_WORKERS:-6}"
EVAL_WORKER_DEVICE="${EVAL_WORKER_DEVICE:-cpu}"
VALIDATION_SEED="${VALIDATION_SEED:-123}"
TEST_SEED="${TEST_SEED:-456}"
FORCE_REEVAL="${FORCE_REEVAL:-0}"
RUN_BC_TEST_ANALYSIS="${RUN_BC_TEST_ANALYSIS:-1}"
SKIP_ANALYSIS_PLOTS="${SKIP_ANALYSIS_PLOTS:-1}"

PACKAGE_ROOT="${PACKAGE_ROOT:-results/offline/td3bc/phase0c/noisy_support_screen}"
CHECKPOINT_PACKAGE_ROOT="${CHECKPOINT_PACKAGE_ROOT:-checkpoints/offline/td3bc/phase0c/noisy_support_screen}"
MANIFEST_ROOT="${MANIFEST_ROOT:-benchmarks/offline_phase0c/noisy_support_screen}"
COMPARISON_OUTPUT_DIR="${COMPARISON_OUTPUT_DIR:-${PACKAGE_ROOT}/analysis_compare}"

run_cmd() {
  echo
  echo "[cmd] $*"
  "$@"
}

float_tag() {
  local value="$1"
  echo "${value//./p}"
}

variant_results_root() {
  local variant="$1"
  case "$variant" in
    deterministic) echo "${PACKAGE_ROOT}/deterministic" ;;
    noisy) echo "${PACKAGE_ROOT}/noisy_std$(float_tag "$NOISY_ACTION_NOISE_STD")_clip$(float_tag "$NOISY_ACTION_NOISE_CLIP")" ;;
    *)
      echo "Unsupported variant: ${variant}" >&2
      exit 1
      ;;
  esac
}

variant_checkpoint_root() {
  local variant="$1"
  case "$variant" in
    deterministic) echo "${CHECKPOINT_PACKAGE_ROOT}/deterministic" ;;
    noisy) echo "${CHECKPOINT_PACKAGE_ROOT}/noisy_std$(float_tag "$NOISY_ACTION_NOISE_STD")_clip$(float_tag "$NOISY_ACTION_NOISE_CLIP")" ;;
    *)
      echo "Unsupported variant: ${variant}" >&2
      exit 1
      ;;
  esac
}

variant_dataset_name() {
  local variant="$1"
  case "$variant" in
    deterministic) echo "$BASE_DATASET_NAME" ;;
    noisy) echo "${BASE_DATASET_NAME}_noise$(float_tag "$NOISY_ACTION_NOISE_STD")clip$(float_tag "$NOISY_ACTION_NOISE_CLIP")" ;;
    *)
      echo "Unsupported variant: ${variant}" >&2
      exit 1
      ;;
  esac
}

variant_noise_std() {
  local variant="$1"
  case "$variant" in
    deterministic) echo "0.0" ;;
    noisy) echo "$NOISY_ACTION_NOISE_STD" ;;
    *)
      echo "Unsupported variant: ${variant}" >&2
      exit 1
      ;;
  esac
}

variant_noise_clip() {
  local variant="$1"
  case "$variant" in
    deterministic) echo "0.5" ;;
    noisy) echo "$NOISY_ACTION_NOISE_CLIP" ;;
    *)
      echo "Unsupported variant: ${variant}" >&2
      exit 1
      ;;
  esac
}

alphas_for_episodes() {
  local episodes="$1"
  case "$episodes" in
    1000) echo "$NOISY_SUPPORT_ALPHAS_1000" ;;
    2000) echo "$NOISY_SUPPORT_ALPHAS_2000" ;;
    *)
      echo "Unsupported noisy-support dataset size: $episodes" >&2
      exit 1
      ;;
  esac
}

run_phase0b_v2_child() {
  local child_mode="$1"
  shift

  local -a env_args=(
    "MODE=${child_mode}"
    "PYTHON_BIN=${PYTHON_BIN}"
    "DEVICE=${DEVICE}"
    "BENCHMARK_KEY=${BENCHMARK_KEY}"
    "FLOW_PATH=${FLOW_PATH}"
    "TASK_GEOMETRY=${TASK_GEOMETRY}"
    "TARGET_SPEED=${TARGET_SPEED}"
    "OBJECTIVE=${OBJECTIVE}"
    "PROBE_LAYOUT=${PROBE_LAYOUT}"
    "HISTORY_LENGTH=${HISTORY_LENGTH}"
    "DATASET_POLICY=${DATASET_POLICY}"
    "DATASET_POLICY_MIXTURE="
    "BASE_DATASET_EPISODES=${BASE_DATASET_EPISODES}"
    "BASE_DATASET_SEED=${BASE_DATASET_SEED}"
    "COLLECT_WORKERS=${COLLECT_WORKERS}"
    "BATCH_SIZE=${BATCH_SIZE}"
    "DROP_LAST_BATCH=${DROP_LAST_BATCH}"
    "SAMPLING_MODE=${SAMPLING_MODE}"
    "HIDDEN_DIM=${HIDDEN_DIM}"
    "ACTOR_LR=${ACTOR_LR}"
    "CRITIC_LR=${CRITIC_LR}"
    "GAMMA=${GAMMA}"
    "TAU=${TAU}"
    "POLICY_NOISE=${POLICY_NOISE}"
    "NOISE_CLIP=${NOISE_CLIP}"
    "POLICY_FREQ=${POLICY_FREQ}"
    "NORMALIZER_EPS=${NORMALIZER_EPS}"
    "GRAD_CLIP_NORM=${GRAD_CLIP_NORM}"
    "LOG_EVERY=${LOG_EVERY}"
    "USE_LAYERNORM=${USE_LAYERNORM}"
    "USE_ASYMMETRIC_CRITIC=${USE_ASYMMETRIC_CRITIC}"
    "PRIVILEGED_ACTOR_UPDATE_MODE=${PRIVILEGED_ACTOR_UPDATE_MODE}"
    "RUN_BASELINE_EVAL=${RUN_BASELINE_EVAL}"
    "EVAL_WORKERS=${EVAL_WORKERS}"
    "EVAL_WORKER_DEVICE=${EVAL_WORKER_DEVICE}"
    "VALIDATION_SEED=${VALIDATION_SEED}"
    "TEST_SEED=${TEST_SEED}"
    "FORCE_REEVAL=${FORCE_REEVAL}"
    "RUN_BC_TEST_ANALYSIS=${RUN_BC_TEST_ANALYSIS}"
    "SKIP_ANALYSIS_PLOTS=${SKIP_ANALYSIS_PLOTS}"
    "TRAIN_EPOCHS=${NOISY_SUPPORT_TRAIN_EPOCHS}"
    "CHECKPOINT_EVERY_EPOCHS=${NOISY_SUPPORT_CHECKPOINT_EVERY_EPOCHS}"
    "VAL_MANIFEST_EPISODES=${NOISY_SUPPORT_VAL_MANIFEST_EPISODES}"
    "TEST_MANIFEST_EPISODES=${NOISY_SUPPORT_TEST_MANIFEST_EPISODES}"
    "MANIFEST_ROOT=${MANIFEST_ROOT}"
  )
  if [[ -n "$PYTHON_PREFIX" ]]; then
    env_args+=("PYTHON_PREFIX=${PYTHON_PREFIX}")
  fi
  while (($#)); do
    env_args+=("$1")
    shift
  done
  run_cmd env "${env_args[@]}" bash scripts/run_offline_td3bc_phase0b_v2.sh
}

ensure_manifests() {
  run_phase0b_v2_child manifests \
    "BASE_DATASET_NAME=${BASE_DATASET_NAME}" \
    "SIZE_ABLATION_EPISODES=${NOISY_SUPPORT_EPISODES}" \
    "SIZE_ABLATION_ALPHAS=$(alphas_for_episodes 1000)" \
    "SIZE_ABLATION_SEEDS=${NOISY_SUPPORT_SEEDS}" \
    "ACTION_NOISE_STD=0.0" \
    "ACTION_NOISE_CLIP=0.5" \
    "CHECKPOINT_ROOT=$(variant_checkpoint_root deterministic)" \
    "RESULTS_ROOT=$(variant_results_root deterministic)"
}

collect_variant() {
  local variant="$1"
  run_phase0b_v2_child collect \
    "BASE_DATASET_NAME=$(variant_dataset_name "$variant")" \
    "SIZE_ABLATION_EPISODES=${NOISY_SUPPORT_EPISODES}" \
    "SIZE_ABLATION_ALPHAS=$(alphas_for_episodes 1000)" \
    "SIZE_ABLATION_SEEDS=${NOISY_SUPPORT_SEEDS}" \
    "ACTION_NOISE_STD=$(variant_noise_std "$variant")" \
    "ACTION_NOISE_CLIP=$(variant_noise_clip "$variant")" \
    "CHECKPOINT_ROOT=$(variant_checkpoint_root "$variant")" \
    "RESULTS_ROOT=$(variant_results_root "$variant")"
}

run_variant_phase() {
  local variant="$1"
  local child_mode="$2"
  local episodes alphas
  for episodes in $NOISY_SUPPORT_EPISODES; do
    alphas="$(alphas_for_episodes "$episodes")"
    run_phase0b_v2_child "$child_mode" \
      "BASE_DATASET_NAME=$(variant_dataset_name "$variant")" \
      "SIZE_ABLATION_EPISODES=${episodes}" \
      "SIZE_ABLATION_ALPHAS=${alphas}" \
      "SIZE_ABLATION_SEEDS=${NOISY_SUPPORT_SEEDS}" \
      "ACTION_NOISE_STD=$(variant_noise_std "$variant")" \
      "ACTION_NOISE_CLIP=$(variant_noise_clip "$variant")" \
      "CHECKPOINT_ROOT=$(variant_checkpoint_root "$variant")" \
      "RESULTS_ROOT=$(variant_results_root "$variant")"
  done
}

analyze_variant() {
  local variant="$1"
  run_phase0b_v2_child analyze \
    "BASE_DATASET_NAME=$(variant_dataset_name "$variant")" \
    "SIZE_ABLATION_EPISODES=${NOISY_SUPPORT_EPISODES}" \
    "SIZE_ABLATION_ALPHAS=0.0 0.15 0.2 0.25" \
    "SIZE_ABLATION_SEEDS=${NOISY_SUPPORT_SEEDS}" \
    "ACTION_NOISE_STD=$(variant_noise_std "$variant")" \
    "ACTION_NOISE_CLIP=$(variant_noise_clip "$variant")" \
    "CHECKPOINT_ROOT=$(variant_checkpoint_root "$variant")" \
    "RESULTS_ROOT=$(variant_results_root "$variant")"
}

run_compare_analysis() {
  mkdir -p "$COMPARISON_OUTPUT_DIR"
  run_cmd "${PYTHON_CMD[@]}" -m scripts.analyze_offline_td3bc_noisy_support \
    --det-results-root "$(variant_results_root deterministic)" \
    --noisy-results-root "$(variant_results_root noisy)" \
    --output-dir "$COMPARISON_OUTPUT_DIR"
}

run_all() {
  ensure_manifests
  collect_variant deterministic
  collect_variant noisy
  run_variant_phase deterministic train
  run_variant_phase noisy train
  run_variant_phase deterministic validate
  run_variant_phase noisy validate
  run_variant_phase deterministic summarize
  run_variant_phase noisy summarize
  analyze_variant deterministic
  analyze_variant noisy
  run_compare_analysis
}

case "$MODE" in
  all)
    run_all
    ;;
  manifests)
    ensure_manifests
    ;;
  collect)
    collect_variant deterministic
    collect_variant noisy
    ;;
  train|validate|summarize)
    run_variant_phase deterministic "$MODE"
    run_variant_phase noisy "$MODE"
    ;;
  analyze)
    analyze_variant deterministic
    analyze_variant noisy
    run_compare_analysis
    ;;
  *)
    echo "Unsupported MODE: ${MODE}"
    echo "Supported MODE values: all, manifests, collect, train, validate, summarize, analyze"
    exit 1
    ;;
esac
