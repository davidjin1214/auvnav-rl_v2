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

MODE="${MODE:-all}"
DEVICE="${DEVICE:-cuda}"

BENCHMARK_KEY="${BENCHMARK_KEY:-single_u10_cross_tgt15}"
FLOW_PATH="${FLOW_PATH:-wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy}"
TASK_GEOMETRY="${TASK_GEOMETRY:-cross_stream}"
TARGET_SPEED="${TARGET_SPEED:-1.5}"
OBJECTIVE="${OBJECTIVE:-efficiency_v2}"
PROBE_LAYOUT="${PROBE_LAYOUT:-s0}"
HISTORY_LENGTH="${HISTORY_LENGTH:-4}"

DATASET_POLICY="${DATASET_POLICY:-crosscomp}"
DATASET_EPISODES="${DATASET_EPISODES:-500}"
DATASET_SEED="${DATASET_SEED:-0}"
COLLECT_WORKERS="${COLLECT_WORKERS:-4}"
DATASET_NAME="${DATASET_NAME:-${DATASET_POLICY}_${PROBE_LAYOUT}_h${HISTORY_LENGTH}_${OBJECTIVE}_re150_u10cross}"
DATASET_DIR="${DATASET_DIR:-offline_data/${DATASET_NAME}}"

TRAIN_MANIFEST_EPISODES="${TRAIN_MANIFEST_EPISODES:-30}"
FINAL_MANIFEST_EPISODES="${FINAL_MANIFEST_EPISODES:-100}"
TRAIN_MANIFEST_DIR="${TRAIN_MANIFEST_DIR:-benchmarks/offline_phase0/train_${TRAIN_MANIFEST_EPISODES}}"
FINAL_MANIFEST_DIR="${FINAL_MANIFEST_DIR:-benchmarks/offline_phase0/final_${FINAL_MANIFEST_EPISODES}}"
TRAIN_MANIFEST_PATH="${TRAIN_MANIFEST_PATH:-${TRAIN_MANIFEST_DIR}/${BENCHMARK_KEY}.json}"
FINAL_MANIFEST_PATH="${FINAL_MANIFEST_PATH:-${FINAL_MANIFEST_DIR}/${BENCHMARK_KEY}.json}"

ALPHAS="${ALPHAS:-0.0 1.0 2.5 5.0}"
SEEDS="${SEEDS:-42 43}"
TOTAL_STEPS="${TOTAL_STEPS:-200000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
ACTOR_LR="${ACTOR_LR:-3e-4}"
CRITIC_LR="${CRITIC_LR:-3e-4}"
GAMMA="${GAMMA:-0.995}"
TAU="${TAU:-0.005}"
POLICY_NOISE="${POLICY_NOISE:-0.2}"
NOISE_CLIP="${NOISE_CLIP:-0.5}"
POLICY_FREQ="${POLICY_FREQ:-2}"
NORMALIZER_EPS="${NORMALIZER_EPS:-1e-3}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-10.0}"
EVAL_EVERY="${EVAL_EVERY:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-30}"
LOG_EVERY="${LOG_EVERY:-1000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10000}"
TENSOR_REPLAY="${TENSOR_REPLAY:-1}"
TRAIN_SKIP_FINAL_EVAL="${TRAIN_SKIP_FINAL_EVAL:-1}"
RUN_BASELINE_EVAL="${RUN_BASELINE_EVAL:-1}"
RUN_FINAL_EVAL="${RUN_FINAL_EVAL:-1}"
EVAL_WORKERS="${EVAL_WORKERS:-4}"
EVAL_WORKER_DEVICE="${EVAL_WORKER_DEVICE:-cpu}"
TRAIN_EVAL_WORKERS="${TRAIN_EVAL_WORKERS:-$EVAL_WORKERS}"
TRAIN_EVAL_WORKER_DEVICE="${TRAIN_EVAL_WORKER_DEVICE:-$EVAL_WORKER_DEVICE}"

USE_LAYERNORM="${USE_LAYERNORM:-0}"
USE_ASYMMETRIC_CRITIC="${USE_ASYMMETRIC_CRITIC:-0}"
PRIVILEGED_ACTOR_UPDATE_MODE="${PRIVILEGED_ACTOR_UPDATE_MODE:-zeros}"

SAVE_ROOT="${SAVE_ROOT:-checkpoints/offline/td3bc/phase0/${DATASET_NAME}}"
RESULTS_ROOT="${RESULTS_ROOT:-results/offline/td3bc/phase0/${DATASET_NAME}}"
RUN_NOTE="${RUN_NOTE:-}"

mkdir -p "$RESULTS_ROOT"

run_cmd() {
  echo
  echo "[cmd] $*"
  "$@"
}

ensure_manifest() {
  local episodes="$1"
  local output_dir="$2"
  local output_path="$3"
  if [[ -f "$output_path" ]]; then
    echo "[skip] manifest exists: $output_path"
    return
  fi
  mkdir -p "$output_dir"
  if [[ -n "$RUN_NOTE" ]]; then
    run_cmd "${PYTHON_CMD[@]}" -m scripts.generate_standard_benchmarks \
      --benchmarks "$BENCHMARK_KEY" \
      --episodes "$episodes" \
      --output-dir "$output_dir" \
      --notes "$RUN_NOTE"
  else
    run_cmd "${PYTHON_CMD[@]}" -m scripts.generate_standard_benchmarks \
      --benchmarks "$BENCHMARK_KEY" \
      --episodes "$episodes" \
      --output-dir "$output_dir"
  fi
}

run_collect() {
  if [[ -f "${DATASET_DIR}/transitions.npz" ]]; then
    echo "[skip] dataset exists: ${DATASET_DIR}/transitions.npz"
    return
  fi
  mkdir -p "$DATASET_DIR"
  run_cmd "${PYTHON_CMD[@]}" -m scripts.collect_offline_data \
    --policy "$DATASET_POLICY" \
    --flow "$FLOW_PATH" \
    --probe-layout "$PROBE_LAYOUT" \
    --history-length "$HISTORY_LENGTH" \
    --task-geometry "$TASK_GEOMETRY" \
    --target-speed "$TARGET_SPEED" \
    --objective "$OBJECTIVE" \
    --episodes "$DATASET_EPISODES" \
    --seed "$DATASET_SEED" \
    --num-workers "$COLLECT_WORKERS" \
    --output-dir "$DATASET_DIR"
}

run_baseline_eval() {
  if [[ "$RUN_BASELINE_EVAL" != "1" ]]; then
    echo "[skip] baseline evaluation disabled"
    return
  fi
  mkdir -p "$RESULTS_ROOT/baselines"
  run_cmd "${PYTHON_CMD[@]}" -m scripts.evaluate_baseline_on_manifest \
    --policy "$DATASET_POLICY" \
    --flow "$FLOW_PATH" \
    --probe-layout "$PROBE_LAYOUT" \
    --history-length "$HISTORY_LENGTH" \
    --task-geometry "$TASK_GEOMETRY" \
    --target-speed "$TARGET_SPEED" \
    --objective "$OBJECTIVE" \
    --manifest "$FINAL_MANIFEST_PATH" \
    --num-workers "$EVAL_WORKERS" \
    --output-json "$RESULTS_ROOT/baselines/${DATASET_POLICY}_final_eval.json"
}

train_one() {
  local alpha="$1"
  local seed="$2"
  local alpha_tag="${alpha//./p}"
  local run_dir="${SAVE_ROOT}/alpha_${alpha_tag}/seed_${seed}"
  local final_eval_json="${RESULTS_ROOT}/alpha_${alpha_tag}/seed_${seed}_final_eval.json"

  if [[ -f "${run_dir}/trainer_state.json" && -f "$final_eval_json" ]]; then
    echo "[skip] run exists: ${run_dir}"
    return
  fi

  mkdir -p "$run_dir"
  mkdir -p "$(dirname "$final_eval_json")"

  extra_layernorm=()
  extra_priv=()
  extra_perf=()
  if [[ "$USE_LAYERNORM" == "1" ]]; then
    extra_layernorm=(--use-layernorm)
  fi
  if [[ "$USE_ASYMMETRIC_CRITIC" == "1" ]]; then
    extra_priv=(--use-asymmetric-critic --privileged-actor-update-mode "$PRIVILEGED_ACTOR_UPDATE_MODE")
  fi
  if [[ "$TENSOR_REPLAY" != "1" ]]; then
    extra_perf+=(--disable-tensor-replay)
  fi
  if [[ "$TRAIN_SKIP_FINAL_EVAL" == "1" ]]; then
    extra_perf+=(--skip-final-eval)
  fi

  run_cmd "${PYTHON_CMD[@]}" -m scripts.train_offline \
    --offline-data "${DATASET_DIR}/transitions.npz" \
    --alpha "$alpha" \
    --flow "$FLOW_PATH" \
    --probe-layout "$PROBE_LAYOUT" \
    --history-length "$HISTORY_LENGTH" \
    --task-geometry "$TASK_GEOMETRY" \
    --target-speed "$TARGET_SPEED" \
    --objective "$OBJECTIVE" \
    --manifest "$TRAIN_MANIFEST_PATH" \
    --total-steps "$TOTAL_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --hidden-dim "$HIDDEN_DIM" \
    --actor-lr "$ACTOR_LR" \
    --critic-lr "$CRITIC_LR" \
    --gamma "$GAMMA" \
    --tau "$TAU" \
    --policy-noise "$POLICY_NOISE" \
    --noise-clip "$NOISE_CLIP" \
    --policy-freq "$POLICY_FREQ" \
    --normalizer-eps "$NORMALIZER_EPS" \
    --grad-clip-norm "$GRAD_CLIP_NORM" \
    --eval-every "$EVAL_EVERY" \
    --eval-episodes "$EVAL_EPISODES" \
    --eval-workers "$TRAIN_EVAL_WORKERS" \
    --eval-worker-device "$TRAIN_EVAL_WORKER_DEVICE" \
    --log-every "$LOG_EVERY" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --save-dir "$run_dir" \
    --seed "$seed" \
    --device "$DEVICE" \
    "${extra_layernorm[@]+"${extra_layernorm[@]}"}" \
    "${extra_priv[@]+"${extra_priv[@]}"}" \
    "${extra_perf[@]+"${extra_perf[@]}"}"

  if [[ "$RUN_FINAL_EVAL" == "1" ]]; then
    run_cmd "${PYTHON_CMD[@]}" -m scripts.evaluate_offline \
      --checkpoint "$run_dir" \
      --manifest "$FINAL_MANIFEST_PATH" \
      --device "$DEVICE" \
      --num-workers "$EVAL_WORKERS" \
      --worker-device "$EVAL_WORKER_DEVICE" \
      --output-json "$final_eval_json"
  else
    echo "[skip] final evaluation disabled: ${run_dir}"
  fi
}

run_train_sweep() {
  for alpha in $ALPHAS; do
    for seed in $SEEDS; do
      echo "[run] alpha=${alpha} seed=${seed}"
      train_one "$alpha" "$seed"
    done
  done
}

print_summary() {
  cat <<EOF

[done] Phase-0 TD3+BC workflow complete.
dataset      : ${DATASET_DIR}
train_manifest: ${TRAIN_MANIFEST_PATH}
final_manifest: ${FINAL_MANIFEST_PATH}
save_root    : ${SAVE_ROOT}
results_root : ${RESULTS_ROOT}

Recommended next checks:
  1. Compare ${RESULTS_ROOT}/baselines/${DATASET_POLICY}_final_eval.json
  2. Compare all ${RESULTS_ROOT}/alpha_*/seed_*_final_eval.json
  3. Inspect each run's eval_log.csv and trainer_state.json
EOF
}

case "$MODE" in
  all)
    ensure_manifest "$TRAIN_MANIFEST_EPISODES" "$TRAIN_MANIFEST_DIR" "$TRAIN_MANIFEST_PATH"
    ensure_manifest "$FINAL_MANIFEST_EPISODES" "$FINAL_MANIFEST_DIR" "$FINAL_MANIFEST_PATH"
    run_collect
    run_baseline_eval
    run_train_sweep
    print_summary
    ;;
  manifest)
    ensure_manifest "$TRAIN_MANIFEST_EPISODES" "$TRAIN_MANIFEST_DIR" "$TRAIN_MANIFEST_PATH"
    ensure_manifest "$FINAL_MANIFEST_EPISODES" "$FINAL_MANIFEST_DIR" "$FINAL_MANIFEST_PATH"
    ;;
  collect)
    run_collect
    ;;
  baseline)
    ensure_manifest "$FINAL_MANIFEST_EPISODES" "$FINAL_MANIFEST_DIR" "$FINAL_MANIFEST_PATH"
    run_baseline_eval
    ;;
  train)
    ensure_manifest "$TRAIN_MANIFEST_EPISODES" "$TRAIN_MANIFEST_DIR" "$TRAIN_MANIFEST_PATH"
    ensure_manifest "$FINAL_MANIFEST_EPISODES" "$FINAL_MANIFEST_DIR" "$FINAL_MANIFEST_PATH"
    run_train_sweep
    ;;
  *)
    echo "Unsupported MODE: ${MODE}"
    echo "Supported MODE values: all, manifest, collect, baseline, train"
    exit 1
    ;;
esac
