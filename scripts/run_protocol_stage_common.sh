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

STUDY_ROOT="${STUDY_ROOT:-protocol_screen_v2}"
if [[ -z "${STAGE_DIR:-}" ]]; then
  echo "run_protocol_stage_common.sh requires STAGE_DIR" >&2
  exit 1
fi
if [[ -z "${BENCHMARK_KEY:-}" ]]; then
  echo "run_protocol_stage_common.sh requires BENCHMARK_KEY" >&2
  exit 1
fi
if [[ -z "${FLOW_PATH:-}" ]]; then
  echo "run_protocol_stage_common.sh requires FLOW_PATH" >&2
  exit 1
fi
if [[ -z "${TASK_GEOMETRY:-}" ]]; then
  echo "run_protocol_stage_common.sh requires TASK_GEOMETRY" >&2
  exit 1
fi

EPISODES="${EPISODES:-30}"
TOTAL_STEPS="${TOTAL_STEPS:-600000}"
RANDOM_STEPS="${RANDOM_STEPS:-5000}"
UPDATE_AFTER="${UPDATE_AFTER:-5000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
NUM_ENVS="${NUM_ENVS:-6}"
EVAL_EVERY="${EVAL_EVERY:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-30}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10000}"
DEVICE="${DEVICE:-cpu}"
SEEDS="${SEEDS:-46 47 50}"
PROBES="${PROBES:-s0 s1 s2}"
OBJECTIVE="${OBJECTIVE:-efficiency_v2}"
ALGO_TAG="${ALGO_TAG:-sac_vanilla}"
TARGET_SPEED="${TARGET_SPEED:-1.5}"
HISTORY_LENGTH="${HISTORY_LENGTH:-4}"
SAVE_ROOT="${SAVE_ROOT:-experiments/${STUDY_ROOT}/${STAGE_DIR}/${OBJECTIVE}/${ALGO_TAG}}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/${STUDY_ROOT}/${STAGE_DIR}/${OBJECTIVE}/${ALGO_TAG}}"
MANIFEST_PATH="benchmarks/${BENCHMARK_KEY}.json"

echo "[${STAGE_LABEL:-stage}] generating manifest: ${MANIFEST_PATH}"
"${PYTHON_CMD[@]}" -m scripts.generate_standard_benchmarks \
  --benchmarks "$BENCHMARK_KEY" \
  --episodes "$EPISODES"

for PROBE in $PROBES; do
  for SEED in $SEEDS; do
    RUN_DIR="${SAVE_ROOT}/${PROBE}_k${HISTORY_LENGTH}/seed_${SEED}"
    CHECKPOINT_DIR=""
    EXTRA_ARGS=()
    if [[ -n "$CHECKPOINT_ROOT" ]]; then
      CHECKPOINT_DIR="${CHECKPOINT_ROOT}/${PROBE}_k${HISTORY_LENGTH}/seed_${SEED}"
      EXTRA_ARGS+=(--checkpoint-dir "$CHECKPOINT_DIR")
    fi
    if [[ -f "${RUN_DIR}/results/final_eval.json" || -f "${RUN_DIR}/final_eval.json" ]]; then
      echo "[skip] ${RUN_DIR}"
      continue
    fi

    echo "[run] stage=${STAGE_LABEL:-stage} probe=${PROBE} seed=${SEED} save_dir=${RUN_DIR}"
    if [[ -n "$CHECKPOINT_DIR" ]]; then
      echo "      checkpoint_dir=${CHECKPOINT_DIR}"
    fi
    "${PYTHON_CMD[@]}" -m scripts.train_sac \
      --flow "$FLOW_PATH" \
      --task-geometry "$TASK_GEOMETRY" \
      --target-speed "$TARGET_SPEED" \
      --objective "$OBJECTIVE" \
      --probe-layout "$PROBE" \
      --history-length "$HISTORY_LENGTH" \
      --total-steps "$TOTAL_STEPS" \
      --random-steps "$RANDOM_STEPS" \
      --update-after "$UPDATE_AFTER" \
      --batch-size "$BATCH_SIZE" \
      --hidden-dim "$HIDDEN_DIM" \
      --num-envs "$NUM_ENVS" \
      --eval-every "$EVAL_EVERY" \
      --eval-episodes "$EVAL_EPISODES" \
      --checkpoint-every "$CHECKPOINT_EVERY" \
      --eval-manifest "$MANIFEST_PATH" \
      --seed "$SEED" \
      --device "$DEVICE" \
      --save-dir "$RUN_DIR" \
      "${EXTRA_ARGS[@]}"
  done
done

echo "[${STAGE_LABEL:-stage}] done"
