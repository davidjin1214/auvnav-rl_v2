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
EPISODES="${EPISODES:-30}"
TOTAL_STEPS="${TOTAL_STEPS:-100000}"
RANDOM_STEPS="${RANDOM_STEPS:-5000}"
UPDATE_AFTER="${UPDATE_AFTER:-5000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
NUM_ENVS="${NUM_ENVS:-4}"
EVAL_EVERY="${EVAL_EVERY:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-30}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10000}"
DEVICE="${DEVICE:-cpu}"
SEEDS="${SEEDS:-42 43 44}"
PROBES="${PROBES:-s0 s1 s2}"
OBJECTIVE="${OBJECTIVE:-efficiency_v2}"
SAVE_ROOT="${SAVE_ROOT:-experiments/protocol_screen_v2/A1_single_u10_upstream_tgt15/${OBJECTIVE}}"

FLOW_PATH="wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy"
MANIFEST_PATH="benchmarks/single_u10_upstream_tgt15.json"

echo "[A1] generating manifest: ${MANIFEST_PATH}"
"${PYTHON_CMD[@]}" -m scripts.generate_standard_benchmarks \
  --benchmarks single_u10_upstream_tgt15 \
  --episodes "$EPISODES"

for PROBE in $PROBES; do
  for SEED in $SEEDS; do
    RUN_DIR="${SAVE_ROOT}/${PROBE}_k4/seed_${SEED}"
    if [[ -f "${RUN_DIR}/trainer_state.json" ]]; then
      echo "[skip] ${RUN_DIR}"
      continue
    fi

    echo "[run] probe=${PROBE} seed=${SEED} save_dir=${RUN_DIR}"
    "${PYTHON_CMD[@]}" -m scripts.train_sac \
      --flow "$FLOW_PATH" \
      --task-geometry upstream \
      --target-speed 1.5 \
      --objective "$OBJECTIVE" \
      --probe-layout "$PROBE" \
      --history-length 4 \
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
      --save-dir "$RUN_DIR"
  done
done

echo "[A1] done"
