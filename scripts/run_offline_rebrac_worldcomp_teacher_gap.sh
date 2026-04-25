#!/usr/bin/env bash
set -euo pipefail

# ReBRAC Stage D — worldcomp teacher-gap follow-up driver.
#
# Mirrors scripts/run_offline_td3bc_phase0c_worldcomp_teacher_gap.sh in
# structure: this driver is a thin orchestrator that delegates per-cell work
# to scripts/run_offline_rebrac_screen.sh as a child, switching env vars to
# select between the deployable and privileged-critic tracks.
#
# Phase split (see docs/rebrac_experiment_plan.md §6.5):
#
#   Phase 1 (must-do, ~7 runs):
#     - epoch_probe: (β1=4.0, β2=2.0) × 2 seeds × 128 epoch on worldcomp-1000,
#       deployable track. Reads val curve at every CHECKPOINT_EVERY_EPOCHS=8
#       to pick TRAIN_EPOCHS=64 vs 96 for deployable_final.
#     - deployable_final: (β1=4.0, β2=2.0) × 5 seeds × <chosen> epoch on
#       worldcomp-1000, deployable track, test manifest 100 episodes.
#
#   Phase 2 (conditional on Phase 1 deployable mean test success):
#     - privileged_final: (β1=4.0, β2=2.0) × N seeds × <same epoch as Phase 1>
#       on worldcomp-1000, privileged-critic track. N defaults to 5 but can
#       be reduced to 3 (scenario A) by overriding PRIVILEGED_FINAL_SEEDS.
#
# Implementation notes:
#   - The agent (auv_nav/rebrac.py) supports asymmetric critic out of the box
#     via AsymmetricQNetwork; the only wiring needed is
#     USE_ASYMMETRIC_CRITIC=1 + PRIVILEGED_ACTOR_UPDATE_MODE=zeros at the
#     screen.sh layer (already added in screen.sh under the same names).
#   - The screen.sh child driver is itself general-purpose: it iterates
#     ACTOR_PENALTY_COEFS × CRITIC_PENALTY_COEFS × DATASET_EPISODES × SEEDS.
#     Stage D pins each axis to length 1 except SEEDS, so we get a 1-cell
#     grid and just sweep over seeds.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${MODE:-all}"

PYTHON_PREFIX="${PYTHON_PREFIX:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DEVICE="${DEVICE:-cuda}"

# Canonical protocol — must match Stage B/C and TD3BC worldcomp teacher-gap.
BENCHMARK_KEY="${BENCHMARK_KEY:-single_u10_cross_tgt15}"
FLOW_PATH="${FLOW_PATH:-wake_data/wake_v8_U1p00_Re150_D12p00_dx0p60_Ti5pct_1200f_roi.npy}"
TASK_GEOMETRY="${TASK_GEOMETRY:-cross_stream}"
TARGET_SPEED="${TARGET_SPEED:-1.5}"
OBJECTIVE="${OBJECTIVE:-efficiency_v2}"
PROBE_LAYOUT="${PROBE_LAYOUT:-s0}"
HISTORY_LENGTH="${HISTORY_LENGTH:-4}"

# worldcomp baseline policy + 1000 episodes (Stage D scope).
DATASET_POLICY="${DATASET_POLICY:-worldcomp}"
DATASET_EPISODES_VALUE="${DATASET_EPISODES_VALUE:-1000}"
DATASET_SEED="${DATASET_SEED:-0}"
COLLECT_WORKERS="${COLLECT_WORKERS:-8}"

# Finalist locked by Stage C (see report §7.9).
ACTOR_PENALTY_COEF="${ACTOR_PENALTY_COEF:-4.0}"
CRITIC_PENALTY_COEF="${CRITIC_PENALTY_COEF:-2.0}"

# Phase 1 epoch probe.
EPOCH_PROBE_SEEDS="${EPOCH_PROBE_SEEDS:-42 43}"
EPOCH_PROBE_TRAIN_EPOCHS="${EPOCH_PROBE_TRAIN_EPOCHS:-128}"
EPOCH_PROBE_CHECKPOINT_EVERY_EPOCHS="${EPOCH_PROBE_CHECKPOINT_EVERY_EPOCHS:-8}"

# Phase 1 deployable formal.
DEPLOYABLE_FINAL_SEEDS="${DEPLOYABLE_FINAL_SEEDS:-42 43 44 45 46}"
DEPLOYABLE_FINAL_TRAIN_EPOCHS="${DEPLOYABLE_FINAL_TRAIN_EPOCHS:-64}"
DEPLOYABLE_FINAL_CHECKPOINT_EVERY_EPOCHS="${DEPLOYABLE_FINAL_CHECKPOINT_EVERY_EPOCHS:-8}"

# Phase 2 privileged-critic.
# - Scenario A (Phase 1 mean > 0.90): override to "42 43 44" for 3-seed marginal check.
# - Scenario B/C (Phase 1 mean ≤ 0.90): keep default "42 43 44 45 46" full.
PRIVILEGED_FINAL_SEEDS="${PRIVILEGED_FINAL_SEEDS:-42 43 44 45 46}"
PRIVILEGED_FINAL_TRAIN_EPOCHS="${PRIVILEGED_FINAL_TRAIN_EPOCHS:-${DEPLOYABLE_FINAL_TRAIN_EPOCHS}}"
PRIVILEGED_FINAL_CHECKPOINT_EVERY_EPOCHS="${PRIVILEGED_FINAL_CHECKPOINT_EVERY_EPOCHS:-${DEPLOYABLE_FINAL_CHECKPOINT_EVERY_EPOCHS}}"
PRIVILEGED_ACTOR_UPDATE_MODE="${PRIVILEGED_ACTOR_UPDATE_MODE:-zeros}"

# Manifest sizes.
EPOCH_PROBE_VAL_MANIFEST_EPISODES="${EPOCH_PROBE_VAL_MANIFEST_EPISODES:-40}"
EPOCH_PROBE_TEST_MANIFEST_EPISODES="${EPOCH_PROBE_TEST_MANIFEST_EPISODES:-40}"
FINAL_VAL_MANIFEST_EPISODES="${FINAL_VAL_MANIFEST_EPISODES:-40}"
FINAL_TEST_MANIFEST_EPISODES="${FINAL_TEST_MANIFEST_EPISODES:-100}"

# Other fixed knobs (kept identical to Stage C / screen.sh defaults).
BATCH_SIZE="${BATCH_SIZE:-256}"
DROP_LAST_BATCH="${DROP_LAST_BATCH:-0}"
SAMPLING_MODE="${SAMPLING_MODE:-shuffle_no_replacement}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-3}"
ACTOR_LR="${ACTOR_LR:-3e-4}"
CRITIC_LR="${CRITIC_LR:-3e-4}"
GAMMA="${GAMMA:-0.99}"
TAU="${TAU:-0.005}"
POLICY_NOISE="${POLICY_NOISE:-0.2}"
NOISE_CLIP="${NOISE_CLIP:-0.5}"
POLICY_FREQ="${POLICY_FREQ:-2}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-10.0}"
NORMALIZER_EPS="${NORMALIZER_EPS:-1e-3}"
LOG_EVERY="${LOG_EVERY:-1000}"
TRAIN_METRICS_WINDOW_FRACTION="${TRAIN_METRICS_WINDOW_FRACTION:-0.25}"

EVAL_WORKERS="${EVAL_WORKERS:-6}"
EVAL_WORKER_DEVICE="${EVAL_WORKER_DEVICE:-cpu}"
VALIDATION_SEED="${VALIDATION_SEED:-123}"
TEST_SEED="${TEST_SEED:-456}"
FORCE_REEVAL="${FORCE_REEVAL:-0}"

# Output trees — separate from Stage B/C to keep diagnostics clean.
PACKAGE_ROOT="${PACKAGE_ROOT:-results/offline/rebrac/worldcomp_teacher_gap}"
CHECKPOINT_PACKAGE_ROOT="${CHECKPOINT_PACKAGE_ROOT:-checkpoints/offline/rebrac/worldcomp_teacher_gap}"

EPOCH_PROBE_PACKAGE_ROOT="${EPOCH_PROBE_PACKAGE_ROOT:-results/offline/rebrac/worldcomp_epoch_probe}"
EPOCH_PROBE_CHECKPOINT_PACKAGE_ROOT="${EPOCH_PROBE_CHECKPOINT_PACKAGE_ROOT:-checkpoints/offline/rebrac/worldcomp_epoch_probe}"
EPOCH_PROBE_MANIFEST_ROOT="${EPOCH_PROBE_MANIFEST_ROOT:-benchmarks/offline_rebrac_worldcomp_epoch_probe}"

DEPLOYABLE_RESULTS_ROOT="${DEPLOYABLE_RESULTS_ROOT:-${PACKAGE_ROOT}/deployable}"
DEPLOYABLE_CHECKPOINT_ROOT="${DEPLOYABLE_CHECKPOINT_ROOT:-${CHECKPOINT_PACKAGE_ROOT}/deployable}"
PRIVILEGED_RESULTS_ROOT="${PRIVILEGED_RESULTS_ROOT:-${PACKAGE_ROOT}/privileged_critic}"
PRIVILEGED_CHECKPOINT_ROOT="${PRIVILEGED_CHECKPOINT_ROOT:-${CHECKPOINT_PACKAGE_ROOT}/privileged_critic}"
FINAL_MANIFEST_ROOT="${FINAL_MANIFEST_ROOT:-benchmarks/offline_rebrac_worldcomp_final}"

run_cmd() {
  echo
  echo "[cmd] $*"
  "$@"
}

# Run scripts/run_offline_rebrac_screen.sh as a child, with overrides to make
# it behave as a single-cell driver for this Stage D phase.
run_screen_child() {
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
    "DATASET_EPISODES=${DATASET_EPISODES_VALUE}"
    "DATASET_SEED=${DATASET_SEED}"
    "COLLECT_WORKERS=${COLLECT_WORKERS}"
    "ACTOR_PENALTY_COEFS=${ACTOR_PENALTY_COEF}"
    "CRITIC_PENALTY_COEFS=${CRITIC_PENALTY_COEF}"
    "BATCH_SIZE=${BATCH_SIZE}"
    "DROP_LAST_BATCH=${DROP_LAST_BATCH}"
    "SAMPLING_MODE=${SAMPLING_MODE}"
    "HIDDEN_DIM=${HIDDEN_DIM}"
    "NUM_HIDDEN_LAYERS=${NUM_HIDDEN_LAYERS}"
    "ACTOR_LR=${ACTOR_LR}"
    "CRITIC_LR=${CRITIC_LR}"
    "GAMMA=${GAMMA}"
    "TAU=${TAU}"
    "POLICY_NOISE=${POLICY_NOISE}"
    "NOISE_CLIP=${NOISE_CLIP}"
    "POLICY_FREQ=${POLICY_FREQ}"
    "GRAD_CLIP_NORM=${GRAD_CLIP_NORM}"
    "NORMALIZER_EPS=${NORMALIZER_EPS}"
    "LOG_EVERY=${LOG_EVERY}"
    "TRAIN_METRICS_WINDOW_FRACTION=${TRAIN_METRICS_WINDOW_FRACTION}"
    "EVAL_WORKERS=${EVAL_WORKERS}"
    "EVAL_WORKER_DEVICE=${EVAL_WORKER_DEVICE}"
    "VALIDATION_SEED=${VALIDATION_SEED}"
    "TEST_SEED=${TEST_SEED}"
    "FORCE_REEVAL=${FORCE_REEVAL}"
  )
  if [[ -n "$PYTHON_PREFIX" ]]; then
    env_args+=("PYTHON_PREFIX=${PYTHON_PREFIX}")
  fi
  while (($#)); do
    env_args+=("$1")
    shift
  done

  run_cmd env "${env_args[@]}" bash scripts/run_offline_rebrac_screen.sh
}

# ----- Phase 1: epoch probe (deployable, 2 seeds × 128 epoch) -----

run_epoch_probe_phase() {
  local child_mode="$1"
  run_screen_child "$child_mode" \
    "SEEDS=${EPOCH_PROBE_SEEDS}" \
    "TRAIN_EPOCHS=${EPOCH_PROBE_TRAIN_EPOCHS}" \
    "CHECKPOINT_EVERY_EPOCHS=${EPOCH_PROBE_CHECKPOINT_EVERY_EPOCHS}" \
    "VAL_MANIFEST_EPISODES=${EPOCH_PROBE_VAL_MANIFEST_EPISODES}" \
    "TEST_MANIFEST_EPISODES=${EPOCH_PROBE_TEST_MANIFEST_EPISODES}" \
    "MANIFEST_ROOT=${EPOCH_PROBE_MANIFEST_ROOT}" \
    "CHECKPOINT_ROOT=${EPOCH_PROBE_CHECKPOINT_PACKAGE_ROOT}" \
    "RESULTS_ROOT=${EPOCH_PROBE_PACKAGE_ROOT}" \
    "USE_ASYMMETRIC_CRITIC=0"
}

# ----- Phase 1: deployable formal (5 seeds × 64 or 96 epoch) -----

run_deployable_final_phase() {
  local child_mode="$1"
  run_screen_child "$child_mode" \
    "SEEDS=${DEPLOYABLE_FINAL_SEEDS}" \
    "TRAIN_EPOCHS=${DEPLOYABLE_FINAL_TRAIN_EPOCHS}" \
    "CHECKPOINT_EVERY_EPOCHS=${DEPLOYABLE_FINAL_CHECKPOINT_EVERY_EPOCHS}" \
    "VAL_MANIFEST_EPISODES=${FINAL_VAL_MANIFEST_EPISODES}" \
    "TEST_MANIFEST_EPISODES=${FINAL_TEST_MANIFEST_EPISODES}" \
    "MANIFEST_ROOT=${FINAL_MANIFEST_ROOT}" \
    "CHECKPOINT_ROOT=${DEPLOYABLE_CHECKPOINT_ROOT}" \
    "RESULTS_ROOT=${DEPLOYABLE_RESULTS_ROOT}" \
    "USE_ASYMMETRIC_CRITIC=0"
}

# ----- Phase 2: privileged-critic formal (3 or 5 seeds, same epoch as Phase 1) -----

run_privileged_final_phase() {
  local child_mode="$1"
  run_screen_child "$child_mode" \
    "SEEDS=${PRIVILEGED_FINAL_SEEDS}" \
    "TRAIN_EPOCHS=${PRIVILEGED_FINAL_TRAIN_EPOCHS}" \
    "CHECKPOINT_EVERY_EPOCHS=${PRIVILEGED_FINAL_CHECKPOINT_EVERY_EPOCHS}" \
    "VAL_MANIFEST_EPISODES=${FINAL_VAL_MANIFEST_EPISODES}" \
    "TEST_MANIFEST_EPISODES=${FINAL_TEST_MANIFEST_EPISODES}" \
    "MANIFEST_ROOT=${FINAL_MANIFEST_ROOT}" \
    "CHECKPOINT_ROOT=${PRIVILEGED_CHECKPOINT_ROOT}" \
    "RESULTS_ROOT=${PRIVILEGED_RESULTS_ROOT}" \
    "USE_ASYMMETRIC_CRITIC=1" \
    "PRIVILEGED_ACTOR_UPDATE_MODE=${PRIVILEGED_ACTOR_UPDATE_MODE}"
}

# ----- High-level entry points -----

run_phase1() {
  # Phase 1: probe + deployable formal. ~7 runs.
  run_epoch_probe_phase manifests
  run_epoch_probe_phase collect
  run_epoch_probe_phase train
  run_epoch_probe_phase validate
  run_epoch_probe_phase test
  run_epoch_probe_phase summarize
  run_deployable_final_phase manifests
  run_deployable_final_phase train
  run_deployable_final_phase validate
  run_deployable_final_phase test
  run_deployable_final_phase summarize
}

run_phase2() {
  # Phase 2: privileged-critic formal. 3-5 runs (size depends on Phase 1 outcome).
  run_privileged_final_phase manifests
  run_privileged_final_phase train
  run_privileged_final_phase validate
  run_privileged_final_phase test
  run_privileged_final_phase summarize
}

case "$MODE" in
  all)
    run_phase1
    run_phase2
    ;;
  phase1)
    run_phase1
    ;;
  phase2)
    run_phase2
    ;;
  epoch_probe_train|epoch_probe_validate|epoch_probe_test|epoch_probe_summarize|epoch_probe_manifests|epoch_probe_collect)
    run_epoch_probe_phase "${MODE#epoch_probe_}"
    ;;
  deployable_train|deployable_validate|deployable_test|deployable_summarize|deployable_manifests)
    run_deployable_final_phase "${MODE#deployable_}"
    ;;
  privileged_train|privileged_validate|privileged_test|privileged_summarize|privileged_manifests)
    run_privileged_final_phase "${MODE#privileged_}"
    ;;
  *)
    echo "Unsupported MODE: $MODE" >&2
    echo "Supported MODE values: all, phase1, phase2," >&2
    echo "  epoch_probe_{manifests,collect,train,validate,test,summarize}," >&2
    echo "  deployable_{manifests,train,validate,test,summarize}," >&2
    echo "  privileged_{manifests,train,validate,test,summarize}" >&2
    exit 1
    ;;
esac
