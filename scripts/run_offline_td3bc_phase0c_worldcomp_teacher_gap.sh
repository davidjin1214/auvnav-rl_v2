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

DATASET_POLICY="${DATASET_POLICY:-worldcomp}"
BASE_DATASET_NAME="${BASE_DATASET_NAME:-${DATASET_POLICY}_${PROBE_LAYOUT}_h${HISTORY_LENGTH}_${OBJECTIVE}_re150_u10cross_fixdone}"
WORLD_DATASET_EPISODES="${WORLD_DATASET_EPISODES:-1000}"
BASE_DATASET_EPISODES="${BASE_DATASET_EPISODES:-${WORLD_DATASET_EPISODES}}"
BASE_DATASET_SEED="${BASE_DATASET_SEED:-0}"
ACTION_NOISE_STD="${ACTION_NOISE_STD:-0.0}"
ACTION_NOISE_CLIP="${ACTION_NOISE_CLIP:-0.5}"
COLLECT_WORKERS="${COLLECT_WORKERS:-8}"

DEPLOYABLE_SCREEN_ALPHAS="${DEPLOYABLE_SCREEN_ALPHAS:-0.0 0.1 0.25 0.5}"
DEPLOYABLE_SCREEN_SEEDS="${DEPLOYABLE_SCREEN_SEEDS:-42 43}"
DEPLOYABLE_SCREEN_TRAIN_EPOCHS="${DEPLOYABLE_SCREEN_TRAIN_EPOCHS:-64}"
DEPLOYABLE_SCREEN_CHECKPOINT_EVERY_EPOCHS="${DEPLOYABLE_SCREEN_CHECKPOINT_EVERY_EPOCHS:-8}"

PRIVILEGED_SCREEN_ALPHAS="${PRIVILEGED_SCREEN_ALPHAS:-}"
PRIVILEGED_SCREEN_TOPK="${PRIVILEGED_SCREEN_TOPK:-2}"
PRIVILEGED_SCREEN_SEEDS="${PRIVILEGED_SCREEN_SEEDS:-42 43}"
PRIVILEGED_SCREEN_TRAIN_EPOCHS="${PRIVILEGED_SCREEN_TRAIN_EPOCHS:-64}"
PRIVILEGED_SCREEN_CHECKPOINT_EVERY_EPOCHS="${PRIVILEGED_SCREEN_CHECKPOINT_EVERY_EPOCHS:-8}"

FORMAL_SEEDS="${FORMAL_SEEDS:-42 43 44 45 46}"
FORMAL_TRAIN_EPOCHS="${FORMAL_TRAIN_EPOCHS:-96}"
FORMAL_CHECKPOINT_EVERY_EPOCHS="${FORMAL_CHECKPOINT_EVERY_EPOCHS:-4}"

SCREEN_VAL_MANIFEST_EPISODES="${SCREEN_VAL_MANIFEST_EPISODES:-40}"
SCREEN_TEST_MANIFEST_EPISODES="${SCREEN_TEST_MANIFEST_EPISODES:-40}"
FINAL_VAL_MANIFEST_EPISODES="${FINAL_VAL_MANIFEST_EPISODES:-40}"
FINAL_TEST_MANIFEST_EPISODES="${FINAL_TEST_MANIFEST_EPISODES:-100}"

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
PRIVILEGED_ACTOR_UPDATE_MODE="${PRIVILEGED_ACTOR_UPDATE_MODE:-zeros}"

RUN_BASELINE_EVAL="${RUN_BASELINE_EVAL:-1}"
EVAL_WORKERS="${EVAL_WORKERS:-6}"
EVAL_WORKER_DEVICE="${EVAL_WORKER_DEVICE:-cpu}"
VALIDATION_SEED="${VALIDATION_SEED:-123}"
TEST_SEED="${TEST_SEED:-456}"
FORCE_REEVAL="${FORCE_REEVAL:-0}"
RUN_BC_TEST_ANALYSIS="${RUN_BC_TEST_ANALYSIS:-1}"
SKIP_ANALYSIS_PLOTS="${SKIP_ANALYSIS_PLOTS:-1}"

PACKAGE_ROOT="${PACKAGE_ROOT:-results/offline/td3bc/phase0c/worldcomp_teacher_gap}"
CHECKPOINT_PACKAGE_ROOT="${CHECKPOINT_PACKAGE_ROOT:-checkpoints/offline/td3bc/phase0c/worldcomp_teacher_gap}"
SCREEN_MANIFEST_ROOT="${SCREEN_MANIFEST_ROOT:-benchmarks/offline_phase0c/worldcomp_teacher_gap_screen}"
FINAL_MANIFEST_ROOT="${FINAL_MANIFEST_ROOT:-benchmarks/offline_phase0c/worldcomp_teacher_gap_final}"

DEPLOYABLE_SCREEN_RESULTS_ROOT="${DEPLOYABLE_SCREEN_RESULTS_ROOT:-${PACKAGE_ROOT}/deployable_screen}"
DEPLOYABLE_SCREEN_CHECKPOINT_ROOT="${DEPLOYABLE_SCREEN_CHECKPOINT_ROOT:-${CHECKPOINT_PACKAGE_ROOT}/deployable_screen}"
PRIVILEGED_SCREEN_RESULTS_ROOT="${PRIVILEGED_SCREEN_RESULTS_ROOT:-${PACKAGE_ROOT}/privileged_screen}"
PRIVILEGED_SCREEN_CHECKPOINT_ROOT="${PRIVILEGED_SCREEN_CHECKPOINT_ROOT:-${CHECKPOINT_PACKAGE_ROOT}/privileged_screen}"
DEPLOYABLE_FINAL_RESULTS_ROOT="${DEPLOYABLE_FINAL_RESULTS_ROOT:-${PACKAGE_ROOT}/deployable_final}"
DEPLOYABLE_FINAL_CHECKPOINT_ROOT="${DEPLOYABLE_FINAL_CHECKPOINT_ROOT:-${CHECKPOINT_PACKAGE_ROOT}/deployable_final}"
PRIVILEGED_FINAL_RESULTS_ROOT="${PRIVILEGED_FINAL_RESULTS_ROOT:-${PACKAGE_ROOT}/privileged_final}"
PRIVILEGED_FINAL_CHECKPOINT_ROOT="${PRIVILEGED_FINAL_CHECKPOINT_ROOT:-${CHECKPOINT_PACKAGE_ROOT}/privileged_final}"
COMPARISON_OUTPUT_DIR="${COMPARISON_OUTPUT_DIR:-${PACKAGE_ROOT}/analysis_compare}"

run_cmd() {
  echo
  echo "[cmd] $*"
  "$@"
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
    "ACTION_NOISE_STD=${ACTION_NOISE_STD}"
    "ACTION_NOISE_CLIP=${ACTION_NOISE_CLIP}"
    "COLLECT_WORKERS=${COLLECT_WORKERS}"
    "BASE_DATASET_NAME=${BASE_DATASET_NAME}"
    "BASE_DATASET_EPISODES=${BASE_DATASET_EPISODES}"
    "BASE_DATASET_SEED=${BASE_DATASET_SEED}"
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
    "RUN_BASELINE_EVAL=${RUN_BASELINE_EVAL}"
    "EVAL_WORKERS=${EVAL_WORKERS}"
    "EVAL_WORKER_DEVICE=${EVAL_WORKER_DEVICE}"
    "VALIDATION_SEED=${VALIDATION_SEED}"
    "TEST_SEED=${TEST_SEED}"
    "FORCE_REEVAL=${FORCE_REEVAL}"
    "RUN_BC_TEST_ANALYSIS=${RUN_BC_TEST_ANALYSIS}"
    "SKIP_ANALYSIS_PLOTS=${SKIP_ANALYSIS_PLOTS}"
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

read_best_alpha() {
  local results_root="$1"
  "${PYTHON_CMD[@]}" - "$results_root" "$BASE_DATASET_NAME" <<'PY'
import json
import sys
from pathlib import Path

results_root = Path(sys.argv[1])
dataset_name = sys.argv[2]
path = results_root / dataset_name / "selection" / "best_alpha.json"
payload = json.loads(path.read_text(encoding="utf-8"))
print(payload["best_alpha"])
PY
}

read_topk_positive_alphas() {
  local results_root="$1"
  local topk="$2"
  "${PYTHON_CMD[@]}" - "$results_root" "$BASE_DATASET_NAME" "$topk" <<'PY'
import json
import sys
from pathlib import Path

results_root = Path(sys.argv[1])
dataset_name = sys.argv[2]
topk = max(1, int(sys.argv[3]))
path = results_root / dataset_name / "selection" / "best_alpha.json"
payload = json.loads(path.read_text(encoding="utf-8"))

rows = payload.get("per_alpha", [])
def score(item):
    return (
        float(item.get("mean_val_success_rate", float("-inf"))),
        float(item.get("mean_val_return", float("-inf"))),
        -float(item.get("mean_val_safety_cost", float("inf"))),
        -float(item.get("mean_val_time_s", float("inf"))),
    )

positives = [item for item in rows if float(item.get("alpha", 0.0)) > 0.0]
positives.sort(key=score, reverse=True)
selected = [str(item["alpha"]) for item in positives[:topk]]
if not selected and rows:
    rows.sort(key=score, reverse=True)
    selected = [str(rows[0]["alpha"])]
print(" ".join(selected))
PY
}

unique_alpha_list() {
  "${PYTHON_CMD[@]}" - "$@" <<'PY'
import sys
items = []
for arg in sys.argv[1:]:
    for token in arg.split():
        if token and token not in items:
            items.append(token)
print(" ".join(items))
PY
}

ensure_screen_manifests() {
  run_phase0b_v2_child manifests \
    "SIZE_ABLATION_EPISODES=${WORLD_DATASET_EPISODES}" \
    "SIZE_ABLATION_ALPHAS=${DEPLOYABLE_SCREEN_ALPHAS}" \
    "SIZE_ABLATION_SEEDS=${DEPLOYABLE_SCREEN_SEEDS}" \
    "TRAIN_EPOCHS=${DEPLOYABLE_SCREEN_TRAIN_EPOCHS}" \
    "CHECKPOINT_EVERY_EPOCHS=${DEPLOYABLE_SCREEN_CHECKPOINT_EVERY_EPOCHS}" \
    "VAL_MANIFEST_EPISODES=${SCREEN_VAL_MANIFEST_EPISODES}" \
    "TEST_MANIFEST_EPISODES=${SCREEN_TEST_MANIFEST_EPISODES}" \
    "MANIFEST_ROOT=${SCREEN_MANIFEST_ROOT}" \
    "CHECKPOINT_ROOT=${DEPLOYABLE_SCREEN_CHECKPOINT_ROOT}" \
    "RESULTS_ROOT=${DEPLOYABLE_SCREEN_RESULTS_ROOT}"
}

ensure_final_manifests() {
  run_phase0b_v2_child manifests \
    "SIZE_ABLATION_EPISODES=${WORLD_DATASET_EPISODES}" \
    "SIZE_ABLATION_ALPHAS=${DEPLOYABLE_SCREEN_ALPHAS}" \
    "SIZE_ABLATION_SEEDS=${FORMAL_SEEDS}" \
    "TRAIN_EPOCHS=${FORMAL_TRAIN_EPOCHS}" \
    "CHECKPOINT_EVERY_EPOCHS=${FORMAL_CHECKPOINT_EVERY_EPOCHS}" \
    "VAL_MANIFEST_EPISODES=${FINAL_VAL_MANIFEST_EPISODES}" \
    "TEST_MANIFEST_EPISODES=${FINAL_TEST_MANIFEST_EPISODES}" \
    "MANIFEST_ROOT=${FINAL_MANIFEST_ROOT}" \
    "CHECKPOINT_ROOT=${DEPLOYABLE_FINAL_CHECKPOINT_ROOT}" \
    "RESULTS_ROOT=${DEPLOYABLE_FINAL_RESULTS_ROOT}"
}

collect_dataset() {
  run_phase0b_v2_child collect \
    "SIZE_ABLATION_EPISODES=${WORLD_DATASET_EPISODES}" \
    "SIZE_ABLATION_ALPHAS=${DEPLOYABLE_SCREEN_ALPHAS}" \
    "SIZE_ABLATION_SEEDS=${DEPLOYABLE_SCREEN_SEEDS}" \
    "TRAIN_EPOCHS=${DEPLOYABLE_SCREEN_TRAIN_EPOCHS}" \
    "CHECKPOINT_EVERY_EPOCHS=${DEPLOYABLE_SCREEN_CHECKPOINT_EVERY_EPOCHS}" \
    "VAL_MANIFEST_EPISODES=${SCREEN_VAL_MANIFEST_EPISODES}" \
    "TEST_MANIFEST_EPISODES=${SCREEN_TEST_MANIFEST_EPISODES}" \
    "MANIFEST_ROOT=${SCREEN_MANIFEST_ROOT}" \
    "CHECKPOINT_ROOT=${DEPLOYABLE_SCREEN_CHECKPOINT_ROOT}" \
    "RESULTS_ROOT=${DEPLOYABLE_SCREEN_RESULTS_ROOT}"
}

run_deployable_screen_phase() {
  local child_mode="$1"
  run_phase0b_v2_child "$child_mode" \
    "SIZE_ABLATION_EPISODES=${WORLD_DATASET_EPISODES}" \
    "SIZE_ABLATION_ALPHAS=${DEPLOYABLE_SCREEN_ALPHAS}" \
    "SIZE_ABLATION_SEEDS=${DEPLOYABLE_SCREEN_SEEDS}" \
    "TRAIN_EPOCHS=${DEPLOYABLE_SCREEN_TRAIN_EPOCHS}" \
    "CHECKPOINT_EVERY_EPOCHS=${DEPLOYABLE_SCREEN_CHECKPOINT_EVERY_EPOCHS}" \
    "VAL_MANIFEST_EPISODES=${SCREEN_VAL_MANIFEST_EPISODES}" \
    "TEST_MANIFEST_EPISODES=${SCREEN_TEST_MANIFEST_EPISODES}" \
    "MANIFEST_ROOT=${SCREEN_MANIFEST_ROOT}" \
    "CHECKPOINT_ROOT=${DEPLOYABLE_SCREEN_CHECKPOINT_ROOT}" \
    "RESULTS_ROOT=${DEPLOYABLE_SCREEN_RESULTS_ROOT}" \
    "USE_ASYMMETRIC_CRITIC=0"
}

effective_privileged_screen_alphas() {
  if [[ -n "${PRIVILEGED_SCREEN_ALPHAS}" ]]; then
    echo "${PRIVILEGED_SCREEN_ALPHAS}"
  else
    read_topk_positive_alphas "${DEPLOYABLE_SCREEN_RESULTS_ROOT}" "${PRIVILEGED_SCREEN_TOPK}"
  fi
}

run_privileged_screen_phase() {
  local child_mode="$1"
  local alphas
  alphas="$(effective_privileged_screen_alphas)"
  if [[ -z "$alphas" ]]; then
    echo "Failed to determine PRIVILEGED_SCREEN_ALPHAS" >&2
    exit 1
  fi
  run_phase0b_v2_child "$child_mode" \
    "SIZE_ABLATION_EPISODES=${WORLD_DATASET_EPISODES}" \
    "SIZE_ABLATION_ALPHAS=${alphas}" \
    "SIZE_ABLATION_SEEDS=${PRIVILEGED_SCREEN_SEEDS}" \
    "TRAIN_EPOCHS=${PRIVILEGED_SCREEN_TRAIN_EPOCHS}" \
    "CHECKPOINT_EVERY_EPOCHS=${PRIVILEGED_SCREEN_CHECKPOINT_EVERY_EPOCHS}" \
    "VAL_MANIFEST_EPISODES=${SCREEN_VAL_MANIFEST_EPISODES}" \
    "TEST_MANIFEST_EPISODES=${SCREEN_TEST_MANIFEST_EPISODES}" \
    "MANIFEST_ROOT=${SCREEN_MANIFEST_ROOT}" \
    "CHECKPOINT_ROOT=${PRIVILEGED_SCREEN_CHECKPOINT_ROOT}" \
    "RESULTS_ROOT=${PRIVILEGED_SCREEN_RESULTS_ROOT}" \
    "USE_ASYMMETRIC_CRITIC=1" \
    "PRIVILEGED_ACTOR_UPDATE_MODE=${PRIVILEGED_ACTOR_UPDATE_MODE}"
}

effective_deployable_final_alphas() {
  local best
  best="$(read_best_alpha "${DEPLOYABLE_SCREEN_RESULTS_ROOT}")"
  unique_alpha_list "0.0 ${best}"
}

run_deployable_final_phase() {
  local child_mode="$1"
  local alphas
  alphas="$(effective_deployable_final_alphas)"
  run_phase0b_v2_child "$child_mode" \
    "SIZE_ABLATION_EPISODES=${WORLD_DATASET_EPISODES}" \
    "SIZE_ABLATION_ALPHAS=${alphas}" \
    "SIZE_ABLATION_SEEDS=${FORMAL_SEEDS}" \
    "TRAIN_EPOCHS=${FORMAL_TRAIN_EPOCHS}" \
    "CHECKPOINT_EVERY_EPOCHS=${FORMAL_CHECKPOINT_EVERY_EPOCHS}" \
    "VAL_MANIFEST_EPISODES=${FINAL_VAL_MANIFEST_EPISODES}" \
    "TEST_MANIFEST_EPISODES=${FINAL_TEST_MANIFEST_EPISODES}" \
    "MANIFEST_ROOT=${FINAL_MANIFEST_ROOT}" \
    "CHECKPOINT_ROOT=${DEPLOYABLE_FINAL_CHECKPOINT_ROOT}" \
    "RESULTS_ROOT=${DEPLOYABLE_FINAL_RESULTS_ROOT}" \
    "USE_ASYMMETRIC_CRITIC=0"
}

effective_privileged_final_alpha() {
  read_best_alpha "${PRIVILEGED_SCREEN_RESULTS_ROOT}"
}

run_privileged_final_phase() {
  local child_mode="$1"
  local alpha
  alpha="$(effective_privileged_final_alpha)"
  run_phase0b_v2_child "$child_mode" \
    "SIZE_ABLATION_EPISODES=${WORLD_DATASET_EPISODES}" \
    "SIZE_ABLATION_ALPHAS=${alpha}" \
    "SIZE_ABLATION_SEEDS=${FORMAL_SEEDS}" \
    "TRAIN_EPOCHS=${FORMAL_TRAIN_EPOCHS}" \
    "CHECKPOINT_EVERY_EPOCHS=${FORMAL_CHECKPOINT_EVERY_EPOCHS}" \
    "VAL_MANIFEST_EPISODES=${FINAL_VAL_MANIFEST_EPISODES}" \
    "TEST_MANIFEST_EPISODES=${FINAL_TEST_MANIFEST_EPISODES}" \
    "MANIFEST_ROOT=${FINAL_MANIFEST_ROOT}" \
    "CHECKPOINT_ROOT=${PRIVILEGED_FINAL_CHECKPOINT_ROOT}" \
    "RESULTS_ROOT=${PRIVILEGED_FINAL_RESULTS_ROOT}" \
    "USE_ASYMMETRIC_CRITIC=1" \
    "PRIVILEGED_ACTOR_UPDATE_MODE=${PRIVILEGED_ACTOR_UPDATE_MODE}"
}

run_compare_analysis() {
  mkdir -p "$COMPARISON_OUTPUT_DIR"
  run_cmd "${PYTHON_CMD[@]}" -m scripts.analyze_offline_td3bc_worldcomp_teacher_gap \
    --deployable-screen-root "$DEPLOYABLE_SCREEN_RESULTS_ROOT" \
    --privileged-screen-root "$PRIVILEGED_SCREEN_RESULTS_ROOT" \
    --deployable-final-root "$DEPLOYABLE_FINAL_RESULTS_ROOT" \
    --privileged-final-root "$PRIVILEGED_FINAL_RESULTS_ROOT" \
    --output-dir "$COMPARISON_OUTPUT_DIR"
}

run_all() {
  ensure_screen_manifests
  ensure_final_manifests
  collect_dataset
  run_deployable_screen_phase train
  run_deployable_screen_phase validate
  run_deployable_screen_phase summarize
  run_deployable_screen_phase analyze
  run_privileged_screen_phase train
  run_privileged_screen_phase validate
  run_privileged_screen_phase summarize
  run_privileged_screen_phase analyze
  run_deployable_final_phase train
  run_deployable_final_phase validate
  run_deployable_final_phase summarize
  run_deployable_final_phase analyze
  run_privileged_final_phase train
  run_privileged_final_phase validate
  run_privileged_final_phase summarize
  run_privileged_final_phase analyze
  run_compare_analysis
}

case "$MODE" in
  all)
    run_all
    ;;
  manifests)
    ensure_screen_manifests
    ensure_final_manifests
    ;;
  collect)
    collect_dataset
    ;;
  screen_deployable_train|screen_deployable_validate|screen_deployable_summarize|screen_deployable_analyze)
    run_deployable_screen_phase "${MODE#screen_deployable_}"
    ;;
  screen_privileged_train|screen_privileged_validate|screen_privileged_summarize|screen_privileged_analyze)
    run_privileged_screen_phase "${MODE#screen_privileged_}"
    ;;
  final_deployable_train|final_deployable_validate|final_deployable_summarize|final_deployable_analyze)
    run_deployable_final_phase "${MODE#final_deployable_}"
    ;;
  final_privileged_train|final_privileged_validate|final_privileged_summarize|final_privileged_analyze)
    run_privileged_final_phase "${MODE#final_privileged_}"
    ;;
  compare)
    run_compare_analysis
    ;;
  *)
    echo "Unsupported MODE: $MODE" >&2
    echo "Supported MODE values: all, manifests, collect," >&2
    echo "  screen_deployable_{train,validate,summarize,analyze}," >&2
    echo "  screen_privileged_{train,validate,summarize,analyze}," >&2
    echo "  final_deployable_{train,validate,summarize,analyze}," >&2
    echo "  final_privileged_{train,validate,summarize,analyze}, compare" >&2
    exit 1
    ;;
esac
