#!/usr/bin/env bash
set -euo pipefail

# ReBRAC broad-validation per-spoke driver.
#
# Spec: docs/superpowers/specs/2026-05-04-rebrac-broad-validation-design.md
# Plan: docs/superpowers/plans/2026-05-04-rebrac-broad-validation-plan.md
#
# Usage:
#   SPOKE_ID=A1 PHASE=p1 bash scripts/run_offline_rebrac_broad.sh
#   SPOKE_ID=B2 PHASE=p1 bash scripts/run_offline_rebrac_broad.sh
#   SPOKE_ID=C3 PHASE=p2_refit ACTOR_PENALTY_COEFS=2.0 CRITIC_PENALTY_COEFS=2.0 bash scripts/run_offline_rebrac_broad.sh
#   SPOKE_ID=C3 PHASE=p2_5seed bash scripts/run_offline_rebrac_broad.sh
#
# All other phase-independent configuration (TRAIN_EPOCHS, BATCH_SIZE, ...) is
# inherited from the same env-var contract as run_offline_rebrac_screen.sh.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SPOKE_ID="${SPOKE_ID:?SPOKE_ID is required (e.g. A1, A2, A2-td3bc, A3, B1, B2, C1, C3)}"
PHASE="${PHASE:?PHASE is required (p1, p2_refit, or p2_5seed)}"
MODE="${MODE:-all}"

PYTHON_PREFIX="${PYTHON_PREFIX:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_CMD=()
if [[ -n "$PYTHON_PREFIX" ]]; then
  read -r -a PYTHON_PREFIX_ARR <<< "$PYTHON_PREFIX"
  PYTHON_CMD+=("${PYTHON_PREFIX_ARR[@]}")
fi
PYTHON_CMD+=("$PYTHON_BIN")

# ---- Spoke registry lookup ----
get_spoke_field() {
  "${PYTHON_CMD[@]}" -m scripts.broad_validation_spoke_registry --get-field "$SPOKE_ID" "$1"
}

DATASET_NAME="$(get_spoke_field dataset_name)"
COLLECTOR_POLICY="$(get_spoke_field collector_policy)"
POLICY_MIXTURE="$(get_spoke_field policy_mixture)"
PROBE_LAYOUT="$(get_spoke_field probe_layout)"
TASK_GEOMETRY="$(get_spoke_field task_geometry)"
FLOW_PATH="$(get_spoke_field flow_path)"
BENCHMARK_KEY="$(get_spoke_field benchmark_key)"
TARGET_SPEED="$(get_spoke_field target_speed)"
ALGO="$(get_spoke_field algo)"
ACTOR_PENALTY_DEFAULT="$(get_spoke_field actor_penalty_coef)"
CRITIC_PENALTY_DEFAULT="$(get_spoke_field critic_penalty_coef)"
TD3BC_ALPHA_DEFAULT="$(get_spoke_field td3bc_alpha)"

DEVICE="${DEVICE:-cuda}"
HISTORY_LENGTH="${HISTORY_LENGTH:-4}"
OBJECTIVE="${OBJECTIVE:-efficiency_v2}"
DATASET_EPISODES="${DATASET_EPISODES:-1000}"
DATASET_SEED="${DATASET_SEED:-0}"
COLLECT_WORKERS="${COLLECT_WORKERS:-8}"

# ---- Phase-driven seed / hyperparam defaults ----
case "$PHASE" in
  p1)
    SEEDS="${SEEDS:-42 44}"
    ACTOR_PENALTY_COEFS="${ACTOR_PENALTY_COEFS:-$ACTOR_PENALTY_DEFAULT}"
    CRITIC_PENALTY_COEFS="${CRITIC_PENALTY_COEFS:-$CRITIC_PENALTY_DEFAULT}"
    TD3BC_ALPHAS="${TD3BC_ALPHAS:-$TD3BC_ALPHA_DEFAULT}"
    ;;
  p2_refit)
    SEEDS="${SEEDS:-42}"
    ACTOR_PENALTY_COEFS="${ACTOR_PENALTY_COEFS:?ACTOR_PENALTY_COEFS required for p2_refit}"
    CRITIC_PENALTY_COEFS="${CRITIC_PENALTY_COEFS:?CRITIC_PENALTY_COEFS required for p2_refit}"
    TD3BC_ALPHAS="${TD3BC_ALPHAS:-$TD3BC_ALPHA_DEFAULT}"
    ;;
  p2_5seed)
    SEEDS="${SEEDS:-42 43 44 45 46}"
    ACTOR_PENALTY_COEFS="${ACTOR_PENALTY_COEFS:-$ACTOR_PENALTY_DEFAULT}"
    CRITIC_PENALTY_COEFS="${CRITIC_PENALTY_COEFS:-$CRITIC_PENALTY_DEFAULT}"
    TD3BC_ALPHAS="${TD3BC_ALPHAS:-$TD3BC_ALPHA_DEFAULT}"
    ;;
  *)
    echo "Unsupported PHASE: ${PHASE}" >&2
    exit 1
    ;;
esac

SAMPLING_MODE="${SAMPLING_MODE:-shuffle_no_replacement}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-64}"
CHECKPOINT_EVERY_EPOCHS="${CHECKPOINT_EVERY_EPOCHS:-8}"
DROP_LAST_BATCH="${DROP_LAST_BATCH:-0}"
BATCH_SIZE="${BATCH_SIZE:-256}"
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

VAL_MANIFEST_EPISODES="${VAL_MANIFEST_EPISODES:-40}"
TEST_MANIFEST_EPISODES="${TEST_MANIFEST_EPISODES:-100}"
MANIFEST_ROOT="${MANIFEST_ROOT:-benchmarks/offline_rebrac_broad}"
VAL_MANIFEST_DIR="${VAL_MANIFEST_DIR:-${MANIFEST_ROOT}/val_${VAL_MANIFEST_EPISODES}}"
TEST_MANIFEST_DIR="${TEST_MANIFEST_DIR:-${MANIFEST_ROOT}/test_${TEST_MANIFEST_EPISODES}}"
VAL_MANIFEST_PATH="${VAL_MANIFEST_PATH:-${VAL_MANIFEST_DIR}/${BENCHMARK_KEY}.json}"
TEST_MANIFEST_PATH="${TEST_MANIFEST_PATH:-${TEST_MANIFEST_DIR}/${BENCHMARK_KEY}.json}"

EVAL_WORKERS="${EVAL_WORKERS:-6}"
EVAL_WORKER_DEVICE="${EVAL_WORKER_DEVICE:-cpu}"
VALIDATION_SEED="${VALIDATION_SEED:-123}"
TEST_SEED="${TEST_SEED:-456}"
FORCE_REEVAL="${FORCE_REEVAL:-0}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/offline/rebrac/broad_validation/${SPOKE_ID}}"
RESULTS_ROOT="${RESULTS_ROOT:-results/offline/rebrac/broad_validation/${SPOKE_ID}}"

DATASET_DIR="offline_data/${DATASET_NAME}"

run_cmd() {
  echo
  echo "[cmd] $*"
  "$@"
}

ensure_manifest() {
  local episodes="$1" output_dir="$2" output_path="$3"
  if [[ -f "$output_path" ]]; then
    echo "[skip] manifest exists: $output_path"
    return
  fi
  mkdir -p "$output_dir"
  run_cmd "${PYTHON_CMD[@]}" -m scripts.generate_standard_benchmarks \
    --benchmarks "$BENCHMARK_KEY" \
    --episodes "$episodes" \
    --output-dir "$output_dir"
}

ensure_manifests() {
  ensure_manifest "$VAL_MANIFEST_EPISODES" "$VAL_MANIFEST_DIR" "$VAL_MANIFEST_PATH"
  ensure_manifest "$TEST_MANIFEST_EPISODES" "$TEST_MANIFEST_DIR" "$TEST_MANIFEST_PATH"
}

ensure_dataset() {
  if [[ -f "${DATASET_DIR}/transitions.npz" ]]; then
    echo "[skip] dataset exists: ${DATASET_DIR}/transitions.npz"
    return
  fi
  mkdir -p "$DATASET_DIR"
  local extra_flags=()
  if [[ -n "$POLICY_MIXTURE" ]]; then
    extra_flags+=(--policy-mixture "$POLICY_MIXTURE")
  fi
  run_cmd "${PYTHON_CMD[@]}" -m scripts.collect_offline_data \
    --policy "$COLLECTOR_POLICY" \
    --flow "$FLOW_PATH" \
    --probe-layout "$PROBE_LAYOUT" \
    --history-length "$HISTORY_LENGTH" \
    --task-geometry "$TASK_GEOMETRY" \
    --target-speed "$TARGET_SPEED" \
    --objective "$OBJECTIVE" \
    --episodes "$DATASET_EPISODES" \
    --seed "$DATASET_SEED" \
    --num-workers "$COLLECT_WORKERS" \
    --output-dir "$DATASET_DIR" \
    "${extra_flags[@]}"
  run_cmd "${PYTHON_CMD[@]}" -m scripts.write_sanity_card \
    --dataset-dir "$DATASET_DIR" \
    --expected-probe-layout "$PROBE_LAYOUT"
}

pair_tag_rebrac() {
  local actor_tag="${1//./p}"
  local critic_tag="${2//./p}"
  echo "actorb_${actor_tag}__criticb_${critic_tag}"
}
pair_tag_td3bc() {
  local alpha_tag="${1//./p}"
  echo "alpha_${alpha_tag}"
}

run_dir_for() {
  local pair="$1" seed="$2"
  echo "${CHECKPOINT_ROOT}/${pair}/seed_${seed}"
}
result_dir_for() {
  local pair="$1"
  echo "${RESULTS_ROOT}/${pair}"
}

compute_schedule() {
  local schedule_lines
  schedule_lines="$(
    "${PYTHON_CMD[@]}" - "$DATASET_DIR" "$BATCH_SIZE" "$TRAIN_EPOCHS" "$CHECKPOINT_EVERY_EPOCHS" "$DROP_LAST_BATCH" <<'PY'
import json, math, sys
from pathlib import Path
dataset_dir = Path(sys.argv[1])
batch_size = max(1, int(sys.argv[2]))
train_epochs = max(1, int(sys.argv[3]))
checkpoint_every_epochs = max(1, int(sys.argv[4]))
drop_last = sys.argv[5] == "1"
metadata_path = dataset_dir / "metadata.json"
num_transitions = None
if metadata_path.exists():
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    value = data.get("num_transitions")
    if value is not None:
        num_transitions = int(value)
if num_transitions is None:
    import numpy as np
    with np.load(dataset_dir / "transitions.npz", mmap_mode="r") as payload:
        num_transitions = int(payload["obs"].shape[0])
if drop_last:
    steps_per_epoch = num_transitions // batch_size
    if steps_per_epoch <= 0:
        raise ValueError("drop_last_batch=True requires num_transitions >= batch_size.")
else:
    steps_per_epoch = max(1, math.ceil(num_transitions / batch_size))
total_steps = steps_per_epoch * train_epochs
checkpoint_every_steps = max(1, steps_per_epoch * checkpoint_every_epochs)
print(f"total_steps={total_steps}")
print(f"checkpoint_every_steps={checkpoint_every_steps}")
PY
  )"
  SCHEDULE_TOTAL_STEPS=""
  SCHEDULE_CHECKPOINT_EVERY_STEPS=""
  while IFS='=' read -r key value; do
    case "$key" in
      total_steps) SCHEDULE_TOTAL_STEPS="$value" ;;
      checkpoint_every_steps) SCHEDULE_CHECKPOINT_EVERY_STEPS="$value" ;;
    esac
  done <<< "$schedule_lines"
}

train_one_rebrac() {
  local actor_beta="$1" critic_beta="$2" seed="$3"
  local pair run_dir
  pair="$(pair_tag_rebrac "$actor_beta" "$critic_beta")"
  run_dir="$(run_dir_for "$pair" "$seed")"
  if [[ -f "${run_dir}/trainer_state.json" && -f "${run_dir}/agent_final.pt" ]]; then
    echo "[skip] trained run exists: ${run_dir}"
    return
  fi
  mkdir -p "$run_dir"
  local extra_flags=()
  if [[ "$DROP_LAST_BATCH" == "1" ]]; then
    extra_flags+=(--drop-last-batch)
  fi
  run_cmd "${PYTHON_CMD[@]}" -m scripts.train_offline \
    --algo rebrac \
    --offline-data "${DATASET_DIR}/transitions.npz" \
    --flow "$FLOW_PATH" \
    --manifest "$VAL_MANIFEST_PATH" \
    --probe-layout "$PROBE_LAYOUT" \
    --history-length "$HISTORY_LENGTH" \
    --task-geometry "$TASK_GEOMETRY" \
    --target-speed "$TARGET_SPEED" \
    --objective "$OBJECTIVE" \
    --sampling-mode "$SAMPLING_MODE" \
    --num-epochs "$TRAIN_EPOCHS" \
    --total-steps "$SCHEDULE_TOTAL_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --hidden-dim "$HIDDEN_DIM" \
    --num-hidden-layers "$NUM_HIDDEN_LAYERS" \
    --actor-lr "$ACTOR_LR" \
    --critic-lr "$CRITIC_LR" \
    --gamma "$GAMMA" \
    --tau "$TAU" \
    --actor-penalty-coef "$actor_beta" \
    --critic-penalty-coef "$critic_beta" \
    --policy-noise "$POLICY_NOISE" \
    --noise-clip "$NOISE_CLIP" \
    --policy-freq "$POLICY_FREQ" \
    --grad-clip-norm "$GRAD_CLIP_NORM" \
    --normalizer-eps "$NORMALIZER_EPS" \
    --eval-every 0 \
    --checkpoint-every "$SCHEDULE_CHECKPOINT_EVERY_STEPS" \
    --skip-final-eval \
    --log-every "$LOG_EVERY" \
    --critic-layernorm \
    --no-actor-layernorm \
    --save-dir "$run_dir" \
    --seed "$seed" \
    --device "$DEVICE" \
    "${extra_flags[@]}"
}

train_one_td3bc() {
  local alpha="$1" seed="$2"
  local pair run_dir
  pair="$(pair_tag_td3bc "$alpha")"
  run_dir="$(run_dir_for "$pair" "$seed")"
  if [[ -f "${run_dir}/trainer_state.json" && -f "${run_dir}/agent_final.pt" ]]; then
    echo "[skip] trained run exists: ${run_dir}"
    return
  fi
  mkdir -p "$run_dir"
  local extra_flags=()
  if [[ "$DROP_LAST_BATCH" == "1" ]]; then
    extra_flags+=(--drop-last-batch)
  fi
  run_cmd "${PYTHON_CMD[@]}" -m scripts.train_offline \
    --algo td3bc \
    --offline-data "${DATASET_DIR}/transitions.npz" \
    --flow "$FLOW_PATH" \
    --manifest "$VAL_MANIFEST_PATH" \
    --probe-layout "$PROBE_LAYOUT" \
    --history-length "$HISTORY_LENGTH" \
    --task-geometry "$TASK_GEOMETRY" \
    --target-speed "$TARGET_SPEED" \
    --objective "$OBJECTIVE" \
    --sampling-mode "$SAMPLING_MODE" \
    --num-epochs "$TRAIN_EPOCHS" \
    --total-steps "$SCHEDULE_TOTAL_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --hidden-dim "$HIDDEN_DIM" \
    --num-hidden-layers "$NUM_HIDDEN_LAYERS" \
    --actor-lr "$ACTOR_LR" \
    --critic-lr "$CRITIC_LR" \
    --gamma "$GAMMA" \
    --tau "$TAU" \
    --alpha "$alpha" \
    --policy-noise "$POLICY_NOISE" \
    --noise-clip "$NOISE_CLIP" \
    --policy-freq "$POLICY_FREQ" \
    --grad-clip-norm "$GRAD_CLIP_NORM" \
    --normalizer-eps "$NORMALIZER_EPS" \
    --eval-every 0 \
    --checkpoint-every "$SCHEDULE_CHECKPOINT_EVERY_STEPS" \
    --skip-final-eval \
    --log-every "$LOG_EVERY" \
    --critic-layernorm \
    --no-actor-layernorm \
    --save-dir "$run_dir" \
    --seed "$seed" \
    --device "$DEVICE" \
    "${extra_flags[@]}"
}

validate_one() {
  local pair="$1" seed="$2"
  local run_dir result_dir val_dir
  run_dir="$(run_dir_for "$pair" "$seed")"
  result_dir="$(result_dir_for "$pair")"
  val_dir="${result_dir}/validation/seed_${seed}"
  if [[ ! -d "$run_dir" ]]; then
    echo "[skip] missing run dir: ${run_dir}"
    return
  fi
  if [[ ! -f "${run_dir}/trainer_state.json" || ! -f "${run_dir}/agent_final.pt" ]]; then
    echo "[warn] skipping ${run_dir}: missing trainer_state.json or agent_final.pt" >&2
    return
  fi
  mkdir -p "$val_dir"
  local checkpoint_files=()
  shopt -s nullglob
  local _step_files=("$run_dir"/agent_step_*.pt)
  shopt -u nullglob
  if [[ "${#_step_files[@]}" -gt 0 ]]; then
    while IFS= read -r _path; do
      checkpoint_files+=("$(basename "$_path")")
    done < <(printf '%s\n' "${_step_files[@]}" | sort)
  fi
  checkpoint_files+=("agent_final.pt")
  for agent_file in "${checkpoint_files[@]}"; do
    local agent_tag="${agent_file%.pt}"
    local output_json="${val_dir}/${agent_tag}.json"
    if [[ -f "$output_json" && "$FORCE_REEVAL" != "1" ]]; then
      echo "[skip] validation exists: ${output_json}"
      continue
    fi
    run_cmd "${PYTHON_CMD[@]}" -m scripts.evaluate_offline \
      --checkpoint "$run_dir" \
      --agent-file "$agent_file" \
      --manifest "$VAL_MANIFEST_PATH" \
      --device "$DEVICE" \
      --num-workers "$EVAL_WORKERS" \
      --worker-device "$EVAL_WORKER_DEVICE" \
      --seed "$VALIDATION_SEED" \
      --output-json "$output_json"
  done
}

select_best_checkpoint() {
  local pair="$1" seed="$2"
  local run_dir result_dir val_dir selection_dir output_json trainer_state_path
  run_dir="$(run_dir_for "$pair" "$seed")"
  result_dir="$(result_dir_for "$pair")"
  val_dir="${result_dir}/validation/seed_${seed}"
  selection_dir="${result_dir}/selection/seed_${seed}"
  output_json="${selection_dir}/selected_checkpoint.json"
  trainer_state_path="${run_dir}/trainer_state.json"
  if [[ -f "$output_json" && "$FORCE_REEVAL" != "1" ]]; then
    echo "[skip] checkpoint selection exists: ${output_json}"
    return
  fi
  if [[ ! -d "$val_dir" ]]; then
    echo "[skip] missing validation dir: ${val_dir}"
    return
  fi
  mkdir -p "$selection_dir"
  local agent_file_sidecar="${selection_dir}/selected_agent_file.txt"
  run_cmd "${PYTHON_CMD[@]}" - "$val_dir" "$trainer_state_path" "$output_json" "$agent_file_sidecar" <<'PY'
import json, sys
from pathlib import Path
eval_dir = Path(sys.argv[1])
trainer_state_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])
agent_file_sidecar = Path(sys.argv[4])
trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
train_step = int(trainer_state.get("train_step", 0))
records = []
for json_path in sorted(eval_dir.glob("*.json")):
    if json_path.name == output_path.name:
        continue
    metrics = json.loads(json_path.read_text(encoding="utf-8"))
    agent_file = f"{json_path.stem}.pt"
    if json_path.stem.startswith("agent_step_"):
        try:
            step = int(json_path.stem.split("_")[-1])
        except ValueError:
            step = None
    elif json_path.stem == "agent_final":
        step = train_step
    else:
        step = None
    records.append({
        "agent_file": agent_file,
        "agent_tag": json_path.stem,
        "train_step": step,
        "eval_success_rate": float(metrics["eval_success_rate"]),
        "eval_return": float(metrics["eval_return"]),
        "eval_safety_cost": float(metrics["eval_safety_cost"]),
        "eval_time_s": float(metrics["eval_time_s"]),
        "metrics": metrics,
    })
if not records:
    raise ValueError(f"No validation metrics found under {eval_dir}")
best = max(records, key=lambda item: (
    item["eval_success_rate"], item["eval_return"],
    -item["eval_safety_cost"], -item["eval_time_s"],
))
payload = {
    "selection_metric": "eval_success_rate -> eval_return -> -eval_safety_cost -> -eval_time_s",
    "num_candidates": len(records),
    "best": best,
    "candidates": records,
}
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
agent_file_sidecar.parent.mkdir(parents=True, exist_ok=True)
agent_file_sidecar.write_text(best["agent_file"] + "\n", encoding="utf-8")
print(f"[write] selection: {output_path}")
print(f"[best] agent={best['agent_file']} success={best['eval_success_rate']:.4f}")
PY
}

test_one() {
  local pair="$1" seed="$2"
  local run_dir result_dir selection_dir selection_path agent_file_sidecar test_dir output_json agent_file
  run_dir="$(run_dir_for "$pair" "$seed")"
  result_dir="$(result_dir_for "$pair")"
  selection_dir="${result_dir}/selection/seed_${seed}"
  selection_path="${selection_dir}/selected_checkpoint.json"
  agent_file_sidecar="${selection_dir}/selected_agent_file.txt"
  test_dir="${result_dir}/test"
  output_json="${test_dir}/seed_${seed}.json"
  if [[ ! -f "$selection_path" ]]; then
    echo "[skip] missing selection: ${selection_path}"
    return
  fi
  if [[ -f "$output_json" && "$FORCE_REEVAL" != "1" ]]; then
    echo "[skip] test exists: ${output_json}"
    return
  fi
  mkdir -p "$test_dir"
  if [[ -f "$agent_file_sidecar" ]]; then
    agent_file="$(< "$agent_file_sidecar")"
    agent_file="${agent_file//$'\n'/}"
  else
    agent_file="$("${PYTHON_CMD[@]}" - "$selection_path" <<'PY'
import json, sys
from pathlib import Path
print(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["best"]["agent_file"])
PY
)"
  fi
  run_cmd "${PYTHON_CMD[@]}" -m scripts.evaluate_offline \
    --checkpoint "$run_dir" \
    --agent-file "$agent_file" \
    --manifest "$TEST_MANIFEST_PATH" \
    --device "$DEVICE" \
    --num-workers "$EVAL_WORKERS" \
    --worker-device "$EVAL_WORKER_DEVICE" \
    --seed "$TEST_SEED" \
    --output-json "$output_json"
}

# ---- Top-level execution ----
ensure_dataset
ensure_manifests
compute_schedule

if [[ "$ALGO" == "rebrac" ]]; then
  for actor_beta in $ACTOR_PENALTY_COEFS; do
    for critic_beta in $CRITIC_PENALTY_COEFS; do
      pair="$(pair_tag_rebrac "$actor_beta" "$critic_beta")"
      for seed in $SEEDS; do
        train_one_rebrac "$actor_beta" "$critic_beta" "$seed"
        validate_one "$pair" "$seed"
        select_best_checkpoint "$pair" "$seed"
        test_one "$pair" "$seed"
      done
    done
  done
elif [[ "$ALGO" == "td3bc" ]]; then
  for alpha in $TD3BC_ALPHAS; do
    pair="$(pair_tag_td3bc "$alpha")"
    for seed in $SEEDS; do
      train_one_td3bc "$alpha" "$seed"
      validate_one "$pair" "$seed"
      select_best_checkpoint "$pair" "$seed"
      test_one "$pair" "$seed"
    done
  done
else
  echo "Unsupported ALGO: ${ALGO}" >&2
  exit 1
fi

echo
echo "[done] SPOKE_ID=${SPOKE_ID} PHASE=${PHASE}"
