#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Phase0b v2 protocol:
# 1. Build fixed validation/test manifests.
# 2. Collect offline datasets if missing.
# 3. Train TD3+BC with epoch-aligned budgets (default: no-replacement sampling).
# 4. Select checkpoints and alpha on validation, then run test only for the selected models.

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

SIZE_ABLATION_EPISODES="${SIZE_ABLATION_EPISODES:-500 1000 2000}"
SIZE_ABLATION_ALPHAS="${SIZE_ABLATION_ALPHAS:-0.0 0.05 0.1 0.25 0.5 1.0}"
SIZE_ABLATION_SEEDS="${SIZE_ABLATION_SEEDS:-42 43 44 45 46}"

BATCH_SIZE="${BATCH_SIZE:-256}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-96}"
CHECKPOINT_EVERY_EPOCHS="${CHECKPOINT_EVERY_EPOCHS:-4}"
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

VAL_MANIFEST_EPISODES="${VAL_MANIFEST_EPISODES:-20}"
TEST_MANIFEST_EPISODES="${TEST_MANIFEST_EPISODES:-100}"
MANIFEST_ROOT="${MANIFEST_ROOT:-benchmarks/offline_phase0b_v2}"
VAL_MANIFEST_DIR="${VAL_MANIFEST_DIR:-${MANIFEST_ROOT}/val_${VAL_MANIFEST_EPISODES}}"
TEST_MANIFEST_DIR="${TEST_MANIFEST_DIR:-${MANIFEST_ROOT}/test_${TEST_MANIFEST_EPISODES}}"
VAL_MANIFEST_PATH="${VAL_MANIFEST_PATH:-${VAL_MANIFEST_DIR}/${BENCHMARK_KEY}.json}"
TEST_MANIFEST_PATH="${TEST_MANIFEST_PATH:-${TEST_MANIFEST_DIR}/${BENCHMARK_KEY}.json}"

RUN_BASELINE_EVAL="${RUN_BASELINE_EVAL:-1}"
EVAL_WORKERS="${EVAL_WORKERS:-8}"
EVAL_WORKER_DEVICE="${EVAL_WORKER_DEVICE:-cpu}"
VALIDATION_SEED="${VALIDATION_SEED:-123}"
TEST_SEED="${TEST_SEED:-456}"
FORCE_REEVAL="${FORCE_REEVAL:-0}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/offline/td3bc/phase0b_v2}"
RESULTS_ROOT="${RESULTS_ROOT:-results/offline/td3bc/phase0b_v2}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${RESULTS_ROOT}/summaries}"
ANALYSIS_OUTPUT_DIR="${ANALYSIS_OUTPUT_DIR:-${RESULTS_ROOT}/analysis}"
RUN_BC_TEST_ANALYSIS="${RUN_BC_TEST_ANALYSIS:-1}"
SKIP_ANALYSIS_PLOTS="${SKIP_ANALYSIS_PLOTS:-0}"

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

dataset_dir_for_name() {
  local dataset_name="$1"
  echo "offline_data/${dataset_name}"
}

checkpoint_dir_for_name() {
  local dataset_name="$1"
  echo "${CHECKPOINT_ROOT}/${dataset_name}"
}

result_dir_for_name() {
  local dataset_name="$1"
  echo "${RESULTS_ROOT}/${dataset_name}"
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
  run_cmd "${PYTHON_CMD[@]}" -m scripts.generate_standard_benchmarks \
    --benchmarks "$BENCHMARK_KEY" \
    --episodes "$episodes" \
    --output-dir "$output_dir"
}

ensure_dataset() {
  local dataset_name="$1"
  local dataset_episodes="$2"
  local dataset_dir
  dataset_dir="$(dataset_dir_for_name "$dataset_name")"
  if [[ -f "${dataset_dir}/transitions.npz" ]]; then
    echo "[skip] dataset exists: ${dataset_dir}/transitions.npz"
    return
  fi

  mkdir -p "$dataset_dir"
  extra_collect=(
    --action-noise-std "$ACTION_NOISE_STD"
    --action-noise-clip "$ACTION_NOISE_CLIP"
  )
  if [[ -n "$DATASET_POLICY_MIXTURE" ]]; then
    extra_collect+=(--policy-mixture "$DATASET_POLICY_MIXTURE")
  fi

  run_cmd "${PYTHON_CMD[@]}" -m scripts.collect_offline_data \
    --policy "$DATASET_POLICY" \
    --flow "$FLOW_PATH" \
    --probe-layout "$PROBE_LAYOUT" \
    --history-length "$HISTORY_LENGTH" \
    --task-geometry "$TASK_GEOMETRY" \
    --target-speed "$TARGET_SPEED" \
    --objective "$OBJECTIVE" \
    --episodes "$dataset_episodes" \
    --seed "$BASE_DATASET_SEED" \
    --num-workers "$COLLECT_WORKERS" \
    "${extra_collect[@]}" \
    --output-dir "$dataset_dir"
}

compute_schedule() {
  local dataset_dir="$1"
  local schedule_lines
  schedule_lines="$(
    "${PYTHON_CMD[@]}" - "$dataset_dir" "$BATCH_SIZE" "$TRAIN_EPOCHS" "$CHECKPOINT_EVERY_EPOCHS" "$DROP_LAST_BATCH" <<'PY'
import json
import math
import sys
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

print(f"num_transitions={num_transitions}")
print(f"steps_per_epoch={steps_per_epoch}")
print(f"total_steps={total_steps}")
print(f"checkpoint_every_steps={checkpoint_every_steps}")
PY
  )"

  SCHEDULE_NUM_TRANSITIONS=""
  SCHEDULE_STEPS_PER_EPOCH=""
  SCHEDULE_TOTAL_STEPS=""
  SCHEDULE_CHECKPOINT_EVERY_STEPS=""
  while IFS='=' read -r key value; do
    case "$key" in
      num_transitions) SCHEDULE_NUM_TRANSITIONS="$value" ;;
      steps_per_epoch) SCHEDULE_STEPS_PER_EPOCH="$value" ;;
      total_steps) SCHEDULE_TOTAL_STEPS="$value" ;;
      checkpoint_every_steps) SCHEDULE_CHECKPOINT_EVERY_STEPS="$value" ;;
    esac
  done <<< "$schedule_lines"
}

run_baseline_eval() {
  local dataset_name="$1"
  local result_dir
  result_dir="$(result_dir_for_name "$dataset_name")"
  local baseline_dir="${result_dir}/baselines"
  if [[ "$RUN_BASELINE_EVAL" != "1" ]]; then
    echo "[skip] baseline evaluation disabled"
    return
  fi

  mkdir -p "$baseline_dir"
  if [[ ! -f "${baseline_dir}/${DATASET_POLICY}_val_eval.json" || "$FORCE_REEVAL" == "1" ]]; then
    run_cmd "${PYTHON_CMD[@]}" -m scripts.evaluate_baseline_on_manifest \
      --policy "$DATASET_POLICY" \
      --flow "$FLOW_PATH" \
      --probe-layout "$PROBE_LAYOUT" \
      --history-length "$HISTORY_LENGTH" \
      --task-geometry "$TASK_GEOMETRY" \
      --target-speed "$TARGET_SPEED" \
      --objective "$OBJECTIVE" \
      --manifest "$VAL_MANIFEST_PATH" \
      --num-workers "$EVAL_WORKERS" \
      --output-json "${baseline_dir}/${DATASET_POLICY}_val_eval.json"
  else
    echo "[skip] baseline val exists: ${baseline_dir}/${DATASET_POLICY}_val_eval.json"
  fi

  if [[ ! -f "${baseline_dir}/${DATASET_POLICY}_test_eval.json" || "$FORCE_REEVAL" == "1" ]]; then
    run_cmd "${PYTHON_CMD[@]}" -m scripts.evaluate_baseline_on_manifest \
      --policy "$DATASET_POLICY" \
      --flow "$FLOW_PATH" \
      --probe-layout "$PROBE_LAYOUT" \
      --history-length "$HISTORY_LENGTH" \
      --task-geometry "$TASK_GEOMETRY" \
      --target-speed "$TARGET_SPEED" \
      --objective "$OBJECTIVE" \
      --manifest "$TEST_MANIFEST_PATH" \
      --num-workers "$EVAL_WORKERS" \
      --output-json "${baseline_dir}/${DATASET_POLICY}_test_eval.json"
  else
    echo "[skip] baseline test exists: ${baseline_dir}/${DATASET_POLICY}_test_eval.json"
  fi
}

train_one() {
  local dataset_name="$1"
  local alpha="$2"
  local seed="$3"
  local dataset_dir run_root alpha_tag run_dir

  dataset_dir="$(dataset_dir_for_name "$dataset_name")"
  run_root="$(checkpoint_dir_for_name "$dataset_name")"
  alpha_tag="${alpha//./p}"
  run_dir="${run_root}/alpha_${alpha_tag}/seed_${seed}"

  if [[ -f "${run_dir}/trainer_state.json" && -f "${run_dir}/agent_final.pt" ]]; then
    echo "[skip] trained run exists: ${run_dir}"
    return
  fi

  compute_schedule "$dataset_dir"
  mkdir -p "$run_dir"

  extra_flags=()
  if [[ "$USE_LAYERNORM" == "1" ]]; then
    extra_flags+=(--use-layernorm)
  fi
  if [[ "$USE_ASYMMETRIC_CRITIC" == "1" ]]; then
    extra_flags+=(--use-asymmetric-critic --privileged-actor-update-mode "$PRIVILEGED_ACTOR_UPDATE_MODE")
  fi
  if [[ "$DROP_LAST_BATCH" == "1" ]]; then
    extra_flags+=(--drop-last-batch)
  fi

  run_cmd "${PYTHON_CMD[@]}" -m scripts.train_offline \
    --offline-data "${dataset_dir}/transitions.npz" \
    --alpha "$alpha" \
    --flow "$FLOW_PATH" \
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
    --actor-lr "$ACTOR_LR" \
    --critic-lr "$CRITIC_LR" \
    --gamma "$GAMMA" \
    --tau "$TAU" \
    --policy-noise "$POLICY_NOISE" \
    --noise-clip "$NOISE_CLIP" \
    --policy-freq "$POLICY_FREQ" \
    --normalizer-eps "$NORMALIZER_EPS" \
    --grad-clip-norm "$GRAD_CLIP_NORM" \
    --eval-every 0 \
    --eval-workers 1 \
    --log-every "$LOG_EVERY" \
    --checkpoint-every "$SCHEDULE_CHECKPOINT_EVERY_STEPS" \
    --skip-final-eval \
    --save-dir "$run_dir" \
    --seed "$seed" \
    --device "$DEVICE" \
    "${extra_flags[@]}"
}

validate_one() {
  local dataset_name="$1"
  local alpha="$2"
  local seed="$3"
  local result_dir run_root alpha_tag run_dir val_dir

  result_dir="$(result_dir_for_name "$dataset_name")"
  run_root="$(checkpoint_dir_for_name "$dataset_name")"
  alpha_tag="${alpha//./p}"
  run_dir="${run_root}/alpha_${alpha_tag}/seed_${seed}"
  val_dir="${result_dir}/validation/alpha_${alpha_tag}/seed_${seed}"

  if [[ ! -d "$run_dir" ]]; then
    echo "[skip] missing run dir: ${run_dir}"
    return
  fi

  mkdir -p "$val_dir"
  mapfile -t checkpoint_files < <(
    "${PYTHON_CMD[@]}" - "$run_dir" <<'PY'
from pathlib import Path
import sys

run_dir = Path(sys.argv[1])
names = [
    path.name
    for path in run_dir.glob("agent_step_*.pt")
]
if (run_dir / "agent_final.pt").exists():
    names.append("agent_final.pt")
for name in sorted(names):
    print(name)
PY
  )

  if [[ "${#checkpoint_files[@]}" -eq 0 ]]; then
    echo "[skip] no checkpoints found under ${run_dir}"
    return
  fi

  local agent_file agent_tag output_json
  for agent_file in "${checkpoint_files[@]}"; do
    agent_tag="${agent_file%.pt}"
    output_json="${val_dir}/${agent_tag}.json"
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
  local dataset_name="$1"
  local alpha="$2"
  local seed="$3"
  local result_dir alpha_tag val_dir selection_dir output_json trainer_state_path

  result_dir="$(result_dir_for_name "$dataset_name")"
  alpha_tag="${alpha//./p}"
  val_dir="${result_dir}/validation/alpha_${alpha_tag}/seed_${seed}"
  selection_dir="${result_dir}/selection/alpha_${alpha_tag}/seed_${seed}"
  output_json="${selection_dir}/selected_checkpoint.json"
  trainer_state_path="$(checkpoint_dir_for_name "$dataset_name")/alpha_${alpha_tag}/seed_${seed}/trainer_state.json"

  if [[ -f "$output_json" && "$FORCE_REEVAL" != "1" ]]; then
    echo "[skip] checkpoint selection exists: ${output_json}"
    return
  fi

  mkdir -p "$selection_dir"
  run_cmd "${PYTHON_CMD[@]}" - "$val_dir" "$trainer_state_path" "$output_json" <<'PY'
import json
import sys
from pathlib import Path

eval_dir = Path(sys.argv[1])
trainer_state_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])

if not eval_dir.exists():
    raise FileNotFoundError(f"Missing validation directory: {eval_dir}")

trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
train_step = int(trainer_state.get("train_step", 0))

records = []
for json_path in sorted(eval_dir.glob("*.json")):
    if json_path.name == output_path.name:
        continue
    metrics = json.loads(json_path.read_text(encoding="utf-8"))
    agent_file = f"{json_path.stem}.pt"
    step = train_step if json_path.stem == "agent_final" else None
    if json_path.stem.startswith("agent_step_"):
        try:
            step = int(json_path.stem.split("_")[-1])
        except ValueError:
            step = None
    records.append(
        {
            "agent_file": agent_file,
            "agent_tag": json_path.stem,
            "train_step": step,
            "eval_success_rate": float(metrics["eval_success_rate"]),
            "eval_return": float(metrics["eval_return"]),
            "eval_safety_cost": float(metrics["eval_safety_cost"]),
            "eval_time_s": float(metrics["eval_time_s"]),
            "metrics": metrics,
        }
    )

if not records:
    raise ValueError(f"No validation metrics found under {eval_dir}")

best = max(
    records,
    key=lambda item: (
        item["eval_success_rate"],
        item["eval_return"],
        -item["eval_safety_cost"],
        -item["eval_time_s"],
    ),
)

payload = {
    "selection_metric": "eval_success_rate -> eval_return -> -eval_safety_cost -> -eval_time_s",
    "num_candidates": len(records),
    "best": best,
    "candidates": records,
}
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"[write] selection: {output_path}")
print(f"[best] agent={best['agent_file']} success={best['eval_success_rate']:.4f} return={best['eval_return']:.4f}")
PY
}

select_best_alpha() {
  local dataset_name="$1"
  local result_dir selection_root output_json

  result_dir="$(result_dir_for_name "$dataset_name")"
  selection_root="${result_dir}/selection"
  output_json="${selection_root}/best_alpha.json"

  if [[ -f "$output_json" && "$FORCE_REEVAL" != "1" ]]; then
    echo "[skip] alpha selection exists: ${output_json}"
    return
  fi

  mkdir -p "$selection_root"
  run_cmd "${PYTHON_CMD[@]}" - "$selection_root" "$output_json" <<'PY'
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

selection_root = Path(sys.argv[1])
output_path = Path(sys.argv[2])

grouped = defaultdict(list)
for selection_path in sorted(selection_root.glob("alpha_*/seed_*/selected_checkpoint.json")):
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    alpha_tag = selection_path.parents[1].name
    best = payload["best"]
    grouped[alpha_tag].append(
        {
            "seed": selection_path.parent.name,
            "agent_file": best["agent_file"],
            "agent_tag": best["agent_tag"],
            "train_step": best["train_step"],
            "val_success_rate": float(best["eval_success_rate"]),
            "val_return": float(best["eval_return"]),
            "val_safety_cost": float(best["eval_safety_cost"]),
            "val_time_s": float(best["eval_time_s"]),
            "selection_path": str(selection_path),
        }
    )

if not grouped:
    raise ValueError(f"No per-run selections found under {selection_root}")

per_alpha = []
for alpha_tag, records in sorted(grouped.items()):
    success = [item["val_success_rate"] for item in records]
    returns = [item["val_return"] for item in records]
    safety = [item["val_safety_cost"] for item in records]
    time_s = [item["val_time_s"] for item in records]
    alpha = float(alpha_tag[len("alpha_") :].replace("p", "."))
    per_alpha.append(
        {
            "alpha_tag": alpha_tag,
            "alpha": alpha,
            "num_seeds": len(records),
            "mean_val_success_rate": statistics.fmean(success),
            "std_val_success_rate": statistics.pstdev(success) if len(success) > 1 else 0.0,
            "mean_val_return": statistics.fmean(returns),
            "mean_val_safety_cost": statistics.fmean(safety),
            "mean_val_time_s": statistics.fmean(time_s),
            "selected_runs": records,
        }
    )

best = max(
    per_alpha,
    key=lambda item: (
        item["mean_val_success_rate"],
        item["mean_val_return"],
        -item["mean_val_safety_cost"],
        -item["mean_val_time_s"],
    ),
)

payload = {
    "selection_metric": "mean_val_success_rate -> mean_val_return -> -mean_val_safety_cost -> -mean_val_time_s",
    "best_alpha_tag": best["alpha_tag"],
    "best_alpha": best["alpha"],
    "per_alpha": per_alpha,
    "selected_runs": best["selected_runs"],
}
output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"[write] alpha selection: {output_path}")
print(f"[best-alpha] {best['alpha_tag']} success={best['mean_val_success_rate']:.4f} return={best['mean_val_return']:.4f}")
PY
}

run_test_for_dataset() {
  local dataset_name="$1"
  local result_dir alpha_selection_path

  result_dir="$(result_dir_for_name "$dataset_name")"
  alpha_selection_path="${result_dir}/selection/best_alpha.json"

  if [[ ! -f "$alpha_selection_path" ]]; then
    echo "[skip] missing alpha selection: ${alpha_selection_path}"
    return
  fi

  mapfile -t test_jobs < <(
    "${PYTHON_CMD[@]}" - "$alpha_selection_path" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for item in payload["selected_runs"]:
    print(f"{item['seed']}|{item['agent_file']}|{payload['best_alpha_tag']}")
PY
  )

  local job seed agent_file alpha_tag run_dir output_json
  for job in "${test_jobs[@]}"; do
    IFS='|' read -r seed agent_file alpha_tag <<< "$job"
    run_dir="$(checkpoint_dir_for_name "$dataset_name")/${alpha_tag}/${seed}"
    output_json="${result_dir}/test_selected/${alpha_tag}/${seed}.json"
    mkdir -p "$(dirname "$output_json")"
    if [[ -f "$output_json" && "$FORCE_REEVAL" != "1" ]]; then
      echo "[skip] test exists: ${output_json}"
      continue
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
  done
}

write_dataset_summary() {
  local dataset_name="$1"
  local result_dir summary_json summary_csv

  result_dir="$(result_dir_for_name "$dataset_name")"
  mkdir -p "$SUMMARY_ROOT"
  summary_json="${SUMMARY_ROOT}/${dataset_name}_summary.json"
  summary_csv="${SUMMARY_ROOT}/${dataset_name}_summary.csv"

  run_cmd "${PYTHON_CMD[@]}" - "$dataset_name" "$result_dir" "$summary_json" "$summary_csv" "$DATASET_POLICY" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path

dataset_name = sys.argv[1]
result_dir = Path(sys.argv[2])
summary_json = Path(sys.argv[3])
summary_csv = Path(sys.argv[4])
baseline_policy = sys.argv[5]

selection = json.loads((result_dir / "selection" / "best_alpha.json").read_text(encoding="utf-8"))
test_root = result_dir / "test_selected" / selection["best_alpha_tag"]
test_records = []
for json_path in sorted(test_root.glob("*.json")):
    metrics = json.loads(json_path.read_text(encoding="utf-8"))
    test_records.append(
        {
            "seed": json_path.stem,
            "eval_success_rate": float(metrics["eval_success_rate"]),
            "eval_return": float(metrics["eval_return"]),
            "eval_safety_cost": float(metrics["eval_safety_cost"]),
            "eval_time_s": float(metrics["eval_time_s"]),
            "eval_path_length_m": float(metrics["eval_path_length_m"]),
            "eval_progress_ratio": float(metrics["eval_progress_ratio"]),
            "eval_path_efficiency": float(metrics["eval_path_efficiency"]),
            "eval_termination_counts": metrics["eval_termination_counts"],
        }
    )

if not test_records:
    raise ValueError(
        f"No test metrics found under {test_root}. "
        "Run MODE=test first, or rerun MODE=summarize after the script updates "
        "that auto-populate missing selected-test evaluations."
    )

baseline_test_path = result_dir / "baselines" / f"{baseline_policy}_test_eval.json"
baseline_test = None
if baseline_test_path.exists():
    baseline_test = json.loads(baseline_test_path.read_text(encoding="utf-8"))

def mean_std(values):
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, std

success = [item["eval_success_rate"] for item in test_records]
returns = [item["eval_return"] for item in test_records]
safety = [item["eval_safety_cost"] for item in test_records]
time_s = [item["eval_time_s"] for item in test_records]
path = [item["eval_path_length_m"] for item in test_records]
progress = [item["eval_progress_ratio"] for item in test_records]
eff = [item["eval_path_efficiency"] for item in test_records]

mean_success, std_success = mean_std(success)
mean_return, std_return = mean_std(returns)
mean_safety, std_safety = mean_std(safety)
mean_time_s, std_time_s = mean_std(time_s)
mean_path, std_path = mean_std(path)
mean_progress, std_progress = mean_std(progress)
mean_eff, std_eff = mean_std(eff)

summary = {
    "dataset": dataset_name,
    "best_alpha_tag": selection["best_alpha_tag"],
    "best_alpha": selection["best_alpha"],
    "num_test_runs": len(test_records),
    "mean_test_success_rate": mean_success,
    "std_test_success_rate": std_success,
    "mean_test_return": mean_return,
    "std_test_return": std_return,
    "mean_test_safety_cost": mean_safety,
    "std_test_safety_cost": std_safety,
    "mean_test_time_s": mean_time_s,
    "std_test_time_s": std_time_s,
    "mean_test_path_length_m": mean_path,
    "std_test_path_length_m": std_path,
    "mean_test_progress_ratio": mean_progress,
    "std_test_progress_ratio": std_progress,
    "mean_test_path_efficiency": mean_eff,
    "std_test_path_efficiency": std_eff,
    "baseline_test_success_rate": None if baseline_test is None else baseline_test["eval_success_rate"],
    "baseline_test_return": None if baseline_test is None else baseline_test["eval_return"],
    "test_minus_baseline_success_rate": None if baseline_test is None else mean_success - float(baseline_test["eval_success_rate"]),
    "selected_runs": selection["selected_runs"],
    "test_runs": test_records,
    "validation_selection": selection,
}

summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
with summary_csv.open("w", encoding="utf-8", newline="") as fp:
    writer = csv.DictWriter(
        fp,
        fieldnames=[
            "dataset",
            "best_alpha",
            "num_test_runs",
            "mean_test_success_rate",
            "std_test_success_rate",
            "mean_test_return",
            "std_test_return",
            "mean_test_safety_cost",
            "mean_test_time_s",
            "mean_test_path_length_m",
            "mean_test_progress_ratio",
            "mean_test_path_efficiency",
            "baseline_test_success_rate",
            "baseline_test_return",
            "test_minus_baseline_success_rate",
        ],
    )
    writer.writeheader()
    writer.writerow(
        {
            "dataset": dataset_name,
            "best_alpha": selection["best_alpha"],
            "num_test_runs": len(test_records),
            "mean_test_success_rate": mean_success,
            "std_test_success_rate": std_success,
            "mean_test_return": mean_return,
            "std_test_return": std_return,
            "mean_test_safety_cost": mean_safety,
            "mean_test_time_s": mean_time_s,
            "mean_test_path_length_m": mean_path,
            "mean_test_progress_ratio": mean_progress,
            "mean_test_path_efficiency": mean_eff,
            "baseline_test_success_rate": None if baseline_test is None else baseline_test["eval_success_rate"],
            "baseline_test_return": None if baseline_test is None else baseline_test["eval_return"],
            "test_minus_baseline_success_rate": None if baseline_test is None else mean_success - float(baseline_test["eval_success_rate"]),
        }
    )

print(f"[write] summary: {summary_json}")
print(f"[write] summary: {summary_csv}")
PY
}

write_global_summary() {
  mkdir -p "$SUMMARY_ROOT"
  run_cmd "${PYTHON_CMD[@]}" - "$SUMMARY_ROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_root = Path(sys.argv[1])
rows = []
for path in sorted(summary_root.glob("*_summary.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows.append(
        {
            "dataset": payload["dataset"],
            "best_alpha": payload["best_alpha"],
            "num_test_runs": payload["num_test_runs"],
            "mean_test_success_rate": payload["mean_test_success_rate"],
            "std_test_success_rate": payload["std_test_success_rate"],
            "mean_test_return": payload["mean_test_return"],
            "mean_test_safety_cost": payload["mean_test_safety_cost"],
            "mean_test_time_s": payload["mean_test_time_s"],
            "baseline_test_success_rate": payload["baseline_test_success_rate"],
            "test_minus_baseline_success_rate": payload["test_minus_baseline_success_rate"],
        }
    )

if not rows:
    sys.exit(0)

output_csv = summary_root / "phase0b_v2_overview.csv"
with output_csv.open("w", encoding="utf-8", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"[write] overview: {output_csv}")
PY
}

run_mode_analyze() {
  local extra_flags=()
  if [[ "$RUN_BC_TEST_ANALYSIS" == "1" ]]; then
    ensure_manifest "$TEST_MANIFEST_EPISODES" "$TEST_MANIFEST_DIR" "$TEST_MANIFEST_PATH"
    extra_flags+=(--run-bc-test --test-manifest "$TEST_MANIFEST_PATH")
  fi
  if [[ "$FORCE_REEVAL" == "1" ]]; then
    extra_flags+=(--force-reeval)
  fi
  if [[ "$SKIP_ANALYSIS_PLOTS" == "1" ]]; then
    extra_flags+=(--skip-plots)
  fi
  run_cmd "${PYTHON_CMD[@]}" -m scripts.analyze_offline_td3bc_phase0b_v2 \
    --results-root "$RESULTS_ROOT" \
    --checkpoints-root "$CHECKPOINT_ROOT" \
    --output-dir "$ANALYSIS_OUTPUT_DIR" \
    --device "$DEVICE" \
    --num-workers "$EVAL_WORKERS" \
    --worker-device "$EVAL_WORKER_DEVICE" \
    --test-seed "$TEST_SEED" \
    "${extra_flags[@]}"
}

run_dataset_pipeline() {
  local dataset_name="$1"
  local dataset_episodes="$2"
  local alpha seed

  ensure_dataset "$dataset_name" "$dataset_episodes"
  run_baseline_eval "$dataset_name"

  for alpha in $SIZE_ABLATION_ALPHAS; do
    for seed in $SIZE_ABLATION_SEEDS; do
      train_one "$dataset_name" "$alpha" "$seed"
    done
  done

  for alpha in $SIZE_ABLATION_ALPHAS; do
    for seed in $SIZE_ABLATION_SEEDS; do
      validate_one "$dataset_name" "$alpha" "$seed"
      select_best_checkpoint "$dataset_name" "$alpha" "$seed"
    done
  done

  select_best_alpha "$dataset_name"
  run_test_for_dataset "$dataset_name"
  write_dataset_summary "$dataset_name"
}

run_size_ablation() {
  local episodes dataset_name
  ensure_manifest "$VAL_MANIFEST_EPISODES" "$VAL_MANIFEST_DIR" "$VAL_MANIFEST_PATH"
  ensure_manifest "$TEST_MANIFEST_EPISODES" "$TEST_MANIFEST_DIR" "$TEST_MANIFEST_PATH"

  for episodes in $SIZE_ABLATION_EPISODES; do
    dataset_name="$(dataset_name_for_episodes "$episodes")"
    run_dataset_pipeline "$dataset_name" "$episodes"
  done

  write_global_summary
}

run_mode_collect() {
  local episodes dataset_name
  for episodes in $SIZE_ABLATION_EPISODES; do
    dataset_name="$(dataset_name_for_episodes "$episodes")"
    ensure_dataset "$dataset_name" "$episodes"
  done
}

run_mode_baseline() {
  ensure_manifest "$VAL_MANIFEST_EPISODES" "$VAL_MANIFEST_DIR" "$VAL_MANIFEST_PATH"
  ensure_manifest "$TEST_MANIFEST_EPISODES" "$TEST_MANIFEST_DIR" "$TEST_MANIFEST_PATH"
  local episodes dataset_name
  for episodes in $SIZE_ABLATION_EPISODES; do
    dataset_name="$(dataset_name_for_episodes "$episodes")"
    run_baseline_eval "$dataset_name"
  done
}

run_mode_train() {
  local episodes dataset_name alpha seed
  for episodes in $SIZE_ABLATION_EPISODES; do
    dataset_name="$(dataset_name_for_episodes "$episodes")"
    ensure_dataset "$dataset_name" "$episodes"
    for alpha in $SIZE_ABLATION_ALPHAS; do
      for seed in $SIZE_ABLATION_SEEDS; do
        train_one "$dataset_name" "$alpha" "$seed"
      done
    done
  done
}

run_mode_validate() {
  ensure_manifest "$VAL_MANIFEST_EPISODES" "$VAL_MANIFEST_DIR" "$VAL_MANIFEST_PATH"
  local episodes dataset_name alpha seed
  for episodes in $SIZE_ABLATION_EPISODES; do
    dataset_name="$(dataset_name_for_episodes "$episodes")"
    for alpha in $SIZE_ABLATION_ALPHAS; do
      for seed in $SIZE_ABLATION_SEEDS; do
        validate_one "$dataset_name" "$alpha" "$seed"
        select_best_checkpoint "$dataset_name" "$alpha" "$seed"
      done
    done
  done
}

run_mode_select() {
  local episodes dataset_name
  for episodes in $SIZE_ABLATION_EPISODES; do
    dataset_name="$(dataset_name_for_episodes "$episodes")"
    select_best_alpha "$dataset_name"
  done
}

run_mode_test() {
  ensure_manifest "$TEST_MANIFEST_EPISODES" "$TEST_MANIFEST_DIR" "$TEST_MANIFEST_PATH"
  local episodes dataset_name
  for episodes in $SIZE_ABLATION_EPISODES; do
    dataset_name="$(dataset_name_for_episodes "$episodes")"
    run_test_for_dataset "$dataset_name"
  done
}

run_mode_summarize() {
  local episodes dataset_name
  ensure_manifest "$TEST_MANIFEST_EPISODES" "$TEST_MANIFEST_DIR" "$TEST_MANIFEST_PATH"
  for episodes in $SIZE_ABLATION_EPISODES; do
    dataset_name="$(dataset_name_for_episodes "$episodes")"
    run_baseline_eval "$dataset_name"
    select_best_alpha "$dataset_name"
    run_test_for_dataset "$dataset_name"
    write_dataset_summary "$dataset_name"
  done
  write_global_summary
}

case "$MODE" in
  all|size_ablation)
    run_size_ablation
    ;;
  manifests)
    ensure_manifest "$VAL_MANIFEST_EPISODES" "$VAL_MANIFEST_DIR" "$VAL_MANIFEST_PATH"
    ensure_manifest "$TEST_MANIFEST_EPISODES" "$TEST_MANIFEST_DIR" "$TEST_MANIFEST_PATH"
    ;;
  collect)
    run_mode_collect
    ;;
  baseline)
    run_mode_baseline
    ;;
  train)
    run_mode_train
    ;;
  validate)
    run_mode_validate
    ;;
  select)
    run_mode_select
    ;;
  test)
    run_mode_test
    ;;
  summarize)
    run_mode_summarize
    ;;
  analyze)
    run_mode_analyze
    ;;
  *)
    echo "Unsupported MODE: ${MODE}"
    echo "Supported MODE values: all, size_ablation, manifests, collect, baseline, train, validate, select, test, summarize, analyze"
    exit 1
    ;;
esac
