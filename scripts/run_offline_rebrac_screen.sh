#!/usr/bin/env bash
set -euo pipefail

# ReBRAC Stage B screening driver.
#
# Protocol is intentionally aligned with scripts/run_offline_td3bc_phase0b_v2.sh
# (which phase0c Stage B reuses):
#
#   1. Build fixed val + test manifests up-front.
#   2. Train per (dataset, actor_beta, critic_beta, seed) with epoch-aligned
#      checkpoint cadence and in-training evaluation disabled.
#   3. Evaluate every saved checkpoint on the val manifest; select the best
#      checkpoint by success_rate -> return -> -safety_cost -> -time.
#   4. Re-evaluate each selected checkpoint on the held-out test manifest.
#   5. Aggregate per (dataset, pair) across seeds and emit a summary that
#      also reports late-training critic_penalty / target_q diagnostics.
#
# Implementation note:
#   The underlying agent (see auv_nav/rebrac.py) is a Q-normalized ReBRAC
#   variant: the actor loss divides the deterministic policy gradient by
#   |Q| detached (matching TD3+BC's Q-scaling), rather than using the
#   canonical ReBRAC paper's unnormalized form. All results under this
#   script should be interpreted under that variant.

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
DATASET_EPISODES="${DATASET_EPISODES:-1000 2000}"
DATASET_SEED="${DATASET_SEED:-0}"
COLLECT_WORKERS="${COLLECT_WORKERS:-8}"

# Three seeds are the minimum that still surface seed variance; screening runs
# at 2 seeds routinely flipped winners under the larger phase0c Stage A val.
SEEDS="${SEEDS:-42 43 44}"

# The grid covers three BC-strength regimes on the TD3BC-equivalent scale
# (action_dim=2, so beta1 in ReBRAC equals ~1/(2*alpha_TD3BC)):
#   beta1=1.0 ~ TD3BC alpha=0.5  (phase0c 500-episode winner)
#   beta1=2.0 ~ TD3BC alpha=0.25 (phase0c 1000-episode winner)
#   beta1=4.0 ~ TD3BC alpha=0.125 (covers the 2000-episode "needs weaker BC" regime)
ACTOR_PENALTY_COEFS="${ACTOR_PENALTY_COEFS:-1.0 2.0 4.0}"
CRITIC_PENALTY_COEFS="${CRITIC_PENALTY_COEFS:-1.0 2.0}"

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

# Asymmetric critic (privileged-critic) support. Off by default so existing
# Stage B/C behavior is unchanged. When USE_ASYMMETRIC_CRITIC=1, the train
# call adds --use-asymmetric-critic and --privileged-actor-update-mode flags.
# Used by Stage D (worldcomp teacher-gap) to share screen.sh as the worker.
USE_ASYMMETRIC_CRITIC="${USE_ASYMMETRIC_CRITIC:-0}"
PRIVILEGED_ACTOR_UPDATE_MODE="${PRIVILEGED_ACTOR_UPDATE_MODE:-zeros}"

# Fraction of the training timeline used to summarize late-training
# critic_penalty / target_q diagnostics. Default is the last 25% of logged
# steps so that early instability does not contaminate the statistic.
TRAIN_METRICS_WINDOW_FRACTION="${TRAIN_METRICS_WINDOW_FRACTION:-0.25}"

VAL_MANIFEST_EPISODES="${VAL_MANIFEST_EPISODES:-40}"
TEST_MANIFEST_EPISODES="${TEST_MANIFEST_EPISODES:-40}"
MANIFEST_ROOT="${MANIFEST_ROOT:-benchmarks/offline_rebrac_screen}"
VAL_MANIFEST_DIR="${VAL_MANIFEST_DIR:-${MANIFEST_ROOT}/val_${VAL_MANIFEST_EPISODES}}"
TEST_MANIFEST_DIR="${TEST_MANIFEST_DIR:-${MANIFEST_ROOT}/test_${TEST_MANIFEST_EPISODES}}"
VAL_MANIFEST_PATH="${VAL_MANIFEST_PATH:-${VAL_MANIFEST_DIR}/${BENCHMARK_KEY}.json}"
TEST_MANIFEST_PATH="${TEST_MANIFEST_PATH:-${TEST_MANIFEST_DIR}/${BENCHMARK_KEY}.json}"

EVAL_WORKERS="${EVAL_WORKERS:-6}"
EVAL_WORKER_DEVICE="${EVAL_WORKER_DEVICE:-cpu}"
VALIDATION_SEED="${VALIDATION_SEED:-123}"
TEST_SEED="${TEST_SEED:-456}"
FORCE_REEVAL="${FORCE_REEVAL:-0}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/offline/rebrac/screening}"
RESULTS_ROOT="${RESULTS_ROOT:-results/offline/rebrac/screening}"
SUMMARY_ROOT="${SUMMARY_ROOT:-${RESULTS_ROOT}/summaries}"

run_cmd() {
  echo
  echo "[cmd] $*"
  "$@"
}

dataset_name_for_episodes() {
  local episodes="$1"
  echo "${DATASET_POLICY}_${PROBE_LAYOUT}_h${HISTORY_LENGTH}_${OBJECTIVE}_re150_u10cross_fixdone_ep${episodes}"
}

dataset_dir_for_name() {
  local dataset_name="$1"
  echo "offline_data/${dataset_name}"
}

pair_tag() {
  local actor_beta="$1"
  local critic_beta="$2"
  local actor_tag critic_tag
  actor_tag="${actor_beta//./p}"
  critic_tag="${critic_beta//./p}"
  echo "actorb_${actor_tag}__criticb_${critic_tag}"
}

run_dir_for() {
  local dataset_name="$1" actor_beta="$2" critic_beta="$3" seed="$4"
  local pair
  pair="$(pair_tag "$actor_beta" "$critic_beta")"
  echo "${CHECKPOINT_ROOT}/${dataset_name}/${pair}/seed_${seed}"
}

result_dir_for() {
  local dataset_name="$1" actor_beta="$2" critic_beta="$3"
  local pair
  pair="$(pair_tag "$actor_beta" "$critic_beta")"
  echo "${RESULTS_ROOT}/${dataset_name}/${pair}"
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
  local episodes="$2"
  local dataset_dir
  dataset_dir="$(dataset_dir_for_name "$dataset_name")"
  if [[ -f "${dataset_dir}/transitions.npz" ]]; then
    echo "[skip] dataset exists: ${dataset_dir}/transitions.npz"
    return
  fi
  mkdir -p "$dataset_dir"
  run_cmd "${PYTHON_CMD[@]}" -m scripts.collect_offline_data \
    --policy "$DATASET_POLICY" \
    --flow "$FLOW_PATH" \
    --probe-layout "$PROBE_LAYOUT" \
    --history-length "$HISTORY_LENGTH" \
    --task-geometry "$TASK_GEOMETRY" \
    --target-speed "$TARGET_SPEED" \
    --objective "$OBJECTIVE" \
    --episodes "$episodes" \
    --seed "$DATASET_SEED" \
    --num-workers "$COLLECT_WORKERS" \
    --output-dir "$dataset_dir"
}

compute_schedule() {
  # Populates globals SCHEDULE_TOTAL_STEPS, SCHEDULE_CHECKPOINT_EVERY_STEPS
  # from the dataset's num_transitions plus driver-level BATCH_SIZE /
  # TRAIN_EPOCHS / CHECKPOINT_EVERY_EPOCHS / DROP_LAST_BATCH knobs.
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

# Single-entry memo for compute_schedule. The outer driver loop iterates
# datasets sequentially and fully completes one dataset's grid+seeds before
# moving on, so a single cached entry keyed on dataset_name avoids the
# redundant Python subprocess on every (actor_beta, critic_beta, seed) cell.
# Bash 3.2 on macOS lacks associative arrays, hence the flat scalars here.
CACHED_SCHEDULE_DATASET=""
CACHED_SCHEDULE_TOTAL_STEPS=""
CACHED_SCHEDULE_CHECKPOINT_EVERY_STEPS=""

cache_schedule() {
  local dataset_name="$1" dataset_dir
  if [[ "$CACHED_SCHEDULE_DATASET" == "$dataset_name" && -n "$CACHED_SCHEDULE_TOTAL_STEPS" ]]; then
    return
  fi
  dataset_dir="$(dataset_dir_for_name "$dataset_name")"
  compute_schedule "$dataset_dir"
  CACHED_SCHEDULE_DATASET="$dataset_name"
  CACHED_SCHEDULE_TOTAL_STEPS="$SCHEDULE_TOTAL_STEPS"
  CACHED_SCHEDULE_CHECKPOINT_EVERY_STEPS="$SCHEDULE_CHECKPOINT_EVERY_STEPS"
}

train_one() {
  local dataset_name="$1" actor_beta="$2" critic_beta="$3" seed="$4"
  local run_dir dataset_dir total_steps checkpoint_every extra_flags=()

  run_dir="$(run_dir_for "$dataset_name" "$actor_beta" "$critic_beta" "$seed")"
  dataset_dir="$(dataset_dir_for_name "$dataset_name")"
  if [[ -f "${run_dir}/trainer_state.json" && -f "${run_dir}/agent_final.pt" ]]; then
    echo "[skip] trained run exists: ${run_dir}"
    return
  fi

  cache_schedule "$dataset_name"
  total_steps="$CACHED_SCHEDULE_TOTAL_STEPS"
  checkpoint_every="$CACHED_SCHEDULE_CHECKPOINT_EVERY_STEPS"
  mkdir -p "$run_dir"
  if [[ "$DROP_LAST_BATCH" == "1" ]]; then
    extra_flags+=(--drop-last-batch)
  fi
  if [[ "$USE_ASYMMETRIC_CRITIC" == "1" ]]; then
    extra_flags+=(--use-asymmetric-critic)
    extra_flags+=(--privileged-actor-update-mode "$PRIVILEGED_ACTOR_UPDATE_MODE")
  fi

  run_cmd "${PYTHON_CMD[@]}" -m scripts.train_offline \
    --algo rebrac \
    --offline-data "${dataset_dir}/transitions.npz" \
    --flow "$FLOW_PATH" \
    --manifest "$VAL_MANIFEST_PATH" \
    --probe-layout "$PROBE_LAYOUT" \
    --history-length "$HISTORY_LENGTH" \
    --task-geometry "$TASK_GEOMETRY" \
    --target-speed "$TARGET_SPEED" \
    --objective "$OBJECTIVE" \
    --sampling-mode "$SAMPLING_MODE" \
    --num-epochs "$TRAIN_EPOCHS" \
    --total-steps "$total_steps" \
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
    --checkpoint-every "$checkpoint_every" \
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
  local dataset_name="$1" actor_beta="$2" critic_beta="$3" seed="$4"
  local run_dir result_dir val_dir

  run_dir="$(run_dir_for "$dataset_name" "$actor_beta" "$critic_beta" "$seed")"
  result_dir="$(result_dir_for "$dataset_name" "$actor_beta" "$critic_beta")"
  val_dir="${result_dir}/validation/seed_${seed}"

  if [[ ! -d "$run_dir" ]]; then
    echo "[skip] missing run dir: ${run_dir}"
    return
  fi

  # Mirror the completion guard used by train_one — a run dir without
  # trainer_state.json or agent_final.pt means training crashed mid-epoch.
  # Warn loudly instead of silently validating partial checkpoints.
  if [[ ! -f "${run_dir}/trainer_state.json" || ! -f "${run_dir}/agent_final.pt" ]]; then
    echo "[warn] skipping ${run_dir}: missing trainer_state.json or agent_final.pt" >&2
    return
  fi

  mkdir -p "$val_dir"
  local checkpoint_files=()
  local -a _step_files=()
  shopt -s nullglob
  _step_files=("$run_dir"/agent_step_*.pt)
  shopt -u nullglob
  if [[ "${#_step_files[@]}" -gt 0 ]]; then
    local _path
    while IFS= read -r _path; do
      checkpoint_files+=("$(basename "$_path")")
    done < <(printf '%s\n' "${_step_files[@]}" | sort)
  fi
  checkpoint_files+=("agent_final.pt")

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
  local dataset_name="$1" actor_beta="$2" critic_beta="$3" seed="$4"
  local run_dir result_dir val_dir selection_dir output_json trainer_state_path

  run_dir="$(run_dir_for "$dataset_name" "$actor_beta" "$critic_beta" "$seed")"
  result_dir="$(result_dir_for "$dataset_name" "$actor_beta" "$critic_beta")"
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
import json
import sys
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
agent_file_sidecar.parent.mkdir(parents=True, exist_ok=True)
agent_file_sidecar.write_text(best["agent_file"] + "\n", encoding="utf-8")
print(f"[write] selection: {output_path}")
print(
    f"[best] agent={best['agent_file']} "
    f"success={best['eval_success_rate']:.4f} return={best['eval_return']:.4f}"
)
PY
}

test_one() {
  local dataset_name="$1" actor_beta="$2" critic_beta="$3" seed="$4"
  local run_dir result_dir selection_dir selection_path agent_file_sidecar test_dir output_json agent_file

  run_dir="$(run_dir_for "$dataset_name" "$actor_beta" "$critic_beta" "$seed")"
  result_dir="$(result_dir_for "$dataset_name" "$actor_beta" "$critic_beta")"
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
  # Prefer the plain-text sidecar; fall back to parsing the JSON for
  # selections written by older runs that pre-date the sidecar.
  if [[ -f "$agent_file_sidecar" ]]; then
    agent_file="$(< "$agent_file_sidecar")"
    agent_file="${agent_file//$'\n'/}"
  else
    agent_file="$(
      "${PYTHON_CMD[@]}" - "$selection_path" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload["best"]["agent_file"])
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

write_summary() {
  mkdir -p "$SUMMARY_ROOT"
  run_cmd "${PYTHON_CMD[@]}" - "$RESULTS_ROOT" "$CHECKPOINT_ROOT" "$SUMMARY_ROOT" "$TRAIN_METRICS_WINDOW_FRACTION" <<'PY'
import csv
import json
import math
import statistics
import sys
from pathlib import Path

results_root = Path(sys.argv[1])
checkpoint_root = Path(sys.argv[2])
summary_root = Path(sys.argv[3])
window_fraction = max(0.01, min(1.0, float(sys.argv[4])))


def _safe_mean(values):
    cleaned = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.fmean(cleaned) if cleaned else None


def _safe_std(values):
    cleaned = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.pstdev(cleaned) if len(cleaned) > 1 else 0.0


def _train_metrics(seed_run_dir: Path) -> dict[str, float]:
    """Compute late-training mean critic_penalty / target_q from train_log.jsonl.

    The window is the last ``window_fraction`` of logged entries (by train_step).
    """
    log_path = seed_run_dir / "train_log.jsonl"
    if not log_path.exists():
        return {
            "mean_critic_penalty": None,
            "mean_target_q": None,
            "mean_critic_penalty_ratio": None,
        }
    rows = []
    with log_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return {
            "mean_critic_penalty": None,
            "mean_target_q": None,
            "mean_critic_penalty_ratio": None,
        }
    rows.sort(key=lambda item: int(item.get("train_step", 0)))
    window = max(1, int(round(len(rows) * window_fraction)))
    tail = rows[-window:]
    critic_penalties = [item.get("critic_penalty") for item in tail]
    target_qs = [item.get("target_q") for item in tail]
    ratios = [
        abs(float(cp)) / max(abs(float(tq)), 1e-6)
        for cp, tq in zip(critic_penalties, target_qs)
        if cp is not None and tq is not None and math.isfinite(float(cp)) and math.isfinite(float(tq))
    ]
    return {
        "mean_critic_penalty": _safe_mean(critic_penalties),
        "mean_target_q": _safe_mean(target_qs),
        "mean_critic_penalty_ratio": _safe_mean(ratios) if ratios else None,
    }


rows = []
for dataset_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
    if dataset_dir.name == "summaries":
        continue
    for pair_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        test_dir = pair_dir / "test"
        if not test_dir.exists():
            continue
        test_paths = sorted(test_dir.glob("seed_*.json"))
        if not test_paths:
            continue
        test_payloads = [
            (path.stem.replace("seed_", ""), json.loads(path.read_text(encoding="utf-8")))
            for path in test_paths
        ]
        success = [float(p["eval_success_rate"]) for _, p in test_payloads]
        returns = [float(p["eval_return"]) for _, p in test_payloads]
        safety = [float(p["eval_safety_cost"]) for _, p in test_payloads]
        time_s = [float(p["eval_time_s"]) for _, p in test_payloads]

        train_stats = []
        for seed, _ in test_payloads:
            seed_run_dir = checkpoint_root / dataset_dir.name / pair_dir.name / f"seed_{seed}"
            train_stats.append(_train_metrics(seed_run_dir))

        rows.append(
            {
                "dataset": dataset_dir.name,
                "pair": pair_dir.name,
                "num_seeds": len(test_payloads),
                "mean_test_success_rate": _safe_mean(success),
                "std_test_success_rate": _safe_std(success),
                "mean_test_return": _safe_mean(returns),
                "std_test_return": _safe_std(returns),
                "mean_test_safety_cost": _safe_mean(safety),
                "mean_test_time_s": _safe_mean(time_s),
                "mean_critic_penalty": _safe_mean(
                    [s["mean_critic_penalty"] for s in train_stats]
                ),
                "mean_target_q": _safe_mean(
                    [s["mean_target_q"] for s in train_stats]
                ),
                "mean_critic_penalty_ratio": _safe_mean(
                    [s["mean_critic_penalty_ratio"] for s in train_stats]
                ),
            }
        )

if not rows:
    sys.exit(0)

rows.sort(
    key=lambda item: (
        item["dataset"],
        -(item["mean_test_success_rate"] or -math.inf),
        -(item["mean_test_return"] or -math.inf),
    )
)

output_csv = summary_root / "overview.csv"
with output_csv.open("w", encoding="utf-8", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

output_json = summary_root / "overview.json"
output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"[write] summary: {output_csv}")
print(f"[write] summary: {output_json}")
PY
}

ensure_manifests() {
  ensure_manifest "$VAL_MANIFEST_EPISODES" "$VAL_MANIFEST_DIR" "$VAL_MANIFEST_PATH"
  ensure_manifest "$TEST_MANIFEST_EPISODES" "$TEST_MANIFEST_DIR" "$TEST_MANIFEST_PATH"
}

run_collect() {
  local episodes dataset_name
  for episodes in $DATASET_EPISODES; do
    dataset_name="$(dataset_name_for_episodes "$episodes")"
    ensure_dataset "$dataset_name" "$episodes"
  done
}

run_train() {
  local episodes dataset_name actor_beta critic_beta seed
  ensure_manifests
  for episodes in $DATASET_EPISODES; do
    dataset_name="$(dataset_name_for_episodes "$episodes")"
    ensure_dataset "$dataset_name" "$episodes"
    for actor_beta in $ACTOR_PENALTY_COEFS; do
      for critic_beta in $CRITIC_PENALTY_COEFS; do
        for seed in $SEEDS; do
          train_one "$dataset_name" "$actor_beta" "$critic_beta" "$seed"
        done
      done
    done
  done
}

run_validate() {
  local episodes dataset_name actor_beta critic_beta seed
  ensure_manifest "$VAL_MANIFEST_EPISODES" "$VAL_MANIFEST_DIR" "$VAL_MANIFEST_PATH"
  for episodes in $DATASET_EPISODES; do
    dataset_name="$(dataset_name_for_episodes "$episodes")"
    for actor_beta in $ACTOR_PENALTY_COEFS; do
      for critic_beta in $CRITIC_PENALTY_COEFS; do
        for seed in $SEEDS; do
          validate_one "$dataset_name" "$actor_beta" "$critic_beta" "$seed"
          select_best_checkpoint "$dataset_name" "$actor_beta" "$critic_beta" "$seed"
        done
      done
    done
  done
}

run_test() {
  local episodes dataset_name actor_beta critic_beta seed
  ensure_manifest "$TEST_MANIFEST_EPISODES" "$TEST_MANIFEST_DIR" "$TEST_MANIFEST_PATH"
  for episodes in $DATASET_EPISODES; do
    dataset_name="$(dataset_name_for_episodes "$episodes")"
    for actor_beta in $ACTOR_PENALTY_COEFS; do
      for critic_beta in $CRITIC_PENALTY_COEFS; do
        for seed in $SEEDS; do
          test_one "$dataset_name" "$actor_beta" "$critic_beta" "$seed"
        done
      done
    done
  done
}

case "$MODE" in
  all)
    run_train
    run_validate
    run_test
    write_summary
    ;;
  manifests)
    ensure_manifests
    ;;
  collect)
    run_collect
    ;;
  train)
    run_train
    ;;
  validate)
    run_validate
    ;;
  test)
    run_test
    ;;
  summarize)
    write_summary
    ;;
  *)
    echo "Unsupported MODE: ${MODE}" >&2
    echo "Supported MODE values: all, manifests, collect, train, validate, test, summarize" >&2
    exit 1
    ;;
esac
