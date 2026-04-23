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
DATASET_EPISODES="${DATASET_EPISODES:-1000 2000}"
DATASET_SEED="${DATASET_SEED:-0}"
COLLECT_WORKERS="${COLLECT_WORKERS:-8}"

SEEDS="${SEEDS:-42 43}"
MANIFEST_EPISODES="${MANIFEST_EPISODES:-40}"
MANIFEST_ROOT="${MANIFEST_ROOT:-benchmarks/offline_rebrac_screen}"
MANIFEST_PATH="${MANIFEST_PATH:-${MANIFEST_ROOT}/${BENCHMARK_KEY}.json}"

SAMPLING_MODE="${SAMPLING_MODE:-shuffle_no_replacement}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-64}"
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

ACTOR_PENALTY_COEFS="${ACTOR_PENALTY_COEFS:-1.0 2.0}"
CRITIC_PENALTY_COEFS="${CRITIC_PENALTY_COEFS:-1.0 2.0}"

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

ensure_manifest() {
  if [[ -f "$MANIFEST_PATH" ]]; then
    echo "[skip] manifest exists: $MANIFEST_PATH"
    return
  fi
  mkdir -p "$MANIFEST_ROOT"
  run_cmd "${PYTHON_CMD[@]}" -m scripts.generate_standard_benchmarks \
    --benchmarks "$BENCHMARK_KEY" \
    --episodes "$MANIFEST_EPISODES" \
    --output-dir "$MANIFEST_ROOT"
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

pair_tag() {
  local actor_beta="$1"
  local critic_beta="$2"
  local actor_tag critic_tag
  actor_tag="${actor_beta//./p}"
  critic_tag="${critic_beta//./p}"
  echo "actorb_${actor_tag}__criticb_${critic_tag}"
}

train_one() {
  local dataset_name="$1"
  local actor_beta="$2"
  local critic_beta="$3"
  local seed="$4"
  local pair run_dir dataset_dir

  pair="$(pair_tag "$actor_beta" "$critic_beta")"
  run_dir="${CHECKPOINT_ROOT}/${dataset_name}/${pair}/seed_${seed}"
  dataset_dir="$(dataset_dir_for_name "$dataset_name")"
  if [[ -f "${run_dir}/trainer_state.json" && -f "${run_dir}/agent_final.pt" ]]; then
    echo "[skip] trained run exists: ${run_dir}"
    return
  fi

  mkdir -p "$run_dir"
  run_cmd "${PYTHON_CMD[@]}" -m scripts.train_offline \
    --algo rebrac \
    --offline-data "${dataset_dir}/transitions.npz" \
    --flow "$FLOW_PATH" \
    --manifest "$MANIFEST_PATH" \
    --probe-layout "$PROBE_LAYOUT" \
    --history-length "$HISTORY_LENGTH" \
    --task-geometry "$TASK_GEOMETRY" \
    --target-speed "$TARGET_SPEED" \
    --objective "$OBJECTIVE" \
    --sampling-mode "$SAMPLING_MODE" \
    --num-epochs "$TRAIN_EPOCHS" \
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
    --checkpoint-every 0 \
    --log-every "$LOG_EVERY" \
    --critic-layernorm \
    --no-actor-layernorm \
    --save-dir "$run_dir" \
    --seed "$seed" \
    --device "$DEVICE"
}

eval_one() {
  local dataset_name="$1"
  local actor_beta="$2"
  local critic_beta="$3"
  local seed="$4"
  local pair run_dir result_dir agent_file

  pair="$(pair_tag "$actor_beta" "$critic_beta")"
  run_dir="${CHECKPOINT_ROOT}/${dataset_name}/${pair}/seed_${seed}"
  result_dir="${RESULTS_ROOT}/${dataset_name}/${pair}"
  mkdir -p "$result_dir"
  if [[ -f "${result_dir}/seed_${seed}.json" ]]; then
    echo "[skip] eval exists: ${result_dir}/seed_${seed}.json"
    return
  fi

  agent_file="agent_best.pt"
  if [[ ! -f "${run_dir}/${agent_file}" ]]; then
    agent_file="agent_final.pt"
  fi

  run_cmd "${PYTHON_CMD[@]}" -m scripts.evaluate_offline \
    --checkpoint "$run_dir" \
    --agent-file "$agent_file" \
    --manifest "$MANIFEST_PATH" \
    --device "$DEVICE" \
    --seed 123 \
    --output-json "${result_dir}/seed_${seed}.json"
}

write_summary() {
  mkdir -p "$SUMMARY_ROOT"
  run_cmd "${PYTHON_CMD[@]}" - "$RESULTS_ROOT" "$SUMMARY_ROOT" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path

results_root = Path(sys.argv[1])
summary_root = Path(sys.argv[2])
rows = []
for dataset_dir in sorted(path for path in results_root.iterdir() if path.is_dir()):
    for pair_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        eval_paths = sorted(pair_dir.glob("seed_*.json"))
        if not eval_paths:
            continue
        payloads = [json.loads(path.read_text(encoding="utf-8")) for path in eval_paths]
        success = [float(item["eval_success_rate"]) for item in payloads]
        returns = [float(item["eval_return"]) for item in payloads]
        rows.append(
            {
                "dataset": dataset_dir.name,
                "pair": pair_dir.name,
                "num_seeds": len(payloads),
                "mean_success_rate": statistics.fmean(success),
                "std_success_rate": statistics.pstdev(success) if len(success) > 1 else 0.0,
                "mean_return": statistics.fmean(returns),
            }
        )

if not rows:
    sys.exit(0)

rows.sort(
    key=lambda item: (
        item["dataset"],
        -item["mean_success_rate"],
        -item["mean_return"],
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

run_train() {
  local episodes dataset_name actor_beta critic_beta seed
  ensure_manifest
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

run_eval() {
  local episodes dataset_name actor_beta critic_beta seed
  ensure_manifest
  for episodes in $DATASET_EPISODES; do
    dataset_name="$(dataset_name_for_episodes "$episodes")"
    for actor_beta in $ACTOR_PENALTY_COEFS; do
      for critic_beta in $CRITIC_PENALTY_COEFS; do
        for seed in $SEEDS; do
          eval_one "$dataset_name" "$actor_beta" "$critic_beta" "$seed"
        done
      done
    done
  done
}

case "$MODE" in
  all)
    run_train
    run_eval
    write_summary
    ;;
  manifests)
    ensure_manifest
    ;;
  collect)
    episodes=""
    for episodes in $DATASET_EPISODES; do
      dataset_name="$(dataset_name_for_episodes "$episodes")"
      ensure_dataset "$dataset_name" "$episodes"
    done
    ;;
  train)
    run_train
    ;;
  eval)
    run_eval
    ;;
  summarize)
    write_summary
    ;;
  *)
    echo "Unsupported MODE: ${MODE}" >&2
    echo "Supported MODE values: all, manifests, collect, train, eval, summarize" >&2
    exit 1
    ;;
esac
