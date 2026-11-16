#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
PROJECT=$(dirname -- "$REPO")
PYTHON="$REPO/.venv/bin/python"
RUN_ROOT="${RUN_ROOT:-$PROJECT/runs/cup-stacking-baseline}"
SOURCE_DATASET="${SOURCE_DATASET:-$PROJECT/datasets/cup_stacking}"
TASK_FILE="$REPO/examples/realman/task_instruction.txt"
SKIP_FIRST_EPISODES="${SKIP_FIRST_EPISODES:-5}"
if [[ ! "$SKIP_FIRST_EPISODES" =~ ^[0-9]+$ ]]; then
  echo 'SKIP_FIRST_EPISODES must be a non-negative integer.' >&2
  exit 2
fi
# A changed instruction gets its own metadata view; source data/videos are shared.
TASK_ID=$(sha256sum "$TASK_FILE" | cut -c1-12)
DATASET="$RUN_ROOT/data-$TASK_ID-skip$SKIP_FIRST_EPISODES"
MODE="${1:-smoke}"
cd "$REPO"

case "$MODE" in
  resume)
    exec bash "$PROJECT/runs/cup-stacking-baseline/resume_training.sh" "${@:2}"
    ;;
  prepare)
    exec "$PYTHON" examples/realman/prepare_baseline.py --source "$SOURCE_DATASET" --output "$DATASET" --skip-first "$SKIP_FIRST_EPISODES"
    ;;
  smoke|train) ;;
  *) echo 'Usage: bash examples/realman/train.sh prepare|smoke|train|resume [--dry-run]' >&2; exit 2 ;;
esac

if [[ "$MODE" == smoke ]]; then
  GPUS=1
else
  GPUS="${NUM_GPUS:-4}"
fi
if [[ ! "$GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo 'NUM_GPUS must be a positive integer.' >&2
  exit 2
fi
# Restrict CUDA visibility before torchrun or any torch import.
if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  CUDA_VISIBLE_DEVICES=$("$PYTHON" examples/realman/select_gpu.py --count "$GPUS")
  export CUDA_VISIBLE_DEVICES
fi
"$PYTHON" -c 'import sys, torch; expected = int(sys.argv[1]); actual = torch.cuda.device_count(); assert actual == expected, f"Expected {expected} visible GPUs, found {actual}. Check CUDA_VISIBLE_DEVICES."' "$GPUS"

# Validate collection semantics before loading models.
"$PYTHON" examples/realman/realman_config.py
"$PYTHON" examples/realman/prepare_baseline.py --source "$SOURCE_DATASET" --output "$DATASET" --skip-first "$SKIP_FIRST_EPISODES"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/data/yuzheng/cache/huggingface
export TORCH_HOME=/data/yuzheng/cache/torch
export TRITON_CACHE_DIR=/data/yuzheng/cache/triton
export PYTHONDONTWRITEBYTECODE=1
export NO_ALBUMENTATIONS_UPDATE=1
export MPLCONFIGDIR=/data/yuzheng/cache/matplotlib
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR"

if [[ "$MODE" == smoke ]]; then
  STEPS=5
  SAVE_STEPS=5
  BATCH=1
  ACCUMULATION=1
else
  STEPS="${MAX_STEPS:-10000}"
  SAVE_STEPS="${SAVE_EVERY:-1000}"
  BATCH="${BATCH_SIZE:-$GPUS}"
  ACCUMULATION="${GRAD_ACCUM:-1}"
fi
if [[ ! "$BATCH" =~ ^[1-9][0-9]*$ || ! "$ACCUMULATION" =~ ^[1-9][0-9]*$ ]] || (( BATCH % GPUS != 0 )); then
  echo 'BATCH_SIZE must be positive and divisible by NUM_GPUS; GRAD_ACCUM must be positive.' >&2
  exit 2
fi
OUTPUT="$RUN_ROOT/${MODE}-$(date +%Y%m%d-%H%M%S)"
if [[ -e "$OUTPUT" ]]; then
  echo "Output already exists: $OUTPUT" >&2
  exit 1
fi
mkdir -p "$OUTPUT"
cp examples/realman/action_semantics.json "$OUTPUT/action_semantics.json"
cp examples/realman/realman_config.py "$OUTPUT/realman_config.py"
cp "$TASK_FILE" "$OUTPUT/task_instruction.txt"
printf 'Mode: %s\nDataset: %s\nOutput: %s\n' "$MODE" "$DATASET" "$OUTPUT"
printf 'GPUs: %s; per-GPU batch: %s; accumulation: %s; effective batch: %s\n' "$GPUS" "$((BATCH / GPUS))" "$ACCUMULATION" "$((BATCH * ACCUMULATION))"
printf '%s\n' "$CUDA_VISIBLE_DEVICES" > "$OUTPUT/gpu_uuids.txt"
LAUNCH=("$PYTHON")
if (( GPUS > 1 )); then
  LAUNCH+=(-m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$GPUS" --max_restarts=0)
fi
# The official launcher defaults to frozen vision/language encoders and no eval
# dataset. Keep the original loss, action head and checkpoint padding behavior.
"${LAUNCH[@]}" gr00t/experiment/launch_finetune.py \
  --base-model-path "$PROJECT/models/GR00T-N1.7-3B" \
  --backbone-model-path "$PROJECT/models/Cosmos-Reason2-2B" \
  --transformers-local-files-only \
  --dataset-path "$DATASET" \
  --modality-config-path examples/realman/realman_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --num-gpus "$GPUS" \
  --output-dir "$OUTPUT" \
  --max-steps "$STEPS" \
  --save-steps "$SAVE_STEPS" \
  --save-total-limit 3 \
  --global-batch-size "$BATCH" \
  --gradient-accumulation-steps "$ACCUMULATION" \
  --learning-rate 0.0001 \
  --dataloader-num-workers 0 \
  --data-loader stock \
  --shard-size 128 \
  --episode-sampling-rate 0.1 \
  2>&1 | tee "$OUTPUT/train.log"
