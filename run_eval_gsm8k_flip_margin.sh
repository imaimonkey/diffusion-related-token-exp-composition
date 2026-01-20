#!/bin/bash
#SBATCH --job-name=eval_gsm8k_flip
#SBATCH --output=logs/eval_gsm8k_flip_%j.out
#SBATCH --error=logs/eval_gsm8k_flip_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --nodelist=ubuntu

set -euo pipefail

# Stream stderr into stdout so `logs/eval_gsm8k_flip_*.out` updates live with tqdm + accuracy logs.
exec 2>&1

mkdir -p logs results

source .venv/bin/activate

MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
METHOD="${METHOD:-flip_margin}"
DATASET="${DATASET:-openai/gsm8k}"
DATASET_CONFIG="${DATASET_CONFIG:-main}"
LIMIT="${LIMIT:-100}"
SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-2}"

STEPS="${STEPS:-256}"
GEN_LENGTH="${GEN_LENGTH:-256}"
BLOCK_LENGTH="${BLOCK_LENGTH:-128}"
TEMPERATURE="${TEMPERATURE:-0.0}"
CFG_SCALE="${CFG_SCALE:-0.0}"
REMASKING="${REMASKING:-low_confidence}"
MASK_ID="${MASK_ID:-126336}"
LOG_EVERY="${LOG_EVERY:-20}"

FLIP_UNMASK_THRESHOLD="${FLIP_UNMASK_THRESHOLD:-0.6}"
FLIP_MARGIN_THRESHOLD="${FLIP_MARGIN_THRESHOLD:-0.1}"
FLIP_THRESHOLD="${FLIP_THRESHOLD:-2}"

OUTPUT="${OUTPUT:-}"
FAST="${FAST:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "=========================================="
echo "Eval GSM8K (flip_margin)"
echo "Model:               $MODEL_PATH"
echo "Method:              $METHOD"
echo "Dataset:             $DATASET"
echo "Dataset config:      $DATASET_CONFIG"
echo "Split:               $SPLIT"
echo "Limit:               $LIMIT"
echo "Batch size:          $BATCH_SIZE"
echo "Steps:               $STEPS"
echo "Gen length:          $GEN_LENGTH"
echo "Block len:           $BLOCK_LENGTH"
echo "Temp:                $TEMPERATURE"
echo "CFG scale:           $CFG_SCALE"
echo "Remasking:           $REMASKING"
echo "Mask id:             $MASK_ID"
echo "Log every:           $LOG_EVERY"
echo "Flip unmask thresh:  $FLIP_UNMASK_THRESHOLD"
echo "Flip margin thresh:  $FLIP_MARGIN_THRESHOLD"
echo "Flip count thresh:   $FLIP_THRESHOLD"
echo "Fast flag:           $FAST"
echo "Output:              ${OUTPUT:-<auto>}"
echo "Extra args:          ${EXTRA_ARGS:-<none>}"
echo "Job ID:              ${SLURM_JOB_ID:-local}"
echo "=========================================="

ARGS=(
  --model-path "$MODEL_PATH"
  --method "$METHOD"
  --dataset "$DATASET"
  --dataset-config "$DATASET_CONFIG"
  --split "$SPLIT"
  --limit "$LIMIT"
  --batch-size "$BATCH_SIZE"
  --steps "$STEPS"
  --gen-length "$GEN_LENGTH"
  --block-length "$BLOCK_LENGTH"
  --temperature "$TEMPERATURE"
  --cfg-scale "$CFG_SCALE"
  --remasking "$REMASKING"
  --mask-id "$MASK_ID"
  --log-every "$LOG_EVERY"
  --flip-unmask-threshold "$FLIP_UNMASK_THRESHOLD"
  --flip-margin-threshold "$FLIP_MARGIN_THRESHOLD"
  --flip-threshold "$FLIP_THRESHOLD"
)

if [[ "$FAST" == "1" ]]; then
  ARGS+=(--fast)
fi

if [[ -n "$OUTPUT" ]]; then
  ARGS+=(--output "$OUTPUT")
fi

python3 -u eval_gsm8k.py "${ARGS[@]}" $EXTRA_ARGS
