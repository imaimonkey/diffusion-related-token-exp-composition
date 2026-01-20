#!/bin/bash
#SBATCH --job-name=trace_gsm8k_wrong
#SBATCH --output=logs/trace_gsm8k_wrong_%j.out
#SBATCH --error=logs/trace_gsm8k_wrong_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --nodelist=ubuntu

set -euo pipefail


mkdir -p logs results

source .venv/bin/activate

MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
RESULTS="${RESULTS:-results/gsm8k_default_20260118_192218.jsonl}"

NUM="${NUM:-10}"
STEPS="${STEPS:-256}"
GEN_LENGTH="${GEN_LENGTH:-256}"
BLOCK_LENGTH="${BLOCK_LENGTH:-128}"
TEMPERATURE="${TEMPERATURE:-0.0}"
CFG_SCALE="${CFG_SCALE:-0.0}"
REMASKING="${REMASKING:-low_confidence}"
MASK_ID="${MASK_ID:-126336}"
VIEW_LEN="${VIEW_LEN:-256}"
MAX_STEPS="${MAX_STEPS:-256}"

EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "=========================================="
echo "Trace GSM8K Wrong"
echo "Model:      $MODEL_PATH"
echo "Results:    $RESULTS"
echo "Num wrong:  $NUM"
echo "Steps:      $STEPS"
echo "Gen length: $GEN_LENGTH"
echo "Block len:  $BLOCK_LENGTH"
echo "Temp:       $TEMPERATURE"
echo "CFG scale:  $CFG_SCALE"
echo "Remasking:  $REMASKING"
echo "Mask id:    $MASK_ID"
echo "View len:   $VIEW_LEN"
echo "Max steps:  $MAX_STEPS"
echo "Extra args: ${EXTRA_ARGS:-<none>}"
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "=========================================="

ARGS=(
  --model-path "$MODEL_PATH"
  --results "$RESULTS"
  --num "$NUM"
  --steps "$STEPS"
  --gen-length "$GEN_LENGTH"
  --block-length "$BLOCK_LENGTH"
  --temperature "$TEMPERATURE"
  --cfg-scale "$CFG_SCALE"
  --remasking "$REMASKING"
  --mask-id "$MASK_ID"
  --view-len "$VIEW_LEN"
  --max-steps "$MAX_STEPS"
)

python3 trace_gsm8k_wrong.py "${ARGS[@]}" $EXTRA_ARGS
