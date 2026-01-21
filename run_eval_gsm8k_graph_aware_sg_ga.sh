#!/bin/bash
#SBATCH --job-name=eval_sg_ga
#SBATCH --output=logs/eval_sg_ga_%j.out
#SBATCH --error=logs/eval_sg_ga_%j.err
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --nodelist=server2

cd /home/kimhj/diffusion-related-token-exp-composition

# 로그 디렉토리 생성

mkdir -p logs


# ============================================================================
# 실험 설정 (Graph-Aware SG-GA/SAGA)
# ============================================================================

# 실행 모드: "dynamic"
MODE="dynamic"

DATASETS=("openai/gsm8k")
METHODS=("graph_aware_sg_ga")
NUM_SAMPLES=""

# Override Config: Block Length=128 (기본 실험 구조 준수)
# Confidence Threshold는 코드 기본값(0.5) 사용 (WINO와 baseline 일치)
OVERRIDE_CONFIG='{"block_length": 128}'

# ============================================================================
# 실행
# ============================================================================

RUN_ID=$(date +"%Y%m%d_%H%M%S")

echo "=========================================="
echo "실험 시작: Graph-Aware SG-GA (SAGA)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Run ID: $RUN_ID"
echo "Override: $OVERRIDE_CONFIG"
echo "=========================================="

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    GPU_POOL=$(seq -s, 0 $((NUM_GPUS-1)))
else
    GPU_POOL=$CUDA_VISIBLE_DEVICES
fi

echo "GPU Pool: $GPU_POOL"
echo ""

uv run python run_with_gpu_pool.py \
    --gpu_pool "$GPU_POOL" \
    --datasets "${DATASETS[@]}" \
    --methods "${METHODS[@]}" \
    --num_samples "$NUM_SAMPLES" \
    --run_id "$RUN_ID" \
    --results_dir results \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --override_config "$OVERRIDE_CONFIG"

echo ""
echo "=========================================="
echo "실험 완료"
echo "=========================================="
