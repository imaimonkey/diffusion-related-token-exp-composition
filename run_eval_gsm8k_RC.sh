#!/bin/bash
#SBATCH --job-name=eval_rc
#SBATCH --output=logs/eval_rc_%j.out
#SBATCH --error=logs/eval_rc_%j.err
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --nodelist=server2

# 작업 디렉토리로 이동 (필요 시 수정)
cd /home/kimhj/diffusion-related-token-exp-composition

# 로그 디렉토리 생성
mkdir -p logs

# ============================================================================
# 실험 설정 (Retrospective Cascading)
# ============================================================================

# 실행 모드: "dynamic" (GPU Pool Work Stealing 사용)
MODE="dynamic"

# 데이터셋
DATASETS=("openai/gsm8k")

# 메서드: Retrospective Cascading
METHODS=("retrospective_cascading")

# 샘플 수 제한 (빈 값 = 전체 실행)
NUM_SAMPLES=""

# Override Config: 기본 실험 세팅(run_eval_gsm8k.sh)에 맞추어 Block Length=128 설정
OVERRIDE_CONFIG='{"block_length": 128}'

# ============================================================================
# 실행
# ============================================================================

RUN_ID=$(date +"%Y%m%d_%H%M%S")
RC_RUN_ID="RC_${RUN_ID}"

echo "=========================================="
echo "실험 시작: Retrospective Cascading"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Run ID: $RC_RUN_ID"
echo "Mode: $MODE"
echo "Override: $OVERRIDE_CONFIG"
echo "=========================================="

# Slurm이 할당한 GPU 자동 감지
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
    --run_id "$RC_RUN_ID" \
    --results_dir results \
    --model GSAI-ML/LLaDA-8B-Instruct \
    --override_config "$OVERRIDE_CONFIG"

echo ""
echo "=========================================="
echo "실험 완료"
echo "=========================================="
