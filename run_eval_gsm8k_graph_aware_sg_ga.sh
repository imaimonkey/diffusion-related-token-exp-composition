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

# SGGA trace on/off (1=enable, 0=disable)
export SGGA_TRACE="${SGGA_TRACE:-1}"

DATASETS=("openai/gsm8k")
METHODS=("graph_aware_sg_ga")
NUM_SAMPLES=""

# Override Config (재현성 위해 knobs 명시)
# - block_length: 비교 기준(보통 128)
# - min_loop_step_for_remask: 초반 remask 지연(진동 감소)
# - min_remask_priority: 낮은 신호 remask 차단
# - numeric_min_remask_priority: 숫자/연산자 remask를 더 보수적으로
# - remask_cooldown_period(+numeric_*): 동일 위치 반복 remask 억제
# - max_remasks_per_pos: 위치별 remask 상한
# - hop2_decay: cascade_depth=2에서 2-hop 연관성 감쇠(과도 확장 방지)
# - max_total_remasks: 샘플당 과도 remask(thrashing) 하드 컷(선택)
OVERRIDE_CONFIG='{"block_length":128,"min_loop_step_for_remask":3,"min_remask_priority":0.02,"numeric_min_remask_priority":0.2,"remask_cooldown_period":3,"numeric_remask_cooldown_period":3,"max_remasks_per_pos":2,"hop2_decay":0.8,"max_total_remasks":120}'
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
