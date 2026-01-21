#!/bin/bash
#SBATCH --job-name=exp_sgga_wino_hybrid
#SBATCH --nodelist=server2
#SBATCH --output=slurm_logs/exp_sgga_wino_hybrid_%j.out
#SBATCH --error=slurm_logs/exp_sgga_wino_hybrid_%j.err
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00

# 작업 디렉토리로 이동
cd /home/kimhj/difffusion-sampling-exp-benchmark-playground

# 로그 디렉토리 생성
mkdir -p slurm_logs

# ============================================================================
# 실험 설정
# ============================================================================

# 실행 모드 선택
# - "fixed": 고정 GPU 할당 (3 GPU, 빠른 시작) - DATASETS의 첫 번째 데이터셋 사용
# - "dynamic": 동적 GPU 할당 (모든 GPU 활용) - DATASETS의 모든 데이터셋 사용
# MODE="fixed"
MODE="dynamic"

# 데이터셋 설정 (배열)
# fixed 모드에서는 첫 번째 요소만 사용됩니다.
DATASETS=(
    # "ScaleAI/gsm1k"
    "openai/gsm8k"
)

# 샘플 수 제한 (빈 값 = 전체 실행)
NUM_SAMPLES=200

# 메서드 설정
METHODS=(
    # "graph_aware"
    # "graph_aware_v2"
    # "graph_aware_sg_ga"
    # "graph_aware_gradient"
    # "margin_budget"
    "wino"
    "sgga_wino_hybrid"
)

# ============================================================================
# 실행 (수정 불필요)
# ============================================================================

RUN_ID=$(date +"%Y%m%d_%H%M%S")

echo "=========================================="
echo "실험 시작"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Run ID: $RUN_ID"
echo "Node: $SLURM_NODELIST"
echo "Mode: $MODE"
echo "=========================================="

nvidia-smi

echo ""

if [ "$MODE" = "fixed" ]; then
    # ========================================================================
    # 고정 GPU 할당 모드 (GSM1K / GSM8K 단일 실행)
    # ========================================================================
    
    # 첫 번째 데이터셋 선택
    TARGET_DATASET=${DATASETS[0]}
    
    echo "=========================================="
    echo "고정 GPU 할당 모드 (3 GPUs)"
    echo "Dataset: $TARGET_DATASET"
    echo "Samples: ${NUM_SAMPLES:-ALL}"
    echo "Methods: graph_aware, margin_budget, wino"
    echo "Run ID:  $RUN_ID"
    echo "=========================================="
    echo ""
    
    # GPU 0: Graph Aware
    echo "[GPU 0] Starting graph_aware..."
    CUDA_VISIBLE_DEVICES=0 uv run python run_experiments.py \
        --model GSAI-ML/LLaDA-8B-Instruct \
        --dataset "$TARGET_DATASET" \
        --methods graph_aware \
        --num_samples "$NUM_SAMPLES" \
        --run_id "$RUN_ID" &
    
    # GPU 1: Margin Budget
    echo "[GPU 1] Starting margin_budget..."
    CUDA_VISIBLE_DEVICES=1 uv run python run_experiments.py \
        --model GSAI-ML/LLaDA-8B-Instruct \
        --dataset "$TARGET_DATASET" \
        --methods margin_budget \
        --num_samples "$NUM_SAMPLES" \
        --run_id "$RUN_ID" &
    
    # GPU 2: Wino
    echo "[GPU 2] Starting wino..."
    CUDA_VISIBLE_DEVICES=2 uv run python run_experiments.py \
        --model GSAI-ML/LLaDA-8B-Instruct \
        --dataset "$TARGET_DATASET" \
        --methods wino \
        --num_samples "$NUM_SAMPLES" \
        --run_id "$RUN_ID" &
    
    wait

elif [ "$MODE" = "dynamic" ]; then
    # ========================================================================
    # 동적 GPU 할당 모드
    # ========================================================================
    
    # Slurm이 할당한 GPU 자동 감지
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        # CUDA_VISIBLE_DEVICES가 없으면 사용 가능한 GPU 개수로 설정
        NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
        GPU_POOL=$(seq -s, 0 $((NUM_GPUS-1)))
    else
        # CUDA_VISIBLE_DEVICES에서 GPU 목록 추출
        GPU_POOL=$CUDA_VISIBLE_DEVICES
    fi
    
    echo "=========================================="
    echo "동적 GPU 할당 모드"
    echo "GPU Pool (자동 감지): $GPU_POOL"
    echo "Datasets: ${DATASETS[@]}"
    echo "Methods: ${METHODS[@]}"
    echo "Samples: ${NUM_SAMPLES:-ALL}"
    echo "=========================================="
    echo ""
    
    uv run python run_with_gpu_pool.py \
        --gpu_pool "$GPU_POOL" \
        --datasets "${DATASETS[@]}" \
        --methods "${METHODS[@]}" \
        --num_samples "$NUM_SAMPLES" \
        --run_id "$RUN_ID" \
        --results_dir results \
        --model GSAI-ML/LLaDA-8B-Instruct

else
    echo "Error: Invalid MODE='$MODE'. Use 'fixed' or 'dynamic'."
    exit 1
fi

echo ""
echo "=========================================="
echo "실험 완료: $(date)"
echo "결과 디렉토리: results/$RUN_ID"
echo "=========================================="

# ============================================================================
# 사용 가이드
# ============================================================================
#
# [고정 모드] - 빠르고 간단
# MODE="fixed"
# DATASET="ScaleAI/gsm1k"
# NUM_SAMPLES=100
#
# [동적 모드] - 여러 실험 자동 관리
# MODE="dynamic"
# DATASETS=("ScaleAI/gsm1k" "openai/gsm8k")
# METHODS=("graph_aware" "margin_budget" "wino")
# DYNAMIC_NUM_SAMPLES=100
#
# GPU는 Slurm이 할당한 것을 자동으로 감지합니다 (#SBATCH --gres=gpu:N)
#
# ============================================================================
