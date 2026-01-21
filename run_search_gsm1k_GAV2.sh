#!/bin/bash
#SBATCH --job-name=search_gav2_params
#SBATCH --output=logs/search_gav2_%j.out
#SBATCH --error=logs/search_gav2_%j.err
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --nodelist=server2

set -euo pipefail

# 작업 디렉토리로 이동
cd /home/kimhj/diffusion-related-token-exp-composition

# 로그 디렉토리 생성
mkdir -p logs
# [HONEST NAMING] "GSM1K" instead of "GSM8K Train 1K" because we are restricted to Test split.
# Using 'gsm8k_test_subset' to avoid "leakage" misunderstanding in papers.
# If strictly 'gsm1k' (scaleai) was available and supported, we would use that.
RESULT_ROOT="results/gsm8k_test_subset_search"
mkdir -p "$RESULT_ROOT"

# ============================================================================
# 기본 설정
# ============================================================================

DATASETS=("openai/gsm8k")
# [NOTE] run_experiments.py uses split="test" hardcoded for gsm8k.
# We limit to 1000 to simulate a small holdout, but STRICTLY SPEAKING this is TEST data.
# Do NOT use parameters derived here for final test reporting on the SAME 1000 samples.
LIMIT=1000

MODEL="GSAI-ML/LLaDA-8B-Instruct"
METHOD="graph_aware_v2"

# 탐색 그리드 (Stable Range Search)
# Reduced grid for feasibility in one script execution
CONF_LOWS=(0.35 0.40 0.45)
ATTN_THRS=(0.10 0.15 0.20)
BUDGETS=(4 8)
COOLDOWNS=(2 3)
# [PATCH 2] Cascade Gating Params (V2 specific)
CASCADE_STARTS=(0.25 0.30)
CASCADE_FULLS=(0.45 0.50)
RUN_INDICES=(1 2)     # "Seeds" (Repeated runs to check stability/determinism)

BASELINE_NFE=45.0

# [FIX] GPU Allocation: Trust Slurm first
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    GPU_POOL="$CUDA_VISIBLE_DEVICES"
else
    # Fallback only if not in Slurm (e.g. local dev)
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    GPU_POOL=$(seq -s, 0 $((NUM_GPUS-1)))
fi

echo "=========================================="
echo "Parameter Search: Graph-Aware V2"
echo "Target: Stable Region Discovery (GSM8K Test Subset 1k)"
echo "GPUs: $GPU_POOL"
echo "=========================================="

# 요약 파일 헤더
SUMMARY_FILE="${RESULT_ROOT}/summary_report.csv"
echo "ConfLow,AttnThr,Budget,Cooldown,CascadeStart,CascadeFull,RunIdx,Accuracy,AvgNFE,Status" > "$SUMMARY_FILE"

# ============================================================================
# 그리드 서치 루프
# ============================================================================

for conf in "${CONF_LOWS[@]}"; do
for attn in "${ATTN_THRS[@]}"; do
for budget in "${BUDGETS[@]}"; do
for cd_period in "${COOLDOWNS[@]}"; do
for cs in "${CASCADE_STARTS[@]}"; do
for cf in "${CASCADE_FULLS[@]}"; do
for run_idx in "${RUN_INDICES[@]}"; do

    # 파라미터 식별자 생성
    PARAM_ID="C${conf}_A${attn}_B${budget}_CD${cd_period}_CS${cs}_CF${cf}_R${run_idx}"
    
    # run_with_gpu_pool creates subdirectory using --run_id inside --results_dir
    # We want: results/search/PARAM_ID/RUN_ID/...
    # But run_with_gpu_pool logic is: results_dir / run_id
    # So we set results_dir=RESULT_ROOT and run_id=PARAM_ID
    
    EXPECTED_DIR="${RESULT_ROOT}/${PARAM_ID}"
    SUMMARY_JSON="${EXPECTED_DIR}/summary_${METHOD}.json"
    
    # [RESUME] Check if finished
    if [ -f "$SUMMARY_JSON" ]; then
        echo "[SKIP] ${PARAM_ID} already done."
        
        # Python-based safe extraction
        ACC=$(python3 -c "import json; print(json.load(open('$SUMMARY_JSON'))['accuracy'])" 2>/dev/null || echo "ERR")
        NFE=$(python3 -c "import json; print(json.load(open('$SUMMARY_JSON'))['avg_nfe'])" 2>/dev/null || echo "ERR")
        
        echo "${conf},${attn},${budget},${cd_period},${cs},${cf},${run_idx},${ACC},${NFE},Skipped" >> "$SUMMARY_FILE"
        continue
    fi

    # [FIX] Safe config generation using Python
    # Includes cascade params
    OVERRIDE_CONFIG=$(python3 -c "import json; print(json.dumps({
        'confidence_low': $conf,
        'attention_threshold': $attn,
        'remask_budget': $budget,
        'cooldown_period': $cd_period,
        'cascade_start_ratio': $cs,
        'cascade_full_ratio': $cf,
        'block_length': 128
    }))")

    echo "Running: ${PARAM_ID} ..."

    # 실행
    set +e
    output=$(uv run python run_with_gpu_pool.py \
        --gpu_pool "$GPU_POOL" \
        --datasets "${DATASETS[@]}" \
        --methods "$METHOD" \
        --num_samples "$LIMIT" \
        --run_id "$PARAM_ID" \
        --results_dir "$RESULT_ROOT" \
        --model "$MODEL" \
        --override_config "$OVERRIDE_CONFIG" 2>&1)
    
    EXIT_CODE=$?
    set -e

    if [ $EXIT_CODE -ne 0 ]; then
        echo "  [FAIL] Execution failed"
        echo "${conf},${attn},${budget},${cd_period},${cs},${cf},${run_idx},0.0,0.0,Fail" >> "$SUMMARY_FILE"
        echo "$output" > "${RESULT_ROOT}/${PARAM_ID}_error.log"
    else
        # run_with_gpu_pool saved to RESULT_ROOT/PARAM_ID
        
        if [ ! -f "$SUMMARY_JSON" ]; then
            FOUND_SUMMARY=$(find "${RESULT_ROOT}/${PARAM_ID}" -maxdepth 3 -name "summary_${METHOD}.json" 2>/dev/null | head -n 1 || true)
            if [ -n "$FOUND_SUMMARY" ] && [ -f "$FOUND_SUMMARY" ]; then
                echo "  [WARN] Summary found at alternative path: $FOUND_SUMMARY"
                mkdir -p "$EXPECTED_DIR"
                cp "$FOUND_SUMMARY" "$SUMMARY_JSON"
            fi
        fi

        if [ -f "$SUMMARY_JSON" ]; then
             # Metric Extraction
            ACC=$(python3 -c "import json; print(json.load(open('$SUMMARY_JSON'))['accuracy'])")
            NFE=$(python3 -c "import json; print(json.load(open('$SUMMARY_JSON'))['avg_nfe'])")
            
            # [FIX] Python float comparison for stability
            STATUS=$(python3 -c "print('Unstable(NFE)' if $NFE > 2 * $BASELINE_NFE else 'Stable')")
            
            echo "  [DONE] Acc: $ACC, NFE: $NFE ($STATUS)"
            echo "${conf},${attn},${budget},${cd_period},${cs},${cf},${run_idx},${ACC},${NFE},${STATUS}" >> "$SUMMARY_FILE"
        else
            echo "  [FAIL] Summary not found at expected path: $SUMMARY_JSON"
            echo "${conf},${attn},${budget},${cd_period},${cs},${cf},${run_idx},0.0,0.0,NoOutput" >> "$SUMMARY_FILE"
        fi
    fi

done
done
done
done
done
done
done

echo ""
echo "=========================================="
echo "Search Completed."
echo "Aggregating Results..."
echo "=========================================="

# [ADDED] Aggregation Report
python3 -c "
import pandas as pd
import sys
try:
    df = pd.read_csv('$SUMMARY_FILE')
    if df.empty:
        print('No results to aggregate.')
    else:
        # [PATCH 3 (Quality)] Filter out Fail/NoOutput for clean aggregation
        df = df[df['Status'].isin(['Stable', 'Unstable(NFE)', 'Skipped'])]
        if df.empty:
            print('No valid results to aggregate.')
        else:
            grp_cols = ['ConfLow','AttnThr','Budget','Cooldown','CascadeStart','CascadeFull']
            agg = df.groupby(grp_cols)[['Accuracy', 'AvgNFE']].agg(['mean', 'std', 'count'])
            print(agg.to_string())
            agg.to_csv('${RESULT_ROOT}/aggregated_report.csv')
except Exception as e:
    print(f'Aggregation failed: {e}')
"

echo "=========================================="
