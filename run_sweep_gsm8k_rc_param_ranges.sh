#!/bin/bash
#SBATCH --job-name=sweep_rc
#SBATCH --output=logs/sweep_rc_%j.out
#SBATCH --error=logs/sweep_rc_%j.err
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --nodelist=server2

set -euo pipefail

cd /home/kimhj/diffusion-related-token-exp-composition
mkdir -p logs

# -----------------------------------------------------------------------------
# 목적
# - RC(=retrospective_cascading)에서 "어느 구간"이 민감한지 빠르게 보기 위한 OAT 스윕
# - RC는 retraction(confidence_low)이 안 걸리면 baseline과 동일하게 수렴하기 쉬움
# -----------------------------------------------------------------------------

MODEL="${MODEL:-GSAI-ML/LLaDA-8B-Instruct}"
DATASET="${DATASET:-openai/gsm8k}"
METHOD="retrospective_cascading"
NUM_SAMPLES="${NUM_SAMPLES:-200}"

BASE_OVERRIDE='{"block_length": 128}'

RUN_ID="$(date +'%Y%m%d_%H%M%S')"
RESULT_ROOT="results/sweep_rc_${RUN_ID}"
mkdir -p "$RESULT_ROOT"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  GPU_POOL="$CUDA_VISIBLE_DEVICES"
else
  NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
  GPU_POOL=$(seq -s, 0 $((NUM_GPUS-1)))
fi

merge_json() {
  python3 - "$1" "$2" <<'PY'
import json, sys
base = json.loads(sys.argv[1])
extra = json.loads(sys.argv[2])
base.update(extra)
print(json.dumps(base, ensure_ascii=False))
PY
}

read_summary_field() {
  local summary_json="$1"
  local key="$2"
  python3 - "$summary_json" "$key" <<'PY'
import json, sys
path, key = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    obj = json.load(f)
print(obj.get(key, ""))
PY
}

run_case() {
  local tag="$1"
  local extra_override="$2"
  local override
  override="$(merge_json "$BASE_OVERRIDE" "$extra_override")"

  echo "------------------------------------------"
  echo "TAG: $tag"
  echo "Override: $override"
  echo "------------------------------------------"

  set +e
  uv run python run_with_gpu_pool.py \
    --gpu_pool "$GPU_POOL" \
    --datasets "$DATASET" \
    --methods "$METHOD" \
    --num_samples "$NUM_SAMPLES" \
    --run_id "$tag" \
    --results_dir "$RESULT_ROOT" \
    --model "$MODEL" \
    --override_config "$override"
  local exit_code=$?
  set -e

  local summary_json="${RESULT_ROOT}/${tag}/summary_${METHOD}.json"
  if [ $exit_code -ne 0 ] || [ ! -f "$summary_json" ]; then
    echo "[FAIL] $tag"
    echo "$tag,FAIL,,,,," >> "$CSV"
    return
  fi

  local acc correct total avg_nfe median_nfe
  acc="$(read_summary_field "$summary_json" "accuracy")"
  correct="$(read_summary_field "$summary_json" "correct_count")"
  total="$(read_summary_field "$summary_json" "total_samples")"
  avg_nfe="$(read_summary_field "$summary_json" "avg_nfe")"
  median_nfe="$(read_summary_field "$summary_json" "median_nfe")"

  echo "$tag,OK,$acc,$correct,$total,$avg_nfe,$median_nfe" >> "$CSV"
}

CSV="${RESULT_ROOT}/sweep_report.csv"
echo "tag,status,accuracy,correct,total,avg_nfe,median_nfe" > "$CSV"

run_case "baseline" "{}"

# 1) Retraction trigger: confidence_low (이 값이 올라갈수록 Primary retraction이 더 자주 발생)
for v in 0.35 0.40 0.45 0.50 0.60 0.70; do
  run_case "conf_low_${v}" "{\"confidence_low\": ${v}}"
done

# 2) Primary/Secondary budget split
for v in 0.25 0.50 0.75 1.00; do
  run_case "primary_ratio_${v}" "{\"primary_budget_ratio\": ${v}}"
done

# 3) Total budget / cooldown
for v in 0 2 4 8 16; do
  run_case "budget_${v}" "{\"remask_budget\": ${v}}"
done
for v in 0 1 3 5; do
  run_case "cooldown_${v}" "{\"cooldown_period\": ${v}}"
done

# 4) Secondary cascade filter: attention_threshold (attention 추출 실패 시 매우 낮은 값이 필요할 수 있음)
for v in 0.005 0.01 0.02 0.05 0.10 0.15 0.20; do
  run_case "attn_${v}" "{\"attention_threshold\": ${v}}"
done

# 5) Temporal decay (연쇄 대상의 "시간적 근접" 가중)
for v in 0.20 0.50 0.80 1.00; do
  run_case "temp_${v}" "{\"temporal_decay\": ${v}}"
done

# 6) A few "make-it-different" combos to ensure retraction activates
run_case "aggressive_1" "{\"confidence_low\": 0.60, \"remask_budget\": 16, \"primary_budget_ratio\": 0.75, \"attention_threshold\": 0.01, \"cooldown_period\": 0}"
run_case "aggressive_2" "{\"confidence_low\": 0.70, \"remask_budget\": 16, \"primary_budget_ratio\": 1.00, \"attention_threshold\": 0.01, \"cooldown_period\": 0}"

echo ""
echo "=========================================="
echo "Sweep completed: $RESULT_ROOT"
echo "CSV: $CSV"
echo "=========================================="

python3 - "$CSV" <<'PY'
import pandas as pd, sys
df = pd.read_csv(sys.argv[1])
df_ok = df[df["status"] == "OK"].copy()
if df_ok.empty:
    print("No successful runs to summarize.")
    raise SystemExit(0)
df_ok = df_ok.sort_values(["accuracy", "avg_nfe"], ascending=[False, True])
print(df_ok.head(30).to_string(index=False))
PY

