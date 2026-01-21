"""
Run experimental benchmarks on GSM1K dataset
This script runs Graph-Aware and Margin-Budget experiments and saves results.
"""

import os
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
import argparse

import torch
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

from modeling_llada import LLaDAModelLM
from exp_decoding import (
    decoding_graph_aware,
    decoding_graph_aware_gradient,
    decoding_graph_aware_v2,
    decoding_graph_aware_sg_ga,
    decoding_margin_budget,
    decoding_sgga_wino_hybrid
)
from decoding import decoding_wino

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================================
# Evaluation Functions (from notebook)
# ============================================================================

FEWSHOT_CONTEXT = [
    {
        "role": "user",
        "content": "Question: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?\nLet's think step by step\nAnswer:",
    },
    {
        "role": "assistant",
        "content": "Angelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.\nFor the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.\nAngelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.\nHowever, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.\nThey also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.\nAnd they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.\nSo Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.\nThey want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75\nThey will need to plan to study 4 days to allow for all the time they need.\nThe answer is 4\n",
    },
    {
        "role": "user",
        "content": "Question: Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the 3 pointers and free throws.  What's the total number of points scored by both teams added together?\nLet's think step by step\nAnswer:",
    },
    {
        "role": "assistant",
        "content": "Mark's team scores 25 2 pointers, meaning they scored 25*2= 50 points in 2 pointers.\nHis team also scores 6 3 pointers, meaning they scored 8*3= 24 points in 3 pointers\nThey scored 10 free throws, and free throws count as one point so they scored 10*1=10 points in free throws.\nAll together his team scored 50+24+10= 84 points\nMark's opponents scored double his team's number of 2 pointers, meaning they scored 50*2=100 points in 2 pointers.\nHis opponents scored half his team's number of 3 pointers, meaning they scored 24/2= 12 points in 3 pointers.\nThey also scored half Mark's team's points in free throws, meaning they scored 10/2=5 points in free throws.\nAll together Mark's opponents scored 100+12+5=117 points\nThe total score for the game is both team's scores added together, so it is 84+117=201 points\nThe answer is 201\n",
    },
    {
        "role": "user",
        "content": "Question: Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles?\nLet's think step by step\nAnswer:",
    },
    {
        "role": "assistant",
        "content": "When Bella buys 2/5 times more marbles, she'll have increased the number of marbles by 2/5*60 = 24\nThe total number of marbles she'll have is 60+24 = 84\nIf Bella currently has 60 marbles, and she has two times as many marbles as frisbees, she has 60/2 = 30 frisbees.\nIf Bella buys 2/5 times more frisbees, she'll have 2/5*30 = 12 more frisbees.\nThe total number of frisbees she'll have will increase to 30+12 = 42\nBella also has 20 more frisbees than deck cards, meaning she has 30-20 = 10 deck cards\nIf she buys 2/5 times more deck cards, she'll have 2/5*10 = 4 more deck cards.\nThe total number of deck cards she'll have is 10+4 = 14\nTogether, Bella will have a total of 14+42+84 = 140 items\nThe answer is 140\n",
    },
    {
        "role": "user",
        "content": "Question: A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 bananas in the first three baskets and 2 less of each fruit in the fourth basket. How many fruits are there?\nLet's think step by step\nAnswer:",
    },
    {
        "role": "assistant",
        "content": "For the first three baskets, the number of apples and oranges in one basket is 9+15=24\nIn total, together with bananas, the number of fruits in one basket is 24+14=38 for the first three baskets.\nSince there are three baskets each having 38 fruits, there are 3*38=114 fruits in the first three baskets.\nThe number of apples in the fourth basket is 9-2=7\nThere are also 15-2=13 oranges in the fourth basket\nThe combined number of oranges and apples in the fourth basket is 13+7=20\nThe fourth basket also contains 14-2=12 bananas.\nIn total, the fourth basket has 20+12=32 fruits.\nThe four baskets together have 32+114=146 fruits.\nThe answer is 146\n",
    },
]

QUESTION_PROMPT_TEMPLATE = "Question: {question}\nLet's think step by step\nAnswer:"


def build_prompt(question, tokenizer):
    """Build prompt for GSM8K question"""
    prompt_text = QUESTION_PROMPT_TEMPLATE.format(question=question)
    context = FEWSHOT_CONTEXT + [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(
        context, add_generation_prompt=True, tokenize=False
    )
    return prompt_text, prompt


def gsm8k_dataset_postprocess(text: str) -> str:
    """Extract answer from dataset"""
    if text is None:
        return "NULL"
    text = str(text)
    if "#### " in text:
        return text.split("#### ", 1)[1].replace(",", "").strip()
    numbers = re.findall(r"\-?\d+\.\d+|\-?\d+", text.replace(",", ""))
    if not numbers:
        return "NULL"
    return numbers[-1]


def gsm8k_postprocess(text: str) -> str:
    """Extract answer from generated text"""
    if text is None:
        return "NULL"
    text = str(text)
    text = text.split("Question:")[0]
    numbers = re.findall(r"\-?\d+\.\d+|\-?\d+", text)
    if not numbers:
        return "NULL"
    return numbers[-1]


class Gsm8kEvaluator:
    """GSM8K evaluator"""
    def is_equal(self, pred, refer):
        try:
            if pred == refer or abs(float(pred) - int(refer)) < 1e-6:
                return True
        except Exception:
            pass
        return False


# ============================================================================
# Experiment Runner
# ============================================================================

def run_experiment(
    method_name: str,
    method_func,
    method_config: dict,
    model,
    tokenizer,
    dataset,
    evaluator,
    results_dir: Path,
    dataset_name: str = "gsm1k",
    shard_index: Optional[int] = None,
    device: str = "cuda"
):
    """
    Run a single experiment
    
    Args:
        method_name: Name of the method (e.g., "graph_aware")
        method_func: Decoding function to use
        method_config: Configuration dict for the method
        model: LLaDA model
        tokenizer: Tokenizer
        dataset: GSM dataset (gsm1k or gsm8k)
        evaluator: Gsm8kEvaluator instance
        results_dir: Directory to save results
        dataset_name: Name of dataset for file naming
        shard_index: Shard index if running in parallel (None for non-sharded)
        device: Device to use
    
    Returns:
        List of result records
    """
    # Clean dataset name for filename
    dataset_short = dataset_name.replace("/", "_").replace("openai_", "").replace("ScaleAI_", "")
    
    # Add shard suffix if sharding
    if shard_index is not None:
        output_path = results_dir / f"{dataset_short}_{method_name}_shard{shard_index}.jsonl"
    else:
        output_path = results_dir / f"{dataset_short}_{method_name}.jsonl"
    
    results = []
    correct = 0
    total = len(dataset)
    
    # Display name for progress bar
    dataset_display = dataset_short.upper()
    
    print(f"\n{'='*60}")
    print(f"[{dataset_display}] {method_name.upper()} 평가 시작")
    print(f"{'='*60}")
    print(f"데이터셋: {dataset_name}")
    print(f"샘플 수: {total}")
    print(f"결과 저장 경로: {output_path}")
    print(f"{'='*60}\n")
    
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, doc in enumerate(tqdm(dataset, desc=f"{dataset_display}/{method_name}")):
            question = doc.get("question", "")
            answer_raw = doc.get("answer", "")
            
            # Build prompt
            prompt_text, prompt = build_prompt(question, tokenizer)
            
            # Tokenize
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            if device == "cuda":
                input_ids = input_ids.cuda()
            
            # Decode
            # Inject tokenizer for V2 if needed
            current_config = method_config.copy()
            if method_name == "graph_aware_v2" and "tokenizer" in current_config:
                current_config["tokenizer"] = tokenizer
            
            # Inject log_dir for SG-GA trace logging and IDs for separation
            # Robust check for SG-GA (handling potential minor name variations)
            if "graph_aware_sg_ga" in method_name:
                print(f"[INJECTION] SG-GA detected. Sample idx={idx}, shard={shard_index}")
                
                # Construct explicit trace log path to prevent directory ambiguity
                trace_filename = "trace_sag_ga.jsonl"
                if shard_index is not None:
                     trace_filename = f"trace_sag_ga_shard{shard_index}.jsonl"
                
                # Force absolute path resolution
                trace_full_path = (results_dir / trace_filename).resolve()
                
                current_config["trace_log_path"] = str(trace_full_path)
                current_config["sample_id"] = idx
                current_config["shard_id"] = shard_index
                
                print(f"[INJECTION] Config updated: trace_path={trace_full_path}, sample_id={idx}")
            
            # DEBUG: Check injection
            if "graph_aware_sg_ga" in method_name:
                print(f"[DEBUG] Calling SG-GA. TracePath: {current_config.get('trace_log_path')}, SampleID: {current_config.get('sample_id')}")

            gen_output, steps = method_func(
                model,
                input_ids,
                **current_config
            )
            
            # Decode text
            gen_text = tokenizer.decode(
                gen_output[0, input_ids.shape[1]:], skip_special_tokens=True
            )
            
            # Evaluate
            prediction = gsm8k_postprocess(gen_text)
            answer = gsm8k_dataset_postprocess(answer_raw) if answer_raw else answer_raw
            is_correct = evaluator.is_equal(prediction, answer)
            
            if is_correct:
                correct += 1
            
            # Save result
            record = {
                "index": idx,
                "question": question,
                "answer_raw": answer_raw,
                "answer": answer,
                "prediction": prediction,
                "correct": is_correct,
                "response": gen_text,
                "steps": steps,
                "prompt_text": prompt_text,
                "method": method_name,
            }
            results.append(record)
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
    
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    avg_nfe = np.mean([r['steps'] for r in results])
    
    print(f"\n{'='*60}")
    print(f"[{dataset_display}] {method_name.upper()} 평가 완료!")
    print(f"{'='*60}")
    print(f"데이터셋: {dataset_name}")
    print(f"결과 파일: {output_path}")
    print(f"정확도: {accuracy:.2f}% ({correct}/{total})")
    print(f"평균 NFE: {avg_nfe:.2f}")
    print(f"{'='*60}\n")
    
    # Save summary (only in non-shard mode; merge_results handles shard mode)
    if shard_index is None:
        summary = {
            "method": method_name,
            "dataset": dataset_name,
            "total_samples": total,
            "accuracy": accuracy,
            "correct_count": correct,
            "avg_nfe": float(avg_nfe),
            "median_nfe": float(np.median([r['steps'] for r in results])),
            "config": method_config,
        }
        
        summary_path = results_dir / f"summary_{method_name}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"요약 파일 저장: {summary_path}\n")
    else:
        print(f"샤드 모드: 요약은 병합 시 생성됩니다.\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run experimental benchmarks")
    parser.add_argument("--model", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                       help="Model path")
    parser.add_argument("--dataset", type=str, default="ScaleAI/gsm1k",
                       help="Dataset name")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of samples (None for all)")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Alias for --limit")
    parser.add_argument("--methods", type=str, nargs="+", 
                       default=["graph_aware", "margin_budget"],
                       choices=["graph_aware", "graph_aware_gradient", "graph_aware_v2", "graph_aware_sg_ga", "margin_budget", "wino", "sgga_wino_hybrid"],
                       help="Methods to run")
    parser.add_argument("--results_dir", type=str, default="results",
                       help="Results directory")
    parser.add_argument("--run_id", type=str, default=None,
                       help="Run ID for grouping results (auto-generated if not provided)")
    parser.add_argument("--shard_index", type=int, default=None,
                       help="Shard index for data parallelism (0-indexed)")
    parser.add_argument("--num_shards", type=int, default=None,
                       help="Total number of shards for data parallelism")
    parser.add_argument("--override_config", type=str, default="{}",
                       help="JSON string to override method config parameters (e.g., '{\"confidence_threshold\": 0.6}')")
    
    args = parser.parse_args()
    
    # Unified limit logic
    limit = args.num_samples if args.num_samples is not None else args.limit
    
    # Generate run ID (timestamp-based directory for this batch)
    if args.run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        run_id = args.run_id

    # Setup
    print(f"\n{'='*60}")
    print("실험 설정")
    print(f"{'='*60}")
    print(f"Run ID: {run_id}")
    print(f"모델: {args.model}")
    print(f"데이터셋: {args.dataset}")
    print(f"샘플 수: {'전체' if limit is None else limit}")
    print(f"실행 방법: {', '.join(args.methods)}")
    print(f"{'='*60}\n")
    
    # Create results directory with run_id subdirectory
    base_results_dir = Path(args.results_dir)
    
    # Sharding 모드에서는 results_dir에 이미 run_id가 포함되어 있음
    if args.shard_index is not None and args.num_shards is not None:
        results_dir = base_results_dir
    else:
        results_dir = base_results_dir / run_id
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"결과 저장 디렉터리: {results_dir}\n")
    
    # Load model
    print("모델 로딩 중...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LLaDAModelLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    if device == "cuda":
        model = model.cuda()
    model.eval()
    print(f"✓ 모델 로딩 완료 (device: {device})\n")
    
    # Load tokenizer
    print("토크나이저 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print("✓ 토크나이저 로딩 완료\n")
    
    # Load dataset
    print(f"데이터셋 로딩 중: {args.dataset}")
    
    # GSM8K requires config name
    if "gsm8k" in args.dataset.lower():
        dataset = load_dataset(args.dataset, "main", split="test")
    else:
        dataset = load_dataset(args.dataset, split="test")
    
    if limit is not None and limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))
    
    # Data sharding for parallelism
    if args.shard_index is not None and args.num_shards is not None:
        total_samples = len(dataset)
        shard_size = (total_samples + args.num_shards - 1) // args.num_shards
        start_idx = args.shard_index * shard_size
        end_idx = min(start_idx + shard_size, total_samples)
        
        dataset = dataset.select(range(start_idx, end_idx))
        print(f"✓ 샤드 {args.shard_index + 1}/{args.num_shards}: "
              f"{len(dataset)}개 샘플 (전체 {total_samples}개 중 {start_idx}~{end_idx})\n")
    else:
        print(f"✓ 데이터셋 로딩 완료: {len(dataset)}개 샘플\n")
    
    # Evaluator
    evaluator = Gsm8kEvaluator()
    
    # Method configurations
    configs = {
        "graph_aware": {
            "gen_length": 256,
            "block_length": 256,
            "temperature": 0.0,
            "mask_id": 126336,
            "attention_threshold": 0.15,
            "cascade_depth": 2,
            "temporal_decay": 0.5,
            "confidence_threshold": 0.6,
            "adaptive": True,
        },
        "graph_aware_gradient": {
            "gen_length": 256,
            "block_length": 256,
            "temperature": 0.0,
            "mask_id": 126336,
            "attention_threshold": 0.15,
            "cascade_depth": 2,
            "temporal_decay": 0.3, # Slightly reduced for gradient method
            "confidence_threshold": 0.6,
            "adaptive": True,
        },
        "margin_budget": {
            "gen_length": 256,
            "block_length": 256,
            "temperature": 0.0,
            "mask_id": 126336,
            "margin_threshold": 0.1,
            "budget_per_step": 5,
            "adaptive_budget": True,
        },
        "graph_aware_v2": {
            "gen_length": 256,
            "block_length": 256,
            "temperature": 0.0,
            "mask_id": 126336,
            # V2-specific parameters
            "early_commit_ratio": 0.25,
            "cascade_start_ratio": 0.3,
            "cascade_full_ratio": 0.5,
            "confidence_high": 0.7,
            "confidence_low": 0.4,
            "attention_threshold": 0.15,
            "temporal_decay_strong": 1.0,
            "temporal_decay_base": 0.5,
            "remask_budget": 8,
            "cooldown_period": 3,
            "protect_symbols": True,
            "tokenizer": None,  # Will be set at runtime
        },
        "graph_aware_sg_ga": {
            "gen_length": 256,
            "block_length": 256,
            "temperature": 0.0,
            "mask_id": 126336,
            # SG-GA Theoretical Parameters
            "alpha": 0.5,        # Fisher scaling
            "tau_low": 0.05,     # Curvature threshold low
            "tau_high": 0.15,    # Curvature threshold high
            "min_budget": 3,
            "max_budget": 15,
            "graph_threshold": 0.2, # Optimized 0.1->0.2
            "confidence_threshold": 0.5, # MATCH WINO BASELINE
        },
        "wino": {
            "gen_length": 256,
            "block_length": 256,
            "temperature": 0.0,
            "threshold": 0.5,
            "threshold_back": 0.8,
        },
        "sgga_wino_hybrid": {
            "gen_length": 256,
            "block_length": 256,
            "temperature": 0.0,
            "mask_id": 126336,
            # WINO parameters
            "threshold": 0.5,
            "threshold_back": 0.9,
            # SGGA parameters
            "graph_threshold": 0.2,
            "alpha": 0.5,
            "min_budget": 3,
            "max_budget": 15,
            "tau_low": 0.05,
            "tau_high": 0.15,
        }
    }
    
    # Apply config overrides from command line (for grid search)
    try:
        override_dict = json.loads(args.override_config)
        if override_dict:
            print(f"\n{'='*60}")
            print("Configuration Overrides Applied:")
            print(f"{'='*60}")
            for key, value in override_dict.items():
                print(f"  {key}: {value}")
            print(f"{'='*60}\n")
            
            # Apply overrides to all method configs
            for method_name in args.methods:
                if method_name in configs:
                    configs[method_name].update(override_dict)
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse override_config JSON: {e}")
        print(f"Proceeding with default configurations.\n")
    
    methods = {
        "graph_aware": decoding_graph_aware,
        "graph_aware_gradient": decoding_graph_aware_gradient,
        "graph_aware_v2": decoding_graph_aware_v2,
        "graph_aware_sg_ga": decoding_graph_aware_sg_ga,
        "margin_budget": decoding_margin_budget,
        "wino": decoding_wino,
        "sgga_wino_hybrid": decoding_sgga_wino_hybrid,
    }
    
    # Run experiments
    all_results = {}
    for method_name in args.methods:
        results = run_experiment(
            method_name=method_name,
            method_func=methods[method_name],
            method_config=configs[method_name],
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            evaluator=evaluator,
            results_dir=results_dir,
            dataset_name=args.dataset,
            shard_index=args.shard_index,
            device=device
        )
        all_results[method_name] = results
    
    print(f"\n{'='*60}")
    print("모든 실험 완료!")
    
    # Auto-Analyze Trace for SG-GA
    for method_name in args.methods:
        if method_name == "graph_aware_sg_ga":
            try:
                import analyze_trace
                print(f"\n[Auto-Analysis] Starting trace analysis for {method_name}...")
                
                # Construct expected trace path
                trace_filename = "trace_sag_ga.jsonl"
                if args.shard_index is not None:
                    trace_filename = f"trace_sag_ga_shard{args.shard_index}.jsonl"
                
                trace_path = results_dir / trace_filename
                
                if trace_path.exists():
                    analyze_trace.analyze_trace(str(trace_path))
                else:
                    print(f"[Auto-Analysis] Warning: Trace file not found at {trace_path}")
            except Exception as e:
                print(f"[Auto-Analysis] Failed: {e}")
                
    print(f"{'='*60}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
