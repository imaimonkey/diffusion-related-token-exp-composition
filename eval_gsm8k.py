#!/usr/bin/env python3
import argparse
import datetime
import json
import os
from typing import Optional

import torch
import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

from decoding import (
    decoding_default,
    decoding_wino,
    generate_with_saber,
    generate_with_dual_cache,
    generate_with_entropy,
    generate_with_margin,
    generate_with_remdm,
    generate_with_flip_margin,
    generate_with_flip_margin_group,
)
from gsm_common import (
    Gsm8kEvaluator,
    build_prompt,
    gsm8k_dataset_postprocess,
    gsm8k_postprocess,
)
from modeling_llada import LLaDAModelLM

DEFAULT_MODEL_PATH = "GSAI-ML/LLaDA-8B-Instruct"


def get_generation_function(method_name):
    if method_name == "default":
        return decoding_default
    if method_name == "wino":
        return decoding_wino
    if method_name == "saber":
        return generate_with_saber
    if method_name == "fast":
        return generate_with_dual_cache
    if method_name == "entropy":
        return generate_with_entropy
    if method_name == "margin":
        return generate_with_margin
    if method_name == "remdm":
        return generate_with_remdm
    if method_name == "flip_margin":
        return generate_with_flip_margin
    if method_name == "flip_margin_group":
        return generate_with_flip_margin_group
    raise ValueError(f"Unknown method: {method_name}")


def build_padding_attention_mask(token_mask: torch.Tensor) -> torch.Tensor:
    """Return a 2D (B, T) boolean mask; the model expands it for SDPA."""
    if token_mask.dim() != 2:
        raise ValueError("token_mask must be 2D (B, T)")
    return token_mask.to(dtype=torch.bool)


def load_dataset_flexible(dataset_name: str, split: str, dataset_config: Optional[str]):
    if dataset_config:
        return load_dataset(dataset_name, dataset_config, split=split)
    try:
        return load_dataset(dataset_name, split=split)
    except Exception:
        return load_dataset(dataset_name, "main", split=split)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GSM8K with GSM8K few-shot prompts and LLaDA decoding."
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument(
        "--method",
        default="default",
        choices=[
            "default",
            "wino",
            "saber",
            "fast",
            "entropy",
            "margin",
            "remdm",
            "flip_margin",
            "flip_margin_group",
        ],
    )
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--dataset", default="openai/gsm8k")
    parser.add_argument("--dataset-config", default="main")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--block-length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument(
        "--remasking", default="low_confidence", choices=["low_confidence", "random"]
    )
    parser.add_argument("--mask-id", type=int, default=126336)
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--saber-n", type=int, default=2)
    parser.add_argument("--saber-mu", type=int, default=8)
    parser.add_argument("--wino-threshold", type=float, default=0.6)
    parser.add_argument("--wino-threshold-back", type=float, default=0.9)
    parser.add_argument("--fast-threshold", type=float, default=None)
    parser.add_argument("--fast-factor", type=float, default=1.0)
    parser.add_argument("--remdm-init-unmask-ratio", type=float, default=0.875)
    parser.add_argument("--remdm-unmask-k", type=int, default=1)
    parser.add_argument("--remdm-loop-steps", type=int, default=32)
    parser.add_argument("--remdm-block-length", type=int, default=128)
    parser.add_argument("--flip-unmask-threshold", type=float, default=0.6)
    parser.add_argument("--flip-margin-threshold", type=float, default=0.2)
    parser.add_argument("--flip-threshold", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    args.model_path = args.model_path or os.environ.get("MODEL_PATH") or DEFAULT_MODEL_PATH
    print(f"==> Loading model: {args.model_path}")
    model_cls = LLaDAModelLM
    if args.fast:
        from modeling_llada_fast import LLaDAModelLM as LLaDAModelLMFast

        model_cls = LLaDAModelLMFast
    model = model_cls.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print(f"==> Loading dataset: {args.dataset}")
    dataset = load_dataset_flexible(args.dataset, args.split, args.dataset_config)
    if args.limit is not None and args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    print(f"==> Loaded {len(dataset)} samples")

    output_path = args.output
    if not output_path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results/gsm8k_{args.method}_{timestamp}.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    evaluator = Gsm8kEvaluator()
    correct = 0
    total = len(dataset)
    processed = 0
    batch_size = max(1, int(args.batch_size))
    if batch_size != args.batch_size:
        print(f"==> Adjusted batch size to {batch_size}")

    if args.method in {"wino", "remdm"} and batch_size > 1:
        print(f"==> Method '{args.method}' does not support batching yet; forcing --batch-size 1")
        batch_size = 1
    if args.method == "fast" and batch_size > 1:
        print("==> Method 'fast' does not support batching yet; forcing --batch-size 1")
        batch_size = 1

    with open(output_path, "w", encoding="utf-8") as f:
        original_padding_side = getattr(tokenizer, "padding_side", "right")
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise RuntimeError("Tokenizer has no pad_token_id and no eos_token_id; cannot pad.")
            tokenizer.pad_token = tokenizer.eos_token

        for start in tqdm.tqdm(range(0, total, batch_size), desc=f"gsm8k/{args.method}"):
            end = min(start + batch_size, total)
            docs = [dataset[i] for i in range(start, end)]
            questions = [doc.get("question", "") for doc in docs]
            answers_raw = [doc.get("answer", "") for doc in docs]
            prompt_texts, prompts = [], []
            for q in questions:
                prompt_text, prompt = build_prompt(q, tokenizer)
                prompt_texts.append(prompt_text)
                prompts.append(prompt)

            enc = tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids = enc.input_ids
            prompt_token_mask = enc.attention_mask.to(dtype=torch.bool)
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                prompt_token_mask = prompt_token_mask.cuda()

            attention_mask = None
            if args.method in {"default", "saber", "entropy", "margin", "flip_margin", "flip_margin_group"} and batch_size > 1:
                gen_token_mask = torch.ones(
                    (input_ids.shape[0], args.gen_length),
                    dtype=torch.bool,
                    device=input_ids.device,
                )
                token_mask_full = torch.cat([prompt_token_mask, gen_token_mask], dim=1)
                attention_mask = build_padding_attention_mask(token_mask_full)

            if args.method == "remdm":
                gen_output, steps = generate_with_remdm(
                    model,
                    input_ids,
                    gen_length=args.gen_length,
                    init_unmask_ratio=args.remdm_init_unmask_ratio,
                    unmask_k=args.remdm_unmask_k,
                    loop_steps=args.remdm_loop_steps,
                    temperature=args.temperature,
                    cfg_scale=args.cfg_scale,
                    remasking=args.remasking,
                    mask_id=args.mask_id,
                    tokenizer=tokenizer,
                    block_length=args.remdm_block_length,
                )
            elif args.method == "saber":
                gen_output, steps = generate_with_saber(
                    model,
                    input_ids,
                    n=args.saber_n,
                    mu=args.saber_mu,
                    gen_length=args.gen_length,
                    block_length=args.block_length,
                    temperature=args.temperature,
                    mask_id=args.mask_id,
                    attention_mask=attention_mask,
                )
            elif args.method == "fast":
                gen_output, steps = generate_with_dual_cache(
                    model,
                    input_ids,
                    steps=args.steps,
                    gen_length=args.gen_length,
                    block_length=args.block_length,
                    temperature=args.temperature,
                    remasking=args.remasking,
                    mask_id=args.mask_id,
                    threshold=args.fast_threshold,
                    factor=args.fast_factor,
                )
            elif args.method == "wino":
                gen_output, steps = decoding_wino(
                    model,
                    input_ids,
                    gen_length=args.gen_length,
                    block_length=args.block_length,
                    temperature=args.temperature,
                    mask_id=args.mask_id,
                    threshold=args.wino_threshold,
                    threshold_back=args.wino_threshold_back,
                )
            elif args.method == "flip_margin":
                gen_output, steps = generate_with_flip_margin(
                    model,
                    input_ids,
                    steps=args.steps,
                    gen_length=args.gen_length,
                    block_length=args.block_length,
                    temperature=args.temperature,
                    cfg_scale=args.cfg_scale,
                    remasking=args.remasking,
                    mask_id=args.mask_id,
                    attention_mask=attention_mask,
                    unmask_threshold=args.flip_unmask_threshold,
                    margin_threshold=args.flip_margin_threshold,
                    flip_threshold=args.flip_threshold,
                )
            elif args.method == "flip_margin_group":
                gen_output, steps = generate_with_flip_margin_group(
                    model,
                    input_ids,
                    steps=args.steps,
                    gen_length=args.gen_length,
                    block_length=args.block_length,
                    temperature=args.temperature,
                    cfg_scale=args.cfg_scale,
                    remasking=args.remasking,
                    mask_id=args.mask_id,
                    attention_mask=attention_mask,
                    unmask_threshold=args.flip_unmask_threshold,
                    margin_threshold=args.flip_margin_threshold,
                    flip_threshold=args.flip_threshold,
                )
            else:
                generation_fn = get_generation_function(args.method)
                gen_output, steps = generation_fn(
                    model,
                    input_ids,
                    steps=args.steps,
                    gen_length=args.gen_length,
                    block_length=args.block_length,
                    temperature=args.temperature,
                    cfg_scale=args.cfg_scale,
                    remasking=args.remasking,
                    mask_id=args.mask_id,
                    attention_mask=attention_mask,
                )

            gen_texts = tokenizer.batch_decode(
                gen_output[:, input_ids.shape[1]:], skip_special_tokens=True
            )

            for offset, (question, answer_raw, prompt_text, gen_text) in enumerate(
                zip(questions, answers_raw, prompt_texts, gen_texts)
            ):
                idx = start + offset
                prediction = gsm8k_postprocess(gen_text)
                answer = gsm8k_dataset_postprocess(answer_raw) if answer_raw else answer_raw
                is_correct = evaluator.is_equal(prediction, answer)
                if is_correct:
                    correct += 1
                processed += 1
                if args.log_every and processed % args.log_every == 0:
                    running_acc = 100.0 * correct / processed
                    print(
                        f"[{processed}/{total}] Accuracy: {running_acc:.2f}% ({correct}/{processed})",
                        flush=True,
                    )

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
                    "method": args.method,
                }
                f.write(json.dumps(record, ensure_ascii=True) + "\n")

        tokenizer.padding_side = original_padding_side

    accuracy = 100.0 * correct / total if total else 0.0
    print(f"Results saved to: {output_path}")
    print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")


if __name__ == "__main__":
    main()
