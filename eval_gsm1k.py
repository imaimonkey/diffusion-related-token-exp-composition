#!/usr/bin/env python3
import argparse
import datetime
import json
import os
import re

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
)
from modeling_llada import LLaDAModelLM
from modeling_llada_fast import LLaDAModelLM as LLaDAModelLMFast

def gsm8k_dataset_postprocess(text: str) -> str:
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
    if text is None:
        return "NULL"
    text = str(text)
    text = text.split("Question:")[0]
    numbers = re.findall(r"\-?\d+\.\d+|\-?\d+", text)
    if not numbers:
        return "NULL"
    return numbers[-1]


class Gsm8kEvaluator:
    def is_equal(self, pred, refer):
        try:
            if pred == refer or abs(float(pred) - int(refer)) < 1e-6:
                return True
        except Exception:
            pass
        return False

DEFAULT_MODEL_PATH = "GSAI-ML/LLaDA-8B-Instruct"

# Template source: opencompass/opencompass/configs/datasets/gsm8k/gsm8k_gen_17d0dc.py
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
    prompt_text = QUESTION_PROMPT_TEMPLATE.format(question=question)
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError("Tokenizer does not support apply_chat_template.")
    context = FEWSHOT_CONTEXT + [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(
        context, add_generation_prompt=True, tokenize=False
    )
    return prompt_text, prompt


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
    raise ValueError(f"Unknown method: {method_name}")

def build_padding_attention_mask(token_mask: torch.Tensor) -> torch.Tensor:
    """Return a 2D (B, T) boolean mask; the model expands it for SDPA."""
    if token_mask.dim() != 2:
        raise ValueError("token_mask must be 2D (B, T)")
    return token_mask.to(dtype=torch.bool)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GSM1K with GSM8K few-shot prompts and LLaDA decoding."
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--method", default="default", choices=[
        "default",
        "wino",
        "saber",
        "fast",
        "entropy",
        "margin",
        "remdm",
    ])
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--dataset", default="ScaleAI/gsm1k")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--block-length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence", choices=["low_confidence", "random"])
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
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    args.model_path = args.model_path or os.environ.get("MODEL_PATH") or DEFAULT_MODEL_PATH
    print(f"==> Loading model: {args.model_path}")
    model_cls = LLaDAModelLMFast if args.fast else LLaDAModelLM
    model = model_cls.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print(f"==> Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split=args.split)
    if args.limit is not None and args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    print(f"==> Loaded {len(dataset)} samples")

    output_path = args.output
    if not output_path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"results/gsm1k_{args.method}_{timestamp}.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    evaluator = Gsm8kEvaluator()
    correct = 0
    total = len(dataset)
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

        for start in tqdm.tqdm(range(0, total, batch_size), desc=f"gsm1k/{args.method}"):
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
            if args.method in {"default", "saber", "entropy", "margin"} and batch_size > 1:
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
                answer = (
                    gsm8k_dataset_postprocess(answer_raw) if answer_raw else answer_raw
                )
                is_correct = evaluator.is_equal(prediction, answer)
                if is_correct:
                    correct += 1

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
