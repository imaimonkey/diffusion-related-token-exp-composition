#!/usr/bin/env python3
import argparse
import json
import os

import torch
from transformers import AutoTokenizer

from decoding import decoding_default
from gsm_common import Gsm8kEvaluator, build_prompt, gsm8k_dataset_postprocess, gsm8k_postprocess
from modeling_llada import LLaDAModelLM

DEFAULT_MODEL_PATH = "GSAI-ML/LLaDA-8B-Instruct"


def format_state(tokens, max_len):
    shown = tokens[:max_len]
    return " ".join(shown)


def token_to_display(tokenizer, token_id: int) -> str:
    try:
        tok = tokenizer.convert_ids_to_tokens([int(token_id)])[0]
    except Exception:
        tok = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    # Make tokenizer artifacts human-readable:
    # - GPT2/byte-level BPE uses "Ġ" for a leading space and "Ċ" for newline.
    # - SentencePiece often uses "▁" for a leading space.
    tok = tok.replace("Ġ", "␠").replace("▁", "␠").replace("Ċ", "↵")
    tok = tok.replace("\n", "↵")
    return f"[{tok}]"


def trace_one(
    model,
    tokenizer,
    question: str,
    *,
    steps: int,
    gen_length: int,
    block_length: int,
    temperature: float,
    cfg_scale: float,
    remasking: str,
    mask_id: int,
    view_len,
    max_steps,
):
    prompt_text, prompt = build_prompt(question, tokenizer)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    out = decoding_default(
        model,
        input_ids,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        cfg_scale=cfg_scale,
        remasking=remasking,
        mask_id=mask_id,
        return_trace=True,
    )
    gen_output, _steps_used, trace = out

    if view_len is None:
        view_len = gen_length
    view_len = int(view_len)

    state = ["[mask]"] * min(view_len, gen_length)
    print(f"prompt_text: {prompt_text.replace(chr(10), ' ')}")
    trace_steps = len(trace)
    print(f"gen_length={gen_length} block_length={block_length} steps={steps}")
    if max_steps is None:
        print(f"trace_steps={trace_steps} (printing all)")
        max_steps = trace_steps
    else:
        print(f"trace_steps={trace_steps} (printing up to max_steps={max_steps})")
    print(f"step0: {format_state(state, view_len)}")

    for step_idx, step_info in enumerate(trace[:max_steps], start=1):
        # step_info: list over batch; we run with batch=1
        info = step_info[0] if step_info else {"positions": [], "token_ids": []}
        positions = info.get("positions", [])
        token_ids = info.get("token_ids", [])

        applied = 0
        for pos, tid in zip(positions, token_ids):
            if 0 <= int(pos) < len(state):
                state[int(pos)] = token_to_display(tokenizer, tid)
                applied += 1

        suffix = f" (unmasked={len(positions)})"
        print(f"step{step_idx}: {format_state(state, view_len)}{suffix}")

    gen_text = tokenizer.decode(gen_output[0, input_ids.shape[1] :], skip_special_tokens=True)
    return gen_text


def main():
    parser = argparse.ArgumentParser(
        description="Trace per-step unmasking for wrong GSM8K samples from a saved results JSONL."
    )
    parser.add_argument("--results", required=True, help="Path to eval output JSONL (from eval_gsm8k.py)")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--num", type=int, default=10, help="How many wrong samples to trace")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--gen-length", type=int, default=256)
    parser.add_argument("--block-length", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--remasking", default="low_confidence", choices=["low_confidence", "random"])
    parser.add_argument("--mask-id", type=int, default=126336)
    parser.add_argument(
        "--view-len",
        type=int,
        default=None,
        help="How many generated tokens to display (default: show all gen tokens)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="How many steps to print (default: print all)",
    )
    args = parser.parse_args()

    args.model_path = args.model_path or os.environ.get("MODEL_PATH") or DEFAULT_MODEL_PATH

    print(f"==> Loading model: {args.model_path}")
    model = LLaDAModelLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    evaluator = Gsm8kEvaluator()

    wrong = []
    with open(args.results, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("correct", False):
                wrong.append(r)
                if len(wrong) >= args.num:
                    break

    print(f"==> Loaded wrong samples: {len(wrong)} (requested {args.num})")

    for k, r in enumerate(wrong, start=1):
        idx = r.get("index")
        question = r.get("question", "")
        answer_raw = r.get("answer_raw", "")
        answer = r.get("answer") or gsm8k_dataset_postprocess(answer_raw)

        print("\n" + "=" * 80)
        print(f"[{k}/{len(wrong)}] index={idx}")
        print(f"gold={answer} pred={r.get('prediction')}")
        print("-" * 80)

        gen_text = trace_one(
            model,
            tokenizer,
            question,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            remasking=args.remasking,
            mask_id=args.mask_id,
            view_len=args.view_len,
            max_steps=args.max_steps,
        )

        pred = gsm8k_postprocess(gen_text)
        ok = evaluator.is_equal(pred, str(answer))
        print("-" * 80)
        print(f"final_pred={pred} correct={ok}")


if __name__ == "__main__":
    main()
