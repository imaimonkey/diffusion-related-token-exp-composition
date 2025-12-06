import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import argparse
import tqdm
from datasets import load_dataset
import torch
import yaml
import jsonlines
from modeling_dream import DreamModel
from transformers import AutoModel, AutoTokenizer
import dataset_utils
from human_eval.evaluation import evaluate_functional_correctness
import tempfile
from dataset_utils.eval_correctness_mbpp.evaluation import evaluate_functional_correctness
from dataset_utils.eval_humaneval.all_evaluate import evaluate_solution, evaluate_solution_et
import datetime

def main():
    parser = argparse.ArgumentParser(description="Unified Config-driven Evaluation Script for Language Models")
    parser.add_argument("--config", type=str, required=True, help="Path to the dataset config YAML file (e.g., configs/gsm8k.yaml)")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(f"==> Loaded config for: {config['dataset_name']}")
    method = config.get('method', 'default')
    model_path = config['model_path']
    print(f"==> Loading model: {model_path}")
    if method == 'default':
        model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    else:
        model = DreamModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = model.to("cuda").eval()
    run_single_task_evaluation(config, model, tokenizer)


def run_single_task_evaluation(config, model, tokenizer):
    dataset_name = config['dataset_name']
    print(f"==> Running Single-Task Evaluation for {dataset_name}")
    dataset_cfg = config['dataset_config']
    doc_to_text_fn = getattr(dataset_utils, dataset_cfg['doc_to_text_fn'])
    extract_answer_fn = getattr(dataset_utils, dataset_cfg.get('extract_answer_fn')) if dataset_cfg.get('extract_answer_fn') else None
    
    gen_cfg = config['generation_args']
    gen_length = gen_cfg.get('gen_length', 256)
    definded_steps = gen_cfg.get('steps', 256)
    temperature = gen_cfg.get('temperature', 0)
    method_name = config['method']
    method_params = config.get('method_args', {}).get(method_name, {})
    print(f"==> Loading dataset...")
    loader_type = dataset_cfg.get('data_loader', 'huggingface')
    dataset_path = os.path.join(config['data_root'], dataset_cfg['load_dataset_args']['path'])
    
    if loader_type == 'huggingface':
        dataset_name_hf = dataset_cfg['load_dataset_args'].get('name')
        dataset = load_dataset(dataset_path, dataset_name_hf, trust_remote_code=True)[dataset_cfg['split']]
    elif dataset_name == 'mbpp':
        loader_fn = getattr(dataset_utils, dataset_cfg['loader_fn'])
        dataset = list(loader_fn(dataset_path))
    elif dataset_name == 'livecodebench':
        dataset = load_dataset("json",data_files = dataset_path, split="train")
    print(f"==> Loaded {len(dataset)} samples from {dataset_name} dataset.")

    print("==> Performing warm-up run with one sample...")
    warmup_doc = dataset[2]
    trailing_prompt = "" 

    context, _, trailing_prompt = doc_to_text_fn(warmup_doc)

    prompt = tokenizer.apply_chat_template(
        context,
        add_generation_prompt=True,
        tokenize=False
    ) + trailing_prompt

    inputs = tokenizer(prompt, return_tensors="pt")
    print(f"==> Prompt: {prompt}")
    input_ids_d = inputs.input_ids.to(device="cuda")
    attention_mask = inputs.attention_mask.to(device="cuda") if getattr(inputs, "attention_mask", None) is not None else None
    output, steps = model.diffusion_generate(
                    input_ids_d,
                    attention_mask=attention_mask,
                    max_new_tokens=gen_length,
                    output_history=True,
                    return_dict_in_generate=True,
                    steps=definded_steps,
                    temperature=temperature,
                    top_p=0.95,
                    alg="entropy",
                    alg_temp=0.,
                )
    
    generations = [ tokenizer.decode(g[len(p) :].tolist()) for p, g in zip(input_ids_d, output.sequences) ] 
    print(generations[0].split(tokenizer.eos_token)[0]) 
    print("==> Warm-up complete.")

    # --- Warm-up complete ---

    total_len = len(dataset)
    raw_outputs, correct_count, total_count, total_steps = [], 0, 0, 0
    trailing_prompt = "" 
    for i in tqdm.tqdm(range(total_len), desc=f"Evaluating {dataset_name} with method '{method_name}'"):
        doc = dataset[i]
        
        context, gt_doc, trailing_prompt = doc_to_text_fn(doc)
        
        prompt = tokenizer.apply_chat_template(
            context,
            add_generation_prompt=True,
            tokenize=False
        ) + trailing_prompt

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids_d = inputs.input_ids.to(device="cuda")
        attention_mask = inputs.attention_mask.to(device="cuda") if getattr(inputs, "attention_mask", None) is not None else None
        output, steps = model.diffusion_generate(
                        input_ids_d,
                        attention_mask=attention_mask,
                        max_new_tokens=gen_length,
                        output_history=True,
                        return_dict_in_generate=True,
                        steps=definded_steps,
                        temperature=temperature,
                        top_p=0.95,
                        alg="entropy",
                        alg_temp=0.,
                    )

        generations = [ tokenizer.decode(g[len(p) :].tolist()) for p, g in zip(input_ids_d,output.sequences) ] 

        gen_str = generations[0].split(tokenizer.eos_token)[0]
        
        total_steps += steps
        
        pred, is_correct = None, "N/A"
        result_item = {'index': i, 'completion':'','full_response': gen_str, 'steps': steps}


        if dataset_name == 'mbpp':
            gen_str = f"```python\n" + gen_str
            gen_code = extract_answer_fn(gen_str, doc['entry_point'])
            result_item['completion'] = gen_code
        else:
            pred = extract_answer_fn(gen_str, doc)
            result_item['completion'] = pred
        if 'task_id' in gt_doc: result_item['task_id'] = gt_doc['task_id']
        if 'question_id' in gt_doc: result_item["question_id"] = gt_doc["question_id"]
        raw_outputs.append(result_item)
    output_path = f"./results/{config['dataset_name']}_{method_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    print(f"Results saved in .jsonl format to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with jsonlines.Writer(open(output_path, "w", encoding="utf-8")) as writer:
        writer.write_all(raw_outputs)

    final_metrics = {}
    output_formatter = config['dataset_config'].get('output_formatter', 'default')
    if output_formatter == 'humaneval':
        if evaluate_functional_correctness is None:
            print("Warning: 'human-eval' library not found. Skipping functional correctness evaluation.")
        else:
            print("\n==> Generations complete. Calling official evaluation script...")
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".jsonl") as temp_f:
                for item in raw_outputs:
                    temp_f.write(json.dumps(item) + "\n")
                    temp_file_path = temp_f.name
            if dataset_name == 'humaneval':
                problem_file = config['dataset_config'].get('problem_file', "./data/humaneval/HumanEval.jsonl")
                problem_file_et = config['dataset_config'].get('problem_file_et', './data/humaneval/HumanEval_ET.jsonl')
                final_metrics = evaluate_solution(raw_outputs,problem_file=problem_file)
                print(f"ET Evaluation...{problem_file_et}")
                final_metrics_et = evaluate_solution_et(raw_outputs,problem_file=problem_file_et)
            elif dataset_name == 'mbpp':
                problem_file = config['dataset_config'].get('problem_file', "./data/mbpp/mbpp_sanitized.jsonl")
                problem_file_et = config['dataset_config'].get('problem_file_et', "./data/mbpp/MBPP_ET.jsonl")
                final_metrics = evaluate_functional_correctness(temp_file_path,problem_file=problem_file,is_mbpp=True)
                final_metrics_et = evaluate_functional_correctness(temp_file_path,problem_file=problem_file_et,is_mbpp=True)
            elif dataset_name == 'livecodebench':
                problem_file = config['dataset_config'].get('problem_file', 'default')    
        os.unlink(temp_file_path)
    external_metrics = final_metrics

    if dataset_name != 'livecodebench':
        print("\n--- Evaluation Summary ---")
        print(f"Dataset: {config['dataset_name']}")
        print(f"Method: {method_name}")
        if external_metrics:
            accuracy = external_metrics['pass@1']
        else:
            print("Accuracy: N/A (no external metrics available)")
        print(f"Accuracy: {accuracy:.4f}" if isinstance(accuracy, float) else f"Accuracy: {accuracy}")

        if final_metrics_et is not None:
            print("\n--- ET Evaluation Summary ---")
            print(f"Accuracy_et: {final_metrics_et['pass@1']:.4f}")
        avg_steps = total_steps / total_len if total_len > 0 else 0
        print(f"Average Steps: {avg_steps:.2f}")


if __name__ == "__main__":
    main()