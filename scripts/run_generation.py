"""Script for generating answers from LLM models on benchmark datasets."""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import shortuuid
import tiktoken
from tqdm import tqdm

from utils.completion import (
    API_ERROR_OUTPUT,
    get_endpoint,
    load_questions,
    make_config,
    registered_api_completion,
    reorg_answer_file,
)


def load_existing_uids(answer_file: str) -> Set[str]:
    """Load UIDs of already processed answers.
    
    Args:
        answer_file: Path to answer file.
    
    Returns:
        Set of existing UIDs.
    """
    uids = set()
    if not os.path.exists(answer_file):
        return uids
    
    with open(answer_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                if "uid" in entry:
                    uids.add(entry["uid"])
            except json.JSONDecodeError:
                continue
    
    return uids


def apply_thinking_mode(
    prompt: str,
    prompt_prefix: Optional[str] = None,
    prompt_suffix: Optional[str] = None,
) -> str:
    """Apply prefix/suffix tokens to prompt based on model settings.
    
    Args:
        prompt: Original prompt text.
        prompt_prefix: Prefix to add at the beginning.
        prompt_suffix: Suffix to add at the end.
    
    Returns:
        Modified prompt.
    """
    if prompt_prefix:
        prompt = prompt_prefix + prompt
    if prompt_suffix:
        prompt = prompt.rstrip() + prompt_suffix
    
    return prompt


def get_answer(
    question: Dict,
    answer_file: str,
    settings: Dict,
) -> None:
    """Get model answer via API and save to file.
    
    Args:
        question: Question dict with 'uid' and 'prompt'.
        answer_file: Path to save answer.
        settings: Model config from api_config.yaml.
    """
    messages = []
    if "sys_prompt" in settings:
        messages.append({"role": "system", "content": settings["sys_prompt"]})

    prompt = question["prompt"]
    prompt_prefix = settings.get("prompt_prefix")
    prompt_suffix = settings.get("prompt_suffix")
    prompt = apply_thinking_mode(prompt, prompt_prefix, prompt_suffix)
    
    messages.append({"role": "user", "content": prompt})

    api_func = registered_api_completion[settings["api_type"]]
    kwargs = {
        **settings,
        "api_dict": get_endpoint(settings["endpoints"]),
        "messages": messages,
    }
    output = api_func(**kwargs)
    
    if output is API_ERROR_OUTPUT or output is None:
        return
    
    messages.append({"role": "assistant", "content": output})

    try:
        encoding = tiktoken.encoding_for_model("gpt-4o")
        token_len = len(encoding.encode(output["answer"], disallowed_special=()))
    except Exception:
        token_len = 0

    answer_dict = {
        "uid": question["uid"],
        "ans_id": shortuuid.uuid(),
        "model": settings["model"],
        "messages": messages,
        "tstamp": time.time(),
        "metadata": {
            "token_len": token_len,
            "timestamp": time.time(),
        },
    }

    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    with open(answer_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(answer_dict, ensure_ascii=False) + "\n")


def run_generation(
    bench_list: List[str],
    model_list: List[str],
    api_config_path: str,
) -> None:
    """Generate answers for all benchmark and model combinations.
    
    Args:
        bench_list: List of benchmark names.
        model_list: List of model names.
        api_config_path: Path to api_config.yaml.
    """
    api_endpoints = make_config(api_config_path)
    
    for bench_name in bench_list:
        bench_config_path = f"configs/bench_configs/{bench_name}.yaml"
        
        if not os.path.exists(bench_config_path):
            print(f"[SKIP] Benchmark config not found: {bench_config_path}")
            continue
        
        bench_config = make_config(bench_config_path)
        question_file = bench_config["prompt_dir"]
        
        if not os.path.exists(question_file):
            print(f"[SKIP] Question file not found: {question_file}")
            continue
        
        questions = load_questions(question_file)
        
        for model_name in model_list:
            if model_name not in api_endpoints:
                print(f"[SKIP] Model not found in api_config.yaml: {model_name}")
                continue
            
            endpoint_settings = api_endpoints[model_name]
            answer_file = os.path.join(bench_config["answer_dir"], f"{model_name}.jsonl")
            
            existing_uids = load_existing_uids(answer_file)
            questions_to_answer = [
                q for q in questions if q["uid"] not in existing_uids
            ]
            
            if not questions_to_answer:
                print(f"[{bench_name}] [{model_name}] ✓ All questions answered")
                continue
            
            print(f"\n{'='*70}")
            print(f"[{bench_name}] [{model_name}] Generating answers")
            print(f"Questions: {len(questions_to_answer)}/{len(questions)}")
            print(f"Model: {endpoint_settings['model']}")
            print(f"Output: {answer_file}")
            print(f"{'='*70}\n")
            
            parallel = endpoint_settings.get("parallel", 1)
            
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = [
                    executor.submit(get_answer, q, answer_file, endpoint_settings)
                    for q in questions_to_answer
                ]
                
                for _ in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc=f"{bench_name}/{model_name}",
                ):
                    pass
            
            reorg_answer_file(answer_file)
            print(f"[{bench_name}] [{model_name}] ✓ Done\n")


def main() -> None:
    """Main entry point for answer generation."""
    parser = argparse.ArgumentParser(
        description="Generate answers for all benchmarks and models"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/gen_answer_config.yaml",
        help="Path to config with bench_list and model_list",
    )
    parser.add_argument(
        "--api-config",
        type=str,
        default="configs/api_config.yaml",
        help="Path to API endpoints config",
    )
    args = parser.parse_args()
    
    config = make_config(args.config)
    bench_list = config.get("bench_list", [])
    model_list = config.get("model_list", [])
    
    if not bench_list or not model_list:
        print("ERROR: bench_list or model_list is empty")
        exit(1)
    
    print(f"\n{'='*70}")
    print("🚀 Starting answer generation")
    print(f"{'='*70}")
    print(f"Benchmarks: {len(bench_list)}")
    print(f"Models: {len(model_list)}")
    print(f"Total combinations: {len(bench_list) * len(model_list)}\n")
    
    run_generation(bench_list, model_list, args.api_config)
    
    print(f"\n{'='*70}")
    print("✓ Generation complete")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
