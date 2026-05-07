"""Script for evaluating model answers using a judge model."""

import argparse
import json
import os
import re
from typing import Dict, List, Optional

from tqdm import tqdm

from utils.completion import (
    API_ERROR_OUTPUT,
    get_endpoint,
    make_config,
    registered_api_completion,
)


def get_score(
    output_str: str,
    regex_patterns: Optional[List[str]] = None,
) -> Optional[str]:
    """Extract score (0 or 1) from judge's response using regex patterns.
    
    Args:
        output_str: Judge's full response.
        regex_patterns: List of regex patterns to match.
    
    Returns:
        "0" or "1" if found, None otherwise.
    """
    if not regex_patterns:
        regex_patterns = [
            r"\*\*boxed\s*$$\s*([01])\s*$$\s*\*\*",
            r"boxed$$\{([01])\}$$",
            r"boxed\s*([01])\s*",
        ]
    
    for pattern in regex_patterns:
        try:
            match = re.search(pattern, output_str)
            if match and match.group(1) in ["0", "1"]:
                return match.group(1)
        except Exception:
            continue
    
    return None


def pairwise_judgment(
    question: Dict,
    answer: Dict,
    configs: Dict,
    settings: Dict,
) -> Optional[Dict]:
    """Evaluate model answer using judge model.
    
    Args:
        question: Question dict with 'prompt' and 'answer'.
        answer: Model answer with 'messages'.
        configs: Judge config.
        settings: Judge model settings.
    
    Returns:
        Dict with 'score' and 'judgment', or None on error.
    """
    model_answer_content = answer["messages"][-1]["content"]
    if isinstance(model_answer_content, dict):
        model_answer = model_answer_content.get("answer", str(model_answer_content))
    else:
        model_answer = str(model_answer_content)
    
    user_prompt = configs["prompt_template"].format(
        question=question.get("prompt", ""),
        ground_truth=question.get("answer", ""),
        candidate_answer=model_answer,
    )
    
    messages = [{"role": "user", "content": user_prompt}]
    
    kwargs = {
        **settings,
        "api_dict": get_endpoint(settings["endpoints"]),
        "messages": messages,
        "temperature": configs["temperature"],
        "max_tokens": configs["max_tokens"],
    }
    
    api_func = registered_api_completion[settings["api_type"]]
    output = api_func(**kwargs)
    
    if output is API_ERROR_OUTPUT or output is None:
        return None

    if isinstance(output.get("answer"), str) and output["answer"] in ["0", "1"]:
        score = output["answer"]
    else:
        score = get_score(output.get("answer", ""), configs.get("regex_patterns", []))

    return {
        "score": score,
        "judgment": output,
        "prompt": messages,
    }


def judge_answer(
    question: Dict,
    answer: Dict,
    configs: Dict,
    settings: Dict,
    output_file: str,
) -> None:
    """Process judgment for single answer and save to file.
    
    Args:
        question: Question dict.
        answer: Answer dict.
        configs: Judge config.
        settings: Judge model settings.
        output_file: Output file path.
    """
    output = {
        "uid": question["uid"],
        "category": question.get("category", "unknown"),
        "judge": configs["judge_model"],
        "model": answer["model"],
        "games": [],
    }
    
    if question.get("subcategory"):
        output["subcategory"] = question["subcategory"]
    
    result = pairwise_judgment(question, answer, configs, settings)
    if result:
        output["games"].append(result)
    
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(output, ensure_ascii=False) + "\n")


def run_judgment(
    bench_list: List[str],
    model_list: List[str],
    judge_config_path: str,
    api_config_path: str,
) -> None:
    """Run judgment for all benchmark and model combinations.
    
    Args:
        bench_list: List of benchmark names.
        model_list: List of model names.
        judge_config_path: Path to judge_config.yaml.
        api_config_path: Path to api_config.yaml.
    """
    judge_config = make_config(judge_config_path)
    api_endpoints = make_config(api_config_path)
    
    judge_model = judge_config["judge_model"]
    
    if judge_model not in api_endpoints:
        print(f"ERROR: Judge model not found: {judge_model}")
        exit(1)
    
    endpoint_settings = api_endpoints[judge_model]
    
    for bench_name in bench_list:
        bench_config_path = f"configs/bench_configs/{bench_name}.yaml"
        
        if not os.path.exists(bench_config_path):
            print(f"[SKIP] Benchmark config not found: {bench_config_path}")
            continue
        
        bench_config = make_config(bench_config_path)
        question_file = bench_config["prompt_dir"]
        answer_dir = bench_config["answer_dir"]
        
        if not os.path.exists(question_file):
            print(f"[SKIP] Question file not found: {question_file}")
            continue
        
        questions = []
        with open(question_file, encoding="utf-8") as f:
            for line in f:
                try:
                    questions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        if not questions:
            print(f"[SKIP] No questions in {question_file}")
            continue
        
        for model_name in model_list:
            answer_file = os.path.join(answer_dir, f"{model_name}.jsonl")
            
            if not os.path.exists(answer_file):
                print(f"[SKIP] [{bench_name}] [{model_name}] Answer file not found")
                continue
            
            model_answers = {}
            answer_count = 0
            with open(answer_file, encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        model_answers[data["uid"]] = data
                        answer_count += 1
                    except json.JSONDecodeError:
                        continue
            
            if answer_count != len(questions):
                print(f"[SKIP] [{bench_name}] [{model_name}] "
                      f"Answer count mismatch: {answer_count} vs {len(questions)}")
                continue
            
            output_dir = f"data/{bench_name}/model_judgment/{judge_model}"
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{model_name}.jsonl")
            
            existing_uids = set()
            if os.path.exists(output_file):
                with open(output_file, encoding="utf-8") as f:
                    for line in f:
                        try:
                            existing_uids.add(json.loads(line)["uid"])
                        except json.JSONDecodeError:
                            continue
            
            questions_to_judge = [q for q in questions if q["uid"] not in existing_uids]
            
            if not questions_to_judge:
                print(f"[{bench_name}] [{model_name}] ✓ All questions judged")
                continue
            
            print(f"\n{'='*70}")
            print(f"[{bench_name}] [{model_name}] Running judgment")
            print(f"Questions: {len(questions_to_judge)}/{len(questions)}")
            print(f"Judge: {judge_model}")
            print(f"Output: {output_file}")
            print(f"{'='*70}\n")
            
            parallel = endpoint_settings.get("parallel", 1)
            
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
                futures = [
                    executor.submit(
                        judge_answer,
                        q,
                        model_answers[q["uid"]],
                        judge_config,
                        endpoint_settings,
                        output_file,
                    )
                    for q in questions_to_judge
                    if q["uid"] in model_answers
                ]
                
                for _ in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc=f"{bench_name}/{model_name}",
                ):
                    pass
            
            print(f"[{bench_name}] [{model_name}] ✓ Done\n")


def main() -> None:
    """Main entry point for judgment."""
    parser = argparse.ArgumentParser(
        description="Run judge evaluation for all benchmarks and models"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/gen_answer_config.yaml",
        help="Path to config with bench_list and model_list",
    )
    parser.add_argument(
        "--judge-config",
        type=str,
        default="configs/judge_config.yaml",
        help="Path to judge config",
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
    print("⚖️  Starting judgment evaluation")
    print(f"{'='*70}")
    print(f"Benchmarks: {len(bench_list)}")
    print(f"Models: {len(model_list)}")
    print(f"Judge: {config.get('judge_model', 'unknown')}\n")
    
    run_judgment(bench_list, model_list, args.judge_config, args.api_config)
    
    print(f"\n{'='*70}")
    print("✓ Judgment complete")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
