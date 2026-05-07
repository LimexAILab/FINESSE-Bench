"""Script for computing per-benchmark metrics with bootstrap confidence intervals."""

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from utils.completion import make_config


def load_scores_from_judgments(judgment_file: str) -> Optional[List[int]]:
    """Load binary scores from judgment file.
    
    Args:
        judgment_file: Path to judgment jsonl file.
    
    Returns:
        List of scores (0 or 1), or None if file doesn't exist.
    """
    if not os.path.exists(judgment_file):
        return None
    
    scores = []
    with open(judgment_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                judgment = json.loads(line)
                if judgment.get("games") and len(judgment["games"]) > 0:
                    score = judgment["games"][0].get("score")
                    if score is not None:
                        try:
                            score_int = int(score)
                            if score_int in [0, 1]:
                                scores.append(score_int)
                        except (ValueError, TypeError):
                            pass
            except json.JSONDecodeError:
                continue
    
    return scores if scores else None


def bootstrap_accuracy(
    scores: List[int],
    num_iterations: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Dict:
    """Compute accuracy with bootstrap confidence intervals.
    
    Args:
        scores: List of binary scores (0 or 1).
        num_iterations: Number of bootstrap iterations.
        ci_level: Confidence level (0.95 for 95%).
        seed: Random seed for reproducibility.
    
    Returns:
        Dict with 'value', 'ci_lower', 'ci_upper', 'std_error'.
    """
    np.random.seed(seed)
    n = len(scores)
    scores_array = np.array(scores)
    
    bootstrap_accuracies = []
    for _ in range(num_iterations):
        sample_indices = np.random.randint(0, n, size=n)
        sample_scores = scores_array[sample_indices]
        bootstrap_accuracies.append(sample_scores.mean())
    
    bootstrap_accuracies = np.array(bootstrap_accuracies)
    
    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_accuracies, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_accuracies, 100 * (1 - alpha / 2))
    
    return {
        "value": float(scores_array.mean()),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "ci_level": ci_level,
        "std_error": float(bootstrap_accuracies.std()),
    }


def count_questions(question_file: str) -> int:
    """Count number of questions in benchmark.
    
    Args:
        question_file: Path to question.jsonl.
    
    Returns:
        Number of questions.
    """
    if not os.path.exists(question_file):
        return 0
    
    count = 0
    with open(question_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    
    return count


def compute_metrics_for_model(
    bench_name: str,
    model_name: str,
    judge_model: str,
    question_file: str,
    judgment_file: str,
    output_file: str,
    num_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42,
    overwrite: bool = False,
) -> Tuple[bool, str]:
    """Compute metrics for single model and save to JSON.
    
    Args:
        bench_name: Benchmark name.
        model_name: Model name.
        judge_model: Judge model name.
        question_file: Path to questions.
        judgment_file: Path to judgments.
        output_file: Path to save metrics.
        num_bootstrap: Number of bootstrap iterations.
        ci_level: Confidence level.
        seed: Random seed.
        overwrite: Whether to overwrite existing metrics.
    
    Returns:
        Tuple (success: bool, message: str).
    """
    if not overwrite and os.path.exists(output_file):
        return False, "metrics already computed"
    
    if not os.path.exists(judgment_file):
        return False, "judgment file not found"
    
    if not os.path.exists(question_file):
        return False, "question file not found"
    
    num_questions = count_questions(question_file)
    if num_questions == 0:
        return False, "no questions in benchmark"
    
    scores = load_scores_from_judgments(judgment_file)
    if not scores:
        return False, "no valid scores"
    
    accuracy_metrics = bootstrap_accuracy(
        scores,
        num_iterations=num_bootstrap,
        ci_level=ci_level,
        seed=seed,
    )
    
    result = {
        "bench_name": bench_name,
        "model_name": model_name,
        "judge_model": judge_model,
        "num_questions": num_questions,
        "num_scored": len(scores),
        "metrics": {"accuracy": accuracy_metrics},
        "bootstrap": {
            "num_iterations": num_bootstrap,
            "seed": seed,
            "method": "percentile",
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    return True, "success"


def run_compute_metrics(
    bench_list: List[str],
    model_list: List[str],
    judge_model: str,
    num_bootstrap: int = 10000,
    ci_level: float = 0.95,
    seed: int = 42,
    overwrite: bool = False,
) -> None:
    """Compute metrics for all benchmark and model combinations.
    
    Args:
        bench_list: List of benchmarks.
        model_list: List of models.
        judge_model: Judge model name.
        num_bootstrap: Number of bootstrap iterations.
        ci_level: Confidence level.
        seed: Random seed.
        overwrite: Whether to overwrite existing results.
    """
    total = len(bench_list) * len(model_list)
    processed, skipped = 0, 0
    
    print(f"\nComputing metrics: {len(bench_list)} benchmarks × {len(model_list)} models")
    print(f"Judge: {judge_model} | Bootstrap: {num_bootstrap} iterations\n")
    
    for bench_name in bench_list:
        bench_config_path = f"configs/bench_configs/{bench_name}.yaml"
        
        if not os.path.exists(bench_config_path):
            skipped += len(model_list)
            continue
        
        bench_config = make_config(bench_config_path)
        question_file = bench_config.get("prompt_dir")
        
        if not question_file or not os.path.exists(question_file):
            skipped += len(model_list)
            continue
        
        for model_name in tqdm(model_list, desc=bench_name):
            judgment_file = f"data/{bench_name}/model_judgment/{judge_model}/{model_name}.jsonl"
            output_file = f"data/{bench_name}/metrics/{judge_model}/{model_name}.json"
            
            success, _ = compute_metrics_for_model(
                bench_name=bench_name,
                model_name=model_name,
                judge_model=judge_model,
                question_file=question_file,
                judgment_file=judgment_file,
                output_file=output_file,
                num_bootstrap=num_bootstrap,
                ci_level=ci_level,
                seed=seed,
                overwrite=overwrite,
            )
            
            if success:
                processed += 1
            else:
                skipped += 1
    
    print(f"\n✓ Processed: {processed}/{total} | ⊘ Skipped: {skipped}/{total}\n")


def main() -> None:
    """Main entry point for metrics computation."""
    parser = argparse.ArgumentParser(
        description="Compute metrics with bootstrap confidence intervals"
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
        "--bench",
        type=str,
        default=None,
        help="Specific benchmark (uses all if not specified)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model (uses all if not specified)",
    )
    parser.add_argument(
        "--num-bootstrap",
        type=int,
        default=100000,
        help="Number of bootstrap iterations",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="Confidence level (default 0.95)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute even if metrics exist",
    )
    args = parser.parse_args()
    
    config = make_config(args.config)
    judge_config = make_config(args.judge_config)
    
    bench_list = [args.bench] if args.bench else config.get("bench_list", [])
    model_list = [args.model] if args.model else config.get("model_list", [])
    judge_model = judge_config.get("judge_model")
    
    if not bench_list or not model_list or not judge_model:
        print("ERROR: Missing required config values")
        exit(1)
    
    run_compute_metrics(
        bench_list=bench_list,
        model_list=model_list,
        judge_model=judge_model,
        num_bootstrap=args.num_bootstrap,
        ci_level=args.ci_level,
        seed=args.seed,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
