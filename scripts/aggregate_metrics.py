"""Script for aggregating metrics across benchmark groups."""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

from utils.completion import make_config


def load_scores_from_judgments(judgment_file: str) -> Optional[List[int]]:
    """Load binary scores from judgment file.
    
    Args:
        judgment_file: Path to judgment jsonl file.
    
    Returns:
        List of scores, or None if file doesn't exist.
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


def stratified_bootstrap_aggregate(
    bench_scores_dict: Dict[str, List[int]],
    num_iterations: int = 10000,
    seed: int = 42,
) -> Dict:
    """Stratified bootstrap aggregation across benchmarks.
    
    Resamples questions within each benchmark, then computes weighted average.
    
    Args:
        bench_scores_dict: Dict mapping benchmark name to scores list.
        num_iterations: Number of bootstrap iterations.
        seed: Random seed.
    
    Returns:
        Dict with 'mean', 'ci_lower', 'ci_upper', 'std'.
    """
    np.random.seed(seed)
    bootstrap_aggregates = []
    
    for _ in range(num_iterations):
        bench_accuracies = []
        bench_weights = []
        
        for bench_name, scores in bench_scores_dict.items():
            n = len(scores)
            sample_indices = np.random.randint(0, n, size=n)
            sample_scores = np.array(scores)[sample_indices]
            
            bench_accuracies.append(sample_scores.mean())
            bench_weights.append(n)
        
        weighted_avg = np.average(bench_accuracies, weights=bench_weights)
        bootstrap_aggregates.append(weighted_avg)
    
    bootstrap_aggregates = np.array(bootstrap_aggregates)
    
    return {
        "mean": float(bootstrap_aggregates.mean()),
        "ci_lower": float(np.percentile(bootstrap_aggregates, 2.5)),
        "ci_upper": float(np.percentile(bootstrap_aggregates, 97.5)),
        "std": float(bootstrap_aggregates.std()),
    }


def assign_tiers(models_metrics: List[Dict]) -> List[Dict]:
    """Assign statistical tiers based on overlapping confidence intervals.
    
    Args:
        models_metrics: List of model metrics (must be sorted by mean desc).
    
    Returns:
        Same list with 'tier' field added.
    """
    if not models_metrics:
        return []
    
    for model in models_metrics:
        model["tier"] = None
    
    current_tier = 1
    i = 0
    
    while i < len(models_metrics):
        if models_metrics[i]["tier"] is None:
            models_metrics[i]["tier"] = current_tier
            tier_upper = models_metrics[i]["ci_upper"]
            
            j = i + 1
            while j < len(models_metrics):
                if models_metrics[j]["ci_lower"] <= tier_upper:
                    models_metrics[j]["tier"] = current_tier
                    tier_upper = max(tier_upper, models_metrics[j]["ci_upper"])
                    j += 1
                else:
                    break
            
            current_tier += 1
            i = j
        else:
            i += 1
    
    return models_metrics


def compute_overlaps(models_metrics: List[Dict]) -> List[Dict]:
    """Compute which models have overlapping confidence intervals.
    
    Args:
        models_metrics: List of model metrics.
    
    Returns:
        Same list with 'overlaps_with' field added.
    """
    for i, model in enumerate(models_metrics):
        model["overlaps_with"] = []
        
        for j, other in enumerate(models_metrics):
            if i == j:
                continue
            
            if not (model["ci_upper"] < other["ci_lower"] or 
                    model["ci_lower"] > other["ci_upper"]):
                model["overlaps_with"].append(other["model"])
    
    return models_metrics


def aggregate_group(
    group_name: str,
    bench_list: List[str],
    model_list: List[str],
    judge_model: str,
    num_bootstrap: int = 10000,
    seed: int = 42,
) -> Optional[Dict]:
    """Aggregate metrics for a benchmark group.
    
    Args:
        group_name: Name of the benchmark group.
        bench_list: Benchmarks in this group.
        model_list: All models to evaluate.
        judge_model: Judge model name.
        num_bootstrap: Bootstrap iterations.
        seed: Random seed.
    
    Returns:
        Dict with aggregated results, or None if insufficient data.
    """
    models_data = defaultdict(dict)
    
    for model_name in model_list:
        has_all = True
        
        for bench_name in bench_list:
            judgment_file = f"data/{bench_name}/model_judgment/{judge_model}/{model_name}.jsonl"
            try:
                scores = load_scores_from_judgments(judgment_file)
            except Exception:
                scores = None
            
            if scores is None:
                has_all = False
                break
            
            models_data[model_name][bench_name] = scores
        
        if not has_all and model_name in models_data:
            del models_data[model_name]
    
    if not models_data:
        return None
    
    models_metrics = []
    
    for model_name, bench_scores in tqdm(models_data.items(), desc=f"[{group_name}] Bootstrap"):
        result = stratified_bootstrap_aggregate(
            bench_scores,
            num_iterations=num_bootstrap,
            seed=seed,
        )
        
        models_metrics.append({
            "model": model_name,
            "mean": result["mean"],
            "ci_lower": result["ci_lower"],
            "ci_upper": result["ci_upper"],
            "std": result["std"],
        })
    
    if not models_metrics:
        return None
    
    models_metrics = sorted(models_metrics, key=lambda x: x["mean"], reverse=True)
    
    for i, model in enumerate(models_metrics):
        model["rank"] = i + 1
    
    models_metrics = compute_overlaps(models_metrics)
    models_metrics = assign_tiers(models_metrics)
    
    return {
        "bench_group": group_name,
        "benchmarks": bench_list,
        "judge_model": judge_model,
        "num_models": len(models_metrics),
        "bootstrap": {
            "num_iterations": num_bootstrap,
            "seed": seed,
            "method": "stratified",
        },
        "models": models_metrics,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


def print_results(result: Dict) -> None:
    """Print aggregation results in readable format.
    
    Args:
        result: Aggregation result dict.
    """
    print(f"\n{'='*80}")
    print(f"Benchmark Group: {result['bench_group']}")
    print(f"Benchmarks: {', '.join(result['benchmarks'])}")
    print(f"Judge: {result['judge_model']}")
    print(f"Models: {result['num_models']}")
    print(f"{'='*80}\n")
    
    tiers = defaultdict(list)
    for model in result["models"]:
        tiers[model["tier"]].append(model)
    
    for tier_num in sorted(tiers.keys()):
        tier_models = tiers[tier_num]
        print(f"Tier {tier_num} (statistically indistinguishable):")
        
        for model in tier_models:
            print(f"  Rank {model['rank']}: {model['model']}")
            print(f"    Accuracy: {model['mean']:.1%} [{model['ci_lower']:.1%}, {model['ci_upper']:.1%}]")
            if model["overlaps_with"]:
                overlaps_count = len(model["overlaps_with"])
                if overlaps_count <= 3:
                    print(f"    Overlaps: {', '.join(model['overlaps_with'])}")
                else:
                    print(f"    Overlaps with {overlaps_count} models")
        
        print()


def run_aggregate_metrics(
    bench_groups: Dict[str, List[str]],
    model_list: List[str],
    judge_model: str,
    output_dir: str = "aggregated_metrics",
    num_bootstrap: int = 10000,
    seed: int = 42,
) -> None:
    """Aggregate metrics for all benchmark groups.
    
    Args:
        bench_groups: Dict mapping group name to benchmark list.
        model_list: List of models.
        judge_model: Judge model name.
        output_dir: Output directory for results.
        num_bootstrap: Bootstrap iterations.
        seed: Random seed.
    """
    print(f"\nAggregating metrics for {len(bench_groups)} benchmark groups")
    print(f"Judge: {judge_model} | Bootstrap: {num_bootstrap} iterations\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for group_name, bench_list in bench_groups.items():
        print(f"\nProcessing group: {group_name}")
        print(f"Benchmarks: {', '.join(bench_list)}")
        
        result = aggregate_group(
            group_name=group_name,
            bench_list=bench_list,
            model_list=model_list,
            judge_model=judge_model,
            num_bootstrap=num_bootstrap,
            seed=seed,
        )
        
        if result is None:
            print(f"[SKIP] {group_name}: No models with complete data")
            continue
        
        output_file = os.path.join(output_dir, f"{group_name}_{judge_model}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"Saved to: {output_file}")
        print_results(result)
    
    print(f"\n{'='*80}")
    print(f"✓ Aggregation complete. Results in: {output_dir}/")
    print(f"{'='*80}\n")


def main() -> None:
    """Main entry point for metrics aggregation."""
    parser = argparse.ArgumentParser(
        description="Aggregate metrics across benchmark groups"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/gen_answer_config.yaml",
        help="Path to config with model_list",
    )
    parser.add_argument(
        "--judge-config",
        type=str,
        default="configs/judge_config.yaml",
        help="Path to judge config",
    )
    parser.add_argument(
        "--groups-config",
        type=str,
        default="configs/bench_groups.yaml",
        help="Path to benchmark groups config",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="aggregated_metrics",
        help="Output directory",
    )
    parser.add_argument(
        "--num-bootstrap",
        type=int,
        default=100000,
        help="Number of bootstrap iterations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()
    
    config = make_config(args.config)
    judge_config = make_config(args.judge_config)
    
    if os.path.exists(args.groups_config):
        bench_groups = make_config(args.groups_config)
    else:
        bench_groups = {
            "exam_like": ["cfa_like_level_1", "cfa_like_level_2", "cfa_like_level_3", "cmt_like_level_2", "VLigaBench-ru"],
            "public_benchs": ["finqa", "convfinqa", "tatqa"],
            "ta_benchs": ["Trading_derivatives", "Trading_TA", "cfte_like_level_1"],
        }
    
    model_list = config.get("model_list", [])
    judge_model = judge_config.get("judge_model")
    
    if not model_list or not judge_model:
        print("ERROR: Missing required config values")
        exit(1)
    
    run_aggregate_metrics(
        bench_groups=bench_groups,
        model_list=model_list,
        judge_model=judge_model,
        output_dir=args.output_dir,
        num_bootstrap=args.num_bootstrap,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
    