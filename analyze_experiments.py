#!/usr/bin/env python3
"""
Experiment Results Analysis Script

Analyzes experiment results from the new output format (experiment_results.json).
Supports both the new unified format and legacy validation_results.json.

Usage:
    python analyze_experiments.py --exp_dir ./experiments --output_dir ./analysis
"""

import json
import os
import argparse
from pathlib import Path
from collections import defaultdict


def load_experiment_results(exp_dir):
    """
    Load all experiment results from directory.
    
    Supports:
    - New format: {run_name}/experiment_results.json
    - Legacy format: {run_name}/validation_results.json or {exp_id} files
    """
    results = {}
    exp_path = Path(exp_dir)
    
    # Try to find experiment directories
    for item in exp_path.iterdir():
        if item.is_dir():
            # New format: experiment_results.json
            unified_file = item / "experiment_results.json"
            if unified_file.exists():
                with open(unified_file) as f:
                    data = json.load(f)
                    results[item.name] = {
                        "validation": data.get("validation", {}),
                        "training": data.get("training", {}),
                        "config": data.get("config", {}),
                    }
                continue
            
            # Fallback: validation_results.json
            val_file = item / "validation_results.json"
            if val_file.exists():
                with open(val_file) as f:
                    results[item.name] = {"validation": json.load(f)}
                continue
        
        # Legacy format: simple JSON files (v0, v1, etc.)
        elif item.is_file() and not item.suffix:
            try:
                with open(item) as f:
                    data = json.load(f)
                    results[item.name] = {"validation": data}
            except json.JSONDecodeError:
                continue
    
    return results


def extract_config_from_name(run_name):
    """Extract configuration from experiment name (legacy support)."""
    config = {
        "preproc": "_preproc" in run_name or run_name.endswith("_preproc"),
        "warmup": "_lr" in run_name,
        "dropout": 0.1,
        "weight_decay": 0.01,
    }
    
    if "drop02" in run_name:
        config["dropout"] = 0.2
    elif "drop005" in run_name:
        config["dropout"] = 0.05
    
    if "wd01" in run_name:
        config["weight_decay"] = 0.1
    elif "wd0005" in run_name:
        config["weight_decay"] = 0.005
    
    return config


def print_results_table(results):
    """Print formatted results table."""
    print("\n" + "=" * 120)
    print("📊 Experiment Results Summary")
    print("=" * 120)
    
    # Header
    header = (
        f"{'Experiment':<20} | {'Status':<12} | "
        f"{'Train Loss':<10} | {'Val Loss':<10} | "
        f"{'PPL(FR|EN)':<10} | {'Pass@1':<8} | {'Pass@1_PPL':<10} | {'AUC':<8}"
    )
    print(header)
    print("-" * 120)
    
    # Sort by name
    sorted_results = sorted(results.items(), key=lambda x: x[0])
    
    for exp_id, data in sorted_results:
        val = data.get("validation", {})
        train = data.get("training", {})
        
        status = train.get("status", "N/A")[:12]
        train_loss = train.get("final_train_loss", float('nan'))
        val_loss = train.get("final_val_loss", float('nan'))
        ppl = val.get("ppl_fr_given_en", float('nan'))
        pass1 = val.get("pass@1", float('nan'))
        pass1_ppl = val.get("pass@1_ppl", float('nan'))
        auc = val.get("auc", float('nan'))
        
        row = (
            f"{exp_id:<20} | {status:<12} | "
            f"{train_loss:<10.4f} | {val_loss:<10.4f} | "
            f"{ppl:<10.2f} | {pass1:<8.4f} | {pass1_ppl:<10.4f} | {auc:<8.4f}"
        )
        print(row)


def analyze_factor_effect(results, factor_key, metric, higher_is_better=True):
    """Analyze effect of a boolean factor."""
    with_factor = []
    without_factor = []
    
    for exp_id, data in results.items():
        val = data.get("validation", {}).get(metric)
        if val is None:
            continue
        
        # Get config
        config = data.get("config", {}).get("training", {})
        if not config:
            config = extract_config_from_name(exp_id)
        
        if config.get(factor_key):
            with_factor.append(val)
        else:
            without_factor.append(val)
    
    if not with_factor or not without_factor:
        return None
    
    avg_with = sum(with_factor) / len(with_factor)
    avg_without = sum(without_factor) / len(without_factor)
    
    if higher_is_better:
        improvement = avg_with - avg_without
    else:
        improvement = avg_without - avg_with
    
    return {
        "avg_with": avg_with,
        "avg_without": avg_without,
        "improvement": improvement,
        "is_positive": improvement > 0,
    }


def print_factor_analysis(results):
    """Print factor effect analysis."""
    print("\n" + "=" * 120)
    print("🔍 Factor Effect Analysis")
    print("=" * 120)
    
    metrics = [
        ("ppl_fr_given_en", "PPL (FR|EN)", False),
        ("pass@1", "Pass@1 (Embed)", True),
        ("pass@1_ppl", "Pass@1 (PPL)", True),
        ("auc", "AUC", True),
    ]
    
    factors = [
        ("preproc", "Data Preprocessing (mono+bucket)"),
        ("warmup", "LR Warmup (10%)"),
    ]
    
    for factor_key, factor_name in factors:
        print(f"\n【{factor_name}】")
        for metric, name, higher_is_better in metrics:
            analysis = analyze_factor_effect(results, factor_key, metric, higher_is_better)
            if analysis:
                direction = "↑" if higher_is_better else "↓"
                effect = "✅ Positive" if analysis["is_positive"] else "❌ Negative"
                print(f"  {name:<20}: With={analysis['avg_with']:.4f}, Without={analysis['avg_without']:.4f} "
                      f"| Effect: {effect} ({analysis['improvement']:+.4f} {direction})")


def find_best_experiments(results):
    """Find best experiments by each metric."""
    print("\n" + "=" * 120)
    print("🏆 Best Experiments by Metric")
    print("=" * 120)
    
    metrics_config = [
        ("ppl_fr_given_en", "PPL (FR|EN)", False),
        ("pass@1", "Pass@1 (Embed)", True),
        ("pass@1_ppl", "Pass@1 (PPL)", True),
        ("mrr_ppl", "MRR (PPL)", True),
        ("auc", "AUC", True),
    ]
    
    for metric, name, higher_is_better in metrics_config:
        # Filter experiments with this metric
        valid_results = [
            (exp_id, data) for exp_id, data in results.items()
            if data.get("validation", {}).get(metric) is not None
        ]
        
        if not valid_results:
            continue
        
        sorted_results = sorted(
            valid_results,
            key=lambda x: x[1].get("validation", {}).get(metric, 
                float('inf') if not higher_is_better else float('-inf')),
            reverse=higher_is_better
        )
        
        top3 = sorted_results[:3]
        
        direction = "(Lower is Better)" if not higher_is_better else "(Higher is Better)"
        print(f"\n【{name}】{direction}")
        for i, (exp_id, data) in enumerate(top3, 1):
            val = data.get("validation", {}).get(metric, 0)
            print(f"  {i}. {exp_id}: {val:.4f}")


def calculate_composite_score(data):
    """Calculate composite score for ranking experiments."""
    val = data.get("validation", {})
    
    # PPL score (inverse, lower is better)
    ppl = val.get("ppl_fr_given_en", 100)
    ppl_score = 1 / (1 + ppl / 100)
    
    # Retrieval score (use PPL-based if available, otherwise embedding)
    pass1_ppl = val.get("pass@1_ppl", 0)
    pass1_emb = val.get("pass@1", 0)
    retrieval_score = max(pass1_ppl, pass1_emb)
    
    # AUC score (0.5 is random baseline)
    auc = val.get("auc", 0.5)
    auc_score = max(0, auc - 0.5) * 2  # Normalize to 0-1
    
    # Composite (weighted)
    return ppl_score * 0.3 + retrieval_score * 0.4 + auc_score * 0.3


def generate_recommendation(results):
    """Generate recommendations based on results."""
    print("\n" + "=" * 120)
    print("💡 Recommendations")
    print("=" * 120)
    
    # Calculate composite scores
    scores = {exp_id: calculate_composite_score(data) 
              for exp_id, data in results.items()}
    
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    print("\n📈 Top 5 Experiments (Composite Score):")
    for i, (exp_id, score) in enumerate(sorted_scores[:5], 1):
        data = results[exp_id]
        val = data.get("validation", {})
        
        print(f"\n  {i}. {exp_id} (Score: {score:.4f})")
        print(f"     PPL(FR|EN)={val.get('ppl_fr_given_en', 0):.2f}, "
              f"Pass@1={val.get('pass@1', 0):.4f}, "
              f"Pass@1_PPL={val.get('pass@1_ppl', 0):.4f}, "
              f"AUC={val.get('auc', 0):.4f}")


def save_summary(results, output_dir):
    """Save analysis summary to file."""
    summary = {
        "total_experiments": len(results),
        "experiments": {},
    }
    
    for exp_id, data in results.items():
        val = data.get("validation", {})
        train = data.get("training", {})
        
        summary["experiments"][exp_id] = {
            "training": {
                "status": train.get("status"),
                "final_train_loss": train.get("final_train_loss"),
                "final_val_loss": train.get("final_val_loss"),
                "best_val_loss": train.get("best_val_loss"),
            },
            "validation": {
                "ppl_fr_given_en": val.get("ppl_fr_given_en"),
                "ppl_en_given_fr": val.get("ppl_en_given_fr"),
                "pass@1": val.get("pass@1"),
                "pass@1_ppl": val.get("pass@1_ppl"),
                "mrr_ppl": val.get("mrr_ppl"),
                "auc": val.get("auc"),
            },
            "composite_score": calculate_composite_score(data),
        }
    
    # Add rankings
    sorted_by_score = sorted(
        summary["experiments"].items(),
        key=lambda x: x[1]["composite_score"] or 0,
        reverse=True
    )
    summary["rankings"] = {
        "by_composite_score": [exp_id for exp_id, _ in sorted_by_score]
    }
    
    output_path = Path(output_dir) / "analysis_summary.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Summary saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--exp_dir", default="./experiments", help="Experiments directory")
    parser.add_argument("--output_dir", default="./analysis", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = load_experiment_results(args.exp_dir)
    
    if not results:
        print("❌ No experiment results found!")
        print(f"   Searched in: {args.exp_dir}")
        return
    
    print(f"📂 Loaded {len(results)} experiment(s)")
    
    # Print analyses
    print_results_table(results)
    print_factor_analysis(results)
    find_best_experiments(results)
    generate_recommendation(results)
    
    # Save summary
    save_summary(results, args.output_dir)


if __name__ == "__main__":
    main()
