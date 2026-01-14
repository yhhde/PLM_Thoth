#!/usr/bin/env python3
"""
Experiment Results Visualization Script

Generates charts and plots for experiment results analysis.
Supports the new unified output format (experiment_results.json).

Usage:
    python visualize_experiments.py --exp_dir ./experiments --output_dir ./figures
"""

import json
import os
import argparse
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


def load_experiment_results(exp_dir):
    """Load all experiment results from directory."""
    results = {}
    exp_path = Path(exp_dir)
    
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
        
        # Legacy format: simple JSON files
        elif item.is_file() and not item.suffix:
            try:
                with open(item) as f:
                    results[item.name] = {"validation": json.load(f)}
            except:
                continue
    
    return results


def plot_metrics_comparison(results, output_dir):
    """Bar chart comparing all experiments on key metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    exp_ids = sorted(results.keys())
    
    metrics = [
        ("ppl_fr_given_en", "PPL (FR|EN)", False, "#e74c3c"),
        ("pass@1_ppl", "Pass@1 (PPL-based)", True, "#3498db"),
        ("mrr_ppl", "MRR (PPL-based)", True, "#2ecc71"),
        ("auc", "AUC", True, "#9b59b6"),
    ]
    
    for idx, (metric, title, higher_is_better, color) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        values = []
        for exp_id in exp_ids:
            val = results[exp_id].get("validation", {}).get(metric, 0)
            values.append(val if val else 0)
        
        bars = ax.bar(exp_ids, values, color=color, alpha=0.8)
        
        # Highlight best
        if values:
            best_idx = np.argmin(values) if not higher_is_better else np.argmax(values)
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(3)
        
        ax.set_xlabel('Experiment')
        ax.set_ylabel(title)
        direction = "↑ Higher Better" if higher_is_better else "↓ Lower Better"
        ax.set_title(f'{title}\n({direction})')
        ax.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Experiment Metrics Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: metrics_comparison.png")


def plot_retrieval_methods_comparison(results, output_dir):
    """Compare embedding vs PPL-based retrieval methods."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    exp_ids = sorted(results.keys())
    
    pass1_embed = []
    pass1_ppl = []
    
    for exp_id in exp_ids:
        val = results[exp_id].get("validation", {})
        pass1_embed.append(val.get("pass@1", 0) or 0)
        pass1_ppl.append(val.get("pass@1_ppl", 0) or 0)
    
    x = np.arange(len(exp_ids))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, pass1_embed, width, label='Embedding-based', color='#e74c3c', alpha=0.8)
    bars2 = ax.bar(x + width/2, pass1_ppl, width, label='PPL-based', color='#3498db', alpha=0.8)
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Pass@1 Score')
    ax.set_title('Retrieval Method Comparison: Embedding vs Conditional Probability\n(Higher is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_ids, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, max(max(pass1_embed), max(pass1_ppl)) * 1.2 if pass1_ppl else 0.1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'retrieval_methods.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: retrieval_methods.png")


def plot_training_summary(results, output_dir):
    """Plot training loss and status summary."""
    # Filter experiments with training data
    train_data = {
        exp_id: data.get("training", {})
        for exp_id, data in results.items()
        if data.get("training", {}).get("final_val_loss") is not None
    }
    
    if not train_data:
        print("⚠️ No training data available for training summary plot")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Final losses
    ax = axes[0]
    exp_ids = sorted(train_data.keys())
    train_losses = [train_data[e].get("final_train_loss", 0) for e in exp_ids]
    val_losses = [train_data[e].get("final_val_loss", 0) for e in exp_ids]
    
    x = np.arange(len(exp_ids))
    width = 0.35
    
    ax.bar(x - width/2, train_losses, width, label='Train Loss', color='#3498db', alpha=0.8)
    ax.bar(x + width/2, val_losses, width, label='Val Loss', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Loss')
    ax.set_title('Final Training and Validation Loss')
    ax.set_xticks(x)
    ax.set_xticklabels(exp_ids, rotation=45, ha='right')
    ax.legend()
    
    # Right: Training status
    ax = axes[1]
    status_counts = {"completed": 0, "early_stopped": 0, "other": 0}
    for e in train_data.values():
        status = e.get("status", "other")
        if status in status_counts:
            status_counts[status] += 1
        else:
            status_counts["other"] += 1
    
    labels = [k for k, v in status_counts.items() if v > 0]
    values = [v for v in status_counts.values() if v > 0]
    colors = ['#2ecc71', '#f39c12', '#95a5a6'][:len(labels)]
    
    ax.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title('Training Status Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: training_summary.png")


def plot_heatmap(results, output_dir):
    """Heatmap of all experiments and metrics."""
    exp_ids = sorted(results.keys())
    
    metrics = [
        ("ppl_fr_given_en", "PPL(FR|EN)", False),
        ("ppl_en_given_fr", "PPL(EN|FR)", False),
        ("pass@1", "Pass@1", True),
        ("pass@1_ppl", "Pass@1_PPL", True),
        ("mrr_ppl", "MRR_PPL", True),
        ("auc", "AUC", True),
    ]
    
    # Build data matrix
    data = np.zeros((len(exp_ids), len(metrics)))
    for i, exp_id in enumerate(exp_ids):
        for j, (metric, _, _) in enumerate(metrics):
            data[i, j] = results[exp_id].get("validation", {}).get(metric, 0) or 0
    
    # Normalize for visualization
    normalized = np.zeros_like(data)
    for j, (_, _, higher_is_better) in enumerate(metrics):
        col = data[:, j]
        if col.max() - col.min() > 1e-8:
            if higher_is_better:
                normalized[:, j] = (col - col.min()) / (col.max() - col.min())
            else:
                normalized[:, j] = (col.max() - col) / (col.max() - col.min())
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(exp_ids) * 0.4)))
    
    im = ax.imshow(normalized, cmap='RdYlGn', aspect='auto')
    
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(exp_ids)))
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_yticklabels(exp_ids)
    
    # Add values
    for i in range(len(exp_ids)):
        for j in range(len(metrics)):
            text_color = "black" if normalized[i, j] > 0.4 else "white"
            ax.text(j, i, f'{data[i, j]:.2f}', ha="center", va="center",
                   color=text_color, fontsize=8)
    
    ax.set_title('Experiments Performance Heatmap\n(Green = Better, Red = Worse)', 
                 fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Normalized Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'experiments_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: experiments_heatmap.png")


def plot_training_history(exp_dir, output_dir):
    """Plot training curves from training_history.csv files."""
    exp_path = Path(exp_dir)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    found_any = False
    colors = plt.cm.tab10.colors
    
    for idx, item in enumerate(sorted(exp_path.iterdir())):
        if not item.is_dir():
            continue
        
        history_file = item / "training_history.csv"
        if not history_file.exists():
            continue
        
        # Load CSV
        try:
            import csv
            with open(history_file) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            if not rows:
                continue
            
            steps = [int(r['step']) for r in rows]
            train_loss = [float(r['train_loss']) for r in rows]
            val_loss = [float(r['val_loss']) for r in rows]
            
            color = colors[idx % len(colors)]
            
            axes[0].plot(steps, train_loss, label=item.name, color=color, alpha=0.8)
            axes[1].plot(steps, val_loss, label=item.name, color=color, alpha=0.8)
            
            found_any = True
        except Exception as e:
            print(f"⚠️ Error loading {history_file}: {e}")
            continue
    
    if not found_any:
        print("⚠️ No training history files found")
        plt.close()
        return
    
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Train Loss')
    axes[0].set_title('Training Loss Curves')
    axes[0].legend(loc='upper right', fontsize=8)
    
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Validation Loss')
    axes[1].set_title('Validation Loss Curves')
    axes[1].legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: training_curves.png")


def main():
    parser = argparse.ArgumentParser(description="Visualize experiment results")
    parser.add_argument("--exp_dir", default="./experiments", help="Experiments directory")
    parser.add_argument("--output_dir", default="./figures", help="Output directory for plots")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = load_experiment_results(args.exp_dir)
    
    if not results:
        print("❌ No experiment results found!")
        return
    
    print(f"📂 Loaded {len(results)} experiment(s)")
    print(f"📊 Generating visualizations...")
    
    # Generate all plots
    plot_metrics_comparison(results, args.output_dir)
    plot_retrieval_methods_comparison(results, args.output_dir)
    plot_training_summary(results, args.output_dir)
    plot_heatmap(results, args.output_dir)
    plot_training_history(args.exp_dir, args.output_dir)
    
    print(f"\n✅ All plots saved to: {args.output_dir}")
    print("\nGenerated files:")
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith('.png'):
            print(f"  - {f}")


if __name__ == "__main__":
    main()
