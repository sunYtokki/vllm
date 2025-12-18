#!/usr/bin/env python3
"""
Analyze and plot results from wandb vllm-encoder-cache-optimization project.
Creates a single combined comparison plot with metrics appropriate for the group type.
"""

import subprocess
import sys

try:
    import wandb
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    import wandb

try:
    import pandas as pd
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
    import seaborn as sns

import numpy as np


def fetch_wandb_runs(project: str, entity: str = None, group: str = None):
    """Fetch all runs from a wandb project."""
    api = wandb.Api()
    
    # Build the path
    if entity:
        path = f"{entity}/{project}"
    else:
        path = project
    
    print(f"Fetching runs from: {path}")
    
    runs = api.runs(path)
    
    data = []
    for run in runs:
        # Check if run matches the group filter
        if group and run.group != group:
            continue
            
        run_data = {
            "run_id": run.id,
            "run_name": run.name,
            "state": run.state,
            "branch": run.config.get("branch", "unknown"),
            "model": run.config.get("model", "unknown"),
            "test_type": run.config.get("test_type", "unknown"),
            "group": run.group,
            "created_at": run.created_at,
        }
        
        # Add summary metrics
        summary = run.summary._json_dict
        run_data.update({
            # KV Cache metrics
            "available_kv_cache_memory_gib": summary.get("available_kv_cache_memory_gib"),
            "kv_cache_size_tokens": summary.get("kv_cache_size_tokens"),
            "max_concurrency": summary.get("max_concurrency"),
            # Performance metrics
            "init_time_seconds": summary.get("init_time_seconds"),
            "avg_accuracy": summary.get("avg_accuracy"),
            "avg_tokens_per_second": summary.get("avg_tokens_per_second"),
            "total_output_tokens": summary.get("total_output_tokens"),
            "total_generation_time_seconds": summary.get("total_generation_time_seconds"),
            # Test counts
            "num_images_tested": summary.get("num_images_tested"),
            "num_videos_tested": summary.get("num_videos_tested"),
        })
        
        data.append(run_data)
    
    return pd.DataFrame(data)


def plot_memory_benchmark(df: pd.DataFrame, output_path: str):
    """Create plot for memory-benchmark group."""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    df_clean = df.dropna(subset=["available_kv_cache_memory_gib", "kv_cache_size_tokens", "max_concurrency"])
    
    if df_clean.empty:
        print("Warning: No runs with complete KV cache metrics found.")
        return
    
    df_clean = df_clean.copy()
    df_clean["model_short"] = df_clean["model"].apply(lambda x: x.split("/")[-1] if "/" in str(x) else x)
    
    models = sorted(df_clean["model_short"].unique())
    branches = df_clean["branch"].unique()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    
    colors = {"main": "#E74C3C", "sun-optimized": "#27AE60"}
    bar_width = 0.35
    x = np.arange(len(models))
    
    # Plot 1: Available KV Cache Memory
    ax1 = axes[0]
    for i, branch in enumerate(["main", "sun-optimized"]):
        if branch in branches:
            branch_data = df_clean[df_clean["branch"] == branch].set_index("model_short")
            values = [branch_data.loc[m, "available_kv_cache_memory_gib"] if m in branch_data.index else 0 for m in models]
            offset = (i - 0.5) * bar_width
            bars = ax1.bar(x + offset, values, bar_width, label=branch, color=colors.get(branch, f"C{i}"), edgecolor='black', linewidth=0.5)
            for bar, val in zip(bars, values):
                if val > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                            f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_ylabel("Available KV Cache Memory (GiB)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Model", fontsize=12)
    ax1.set_title("Available KV Cache Memory", fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace("-Instruct", "").replace("-FP8", "\n(FP8)") for m in models], fontsize=10)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_ylim(0, max(df_clean["available_kv_cache_memory_gib"]) * 1.25)
    
    # Plot 2: KV Cache Size (Tokens)
    ax2 = axes[1]
    for i, branch in enumerate(["main", "sun-optimized"]):
        if branch in branches:
            branch_data = df_clean[df_clean["branch"] == branch].set_index("model_short")
            values = [branch_data.loc[m, "kv_cache_size_tokens"] if m in branch_data.index else 0 for m in models]
            offset = (i - 0.5) * bar_width
            bars = ax2.bar(x + offset, values, bar_width, label=branch, color=colors.get(branch, f"C{i}"), edgecolor='black', linewidth=0.5)
            for bar, val in zip(bars, values):
                if val > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2000, 
                            f'{int(val):,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_ylabel("KV Cache Size (Tokens)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Model", fontsize=12)
    ax2.set_title("KV Cache Size (Tokens)", fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.replace("-Instruct", "").replace("-FP8", "\n(FP8)") for m in models], fontsize=10)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}K'))
    ax2.set_ylim(0, max(df_clean["kv_cache_size_tokens"]) * 1.25)
    
    # Plot 3: Max Concurrency
    ax3 = axes[2]
    for i, branch in enumerate(["main", "sun-optimized"]):
        if branch in branches:
            branch_data = df_clean[df_clean["branch"] == branch].set_index("model_short")
            values = [branch_data.loc[m, "max_concurrency"] if m in branch_data.index else 0 for m in models]
            offset = (i - 0.5) * bar_width
            bars = ax3.bar(x + offset, values, bar_width, label=branch, color=colors.get(branch, f"C{i}"), edgecolor='black', linewidth=0.5)
            for bar, val in zip(bars, values):
                if val > 0:
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                            f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax3.set_ylabel("Max Concurrency", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Model", fontsize=12)
    ax3.set_title("Max Concurrency", fontsize=14, fontweight='bold', pad=15)
    ax3.set_xticks(x)
    ax3.set_xticklabels([m.replace("-Instruct", "").replace("-FP8", "\n(FP8)") for m in models], fontsize=10)
    ax3.legend(loc='upper left', fontsize=10)
    ax3.set_ylim(0, max(df_clean["max_concurrency"]) * 1.25)
    
    fig.suptitle("Encoder Cache Optimization: Memory Benchmark", fontsize=16, fontweight='bold', y=1.02)
    
    # Add improvement annotations
    improvements = []
    for model in models:
        main_data = df_clean[(df_clean["branch"] == "main") & (df_clean["model_short"] == model)]
        opt_data = df_clean[(df_clean["branch"] == "sun-optimized") & (df_clean["model_short"] == model)]
        if not main_data.empty and not opt_data.empty:
            main_mem = main_data["available_kv_cache_memory_gib"].iloc[0]
            opt_mem = opt_data["available_kv_cache_memory_gib"].iloc[0]
            improvement = ((opt_mem - main_mem) / main_mem) * 100
            improvements.append(f"{model.replace('-Instruct', '').replace('-FP8', '')}: +{improvement:.0f}%")
    
    if improvements:
        fig.text(0.5, -0.02, "Memory Improvement: " + " | ".join(improvements), 
                ha='center', fontsize=12, style='italic', color='#27AE60', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")
    plt.show()


def plot_functional_test(df: pd.DataFrame, output_path: str, test_type: str = "image"):
    """Create plot for functional-image or functional-video groups.
    
    Shows 4 key metrics:
    1. Available KV Cache Memory (GiB)
    2. KV Cache Size (Tokens)
    3. Max Concurrency
    4. Average Accuracy (%) - guardrail metric
    """
    
    plt.style.use('seaborn-v0_8-whitegrid')
    
    df_clean = df.copy()
    df_clean["model_short"] = df_clean["model"].apply(lambda x: x.split("/")[-1] if "/" in str(x) else x)
    
    models = sorted(df_clean["model_short"].unique())
    branches = df_clean["branch"].unique()
    
    # Create figure with 2x2 layout for 4 metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {"main": "#E74C3C", "sun-optimized": "#27AE60"}
    bar_width = 0.35
    x = np.arange(len(models))
    
    # --- Plot 1: Available KV Cache Memory ---
    ax1 = axes[0, 0]
    df_mem = df_clean.dropna(subset=["available_kv_cache_memory_gib"])
    if not df_mem.empty:
        for i, branch in enumerate(["main", "sun-optimized"]):
            if branch in branches:
                branch_data = df_mem[df_mem["branch"] == branch].set_index("model_short")
                values = [branch_data.loc[m, "available_kv_cache_memory_gib"] if m in branch_data.index else 0 for m in models]
                offset = (i - 0.5) * bar_width
                bars = ax1.bar(x + offset, values, bar_width, label=branch, color=colors.get(branch, f"C{i}"), edgecolor='black', linewidth=0.5)
                for bar, val in zip(bars, values):
                    if val > 0:
                        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax1.set_ylabel("KV Cache Memory (GiB)", fontsize=12, fontweight='bold')
        ax1.set_title("Available KV Cache Memory", fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([m.replace("-Instruct", "") for m in models], fontsize=10, rotation=15)
        ax1.legend(loc='upper left', fontsize=10)
        if df_mem["available_kv_cache_memory_gib"].max() > 0:
            ax1.set_ylim(0, df_mem["available_kv_cache_memory_gib"].max() * 1.25)
    
    # --- Plot 2: KV Cache Tokens ---
    ax2 = axes[0, 1]
    df_tokens = df_clean.dropna(subset=["kv_cache_size_tokens"])
    if not df_tokens.empty:
        for i, branch in enumerate(["main", "sun-optimized"]):
            if branch in branches:
                branch_data = df_tokens[df_tokens["branch"] == branch].set_index("model_short")
                values = [branch_data.loc[m, "kv_cache_size_tokens"] if m in branch_data.index else 0 for m in models]
                offset = (i - 0.5) * bar_width
                bars = ax2.bar(x + offset, values, bar_width, label=branch, color=colors.get(branch, f"C{i}"), edgecolor='black', linewidth=0.5)
                for bar, val in zip(bars, values):
                    if val > 0:
                        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1500, 
                                f'{int(val/1000)}K', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax2.set_ylabel("KV Cache Size (Tokens)", fontsize=12, fontweight='bold')
        ax2.set_title("KV Cache Size (Tokens)", fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([m.replace("-Instruct", "") for m in models], fontsize=10, rotation=15)
        ax2.legend(loc='upper left', fontsize=10)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}K'))
        if df_tokens["kv_cache_size_tokens"].max() > 0:
            ax2.set_ylim(0, df_tokens["kv_cache_size_tokens"].max() * 1.25)
    
    # --- Plot 3: Max Concurrency ---
    ax3 = axes[1, 0]
    df_conc = df_clean.dropna(subset=["max_concurrency"])
    if not df_conc.empty:
        for i, branch in enumerate(["main", "sun-optimized"]):
            if branch in branches:
                branch_data = df_conc[df_conc["branch"] == branch].set_index("model_short")
                values = [branch_data.loc[m, "max_concurrency"] if m in branch_data.index else 0 for m in models]
                offset = (i - 0.5) * bar_width
                bars = ax3.bar(x + offset, values, bar_width, label=branch, color=colors.get(branch, f"C{i}"), edgecolor='black', linewidth=0.5)
                for bar, val in zip(bars, values):
                    if val > 0:
                        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax3.set_ylabel("Max Concurrency", fontsize=12, fontweight='bold')
        ax3.set_title("Max Concurrency", fontsize=13, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([m.replace("-Instruct", "") for m in models], fontsize=10, rotation=15)
        ax3.legend(loc='upper left', fontsize=10)
        if df_conc["max_concurrency"].max() > 0:
            ax3.set_ylim(0, df_conc["max_concurrency"].max() * 1.25)
    
    # --- Plot 4: Average Accuracy (Guardrail) ---
    ax4 = axes[1, 1]
    df_acc = df_clean.dropna(subset=["avg_accuracy"])
    if not df_acc.empty:
        for i, branch in enumerate(["main", "sun-optimized"]):
            if branch in branches:
                branch_data = df_acc[df_acc["branch"] == branch].set_index("model_short")
                values = [branch_data.loc[m, "avg_accuracy"] * 100 if m in branch_data.index else 0 for m in models]
                offset = (i - 0.5) * bar_width
                bars = ax4.bar(x + offset, values, bar_width, label=branch, color=colors.get(branch, f"C{i}"), edgecolor='black', linewidth=0.5)
                for bar, val in zip(bars, values):
                    if val > 0:
                        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, 
                                f'{val:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax4.set_ylabel("Average Accuracy (%)", fontsize=12, fontweight='bold')
        ax4.set_title("Average Accuracy (Guardrail)", fontsize=13, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels([m.replace("-Instruct", "") for m in models], fontsize=10, rotation=15)
        ax4.legend(loc='upper left', fontsize=10)
        ax4.set_ylim(0, 105)
        # Add a horizontal line at some threshold (e.g., minimum acceptable accuracy)
        ax4.axhline(y=50, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Min threshold')
    
    title = f"Functional {test_type.capitalize()} Test: sun-optimized vs main"
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    # Add improvement summary at bottom
    improvements = []
    for model in models:
        main_data = df_clean[(df_clean["branch"] == "main") & (df_clean["model_short"] == model)]
        opt_data = df_clean[(df_clean["branch"] == "sun-optimized") & (df_clean["model_short"] == model)]
        if not main_data.empty and not opt_data.empty:
            main_mem = main_data["available_kv_cache_memory_gib"].iloc[0] if "available_kv_cache_memory_gib" in main_data else None
            opt_mem = opt_data["available_kv_cache_memory_gib"].iloc[0] if "available_kv_cache_memory_gib" in opt_data else None
            if pd.notna(main_mem) and pd.notna(opt_mem) and main_mem > 0:
                improvement = ((opt_mem - main_mem) / main_mem) * 100
                model_name = model.replace('-Instruct', '').replace('-FP8', '')
                improvements.append(f"{model_name}: +{improvement:.0f}%")
    
    if improvements:
        fig.text(0.5, -0.02, "Memory Improvement: " + " | ".join(improvements), 
                ha='center', fontsize=12, style='italic', color='#27AE60', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_path}")
    plt.show()


def print_comparison_table(df: pd.DataFrame, group: str = None):
    """Print a formatted comparison table."""
    
    print("\n" + "=" * 120)
    print("METRICS COMPARISON")
    print("=" * 120)
    
    # Select columns based on group type
    # For functional tests: 4 key metrics (3 KV cache + accuracy guardrail)
    if group and "functional" in group.lower():
        cols = ["branch", "model", "available_kv_cache_memory_gib", "kv_cache_size_tokens", 
                "max_concurrency", "avg_accuracy"]
    else:
        cols = ["branch", "model", "available_kv_cache_memory_gib", "kv_cache_size_tokens", "max_concurrency"]
    
    cols_available = [c for c in cols if c in df.columns]
    
    df_display = df[cols_available].copy()
    df_display = df_display.sort_values(["model", "branch"])
    
    # Format for display
    if "avg_accuracy" in df_display.columns:
        df_display["avg_accuracy"] = df_display["avg_accuracy"].apply(
            lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A"
        )
    if "avg_tokens_per_second" in df_display.columns:
        df_display["avg_tokens_per_second"] = df_display["avg_tokens_per_second"].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"
        )
    if "init_time_seconds" in df_display.columns:
        df_display["init_time_seconds"] = df_display["init_time_seconds"].apply(
            lambda x: f"{x:.1f}s" if pd.notna(x) else "N/A"
        )
    
    print(df_display.to_string(index=False))
    print("=" * 120)
    
    # Print improvements
    print("\n" + "=" * 120)
    print("IMPROVEMENT SUMMARY (sun-optimized vs main)")
    print("=" * 120)
    
    df_clean = df.dropna(subset=["available_kv_cache_memory_gib"])
    models = df_clean["model"].unique()
    
    for model in sorted(models):
        main_data = df_clean[(df_clean["branch"] == "main") & (df_clean["model"] == model)]
        opt_data = df_clean[(df_clean["branch"] == "sun-optimized") & (df_clean["model"] == model)]
        
        if not main_data.empty and not opt_data.empty:
            model_short = model.split("/")[-1]
            print(f"\n{model_short}:")
            
            # Memory metrics
            if "available_kv_cache_memory_gib" in main_data.columns:
                main_val = main_data["available_kv_cache_memory_gib"].iloc[0]
                opt_val = opt_data["available_kv_cache_memory_gib"].iloc[0]
                if pd.notna(main_val) and pd.notna(opt_val):
                    improvement = ((opt_val - main_val) / main_val) * 100
                    print(f"  KV Cache Memory:    {main_val:.2f} GiB → {opt_val:.2f} GiB  (+{improvement:.1f}%)")
            
            if "kv_cache_size_tokens" in main_data.columns:
                main_val = main_data["kv_cache_size_tokens"].iloc[0]
                opt_val = opt_data["kv_cache_size_tokens"].iloc[0]
                if pd.notna(main_val) and pd.notna(opt_val):
                    improvement = ((opt_val - main_val) / main_val) * 100
                    print(f"  KV Cache Tokens:    {int(main_val):,} → {int(opt_val):,}  (+{improvement:.1f}%)")
            
            if "max_concurrency" in main_data.columns:
                main_val = main_data["max_concurrency"].iloc[0]
                opt_val = opt_data["max_concurrency"].iloc[0]
                if pd.notna(main_val) and pd.notna(opt_val):
                    improvement = ((opt_val - main_val) / main_val) * 100
                    print(f"  Max Concurrency:    {main_val:.2f} → {opt_val:.2f}  (+{improvement:.1f}%)")
            
            # Functional test metrics
            if "avg_accuracy" in main_data.columns:
                main_val = main_data["avg_accuracy"].iloc[0]
                opt_val = opt_data["avg_accuracy"].iloc[0]
                if pd.notna(main_val) and pd.notna(opt_val):
                    diff = (opt_val - main_val) * 100
                    print(f"  Avg Accuracy:       {main_val*100:.1f}% → {opt_val*100:.1f}%  ({diff:+.1f}pp)")
            
            if "avg_tokens_per_second" in main_data.columns:
                main_val = main_data["avg_tokens_per_second"].iloc[0]
                opt_val = opt_data["avg_tokens_per_second"].iloc[0]
                if pd.notna(main_val) and pd.notna(opt_val):
                    improvement = ((opt_val - main_val) / main_val) * 100
                    print(f"  Tokens/Second:      {main_val:.1f} → {opt_val:.1f}  ({improvement:+.1f}%)")
    
    print("\n" + "=" * 120)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze wandb vllm-encoder-cache-optimization results")
    parser.add_argument("--entity", type=str, default="wl2904-columbia-university",
                        help="wandb entity (username or team)")
    parser.add_argument("--project", type=str, default="vllm-encoder-cache-optimization",
                        help="wandb project name")
    parser.add_argument("--group", type=str, default=None,
                        help="Filter by run group (e.g., 'memory-benchmark', 'functional-image', 'functional-video')")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path for the plot (auto-generated if not specified)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip generating plots")
    args = parser.parse_args()
    
    print("=" * 60)
    print("WandB Encoder Cache Optimization Analysis")
    print("=" * 60)
    print(f"Entity: {args.entity}")
    print(f"Project: {args.project}")
    if args.group:
        print(f"Group: {args.group}")
    print("=" * 60)
    
    # Fetch runs
    df = fetch_wandb_runs(args.project, entity=args.entity, group=args.group)
    
    if df.empty:
        print("No runs found!")
        return
    
    print(f"\nFound {len(df)} runs")
    
    # Print comparison table
    print_comparison_table(df, group=args.group)
    
    # Generate plot
    if not args.no_plot:
        # Auto-generate output path based on group
        if args.output:
            output_path = args.output
        else:
            group_name = args.group.replace("-", "_") if args.group else "all"
            output_path = f"{group_name}_comparison.png"
        
        if args.group and "functional" in args.group.lower():
            test_type = "video" if "video" in args.group.lower() else "image"
            plot_functional_test(df, output_path, test_type=test_type)
        else:
            plot_memory_benchmark(df, output_path)
    
    # Save to CSV
    csv_path = (args.output or f"{args.group or 'all'}_comparison").replace(".png", "") + "_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nData saved to: {csv_path}")


if __name__ == "__main__":
    main()
