#!/usr/bin/env python3
"""
Benchmark script with Weights & Biases (wandb) tracking for encoder cache optimization.

This script runs vLLM with multimodal models and logs KV cache metrics to wandb
for comparing main vs optimized branches.

Usage:
    # First, login to wandb (one-time setup)
    wandb login

    # Run benchmark
    python benchmark_with_wandb.py --model Qwen/Qwen3-VL-4B-Instruct --branch optimized

    # Compare on wandb dashboard
"""

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime

try:
    import wandb
except ImportError:
    print("wandb not installed. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    import wandb


def get_git_branch():
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_git_commit():
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def run_vllm_benchmark(model: str, gpu_memory_util: float = 0.9, 
                       max_model_len: int = 4096, timeout: int = 120,
                       enforce_eager: bool = True) -> dict:
    """
    Run vLLM serve command and parse output for KV cache metrics.
    
    Returns dict with metrics:
        - available_kv_cache_memory_gib
        - kv_cache_size_tokens
        - max_concurrency
        - encoder_cache_memory_gib (if present)
    """
    cmd = [
        "vllm", "serve", model,
        "--gpu-memory-utilization", str(gpu_memory_util),
        "--max-model-len", str(max_model_len),
        "--load-format", "dummy",  # Don't load actual weights for speed
    ]
    
    if enforce_eager:
        cmd.append("--enforce-eager")
    
    print(f"Running: {' '.join(cmd)}")
    
    metrics = {
        "available_kv_cache_memory_gib": None,
        "kv_cache_size_tokens": None,
        "max_concurrency": None,
        "encoder_cache_memory_gib": None,
    }
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        
        start_time = time.time()
        output_lines = []
        
        while True:
            if time.time() - start_time > timeout:
                print(f"Timeout after {timeout}s, terminating...")
                process.terminate()
                break
            
            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    break
                continue
            
            print(line, end="")
            output_lines.append(line)
            
            # Parse metrics from output
            # Available KV cache memory: X.XX GiB
            if "Available KV cache memory:" in line:
                match = re.search(r"Available KV cache memory:\s*([\d.]+)\s*GiB", line)
                if match:
                    metrics["available_kv_cache_memory_gib"] = float(match.group(1))
            
            # GPU KV cache size: XXX,XXX tokens
            if "GPU KV cache size:" in line:
                match = re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", line)
                if match:
                    metrics["kv_cache_size_tokens"] = int(match.group(1).replace(",", ""))
            
            # Maximum concurrency for X,XXX tokens per request: X.XXx
            if "Maximum concurrency" in line:
                match = re.search(r"Maximum concurrency.*:\s*([\d.]+)x", line)
                if match:
                    metrics["max_concurrency"] = float(match.group(1))
            
            # Encoder cache memory (if present in optimized branch)
            if "Encoder cache" in line.lower() and "memory" in line.lower():
                match = re.search(r"([\d.]+)\s*GiB", line)
                if match:
                    metrics["encoder_cache_memory_gib"] = float(match.group(1))
            
            # Stop once we have all main metrics
            if all([
                metrics["available_kv_cache_memory_gib"],
                metrics["kv_cache_size_tokens"],
                metrics["max_concurrency"],
            ]):
                print("\nAll metrics collected, terminating server...")
                process.terminate()
                break
        
        process.wait(timeout=5)
        
    except Exception as e:
        print(f"Error running benchmark: {e}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM encoder cache with wandb tracking")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-4B-Instruct",
                        help="Model to benchmark")
    parser.add_argument("--branch", type=str, default=None,
                        help="Branch name for tracking (auto-detected if not provided)")
    parser.add_argument("--gpu-memory-util", type=float, default=0.95,
                        help="GPU memory utilization")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Maximum model context length")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Timeout in seconds")
    parser.add_argument("--project", type=str, default="vllm-encoder-cache-optimization",
                        help="wandb project name")
    parser.add_argument("--enforce-eager", action="store_true", default=True,
                        help="Use eager execution (required for Qwen3-VL)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Skip wandb logging (just print metrics)")
    
    args = parser.parse_args()
    
    # Auto-detect branch if not provided
    branch = args.branch or get_git_branch()
    commit = get_git_commit()
    
    print("=" * 60)
    print(f"Encoder Cache Optimization Benchmark")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Branch: {branch}")
    print(f"Commit: {commit}")
    print(f"GPU Memory Util: {args.gpu_memory_util}")
    print(f"Max Model Len: {args.max_model_len}")
    print("=" * 60)
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.project,
            name=f"{branch}-{args.model.split('/')[-1]}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config={
                "model": args.model,
                "branch": branch,
                "commit": commit,
                "gpu_memory_utilization": args.gpu_memory_util,
                "max_model_len": args.max_model_len,
                "enforce_eager": args.enforce_eager,
            },
            tags=[branch, args.model.split("/")[-1]],
        )
    
    # Run benchmark
    metrics = run_vllm_benchmark(
        model=args.model,
        gpu_memory_util=args.gpu_memory_util,
        max_model_len=args.max_model_len,
        timeout=args.timeout,
        enforce_eager=args.enforce_eager,
    )
    
    # Log metrics
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    for key, value in metrics.items():
        if value is not None:
            print(f"  {key}: {value}")
    
    if not args.no_wandb:
        # Log to wandb
        wandb.log(metrics)
        
        # Also log as summary for easy comparison
        for key, value in metrics.items():
            if value is not None:
                wandb.run.summary[key] = value
        
        wandb.finish()
        print(f"\nResults logged to wandb project: {args.project}")
        print(f"View at: https://wandb.ai/{args.project}")
    
    return metrics


if __name__ == "__main__":
    main()

