#!/usr/bin/env python3
"""
Functional test for Qwen3-VL with real image input + wandb logging.
Tests the encoder cache optimization with actual inference.
"""

import io
import logging
import os
import re
import subprocess
import sys
import time

import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams


def parse_kv_cache_metrics(log_text: str) -> dict:
    """Parse KV cache metrics from vLLM log output."""
    metrics = {
        "available_kv_cache_memory_gib": None,
        "kv_cache_size_tokens": None,
        "max_concurrency": None,
    }
    
    # Available KV cache memory: X.XX GiB
    # Use robust regex to handle potential formatting variations or color codes
    match = re.search(r"Available.*?KV cache memory.*?([\d.]+).*?GiB", log_text, re.IGNORECASE)
    if match:
        metrics["available_kv_cache_memory_gib"] = float(match.group(1))
    
    # GPU KV cache size: XXX,XXX tokens
    match = re.search(r"GPU.*?KV cache size.*?([\d,]+).*?tokens", log_text, re.IGNORECASE)
    if match:
        metrics["kv_cache_size_tokens"] = int(match.group(1).replace(",", ""))
    
    # Maximum concurrency for X,XXX tokens per request: X.XXx
    match = re.search(r"Maximum concurrency.*?:.*?([\d.]+)", log_text, re.IGNORECASE)
    if match:
        metrics["max_concurrency"] = float(match.group(1))
    
    return metrics

try:
    import wandb
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    import wandb

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

CHECKPOINT = "Qwen/Qwen3-VL-2B-Instruct"


def get_git_branch():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__))
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__))
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True, return_video_metadata=True
    )
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    return {"prompt": text, "multi_modal_data": mm_data, "mm_processor_kwargs": video_kwargs}


def calculate_accuracy(output_text: str, expected_keywords: list) -> dict:
    """
    Calculate accuracy based on expected keywords.
    """
    output_lower = output_text.lower()
    
    found = []
    missing = []
    for keyword in expected_keywords:
        if keyword.lower() in output_lower:
            found.append(keyword)
        else:
            missing.append(keyword)
    
    accuracy = len(found) / len(expected_keywords) if expected_keywords else 0
    
    return {
        "accuracy": accuracy,
        "keywords_found": len(found),
        "keywords_total": len(expected_keywords),
        "found_keywords": found,
        "missing_keywords": missing,
    }


# Image dataset for testing (20 diverse images using reliable URLs)
IMAGE_DATASET = [
    # 1. Receipt (OCR) - Qwen official
    {
        "name": "receipt",
        "image_url": "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-VL/receipt.png",
        "prompt": "Read all the text in the image.",
        "expected_keywords": ["TOTAL", "CINNAMON", "SUGAR", "CASH"],
    },
    # 2. Dog - Unsplash
    {
        "name": "dog",
        "image_url": "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=800",
        "prompt": "What animal is in this image? Describe it briefly.",
        "expected_keywords": ["dog", "puppy", "animal", "pet"],
    },
    # 3. Cat - Unsplash
    {
        "name": "cat",
        "image_url": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=800",
        "prompt": "What animal is in this image? What color is it?",
        "expected_keywords": ["cat", "orange", "ginger", "animal", "pet"],
    },
    # 4. Pizza - Unsplash
    {
        "name": "pizza",
        "image_url": "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=800",
        "prompt": "What food is shown in this image?",
        "expected_keywords": ["pizza", "cheese", "food"],
    },
    # 5. Coffee - Unsplash
    {
        "name": "coffee",
        "image_url": "https://images.unsplash.com/photo-1509042239860-f550ce710b93?w=800",
        "prompt": "What beverage is shown in this image?",
        "expected_keywords": ["coffee", "cup", "drink", "latte"],
    },
    # 6. Beach - Unsplash
    {
        "name": "beach",
        "image_url": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=800",
        "prompt": "What type of scene is shown? Describe the setting.",
        "expected_keywords": ["beach", "ocean", "sea", "sand", "water"],
    },
    # 7. Mountain - Unsplash
    {
        "name": "mountain",
        "image_url": "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?w=800",
        "prompt": "What is shown in this image?",
        "expected_keywords": ["mountain", "peak", "nature", "landscape"],
    },
    # 8. Car - Unsplash
    {
        "name": "car",
        "image_url": "https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=800",
        "prompt": "What type of vehicle is shown?",
        "expected_keywords": ["car", "vehicle", "automobile"],
    },
    # 9. Flower - Unsplash
    {
        "name": "flower",
        "image_url": "https://images.unsplash.com/photo-1490750967868-88aa4486c946?w=800",
        "prompt": "What is shown in this image?",
        "expected_keywords": ["flower", "pink", "rose", "petal"],
    },
    # 10. Laptop - Unsplash
    {
        "name": "laptop",
        "image_url": "https://images.unsplash.com/photo-1496181133206-80ce9b88a853?w=800",
        "prompt": "What device is shown in this image?",
        "expected_keywords": ["laptop", "computer", "macbook", "keyboard"],
    },
    # 11. Apple fruit - Unsplash
    {
        "name": "apple",
        "image_url": "https://images.unsplash.com/photo-1560806887-1e4cd0b6cbd6?w=800",
        "prompt": "What fruit is shown in this image?",
        "expected_keywords": ["apple", "fruit", "red", "green"],
    },
    # 12. Book - Unsplash
    {
        "name": "book",
        "image_url": "https://images.unsplash.com/photo-1544716278-ca5e3f4abd8c?w=800",
        "prompt": "What object is shown in this image?",
        "expected_keywords": ["book", "reading", "pages"],
    },
    # 13. Bird - Unsplash
    {
        "name": "bird",
        "image_url": "https://images.unsplash.com/photo-1444464666168-49d633b86797?w=800",
        "prompt": "What animal is shown in this image?",
        "expected_keywords": ["bird", "flying", "wings", "sky"],
    },
    # 14. Bicycle - Unsplash
    {
        "name": "bicycle",
        "image_url": "https://images.unsplash.com/photo-1485965120184-e220f721d03e?w=800",
        "prompt": "What vehicle is shown?",
        "expected_keywords": ["bicycle", "bike", "wheel"],
    },
    # 15. Burger - Unsplash
    {
        "name": "burger",
        "image_url": "https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=800",
        "prompt": "What food is shown in this image?",
        "expected_keywords": ["burger", "hamburger", "food", "bun"],
    },
    # 16. City - Unsplash
    {
        "name": "city",
        "image_url": "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=800",
        "prompt": "What type of scene is shown?",
        "expected_keywords": ["city", "building", "urban", "street"],
    },
    # 17. Forest - Unsplash
    {
        "name": "forest",
        "image_url": "https://images.unsplash.com/photo-1448375240586-882707db888b?w=800",
        "prompt": "What type of environment is shown?",
        "expected_keywords": ["forest", "tree", "nature", "green"],
    },
    # 18. Phone - Unsplash
    {
        "name": "phone",
        "image_url": "https://images.unsplash.com/photo-1511707171634-5f897ff02aa9?w=800",
        "prompt": "What device is shown?",
        "expected_keywords": ["phone", "iphone", "smartphone", "mobile"],
    },
    # 19. Cake - Unsplash
    {
        "name": "cake",
        "image_url": "https://images.unsplash.com/photo-1578985545062-69928b1d9587?w=800",
        "prompt": "What food is shown in this image?",
        "expected_keywords": ["cake", "chocolate", "dessert"],
    },
    # 20. Watch - Unsplash
    {
        "name": "watch",
        "image_url": "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=800",
        "prompt": "What object is shown in this image?",
        "expected_keywords": ["watch", "time", "wrist"],
    },
]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", type=str, default=None, help="Branch name for wandb")
    parser.add_argument("--model", type=str, default=CHECKPOINT, help="Model checkpoint")
    parser.add_argument("--project", type=str, default="vllm-encoder-cache-optimization", 
                        help="wandb project")
    parser.add_argument("--no-wandb", action="store_true", help="Skip wandb logging")
    parser.add_argument("--num-images", type=int, default=None, 
                        help="Number of images to test (default: all)")
    # Manual KV cache metrics (from vLLM console output)
    parser.add_argument("--kv-cache-memory", type=float, default=None,
                        help="Available KV cache memory in GiB (from vLLM logs)")
    parser.add_argument("--kv-cache-tokens", type=int, default=None,
                        help="KV cache size in tokens (from vLLM logs)")
    parser.add_argument("--max-concurrency", type=float, default=None,
                        help="Max concurrency (from vLLM logs)")
    args = parser.parse_args()

    branch = args.branch or get_git_branch()
    commit = get_git_commit()
    model = args.model

    print("=" * 60)
    print("Functional Test: Qwen3-VL with Real Image")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Branch: {branch}")
    print(f"Commit: {commit}")
    print("=" * 60)

    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.project,
            name=f"functional-{branch}-{model.split('/')[-1]}-{time.strftime('%Y%m%d-%H%M%S')}",
            config={
                "model": model,
                "branch": branch,
                "commit": commit,
                "test_type": "functional_image",
                "gpu_memory_utilization": 0.95,
                "max_model_len": 4096,
            },
            tags=[branch, "functional", "image", model.split("/")[-1]],
        )

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(model)

    print("Initializing vLLM engine...")
    
    # Robust log capture using a temporary file on disk
    # This ensures we catch output from subprocesses (spawned workers) and both stdout/stderr
    import tempfile
    
    # Create a temp file to capture output
    fd, temp_path = tempfile.mkstemp()
    
    # Save original FDs
    original_stdout_fd = sys.stdout.fileno()
    original_stderr_fd = sys.stderr.fileno()
    
    saved_stdout_fd = os.dup(original_stdout_fd)
    saved_stderr_fd = os.dup(original_stderr_fd)
    
    capture_success = False
    
    try:
        # Redirect stdout and stderr to the temp file
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Redirect both to the same file
        os.dup2(fd, original_stdout_fd)
        os.dup2(fd, original_stderr_fd)
        
        start_time = time.time()
        llm = LLM(
            model=model,
            trust_remote_code=True,
            gpu_memory_utilization=0.95,
            max_model_len=4096,
            enforce_eager=True,
            tensor_parallel_size=1,
            seed=0,
        )
        init_time = time.time() - start_time
        capture_success = True
        
    except Exception as e:
        # If something goes wrong, we'll see it after restoring stdout/stderr
        init_error = e
        init_time = 0
        
    finally:
        # Restore stdout/stderr
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout_fd, original_stdout_fd)
        os.dup2(saved_stderr_fd, original_stderr_fd)
        
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(fd)
    
    if not capture_success:
        print(f"ERROR: LLM initialization failed: {init_error}")
        sys.exit(1)
    
    # Read captured logs
    log_contents = ""
    try:
        with open(temp_path, 'r', errors='replace') as f:
            log_contents = f.read()
        os.unlink(temp_path)
    except Exception as e:
        print(f"Warning: Could not read captured logs: {e}")
        
    # Re-emit captured logs so they appear in console
    print(log_contents)
    
    # Parse metrics from logs
    parsed_metrics = parse_kv_cache_metrics(log_contents)
    
    # Try to get KV cache metrics directly from the LLM object
    kv_cache_metrics = {
        "available_kv_cache_memory_gib": parsed_metrics.get("available_kv_cache_memory_gib"),
        "kv_cache_size_tokens": parsed_metrics.get("kv_cache_size_tokens"),
        "max_concurrency": parsed_metrics.get("max_concurrency"),
    }
    
    try:
        # Access cache config from LLM engine
        cache_config = llm.llm_engine.cache_config
        vllm_config = llm.llm_engine.vllm_config
        model_config = vllm_config.model_config
        
        # Get num_gpu_blocks (set after initialization)
        num_gpu_blocks = cache_config.num_gpu_blocks
        block_size = cache_config.block_size
        
        if num_gpu_blocks is not None and num_gpu_blocks > 0:
            # Calculate KV cache size in tokens
            kv_cache_tokens = num_gpu_blocks * block_size
            kv_cache_metrics["kv_cache_size_tokens"] = kv_cache_tokens
            
            # Calculate max concurrency
            max_model_len = model_config.max_model_len
            if max_model_len > 0:
                max_concurrency = kv_cache_tokens / max_model_len
                kv_cache_metrics["max_concurrency"] = round(max_concurrency, 2)
            
            # NOTE: available_kv_cache_memory_gib is determined by GPU profiling
            # and printed by the subprocess. It cannot be calculated from tokens.
            # Use --kv-cache-memory to manually provide this value from the logs:
            #   (EngineCore_DP0) INFO ... Available KV cache memory: X.XX GiB
            # For now, leave as None unless manually provided.
        
        print(f"  [INFO] num_gpu_blocks: {num_gpu_blocks}, block_size: {block_size}")
    except Exception as e:
        print(f"  [DEBUG] Could not access cache config: {e}")
    
    # Override with manual args if provided
    if args.kv_cache_memory is not None:
        kv_cache_metrics["available_kv_cache_memory_gib"] = args.kv_cache_memory
    if args.kv_cache_tokens is not None:
        kv_cache_metrics["kv_cache_size_tokens"] = args.kv_cache_tokens
    if args.max_concurrency is not None:
        kv_cache_metrics["max_concurrency"] = args.max_concurrency
    
    print(f"Engine initialized in {init_time:.2f}s")
    
    print("\n" + "=" * 60)
    print("KV CACHE METRICS (logged once - constant for all images)")
    print("=" * 60)
    print("Note: These metrics are set at engine initialization and")
    print("      do not change per-image. They reflect the memory")
    print("      optimization from the encoder cache.")
    print("-" * 60)
    
    # Check if we captured any metrics
    metrics_found = any(v is not None for v in kv_cache_metrics.values())
    for k, v in kv_cache_metrics.items():
        if v is not None:
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: (not available - see console logs)")
    
    if kv_cache_metrics.get("available_kv_cache_memory_gib") is None:
        print("")
        print("  TIP: To log available_kv_cache_memory_gib, find this line in console:")
        print("    '(EngineCore_DP0) INFO ... Available KV cache memory: X.XX GiB'")
        print("  Then re-run with: --kv-cache-memory 12.46")
    print("=" * 60)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=256, top_k=-1)

    # Run through images in dataset
    dataset = IMAGE_DATASET[:args.num_images] if args.num_images else IMAGE_DATASET
    
    all_results = []
    total_accuracy = 0
    total_tokens = 0
    total_gen_time = 0

    print(f"\n{'='*60}")
    print(f"Testing on {len(dataset)} images")
    print(f"{'='*60}")

    for i, image_data in enumerate(dataset):
        print(f"\n[{i+1}/{len(dataset)}] Testing: {image_data['name']}")
        print(f"  Image: {image_data['image_url'][:60]}...")
        print(f"  Prompt: {image_data['prompt']}")

        try:
            # Prepare input for this image
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image_data["image_url"]},
                    {"type": "text", "text": image_data["prompt"]}
                ]
            }]
            inputs = [prepare_inputs_for_vllm(messages, processor)]

            # Run generation
            gen_start = time.time()
            outputs = llm.generate(inputs, sampling_params=sampling_params)
            gen_time = time.time() - gen_start

            output_text = outputs[0].outputs[0].text
            num_tokens = len(outputs[0].outputs[0].token_ids)

            # Calculate accuracy for this image
            accuracy_results = calculate_accuracy(output_text, image_data["expected_keywords"])
        except Exception as e:
            print(f"  ERROR: Failed to process image - {str(e)[:100]}")
            print(f"  Skipping this image...")
            continue

        print(f"  Output: {output_text[:100]}...")
        print(f"  Accuracy: {accuracy_results['accuracy']:.2%} ({accuracy_results['keywords_found']}/{accuracy_results['keywords_total']})")
        print(f"  Found: {accuracy_results['found_keywords']}")
        print(f"  Missing: {accuracy_results['missing_keywords']}")
        print(f"  Tokens: {num_tokens}, Time: {gen_time:.2f}s")

        # Track results
        result = {
            "image_name": image_data["name"],
            "accuracy": accuracy_results["accuracy"],
            "keywords_found": accuracy_results["keywords_found"],
            "keywords_total": accuracy_results["keywords_total"],
            "output_tokens": num_tokens,
            "generation_time": gen_time,
            "output_text": output_text,
        }
        all_results.append(result)

        total_accuracy += accuracy_results["accuracy"]
        total_tokens += num_tokens
        total_gen_time += gen_time

        # Log individual image to wandb (per-image metrics only)
        if not args.no_wandb:
            wandb.log({
                f"image_{image_data['name']}_accuracy": accuracy_results["accuracy"],
                f"image_{image_data['name']}_tokens": num_tokens,
                f"image_{image_data['name']}_time": gen_time,
            })

    # Aggregate metrics
    avg_accuracy = total_accuracy / len(dataset) if dataset else 0
    avg_tokens_per_sec = total_tokens / total_gen_time if total_gen_time > 0 else 0

    metrics = {
        "init_time_seconds": init_time,
        "total_generation_time_seconds": total_gen_time,
        "total_output_tokens": total_tokens,
        "avg_tokens_per_second": avg_tokens_per_sec,
        "avg_accuracy": avg_accuracy,
        "num_images_tested": len(dataset),
        "success": True,
    }
    
    # Add KV cache metrics
    for k, v in kv_cache_metrics.items():
        if v is not None:
            metrics[k] = v

    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # Log to wandb
    if not args.no_wandb:
        wandb.log(metrics)
        
        # Log KV cache metrics ONCE to summary (constant for all images)
        # These metrics reflect the encoder cache memory optimization
        if kv_cache_metrics.get("available_kv_cache_memory_gib") is not None:
            wandb.run.summary["available_kv_cache_memory_gib"] = kv_cache_metrics["available_kv_cache_memory_gib"]
        if kv_cache_metrics.get("kv_cache_size_tokens") is not None:
            wandb.run.summary["kv_cache_size_tokens"] = kv_cache_metrics["kv_cache_size_tokens"]
        if kv_cache_metrics.get("max_concurrency") is not None:
            wandb.run.summary["max_concurrency"] = kv_cache_metrics["max_concurrency"]
        
        # Log all other metrics to summary
        for k, v in metrics.items():
            wandb.run.summary[k] = v
        
        # Create a table of per-image results
        table = wandb.Table(columns=["image", "accuracy", "keywords_found", "tokens", "time_s"])
        for r in all_results:
            table.add_data(r["image_name"], r["accuracy"], r["keywords_found"], r["output_tokens"], r["generation_time"])
        wandb.log({"image_results": table})
        
        wandb.finish()
        print(f"\nResults logged to wandb project: {args.project}")

    return metrics


if __name__ == "__main__":
    main()

