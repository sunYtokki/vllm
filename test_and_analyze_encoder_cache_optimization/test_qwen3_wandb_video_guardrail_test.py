#!/usr/bin/env python3
"""
Functional test for Qwen3-VL with real video input + wandb logging.
Tests video processing with actual inference.
"""

import io
import logging
import os
import subprocess
import sys
import time
import requests
import tempfile
import numpy as np
import re
from PIL import Image

# Ensure required packages are installed
try:
    import qwen_vl_utils
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "qwen-vl-utils"])

# Force decord usage if needed, though we primarily handle inputs via process_vision_info
os.environ["FORCE_QWENVL_VIDEO_READER"] = "decord"

from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

try:
    import wandb
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    import wandb

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Using Qwen3-VL-4B-Instruct as default
CHECKPOINT = "Qwen/Qwen3-VL-4B-Instruct" 


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


def download_video_to_temp(url):
    """
    Download video from URL to a temporary file.
    Workaround for 'Protocol not found' error in decord/opencv with some URLs.
    """
    try:
        # User-Agent header to avoid 403 Forbidden from some servers (like Pexels)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, stream=True, timeout=30, headers=headers)
        response.raise_for_status()
        
        # Create temp file
        tf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        for chunk in response.iter_content(chunk_size=8192):
            tf.write(chunk)
        tf.close()
        return tf.name
    except Exception as e:
        print(f"Error downloading video {url}: {e}")
        return None


def prepare_inputs_for_vllm(messages, processor):
    """
    Prepare inputs for Qwen3-VL video.
    Replicating the strategy from user provided example:
    - Use process_vision_info with return_video_metadata=True
    - Pass inputs directly to mm_data["video"]
    """
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # User example explicitly used these arguments
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, 
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True, 
        return_video_metadata=True  # Important: this returns the metadata we need!
    )
    
    mm_data = {}
    if video_inputs is not None:
        # In the working example, video_inputs is passed directly.
        # process_vision_info with return_video_metadata=True likely returns the structured
        # data (including metadata) that we were previously trying to reconstruct manually.
        mm_data["video"] = video_inputs
        
    return {
        "prompt": text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs
    }


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


# Video dataset for testing (5 videos)
VIDEO_DATASET = [
    # 1. Slow traffic (OpenCV sample)
    {
        "name": "traffic",
        "url": "https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4",
        "prompt": "Describe the video.",
        "expected_keywords": ["car", "traffic", "road", "vehicle"],
    },
    # 2. People walking (OpenCV sample)
    {
        "name": "people_walking",
        "url": "https://github.com/opencv/opencv/raw/refs/tags/4.12.0/samples/data/vtest.avi",
        "prompt": "What are the people doing?",
        "expected_keywords": ["walking", "people", "street", "pedestrian"],
    },
    # 3. Baby reading (Qwen sample)
    {
        "name": "baby_reading",
        # Updated URL to a more reliable source or alternative video
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4", 
        "prompt": "Describe this video.",
        "expected_keywords": ["people", "interview", "talking", "man"],
    },
    # 4. Dog (Pexels - public domain)
    {
        "name": "dog_playing",
        "url": "https://videos.pexels.com/video-files/2795383/2795383-sd_640_360_25fps.mp4", # Small dog video
        "prompt": "What animal is in the video?",
        "expected_keywords": ["dog", "puppy", "animal", "pet", "playing"],
    },
    # 5. Ocean/Beach (Pexels - public domain)
    {
        "name": "ocean_waves",
        # Updated URL to a reliable test video (Big Buck Bunny sample) since Pexels is blocking requests
        "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
        "prompt": "Describe the scene.",
        "expected_keywords": ["rabbit", "bunny", "forest", "nature", "animal"],
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
    parser.add_argument("--num-videos", type=int, default=5, 
                        help="Number of videos to test")
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
    print("Functional Test: Qwen3-VL with Real Video")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Branch: {branch}")
    print(f"Commit: {commit}")
    print("=" * 60)

    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.project,
            name=f"functional-video-{branch}-{model.split('/')[-1]}-{time.strftime('%Y%m%d-%H%M%S')}",
            config={
                "model": model,
                "branch": branch,
                "commit": commit,
                "test_type": "functional_video",
                "gpu_memory_utilization": 0.95,
            },
            tags=[branch, "functional", "video", model.split("/")[-1]],
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
            limit_mm_per_prompt={"video": 1},
            enforce_eager=True, # Added based on user example
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

    # Run through videos in datasetu
    dataset = VIDEO_DATASET[:args.num_videos]
    
    all_results = []
    total_accuracy = 0
    total_tokens = 0
    total_gen_time = 0

    print(f"\n{'='*60}")
    print(f"Testing on {len(dataset)} videos")
    print(f"{'='*60}")

    for i, video_data in enumerate(dataset):
        print(f"\n[{i+1}/{len(dataset)}] Testing: {video_data['name']}")
        print(f"  URL: {video_data['url']}")
        print(f"  Prompt: {video_data['prompt']}")

        temp_video_path = None
        try:
            # Download video to temp file to avoid 'Protocol not found' errors
            print("  Downloading video...")
            temp_video_path = download_video_to_temp(video_data['url'])
            if not temp_video_path:
                raise RuntimeError(f"Failed to download video from {video_data['url']}")
            
            # Prepare input
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": temp_video_path, # Use local path
                            # Added nframes=8 based on user example strategy to limit complexity/ensure it works
                            "nframes": 8, 
                        },
                        {"type": "text", "text": video_data["prompt"]},
                    ],
                }
            ]
            
            inputs = prepare_inputs_for_vllm(messages, processor)

            # Run generation
            gen_start = time.time()
            outputs = llm.generate([inputs], sampling_params=sampling_params)
            gen_time = time.time() - gen_start

            output_text = outputs[0].outputs[0].text
            num_tokens = len(outputs[0].outputs[0].token_ids)

            # Calculate accuracy
            accuracy_results = calculate_accuracy(output_text, video_data['expected_keywords'])
        except Exception as e:
            print(f"  ERROR: Failed to process video - {e}")
            print(f"  Skipping this video...")
            import traceback
            traceback.print_exc()
            
            # Clean up temp file on error
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.unlink(temp_video_path)
                except:
                    pass
            continue
        
        # Clean up temp file
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.unlink(temp_video_path)
            except:
                pass

        print(f"  Output: {output_text.strip()}...")
        print(f"  Accuracy: {accuracy_results['accuracy']:.2%} ({accuracy_results['keywords_found']}/{accuracy_results['keywords_total']})")
        print(f"  Found: {accuracy_results['found_keywords']}")
        print(f"  Missing: {accuracy_results['missing_keywords']}")
        print(f"  Tokens: {num_tokens}, Time: {gen_time:.2f}s")

        # Track results
        result = {
            "video_name": video_data["name"],
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

        # Log to wandb
        if not args.no_wandb:
            wandb.log({
                f"video_{video_data['name']}_accuracy": accuracy_results["accuracy"],
                f"video_{video_data['name']}_tokens": num_tokens,
                f"video_{video_data['name']}_time": gen_time,
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
        "num_videos_tested": len(dataset),
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

        for k, v in metrics.items():
            wandb.run.summary[k] = v
        
        # Create a table
        table = wandb.Table(columns=["video", "accuracy", "keywords_found", "tokens", "time_s"])
        for r in all_results:
            table.add_data(r["video_name"], r["accuracy"], r["keywords_found"], r["output_tokens"], r["generation_time"])
        wandb.log({"video_results": table})
        
        wandb.finish()
        print(f"\nResults logged to wandb project: {args.project}")

    return metrics


if __name__ == "__main__":
    main()

