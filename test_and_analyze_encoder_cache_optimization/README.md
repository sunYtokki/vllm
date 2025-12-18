# Encoder Cache Optimization: Tests & Analysis

This directory contains test scripts and analysis tools for validating the **vLLM encoder cache optimization** for Qwen3-VL models. The tests measure KV cache memory improvements and ensure model quality is preserved (guardrail tests).

## Overview

The encoder cache optimization reduces GPU memory consumption by efficiently caching vision encoder outputs, resulting in:
- **More available KV cache memory**
- **Higher token capacity**
- **Increased max concurrency** (more simultaneous requests)

---

## Directory Contents

```
test_and_analyze_encoder_cache_optimization/
├── README.md                              # This file
├── analyze_wandb_results.py               # Analysis & plotting tool
├── test_qwen3_wandb_benchmark_test.py     # Memory benchmark test
├── test_qwen3_wandb_image_guardrail_test.py  # Functional image test
└── test_qwen3_wandb_video_guardrail_test.py  # Functional video test
```

---

## Test Scripts

### 1. Memory Benchmark Test
**File:** `test_qwen3_wandb_benchmark_test.py`

Measures raw KV cache metrics without running inference. Fast way to compare memory improvements across branches.

```bash
python test_qwen3_wandb_benchmark_test.py \
    --model Qwen/Qwen3-VL-2B-Instruct \
    --branch sun-optimized
```

**Metrics logged:**
- `available_kv_cache_memory_gib` - Available KV cache memory in GiB
- `kv_cache_size_tokens` - KV cache capacity in tokens
- `max_concurrency` - Maximum concurrent requests supported

---

### 2. Functional Image Test (Guardrail)
**File:** `test_qwen3_wandb_image_guardrail_test.py`

Tests image understanding quality while measuring KV cache metrics. Ensures optimization doesn't degrade model accuracy.

```bash
python test_qwen3_wandb_image_guardrail_test.py \
    --model Qwen/Qwen3-VL-2B-Instruct \
    --branch sun-optimized \
    --num-images 10
```

**Metrics logged:**
- KV cache metrics (memory, tokens, concurrency)
- `avg_accuracy` - Average accuracy on keyword matching
- Per-image accuracy and generation time

---

### 3. Functional Video Test (Guardrail)
**File:** `test_qwen3_wandb_video_guardrail_test.py`

Tests video understanding quality while measuring KV cache metrics.

```bash
python test_qwen3_wandb_video_guardrail_test.py \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --branch sun-optimized \
    --num-videos 5
```

**Metrics logged:**
- KV cache metrics (memory, tokens, concurrency)
- `avg_accuracy` - Average accuracy on keyword matching
- Per-video accuracy and generation time

---

## Analysis Tool

### `analyze_wandb_results.py`

Fetches results from Weights & Biases and generates comparison plots.

#### Memory Benchmark Analysis
```bash
python analyze_wandb_results.py --group memory-benchmark
```
**Output:** 3-panel bar chart comparing KV cache metrics across branches

#### Functional Test Analysis
```bash
# Image guardrail tests
python analyze_wandb_results.py --group functional-image

# Video guardrail tests
python analyze_wandb_results.py --group functional-video
```
**Output:** 2x2 panel chart with:
1. Available KV Cache Memory (GiB)
2. KV Cache Size (Tokens)
3. Max Concurrency
4. Average Accuracy (%) - Guardrail metric

#### Custom Output Path
```bash
python analyze_wandb_results.py --group memory-benchmark --output my_comparison.png
```

---

## Weights & Biases Integration

All tests log results to **Weights & Biases** for tracking and comparison.

**Project:** `vllm-encoder-cache-optimization`

**Groups:**
| Group | Description |
|-------|-------------|
| `memory-benchmark` | Memory-only benchmark runs |
| `functional-image` | Image guardrail test runs |
| `functional-video` | Video guardrail test runs |

---

## Key Metrics

### KV Cache Metrics (Optimization Proof)
| Metric | Description |
|--------|-------------|
| `available_kv_cache_memory_gib` | GPU memory available for KV cache |
| `kv_cache_size_tokens` | Number of tokens the KV cache can hold |
| `max_concurrency` | Max concurrent requests (tokens / max_model_len) |

### Guardrail Metric (Quality Assurance)
| Metric | Description |
|--------|-------------|
| `avg_accuracy` | Keyword-based accuracy on test prompts |

---

## Supported Models

| Model | Size | Notes |
|-------|------|-------|
| `Qwen/Qwen3-VL-2B-Instruct` | 2B | Fastest for testing |
| `Qwen/Qwen3-VL-4B-Instruct` | 4B | Default for video tests |
| `Qwen/Qwen3-VL-8B-Instruct-FP8` | 8B | FP8 quantized |

---

## Results

### Memory Benchmark (`memory-benchmark` group)

KV cache memory improvements comparing `sun-optimized` vs `main` branch:

| Model | Main (GiB) | Optimized (GiB) | KV Tokens | Max Concurrency | Improvement |
|-------|------------|-----------------|-----------|-----------------|-------------|
| Qwen3-VL-8B-Instruct-FP8 | 4.12 | 8.45 | 29,968 → 61,536 | 7.32 → 15.02 | **+105.1%** |
| Qwen3-VL-4B-Instruct | 7.57 | 10.29 | 55,136 → 74,912 | 13.46 → 18.29 | **+35.9%** |
| Qwen3-VL-2B-Instruct | 12.55 | 14.69 | 117,520 → 137,520 | 28.69 → 33.57 | **+17.0%** |

---

### Functional Image Test (`functional-image` group)

Image understanding quality guardrail test results:

| Model | Main (GiB) | Optimized (GiB) | Memory Gain | Accuracy (main) | Accuracy (opt) |
|-------|------------|-----------------|-------------|-----------------|----------------|
| Qwen3-VL-8B-Instruct-FP8 | 3.93 | 8.31 | **+111.5%** | 71.8% | 71.8% ✓ |
| Qwen3-VL-4B-Instruct | 7.46 | 10.20 | **+36.7%** | 75.3% | 75.3% ✓ |
| Qwen3-VL-2B-Instruct | 12.46 | 14.62 | **+17.3%** | 66.4% | 66.4% ✓ |

**Key Finding:** Accuracy is preserved (0.0 percentage point change) across all models with the optimization.

---

### Functional Video Test (`functional-video` group)

Video understanding quality guardrail test results:

| Model | Main (GiB) | Optimized (GiB) | Memory Gain | Accuracy (main) | Accuracy (opt) |
|-------|------------|-----------------|-------------|-----------------|----------------|
| Qwen3-VL-8B-Instruct-FP8 | 3.93 | 8.31 | **+111.5%** | 33.0% | 33.0% ✓ |
| Qwen3-VL-4B-Instruct | 7.46 | 10.20 | **+36.7%** | 29.0% | 29.0% ✓ |
| Qwen3-VL-2B-Instruct | 12.46 | 14.62 | **+17.3%** | 36.0% | 36.0% ✓ |

**Key Finding:** Video accuracy is identical between branches, confirming the optimization preserves model quality.

---

### Summary

| Metric | 8B-FP8 | 4B | 2B |
|--------|--------|-----|-----|
| **Memory Improvement** | +105-111% | +35-37% | +17% |
| **Image Accuracy Preserved** | ✓ 71.8% | ✓ 75.3% | ✓ 66.4% |
| **Video Accuracy Preserved** | ✓ 33.0% | ✓ 29.0% | ✓ 36.0% |

---

## Quick Start

### Run All Benchmarks
```bash
cd test_and_analyze_encoder_cache_optimization

# Run memory benchmark for all models
for model in "Qwen/Qwen3-VL-2B-Instruct" "Qwen/Qwen3-VL-4B-Instruct" "Qwen/Qwen3-VL-8B-Instruct-FP8"; do
    python test_qwen3_wandb_benchmark_test.py --model "$model" --branch sun-optimized
done

# Run image guardrail test
python test_qwen3_wandb_image_guardrail_test.py --model Qwen/Qwen3-VL-2B-Instruct --num-images 10

# Run video guardrail test
python test_qwen3_wandb_video_guardrail_test.py --model Qwen/Qwen3-VL-4B-Instruct --num-videos 5
```

### Analyze Results
```bash
# Generate comparison plots
python analyze_wandb_results.py --group memory-benchmark
python analyze_wandb_results.py --group functional-image
python analyze_wandb_results.py --group functional-video
```

---

## Skipping WandB Logging

For local testing without WandB:

```bash
python test_qwen3_wandb_image_guardrail_test.py --no-wandb
```

---

## Requirements

```bash
pip install wandb transformers qwen-vl-utils pandas matplotlib seaborn
```

For video tests, also install:
```bash
pip install decord  # Preferred video backend
# or
pip install torchvision  # Fallback
```
