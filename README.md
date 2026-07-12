# Unlocking Parallelism in Autoregressive Language Models via Speculative Decoding with Progressive Tree Drafting

**Anonymous Submission for Peer Review**

This repository contains the implementation code for the paper "Unlocking Parallelism in Autoregressive Language Models
via Speculative Decoding with Progressive Tree Drafting" submitted to IJCAI 2026.

## Repository Structure

```
.
├── tree_draft/           # Core PTD implementation
│   ├── trees/           # Tree data structures and algorithms
│   ├── models/          # Model-specific implementations (LLaMA, Qwen)
│   └── decode_strategy/ # Decoding strategies (PTD, autoregressive baseline)
├── data/                q# Benchmark datasets
│   ├── mt-bench/       # Multi-turn conversation benchmark
│   ├── gsm/            # GSM8K math reasoning
│   ├── mbpp/           # Python code generation
│   └── mt-*/           # MT-bench subtasks
├── configs/            # Configuration files
├── run/                # Example scripts for reproducing experiments
├── results/            # Result processing and visualization scripts
└── main.py            # Main entry point
```

## Requirements

### Environment Setup

```bash
pip install -r requirements.txt
```

### Dependencies

- Python 3.9+
- PyTorch 2.0+
- Transformers 4.54.0
- Additional dependencies listed in `requirements.txt`

### Hardware Requirements

- GPU with at least 24GB memory for 7B models
- GPU with at least 40GB memory for 13B+ models
- Multi-GPU support available via `--num-gpus-per-model` flag

## Quick Start

### 1. Basic Usage

Run PTD on LLaMA-7B-chat with MT-bench:

```bash
python main.py \
    --model-path /path/to/llama-7b-chat-hf \
    --question-file data/mt-bench/mt-bench.jsonl \
    --run-mode draft \
    --sample-number -1
```

### 2. Autoregressive Baseline

For comparison with standard autoregressive decoding:

```bash
python main.py \
    --model-path /path/to/llama-7b-chat-hf \
    --question-file data/mt-bench/mt-bench.jsonl \
    --run-mode ar \
    --sample-number -1
```

## Configuration

### Tree Structure Configuration

PTD uses a progressive tree structure for token drafting. You can configure the tree in two ways:

#### Method 1: Environment Variables (Simple)

```bash
export MAX_DEPTH=6
export MAX_CHILD_NUM=4
python main.py \
    --model-path /path/to/llama-7b-chat-hf \
    --question-file data/mt-bench/mt-bench.jsonl \
    --run-mode draft
```

- `MAX_DEPTH`: Maximum tree depth (default: 6)
- `MAX_CHILD_NUM`: Maximum number of child nodes per parent (default: 4)

#### Method 2: Configuration File (Advanced)

Modify `configs/tree_config.json` for fine-grained control:

```json
{
  "ini_max_depth": 2,
  "max_depth": 6,
  "exclude_root": 1,
  "node_config": {
    "max_child_num": [
      4,
      4,
      4,
      4,
      4,
      4
    ],
    "min_child_num": [
      4,
      0,
      0,
      0,
      0,
      0
    ],
    "default_max_child_num": 4,
    "default_min_child_num": 4
  },
  "tree_update_method": "argmax"
}
```

You can specify a custom config file:

```bash
python main.py \
    --config-file /path/to/custom_config.json \
    --model-path /path/to/model \
    --question-file data/mt-bench/mt-bench.jsonl
```

## Benchmarks

This implementation supports multiple benchmarks:

| Benchmark | Description               | Data Path                      |
|-----------|---------------------------|--------------------------------|
| MT-Bench  | Multi-turn conversations  | `data/mt-bench/mt-bench.jsonl` |
| GSM8K     | Math reasoning            | `data/gsm/test.jsonl`          |
| MBPP      | Python code generation    | `data/mbpp/mbpp.jsonl`         |
| HumanEval | HumanEval code generation | `data/humaneval/`              |

For downloading datasets, please `cd data/` and `bash download_datasets.sh`.

## Reproducing Paper Results

Example scripts for reproducing experiments are provided in the `run/` directory. These scripts demonstrate:

- Multiple model sizes (7B, 13B, etc.)
- Benchmark-specific evaluations
- Different tree configurations (depth and width)

### Example: Running Tree Configuration Experiments

```bash
bash run/run-llama-7b-mt.sh
```

## Command Line Arguments

### Model and Data

- `--model-path`: Path to model weights (local path or HuggingFace repo ID)
- `--question-file`: Path to benchmark data file
- `--sample-number`: Number of samples to process (-1 for all, -2 for deterministic subset)
- `--config-file`: Path to tree configuration file

### Inference Settings

- `--run-mode`: Inference mode (`draft` for PTD, `ar` for autoregressive)
- `--temperature`: Sampling temperature (0.0 for greedy decoding)
- `--max-new-tokens`: Maximum number of tokens to generate (default: 1024)
- `--batch-size`: Batch size for inference (default: 1)

### Hardware Configuration

- `--num-gpus-per-model`: GPUs per model instance (default: 1)
- `--num-gpus-total`: Total number of GPUs available
- `--max-gpu-memory`: Max GPU memory per GPU (e.g., "20GB")
- `--dtype`: Data type (`float16`, `bfloat16`, `float32`)
- `--cpu-offloading`: Enable CPU offloading for large models

### Logging

- `--save-log`: Save logs (0 or 1, default: 1)
- `--log-path`: Directory for log files (default: `logs/`)
- `--log-level`: Logging level (`INFO`, `DEBUG`, `WARNING`)

## Output and Evaluation

Results are saved in the log directory specified by `--log-path`. Each run generates:

- Generation results with timing information
- Token acceptance statistics
- Detailed profiling data

## Supported Models

The implementation currently supports:

- **LLaMA-2 and LLaMA-3 family**
- **Qwen-2 and Qwen-3 family**

To add support for new models, implement the model-specific interface in `tree_draft/models/`.

