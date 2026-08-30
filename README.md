# Unlocking Parallelism in Autoregressive Language Models via Speculative Decoding with Progressive Tree Drafting

This repository contains the implementation code for the paper "Unlocking Parallelism in Autoregressive Language Models
via Speculative Decoding with Progressive Tree Drafting" submitted to COLM 2026.

## Repository Structure

```
.
├── tree_draft/           # Core PTD implementation
│   ├── trees/           # Tree data structures and algorithms
│   ├── models/          # Model-specific implementations (LLaMA, Qwen)
│   └── decode_strategy/ # Decoding strategies (PTD, autoregressive baseline)
├── data/                # Included benchmark inputs
│   ├── mt-bench/       # Multi-turn conversation benchmark
│   ├── gsm/            # GSM8K math reasoning
│   └── mt-*/           # MT-bench subtasks
├── configs/            # Tree configuration files
├── run/                # Portable and legacy experiment scripts
├── docs/               # Reproducibility notes
└── main.py            # Main entry point
```

## Requirements

### Environment Setup

```bash
# Install a PyTorch wheel matching your CUDA version first.
pip install -e ".[runtime]"
```

For development, install the test and lint tools as well:

```bash
pip install -e ".[runtime,dev]"
```

### Dependencies

- Python 3.9+
- PyTorch 2.0+
- Transformers 4.54.0+
- Additional dependencies listed in `requirements.txt`

### Hardware Requirements

- GPU memory requirements depend on the model, dtype, and offloading settings.
- 24GB is a practical starting point for many 7B configurations.
- Multi-GPU support available via `--num-gpus-per-model` flag

## Quick Start

### 1. Basic Usage

Run PTD with a local model or a Hugging Face model ID:

```bash
python main.py \
    --model-path /path/to/model \
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
| HumanEval | HumanEval code generation | `data/humaneval/`              |

The included files are small benchmark inputs. The repository does not bundle
model weights or a dataset downloader; follow each benchmark's official
license and source instructions when obtaining additional data.

## Reproducing Paper Results

Example scripts for reproducing experiments are provided in the `run/` directory. These scripts demonstrate:

- Multiple model sizes (7B, 13B, etc.)
- Benchmark-specific evaluations
- Different tree configurations (depth and width)

### Example: Running Tree Configuration Experiments

The portable entry point avoids author-specific paths:

```bash
MODEL_PATH=/path/to/model \
QUESTION_FILE=data/mt-bench/mt-bench.jsonl \
RUN_MODE=draft \
SAMPLE_NUMBER=1 \
bash run/run_example.sh
```

Set `RUN_MODE=ar` for the autoregressive baseline and `SAMPLE_NUMBER=-1` to
process the complete benchmark. Older scripts in `run/` retain the original
paper hardware paths and should be treated as historical examples.

## Command Line Arguments

### Model and Data

- `--model-path`: Path to model weights (local path or Hugging Face repo ID); required
- `--question-file`: Path to benchmark data file
- `--sample-number`: Number of samples to process (-1 for all, -2 for the default deterministic subset)
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

## Prompt templates

Prompt formatting must match the selected model. The public implementation
selects the FastChat conversation template from the actual `model_path` rather
than using a hard-coded Llama-2 model ID. For Qwen2/Qwen3 or custom chat
models, verify the tokenizer's official template and prefer
`tokenizer.apply_chat_template(..., add_generation_prompt=True)` when it is
available. See `docs/reproducibility.md` for experiment-recording guidance.

## Development

```bash
python main.py --help
python -m compileall -q main.py tree_draft
pytest
```

See `CONTRIBUTING.md` before opening a pull request. Do not commit model
weights, local filesystem paths, generated logs, or unreviewed result dumps.

## Citation

If PTD is useful in your research, please cite the accompanying paper. The
repository includes machine-readable metadata in [`CITATION.cff`](CITATION.cff)
and the paper is available at
[arXiv:2607.10661](https://arxiv.org/abs/2607.10661).

```bibtex
@article{gao2026unlocking,
  title   = {Unlocking Parallelism in Autoregressive Language Models via Speculative Decoding with Progressive Tree Drafting},
  author  = {Gao, Zipeng and Zheng, Zhi and Xia, Qingrong and Lin, Junda and Zhao, Ziwei and Xu, Tong and Wang, Zhefeng and Chen, Enhong},
  journal = {arXiv preprint arXiv:2607.10661},
  year    = {2026},
  doi     = {10.48550/arXiv.2607.10661}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE). Model
weights, benchmark datasets, and third-party dependencies may have separate
licenses; review their terms before redistribution.
