# Reproducing experiments

The repository contains benchmark inputs and tree configurations. Model
weights are intentionally not included; use the exact model revision from the
paper and record its Hugging Face ID or local snapshot path.

## Minimal run

```bash
MODEL_PATH=/path/to/model \
QUESTION_FILE=data/mt-bench/mt-bench.jsonl \
RUN_MODE=draft \
bash run/run_example.sh
```

Use `RUN_MODE=ar` for the autoregressive baseline. Set `SAMPLE_NUMBER=-1`
to process the full benchmark or use a positive number for a sampled run.

For each reported result, record the model revision, Transformers/PyTorch
versions, CUDA device, seed, tree configuration, temperature, maximum output
length, and the complete command line. Chat models should use their
model-specific tokenizer or FastChat conversation template; do not apply a
Llama-2 template to Qwen models.
