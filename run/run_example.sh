#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to a local model directory or Hugging Face model ID}"

QUESTION_FILE="${QUESTION_FILE:-data/mt-bench/mt-bench.jsonl}"
RUN_MODE="${RUN_MODE:-draft}"
SAMPLE_NUMBER="${SAMPLE_NUMBER:--2}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
OUTPUT_DIR="${OUTPUT_DIR:-logs}"

mkdir -p "$OUTPUT_DIR"

python main.py \
  --model-path "$MODEL_PATH" \
  --question-file "$QUESTION_FILE" \
  --run-mode "$RUN_MODE" \
  --sample-number "$SAMPLE_NUMBER" \
  --temperature "$TEMPERATURE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --log-path "$OUTPUT_DIR"
