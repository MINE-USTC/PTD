# Changelog

## Unreleased

- Added a portable `run/run_example.sh` entry point.
- Made the CLI usable without importing the GPU inference stack for `--help`.
- Selected chat templates from the actual model path instead of a hard-coded
  Llama-2 model ID.
- Added packaging metadata, CPU-only tests, CI, reproducibility notes, and
  contribution guidelines.
- Added the Apache-2.0 license and arXiv metadata in `CITATION.cff`.
- Removed broken MBPP scripts, the unused machine-local `run.env`, and dead
  imports/empty loader code.
