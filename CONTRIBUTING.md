# Contributing

Thanks for helping improve PTD. Please keep changes focused and include a
small reproduction or test when changing decoding, tree construction, model
adapters, or configuration behavior.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[runtime,dev]"
```

Use a PyTorch wheel that matches your CUDA version. GPU model weights are not
required for the unit tests.

## Before opening a pull request

```bash
python main.py --help
python -m compileall -q main.py tree_draft
pytest
```

Please describe the hardware, model revision, benchmark, and command used for
any performance or reproducibility change. Do not commit model weights, local
paths, generated logs, or benchmark outputs unless they are intentionally
reviewed artifacts.
