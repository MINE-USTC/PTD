import subprocess
import sys

from main import build_parser


def test_model_path_is_not_machine_specific():
    args = build_parser().parse_args([])
    assert args.model_path is None


def test_help_does_not_require_runtime_dependencies():
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--model-path" in result.stdout
