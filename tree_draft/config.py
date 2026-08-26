"""
TreeDraft unified configuration module.

Configuration priority (highest to lowest):
1. Environment variables
2. CLI arguments (via override methods)
3. Config file
4. Defaults
"""
import os
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
try:
    from loguru import logger
except ImportError:  # Keep config inspection usable before runtime dependencies are installed.
    import logging

    logger = logging.getLogger("tree_draft")


@dataclass
class NodeConfig:
    """Node configuration."""
    max_child_num: List[int] = field(default_factory=lambda: [4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
    min_child_num: List[int] = field(default_factory=lambda: [4, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    default_max_child_num: int = 4
    default_min_child_num: int = 4
    update_sub_nodes: bool = False
    ignore_exceed: bool = False


@dataclass
class TreeConfig:
    """Tree structure configuration."""
    ini_max_depth: int = 2
    max_depth: int = 6
    exclude_root: int = 1
    node_config: NodeConfig = field(default_factory=NodeConfig)
    tree_config: Dict[str, Any] = field(default_factory=lambda: {"root_visible": False})
    tree_update_method: str = 'argmax'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict (for backward compatibility)."""
        result = asdict(self)
        result['node_config'] = asdict(self.node_config)
        return result


@dataclass
class InferenceConfig:
    """Inference configuration."""
    run_mode: str = 'draft'  # 'draft' or 'ar'
    temperature: float = 0.0
    max_new_tokens: int = 1024
    batch_size: int = 1
    sample_number: int = -2
    num_choices: int = 1

    # Distributed
    num_gpus_per_model: int = 1
    num_gpus_total: int = 1
    local_rank: int = 0
    dist_workers: int = 1

    # Model loading
    dtype: str = 'bfloat16'
    use_pp: int = 0
    use_tp_ds: int = 0
    use_flash: int = 0
    cpu_offloading: bool = False
    max_gpu_memory: Optional[str] = None


@dataclass
class LogConfig:
    """Logging configuration."""
    save_log: bool = True
    log_path: str = 'logs/'
    log_level: str = 'INFO'
    chat: int = 1


@dataclass
class ModelConfig:
    """Model configuration."""
    model_path: Optional[str] = None
    question_file: str = 'data/mt-bench/mt-bench.jsonl'
    corpus_path: Optional[str] = None

    # Derived (computed from above)
    model_arch: Optional[str] = None
    bench_name: Optional[str] = None
    inference_id: Optional[str] = None


@dataclass
class PTDConfig:
    """Progressive Tree Drafting full configuration."""
    tree: TreeConfig = field(default_factory=TreeConfig)
    # cache: CacheConfig = field(default_factory=CacheConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    log: LogConfig = field(default_factory=LogConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    # Feature flags (env overrides)
    pld_ini: int = 0
    rest_ini: int = 0
    ptd_enabled: int = 1
    draw_dist: int = 0
    color_print: int = 0
    enable_profiling: bool = False

    @classmethod
    def from_config_file(cls, config_path: Optional[str] = None) -> 'PTDConfig':
        """
        Load config from file.

        Args:
            config_path: Path to config file; None uses default path.

        Returns:
            PTDConfig instance.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "tree_config.json"

        try:
            with open(config_path, 'r') as f:
                data = json.load(f)

            node_config = NodeConfig(**data.get('node_config', {}))

            tree_data = {
                'ini_max_depth': data.get('ini_max_depth', 2),
                'max_depth': data.get('max_depth', 6),
                'exclude_root': data.get('exclude_root', 1),
                'node_config': node_config,
                'tree_config': data.get('tree_config', {"root_visible": False}),
                'tree_update_method': data.get('tree_update_method', 'argmax'),
            }
            tree_config = TreeConfig(**tree_data)

            config = cls(tree=tree_config)

            logger.info(f"Config loaded: {config_path}")
            return config

        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return cls()
        except json.JSONDecodeError as e:
            logger.error(f"Config parse error: {e}")
            raise ValueError(f"Invalid config file: {config_path}") from e
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def apply_env_overrides(self) -> 'PTDConfig':
        """
        Apply environment variable overrides (highest priority).
        """
        if (val := os.getenv('MAX_DEPTH')) is not None:
            self.tree.max_depth = int(val)
            logger.info(f"Env override: MAX_DEPTH={val}")
        if (val := os.getenv('TREE_UPDATE_METHOD')) is not None:
            self.tree.tree_update_method = val.strip().lower()
            logger.info(f"Env override: TREE_UPDATE_METHOD={val}")
        if (val := os.getenv('MAX_CHILD_NUM')) is not None:
            child_num = int(val)
            for i in range(len(self.tree.node_config.max_child_num)):
                self.tree.node_config.max_child_num[i] = child_num
            self.tree.node_config.min_child_num[0] = child_num
            self.tree.node_config.default_max_child_num = child_num
            logger.info(f"Env override: MAX_CHILD_NUM={val}")

        if (val := os.getenv('PLD_INI')) is not None:
            self.pld_ini = int(val)
            logger.debug(f"Env: PLD_INI={val}")

        if (val := os.getenv('REST_INI')) is not None:
            self.rest_ini = int(val)
            logger.debug(f"Env: REST_INI={val}")

        if (val := os.getenv('PTD')) is not None:
            self.ptd_enabled = int(val)
            logger.debug(f"Env: PTD={val}")

        if (val := os.getenv('DRAW_DIST')) is not None:
            self.draw_dist = int(val)
            logger.debug(f"Env: DRAW_DIST={val}")

        if (val := os.getenv('COLOR_PRINT')) is not None:
            self.color_print = int(val)
            logger.debug(f"Env: COLOR_PRINT={val}")

        if (val := os.getenv('ENABLE_PROFILING')) is not None:
            self.enable_profiling = val.lower() in ('1', 'true', 'yes')
            logger.debug(f"Env: ENABLE_PROFILING={val}")

        if (val := os.getenv('CUDA_VISIBLE_DEVICES')) is not None:
            logger.info(f"CUDA devices: {val}")

        return self

    def override_from_args(self, args) -> 'PTDConfig':
        """
        Override config from CLI args.

        Args:
            args: argparse.Namespace from parser.
        """
        if hasattr(args, 'run_mode'):
            self.inference.run_mode = args.run_mode
        if hasattr(args, 'temperature'):
            self.inference.temperature = args.temperature
        if hasattr(args, 'max_new_tokens'):
            self.inference.max_new_tokens = args.max_new_tokens
        if hasattr(args, 'batch_size'):
            self.inference.batch_size = args.batch_size
        if hasattr(args, 'sample_number'):
            self.inference.sample_number = args.sample_number
        if hasattr(args, 'num_gpus_per_model'):
            self.inference.num_gpus_per_model = args.num_gpus_per_model
        if hasattr(args, 'num_gpus_total'):
            self.inference.num_gpus_total = args.num_gpus_total
        if hasattr(args, 'dtype'):
            self.inference.dtype = args.dtype
        if hasattr(args, 'use_pp'):
            self.inference.use_pp = args.use_pp
        if hasattr(args, 'use_tp_ds'):
            self.inference.use_tp_ds = args.use_tp_ds
        if hasattr(args, 'use_flash'):
            self.inference.use_flash = args.use_flash
        if hasattr(args, 'cpu_offloading'):
            self.inference.cpu_offloading = args.cpu_offloading
        if hasattr(args, 'max_gpu_memory'):
            self.inference.max_gpu_memory = args.max_gpu_memory
        if hasattr(args, 'local_rank'):
            self.inference.local_rank = args.local_rank
        if hasattr(args, 'dist_workers'):
            self.inference.dist_workers = args.dist_workers

        if getattr(args, 'model_path', None):
            self.model.model_path = args.model_path
        if hasattr(args, 'question_file'):
            self.model.question_file = args.question_file
        if hasattr(args, 'corpus_path'):
            self.model.corpus_path = args.corpus_path

        if hasattr(args, 'model_arch'):
            self.model.model_arch = args.model_arch
        if hasattr(args, 'bench_name'):
            self.model.bench_name = args.bench_name
        if hasattr(args, 'inference_id'):
            self.model.inference_id = args.inference_id

        if hasattr(args, 'save_log'):
            self.log.save_log = bool(args.save_log)
        if hasattr(args, 'log_path'):
            self.log.log_path = args.log_path
        if hasattr(args, 'log_level'):
            self.log.log_level = args.log_level
        if hasattr(args, 'chat'):
            self.log.chat = args.chat

        return self

    @classmethod
    def create_from_args(cls, args, config_file: Optional[str] = None) -> 'PTDConfig':
        """
        Create config from CLI args (convenience).
        Priority: env > CLI > config file > defaults.

        Args:
            args: CLI namespace.
            config_file: Optional config file path.

        Returns:
            PTDConfig instance.
        """
        config = cls.from_config_file(config_file)
        config.apply_env_overrides()
        config.override_from_args(args)
        config._ensure_derived_model_attrs()
        return config

    def _ensure_derived_model_attrs(self) -> None:
        """Compute and set model_arch, bench_name, inference_id from paths and inference settings."""
        self.model.model_arch = self._compute_model_arch()
        self.model.bench_name = self._compute_bench_name()
        self.model.inference_id = self._compute_inference_id()

    def _compute_model_arch(self) -> str:
        """Derive model_arch from model_path (basename)."""
        p = self.model.model_path
        if not p:
            return ""
        if p.endswith(os.sep) or p.endswith("/"):
            p = p.rstrip("/").rstrip(os.sep)
        return os.path.basename(p)

    def _compute_bench_name(self) -> str:
        """Derive bench_name from question_file (first dir under 'data')."""
        path = self.model.question_file
        norm_path = os.path.normpath(path)
        parts = norm_path.split(os.sep)
        indices = [i for i, p in enumerate(parts) if p == "data"]
        if not indices:
            raise ValueError("No 'data' directory found in path!")
        last_idx = indices[-1]
        if last_idx + 1 >= len(parts):
            raise ValueError("No directory found after the last 'data'!")
        return parts[last_idx + 1]

    def _compute_inference_id(self) -> str:
        """Build inference_id from model_arch, bench_name, run_mode, etc."""
        return (
            f"{self._compute_model_arch()}-"
            f"{self._compute_bench_name()}-"
            f"run_mode-{self.inference.run_mode}-"
            f"sample_number-{self.inference.sample_number}-"
            f"max_new_tokens-{self.inference.max_new_tokens}-"
            f"temperature-{self.inference.temperature}"
        )

    def validate(self) -> bool:
        """
        Validate config.

        Returns:
            True if valid.
        """
        errors = []

        if self.tree.max_depth < 1:
            errors.append(f"max_depth must be >= 1, got {self.tree.max_depth}")

        if self.tree.ini_max_depth > self.tree.max_depth:
            errors.append(
                f"ini_max_depth ({self.tree.ini_max_depth}) must not exceed max_depth ({self.tree.max_depth})")
        if self.tree.tree_update_method not in ('argmax', 'sample', 'topk', 'topp'):
            errors.append(
                f"Unsupported tree_update_method: {self.tree.tree_update_method}, expected one of argmax/sample/topk/topp")

        if self.inference.run_mode not in ['draft', 'ar']:
            errors.append(f"Unsupported run_mode: {self.inference.run_mode}")

        if self.inference.batch_size < 1:
            errors.append(f"batch_size must be >= 1, got {self.inference.batch_size}")

        if self.inference.dtype not in ['float32', 'float64', 'float16', 'bfloat16']:
            errors.append(f"Unsupported dtype: {self.inference.dtype}")

        if not self.model.model_path:
            errors.append("model_path is required; pass --model-path or set it in a config file")
        else:
            try:
                if (not Path(self.model.model_path).exists()
                        and not self.model.model_path.startswith(('http://', 'https://'))):
                    logger.warning(
                        f"Model path does not exist locally; it will be resolved as a Hub ID: "
                        f"{self.model.model_path}"
                    )
            except (PermissionError, OSError) as e:
                logger.debug(f"Cannot access model path {self.model.model_path}: {e}")

        if errors:
            for error in errors:
                logger.error(f"Validation failed: {error}")
            return False

        return True

    def log_config(self):
        """Print current config (for debugging)."""
        logger.info("=" * 80)
        logger.info("TreeDraft config:")
        logger.info(f"  run_mode: {self.inference.run_mode}")
        logger.info(f"  model_path: {self.model.model_path}")
        logger.info(f"  question_file: {self.model.question_file}")
        logger.info(f"  max_depth: {self.tree.max_depth}")
        logger.info(f"  tree_update_method: {self.tree.tree_update_method}")
        logger.info(f"  default_max_child_num: {self.tree.node_config.default_max_child_num}")
        logger.info(f"  batch_size: {self.inference.batch_size}")
        logger.info(f"  temperature: {self.inference.temperature}")
        logger.info(f"  max_new_tokens: {self.inference.max_new_tokens}")
        logger.info(f"  dtype: {self.inference.dtype}")
        logger.info("=" * 80)


_global_config: Optional[PTDConfig] = None


def get_global_config() -> PTDConfig:
    """Return global config (must have been set)."""
    global _global_config
    if _global_config is None:
        raise RuntimeError("Global config not set; call set_global_config() first")
    return _global_config


def set_global_config(config: PTDConfig):
    """Set global config instance."""
    global _global_config
    _global_config = config


def reset_global_config():
    """Reset global config (mainly for tests)."""
    global _global_config
    _global_config = None
