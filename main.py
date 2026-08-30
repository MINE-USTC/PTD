import argparse


def build_parser():
    """Build the CLI parser without importing GPU inference dependencies."""
    parser = argparse.ArgumentParser(description="Progressive Tree Drafting for Speculative Decoding")
    # model and data
    parser.add_argument("--model-path", type=str, default=None,
                        help="Local model directory or Hugging Face model ID (required).")
    parser.add_argument('--question-file', type=str,
                        default="data/mt-bench/mt-bench.jsonl",
                        help="The path to the question file.")
    parser.add_argument('--config-file', type=str, default=None,
                        help="Path to tree config file (default: configs/tree_config.json)")

    parser.add_argument('--sample-number', type=int, default=-2, help="The number of samples to generate.")
    parser.add_argument("--cpu-offloading", action="store_true")
    parser.add_argument("--num-choices", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="The maximum number of new generated tokens.", )
    parser.add_argument("--num-gpus-per-model", type=int, default=1,
                        help="The number of GPUs per model.", )
    parser.add_argument("--num-gpus-total", type=int, default=1,
                        help="The total number of GPUs.")
    parser.add_argument("--max-gpu-memory", type=str,
                        help="Maximum GPU memory used for model weights per GPU.", )
    parser.add_argument("--dtype", type=str, choices=["float32", "float64", "float16", "bfloat16"],
                        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
                        default='bfloat16', )
    parser.add_argument("--local-rank", type=int, default=0, )
    parser.add_argument("--use-pp", type=int, default=0, )
    parser.add_argument("--use-tp-ds", type=int, default=0, )
    parser.add_argument("--use-flash", type=int, default=0, )
    parser.add_argument("--dist-workers", default=1)
    # logs
    parser.add_argument('--save-log', type=int, default=1)
    parser.add_argument('--chat', type=int, default=1)
    parser.add_argument('--log-path', type=str, default='logs/', help="The path to the log file.")
    parser.add_argument("--log-level", default='INFO')
    # inference
    parser.add_argument('--run-mode', type=str, default='draft', choices=['draft', 'ar'],
                        help="Inference mode: 'draft' for tree drafting, 'ar' for autoregressive")
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1, )
    parser.add_argument('--corpus-path', type=str, default=None)
    return parser


def main():
    args = build_parser().parse_args()

    # Keep --help usable without importing torch/Transformers/FastChat.
    import tree_draft
    from tree_draft.config import PTDConfig
    from tree_draft.inference_modes import greedy_decoding, base_tree_draft, ini_model

    # Build config (priority: env > CLI > config file > defaults)
    config = PTDConfig.create_from_args(args, config_file=args.config_file)

    if not config.validate():
        raise ValueError("Config validation failed; check parameters")

    config.log_config()

    # Sync derived fields to args for ini_model / ini_model_paras
    args.model_arch = config.model.model_arch
    args.bench_name = config.model.bench_name
    args.inference_id = config.model.inference_id

    tree_draft.augment_all()
    model, tokenizer = ini_model(args)

    if config.inference.run_mode == 'ar':
        greedy_decoding(args, model, config)
    elif config.inference.run_mode == 'draft':
        if config.model.corpus_path is not None:
            tree_draft.start_cache_server(config.model.corpus_path, tokenizer_path=config.model.model_path)
        else:
            tree_draft.start_cache_server(config.inference.batch_size)
        base_tree_draft(args, model, config)
    else:
        raise NotImplementedError(f"Unsupported run_mode: {config.inference.run_mode}")


if __name__ == "__main__":
    main()
