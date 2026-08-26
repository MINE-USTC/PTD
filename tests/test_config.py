from argparse import Namespace

from tree_draft.config import PTDConfig


def test_config_requires_a_model_path():
    args = Namespace(
        model_path=None,
        question_file="data/gsm/test.jsonl",
        run_mode="draft",
        temperature=0.5,
        max_new_tokens=32,
        batch_size=1,
        sample_number=1,
        num_gpus_per_model=1,
        num_gpus_total=1,
        dtype="float32",
        use_pp=0,
        use_tp_ds=0,
        use_flash=0,
        cpu_offloading=False,
        max_gpu_memory=None,
        local_rank=0,
        dist_workers=1,
        save_log=0,
        log_path="logs",
        log_level="INFO",
        chat=1,
        corpus_path=None,
    )
    config = PTDConfig.create_from_args(args)
    assert not config.validate()


def test_inference_id_contains_temperature(tmp_path):
    args = Namespace(
        model_path=str(tmp_path / "model"),
        question_file="data/gsm/test.jsonl",
        run_mode="draft",
        temperature=0.5,
        max_new_tokens=32,
        batch_size=1,
        sample_number=1,
        num_gpus_per_model=2,
        num_gpus_total=2,
        dtype="float32",
        use_pp=0,
        use_tp_ds=0,
        use_flash=0,
        cpu_offloading=False,
        max_gpu_memory=None,
        local_rank=0,
        dist_workers=1,
        save_log=0,
        log_path="logs",
        log_level="INFO",
        chat=1,
        corpus_path=None,
    )
    config = PTDConfig.create_from_args(args)
    assert "temperature-0.5" in config.model.inference_id
    assert "temperature-2" not in config.model.inference_id
