import os

from fastchat.model import get_conversation_template

from .utils import *
from .inference_profile import ProfileKeys, InferProfile
from .config import PTDConfig


if os.environ.get('CUDA_VISIBLE_DEVICES') is None:
    devices = '0'
else:
    devices = os.environ['CUDA_VISIBLE_DEVICES'].split(',')


def set_log(config: PTDConfig):
    """
    Set up logging by run_mode (log path, file, level).
    Returns (log_file_name, logger_id) for caller to finalize (e.g. chmod, print path).
    """
    bench_name = config.model.bench_name
    if config.log.save_log:
        logger.remove()
        logger.add(sys.stdout, level=config.log.log_level)
        if not os.path.exists(config.log.log_path):
            os.makedirs(config.log.log_path)
        if config.inference.run_mode == 'ar':
            log_dir = os.path.join('log', 'ar', bench_name)
            os.makedirs(log_dir, exist_ok=True)
            log_file_name = os.path.join(
                log_dir,
                f'{config.model.model_arch}-{bench_name}-{config.inference.max_new_tokens}.log'
            )
        else:
            log_dir = os.path.join(config.log.log_path, bench_name)
            os.makedirs(log_dir, exist_ok=True)
            now = time.strftime('%Y%m%d%H%M%S', time.localtime())
            log_file_name = os.path.join(log_dir, config.model.inference_id + '-' + now + '.log')
        logger_id = logger.add(log_file_name, level=config.log.log_level.upper())
    else:
        log_file_name = None
        logger.remove()
        logger_id = logger.add(sys.stdout, level=config.log.log_level.upper())
    return log_file_name, logger_id


def base_tree_draft(args, model, config: PTDConfig):
    """
    Tree Draft inference mode.

    Args:
        args: CLI args (legacy compatibility).
        model: Model instance.
        config: PTDConfig (required).
    """
    logger_path, logger_id = set_log(config)

    tree_config = config.tree.to_dict()
    model.tree_config = tree_config

    Node.load_config(tree_config['node_config'])
    Tree.load_config(tree_config)
    logger.success(f"Draft tree config: {tree_config}")

    model.draft_trees = []
    for i in range(config.inference.batch_size):
        draft_tree = DraftTree(**tree_config)
        model.draft_trees.append(draft_tree)

    args.inference_id = config.model.inference_id
    model = ini_model_paras(model, args)
    logger.info('==========' * 5 + 'decoding parameters' + '==========' * 5)
    logger.info(f'{args}')
    logger.info('==========' * 5 + 'model parameters' + '==========' * 5)
    logger.info(model_info(model))
    questions = load_prompts(config.model.question_file)
    decode_kwargs = {
        'temperature': config.inference.temperature,
        'max_new_tokens': config.inference.max_new_tokens,
        'pad_token_id': model.tokenizer.pad_token_id,
    }
    setup_seed(121)
    sample_idx, sampled_questions = sample_questions(questions, config.inference.sample_number)
    logger.success(f'sampled indices of the benchmark {config.model.bench_name}:', sample_idx)
    sampled_questions = extract_all_turn_questions(sampled_questions, config.model.bench_name)
    sampled_questions_batch = [
        sampled_questions[i:i + config.inference.batch_size]
        for i in range(0, len(sampled_questions), config.inference.batch_size)
    ]
    get_model_answers(
        model,
        model.tokenizer,
        sampled_questions_batch,
        config.model.bench_name,
        config.model.model_path,
        **decode_kwargs,
    )
    logger.info(args)

    if config.log.save_log:
        logger.success('Finished! All logs saved at ' + logger_path)
        logger.remove(logger_id)
        os.chmod(logger_path, 0o444)


def greedy_decoding(args, model, config: PTDConfig):
    """
    Autoregressive greedy decoding mode.

    Args:
        args: CLI args (legacy compatibility).
        model: Model instance.
        config: PTDConfig (required).
    """
    _, logger_id = set_log(config)
    args.inference_id = config.model.inference_id
    model = ini_model_paras(model, args)
    logger.info('==========' * 5 + 'decoding parameters' + '==========' * 5)
    logger.info(f'{args}')
    logger.info('==========' * 5 + 'model parameters' + '==========' * 5)
    logger.info(model_info(model))
    questions = load_prompts(config.model.question_file)
    decode_kwargs = {
        'temperature': config.inference.temperature,
        'max_new_tokens': config.inference.max_new_tokens,
        'pad_token_id': model.tokenizer.pad_token_id,
    }
    setup_seed(121)
    sample_idx, sampled_questions = sample_questions(questions, config.inference.sample_number)
    logger.success(f'sampled indices of the benchmark {config.model.bench_name}:', sample_idx)
    sampled_questions = extract_all_turn_questions(sampled_questions, config.model.bench_name)
    sampled_questions_batch = [
        sampled_questions[i:i + config.inference.batch_size]
        for i in range(0, len(sampled_questions), config.inference.batch_size)
    ]
    get_model_answers(
        model,
        model.tokenizer,
        sampled_questions_batch,
        config.model.bench_name,
        config.model.model_path,
        **decode_kwargs,
    )
    logger.info(args)
    logger.remove(logger_id)


def get_model_answers(
        model,
        tokenizer,
        sampled_questions_batch,
        bench_name,
        model_path,
        **kwargs
):
    ds_local_rank = int(os.getenv('LOCAL_RANK', '0'))
    profile = InferProfile()

    for batch_idx, questions in enumerate(sampled_questions_batch):

        if isinstance(questions, list) and len(questions) > 0:
            conv_range = len(questions[0])
        else:
            conv_range = 1
        convs = []
        for turn in range(conv_range):
            inputs_list = []
            for j, question in enumerate(questions):
                if len(convs) < len(questions):
                    conv = get_conversation_template(model_path)
                    conv.append_message(conv.roles[0], question[turn])
                    conv.append_message(conv.roles[1], None)
                    convs.append(conv)
                else:
                    convs[j].append_message(convs[j].roles[0], question[turn])
                    convs[j].append_message(convs[j].roles[1], None)

                prompt = convs[j].get_prompt()
                prompt = prompt.replace('<s>', ' ').replace('</s>', ' ')
                inputs_list.append(prompt)

            logger.info(f'Processing batch {batch_idx + 1}/{len(sampled_questions_batch)}')
            batch_inputs = tokenizer(inputs_list, return_tensors='pt', padding=True, truncation=True).to(model.device)

            prompt_len = batch_inputs['attention_mask'].sum(dim=1).tolist()

            if kwargs['temperature'] < 1e-4:
                kwargs['do_sample'] = False
            else:
                kwargs['do_sample'] = True

            start_time = time.perf_counter()

            outputs = model.generate(**batch_inputs, **kwargs)

            gap_time = time.perf_counter() - start_time
            output_ids = outputs[0]
            model_profile = outputs[2] if len(outputs) > 2 else None

            # Tree Draft: log profile details
            logger.debug("\n" + str(model_profile))
            model_profile.log_profile(logger, f"Current Summary (Batch {batch_idx + 1}, Turn {turn + 1})")

            output_str = []
            for _, output_id in enumerate(output_ids):
                output_str.append(tokenizer.decode(output_id[prompt_len[_]:], skip_special_tokens=True))

            output_len = output_ids.ne(tokenizer.pad_token_id).sum(dim=1).tolist()
            gen_counts = [output_len[i] - prompt_len[i] for i in range(len(output_len))]

            profile.incremental_updates(model_profile)

            logger.success('==========' * 5 + 'output:' + '==========' * 5)
            for i, output in enumerate(output_str):
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output_str[i] = output
                logger.success(f'Batch {batch_idx + 1} - Question {i + 1} - Turn {turn + 1}: {output}\n')

            logger.success('==========' * 5 + 'output end' + '==========' * 5)

            if get_device(model.draft_config) == 0 and ds_local_rank == 0:
                logger.success(
                    f"Batch {batch_idx + 1} \t time: {gap_time:.4f} \t generated tokens_count: {sum(gen_counts)} \t"
                    f"Throughput: {(sum(gen_counts) / gap_time):.4f}"
                )

            assert len(output_str) == len(convs)
            for i in range(len(output_str)):
                convs[i].messages[-1][-1] = output_str[i]

        profile.log_profile(logger, stage=f"Profile So Far (Batch {batch_idx + 1})")

    profile.log_profile(logger, stage="FINAL PROFILE")
    profile.log_profile_time(logger)
