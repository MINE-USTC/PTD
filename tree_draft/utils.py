import math
import subprocess
import sys
import warnings
from argparse import Namespace
from statistics import stdev

import psutil
from datasets import load_dataset
from fastchat.llm_judge.common import load_questions
from fastchat.model.model_adapter import Llama2Adapter, QwenChatAdapter, \
    raise_warning_for_incompatible_cpu_offloading_configuration
from fastchat.utils import get_gpu_memory, str_to_torch_dtype
from transformers import GenerationMixin
from transformers.models.llama import modeling_llama
from transformers.models.qwen2 import modeling_qwen2
from transformers.models.qwen3 import modeling_qwen3

from .decode_strategy import *
from .models.Llama import llama_ptd
from .models.Qwen import qwen_ptd
from .models.Qwen3 import qwen3_ptd
from .models import ptd_common


FUNC_MAP = {}


def greedy_search_proxy(self, *args, **kwargs):
    if self.draft_config.run_mode in ["draft"]:
        return tree_draft_greedy_search(self, chat=True, *args, **kwargs)
    else:
        return FUNC_MAP["sample"](self, *args, **kwargs)


def augment_llama():
    modeling_llama.LlamaForCausalLM.PTDforward = llama_ptd.PTDforward
    modeling_llama.LlamaModel.LlamaModelPTDforward = llama_ptd.LlamaModelPTDforward
    modeling_llama.LlamaModel.prepare_decoder_attention_mask = ptd_common.prepare_decoder_attention_mask_batch


def augment_qwen():
    modeling_qwen2.Qwen2ForCausalLM.PTDforward = qwen_ptd.PTDfroward
    modeling_qwen2.Qwen2Model.QwenModelPTDForward = qwen_ptd.QwenModelPTDForward
    modeling_qwen2.Qwen2Model.prepare_decoder_attention_mask = ptd_common.prepare_decoder_attention_mask_batch


def augment_qwen3():
    modeling_qwen3.Qwen3ForCausalLM.PTDforward = qwen3_ptd.PTDfroward
    modeling_qwen3.Qwen3Model.QwenModelPTDForward = qwen3_ptd.QwenModelPTDForward
    modeling_qwen3.Qwen3Model.prepare_decoder_attention_mask = ptd_common.prepare_decoder_attention_mask_batch


def augment_generate():
    FUNC_MAP["sample"] = sample
    GenerationMixin._sample = greedy_search_proxy
    return


def augment_all():
    augment_llama()
    augment_qwen()
    augment_qwen3()
    augment_generate()


def get_device(config):
    if "local_rank" not in config:
        return 0
    local_rank = config.local_rank
    return local_rank


def ini_model_paras(model, args):
    model.draft_config = Namespace()
    model.cache = {}
    model.cache_other = {}

    attributes = [
        'run_mode', 'inference_id', 'model_arch',
        'dist_workers', 'use_flash']
    for attr in attributes:
        setattr(model.draft_config, attr, getattr(args, attr))

    return model


def model_info(model):
    info = "\n"
    for key, val in vars(model.draft_config).items():
        info += f'{key}:\t\t{val}\n'

    return info


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def set_user_query(query, messages=None):
    if messages is None:
        messages = [
            {"role": "user", "content": query},
        ]
    else:
        messages.append({"role": "user", "content": query})

    return messages


def update_agent_response(query, message):
    assert message is not None
    message.append({'role': 'assistant', 'content': query})
    return message


def extract_all_turn_questions(sampled_questions, bench_name):
    questions = []
    for question in sampled_questions:
        questions.append(get_all_turn_question(question, bench_name))
    return questions


def get_all_turn_question(question, bench_name):
    if bench_name.lower() == 'humaneval':
        return question
    if bench_name.lower() == 'mt-bench':
        return question['turns'][:]
    if bench_name.lower() == 'mbpp':
        return question
    if bench_name.lower() == 'gsm':
        return question['question']
    if 'mt' in bench_name:
        return question['turns'][:]
    if bench_name.lower() == 'plain':
        return question


def extract_batch_question(questions, bench_name, j):
    if isinstance(questions, str):
        question_batch = [questions]
    elif isinstance(questions, list):
        question_batch = questions
    else:
        raise RuntimeError(f'Unsupported input type: {type(questions)}')
    res = []
    for question in question_batch:
        res.append(extract_question(question, bench_name, j))
    return res


def extract_question(question, bench_name, j=0):
    if bench_name.lower() == 'humaneval':
        return question
    if bench_name.lower() == 'mt-bench':
        return question['turns'][j]
    if bench_name.lower() == 'mbpp':
        return question['text']
    if bench_name.lower() == 'gsm':
        return question['question']
    if 'mt-bench' in bench_name:
        return question['turns'][j]
    if bench_name.lower() == 'longbench':
        return question
    if bench_name.lower() == 'hagrid':
        return question
    if bench_name.lower() == 'cnn_dailymail':
        return question
    if bench_name.lower() == 'wmt20-en-zh':
        return question

    raise RuntimeError(f'Unsupported benchmark: {bench_name}')

def polish_dialogue(question, bench_name, turn):
    if bench_name == "HumanEval":
        assert isinstance(question, str)
        q = question
        qs = "Continue complete the following python functions: \n" + q

    elif bench_name == 'mbpp':
        qs = question
    elif bench_name == 'ifeval':
        qs = question['prompt']
    elif bench_name == 'gsm':
        qs = question['prompt']
    else:
        if isinstance(question, dict) and 'turns' in question:  # multi-turn e.g. mt-bench
            qs = question["turns"][turn]
        elif isinstance(question, dict) and 'context' in question:  # single turn with context e.g. dolly-15k

            if not question['context']:
                qs = question['instruction']
            else:
                context = question['context']
                qs = 'Context: ' + context + "\nInstruction: " + question['instruction'] + \
                     '\nPlease response to the Instruction according to the Context.'

        elif isinstance(question, str):
            qs = question
        else:
            raise RuntimeError(f'Unsupported benchmark: {bench_name}')

    return qs


def sample_and_pad(l, sample_list, c):
    _ = random.sample(sample_list, c)
    return [l[0] + _]


def pad_random_words(sentence, pool, c):
    s = random.sample(pool, c)
    return sentence + ' ' + ' '.join(s)


# load eval datasets
def load_human_eval(file_path, begin, end):
    all_data = load_dataset(file_path)
    return [all_data['test'][i]['prompt'] for i in range(len(all_data['test']['prompt'][begin:end]))]


def load_quac(file_path, begin, end):
    pass


def load_mbpp_eval(file_path, begin, end):
    all_data = load_questions(file_path, begin, end)
    return all_data


def load_text(question_file):
    questions = []
    with open(question_file, 'r') as file:
        for line in file:
            if line:
                questions.append(line.strip())
    return questions


def load_longbench_4k_8k(file_path=None, task_names=None, max_samples=None):
    """
    Load LongBench tasks and extract input texts with 4K–8K word-level length.
    Returns a list of input strings.
    """
    # All available LongBench tasks
    all_tasks = [
        "narrativeqa", "qasper", "multi_news", "gov_report", "hotpotqa", "2wikimqa",
        "musique", "trec", "samsum", "lsht", "passage_count", "passage_retrieval_en",
        "passage_retrieval_zh", "financial_qa", "multidoc_qa", "legal_contract_qa",
        "vcsum"
    ]

    if task_names is None:
        task_names = all_tasks

    inputs = []
    MIN_LEN = int(os.environ.get("MIN_LEN", 0))
    MAX_LEN = int(os.environ.get("MAX_LEN", 0))

    # task_names = ["gov_report.jsonl", "multi_news.jsonl", "narrativeqa.jsonl"]
    task_names = ["samsum_e.jsonl"]

    for task in task_names:
        print(f"Loading task: {task}")

        ds = load_dataset('json', data_files=os.path.join(file_path, task))
        text = ds['train']['context']
        for t in text:
            # Approximate token length using whitespace split
            length = len(t.split())
            if MIN_LEN <= length < MAX_LEN:
                inputs.append(t)
        print(f'get {len(inputs)} samples')
    # Random subsampling
    if max_samples and len(inputs) > max_samples:
        random.seed(42)
        inputs = random.sample(inputs, max_samples)

    return inputs


def load_cnn(file_path):
    input = []
    data = load_dataset(file_path)
    for task in data['test']:
        input.append("The article is: \n" + task['article'] + "\nPlease summarize this article.")
    return input


def load_hagrid(file_path):
    input = []
    data = load_dataset(file_path)
    for task in data['validation']:
        context = []
        for quote in task['quotes']:
            context.append(str(quote['idx']) + ' ' + quote['text'])
        input.append("The context is: \n" + '\n'.join(context) + "\nPlease answer the following question according "
                                                                 "to the Context. The question is: \n" + task['query'])

    return input


def load_wmt(file_path):
    input = []
    data = load_dataset(file_path)
    for task in data['test']['translation']:
        input.append(f"Please translate the following sentence into chinese: \“{task['en']}\”")

    return input

def load_prompts(file_path, begin=None, end=None):
    if 'mt-bench' in file_path:
        return load_questions(file_path, begin, end)
    elif 'HumanEval'.lower() in file_path.lower():
        return load_human_eval(file_path, begin, end)
    elif 'gsm' in file_path.lower():
        return load_questions(file_path, begin, end)
    elif "mbpp" in file_path.lower():
        return load_mbpp_eval(file_path, begin, end)
    elif 'longbench' in file_path.lower():
        return load_longbench_4k_8k(file_path=os.path.join(os.getcwd(), file_path))
    elif 'cnn_dailymail' in file_path.lower():
        return load_cnn(file_path=os.path.join(os.getcwd(), file_path))
    elif 'hagrid' in file_path.lower():
        return load_hagrid(file_path)
    elif 'wmt' in file_path.lower():
        return load_wmt(file_path)
    else:
        raise RuntimeError(f'Unsupported benchmark: {file_path}')


def get_model_arch(args):
    if args.model_path is not None:
        if args.model_path[-1] == '/':
            p = args.model_path[:-1]
        else:
            p = args.model_path
    return os.path.basename(p)


def get_inference_id(args):
    return f'{args.model_arch}-{args.bench_name}' \
           f'-run_mode-{args.run_mode}' \
           f'-sample_number-{args.sample_number}' \
           f'-max_new_tokens-{args.max_new_tokens}' \
           f'-temperature-{args.num_gpus_per_model}'





# load model
def load_model(
        model_path: str,
        device: str = "cuda",
        device_map: str = "",
        num_gpus: int = 1,
        max_gpu_memory: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        load_8bit: bool = False,
        cpu_offloading: bool = False,
        revision: str = "main",
        use_flash: bool = False
):
    """Load a model from Hugging Face."""

    # adapter = Llama2Adapter()
    if 'qwen' in model_path.lower():
        adapter = QwenChatAdapter()
    if 'llama' in model_path.lower():
        adapter = Llama2Adapter()

    # Handle device mapping
    cpu_offloading = raise_warning_for_incompatible_cpu_offloading_configuration(
        device, load_8bit, cpu_offloading
    )

    if device.startswith("cuda"):
        kwargs = {"torch_dtype": torch.float16}
        if num_gpus != 1:
            kwargs["device_map"] = "auto"
            if max_gpu_memory is None:
                kwargs[
                    "device_map"
                ] = "sequential"  # This is important for not the same VRAM sizes
                available_gpu_memory = get_gpu_memory(num_gpus)
                kwargs["max_memory"] = {
                    i: str(int(available_gpu_memory[i] * 0.85)) + "GiB"
                    for i in range(num_gpus)
                }
            else:
                kwargs["max_memory"] = {i: max_gpu_memory for i in range(num_gpus)}

    if cpu_offloading:
        # raises an error on incompatible platforms
        from transformers import BitsAndBytesConfig

        if "max_memory" in kwargs:
            kwargs["max_memory"]["cpu"] = (
                    str(math.floor(psutil.virtual_memory().available / 2 ** 20)) + "Mib"
            )
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit_fp32_cpu_offload=cpu_offloading
        )
        kwargs["load_in_8bit"] = load_8bit
    elif load_8bit:
        if num_gpus != 1:
            warnings.warn(
                "8-bit quantization is not supported for multi-gpu inference."
            )
        else:
            model, tokenizer = adapter.load_compress_model(
                model_path=model_path,
                device=device,
                torch_dtype=kwargs["torch_dtype"],
                revision=revision,
            )

            return model, tokenizer
    kwargs["revision"] = revision

    if dtype is not None:  # Overwrite dtype if it is provided in the arguments.
        kwargs["torch_dtype"] = dtype
    if use_flash:
        kwargs["use_flash_attention_2"] = use_flash
    if len(device_map) > 0:
        kwargs["device_map"] = device_map
    # if 'qwen' in model_path.lower():
    #     kwargs['trust_remote_code'] = True
    # Load model
    model, tokenizer = adapter.load_model(model_path, kwargs)

    if len(device_map) > 0:
        return model, tokenizer

    if (device.startswith("cuda") and num_gpus == 1 and not cpu_offloading) or device in (
            "mps",
            "xpu",
            "npu",
    ):
        model.to(device)

    if device == "xpu":
        model = torch.xpu.optimize(model, dtype=kwargs["torch_dtype"], inplace=True)

    return model, tokenizer


def ini_model(args):
    dtype = str_to_torch_dtype(args.dtype)

    if args.use_pp:
        model_, tokenizer_ = load_model(
            args.model_path,
            use_flash=args.use_flash,
            device=f"cuda",
            device_map="balanced",
            num_gpus=args.num_gpus_per_model,
            max_gpu_memory=args.max_gpu_memory,
            dtype=dtype,
            load_8bit=False,
            cpu_offloading=args.cpu_offloading,
        )

    elif args.use_tp_ds:
        import deepspeed
        torch.cuda.set_device(int(os.getenv('LOCAL_RANK', '0')))
        model_, tokenizer_ = load_model(
            args.model_path,
            use_flash=args.use_flash,
            device_map="cpu",
            num_gpus=args.num_gpus_per_model,
            max_gpu_memory=args.max_gpu_memory,
            dtype=dtype,
            load_8bit=False,
            cpu_offloading=args.cpu_offloading,
        )
        model_ = deepspeed.init_inference(
            model_,
            mp_size=int(os.getenv("WORLD_SIZE", "1")),
            dtype=torch.half
        )
    else:
        model_, tokenizer_ = load_model(
            args.model_path,
            use_flash=args.use_flash,
            device=f"cuda:{get_device(args)}",
            num_gpus=args.num_gpus_per_model,
            max_gpu_memory=args.max_gpu_memory,
            dtype=dtype,
            load_8bit=False,
            cpu_offloading=args.cpu_offloading,
        )
        logger.info('model load finished!')
    if tokenizer_.padding_side is None:
        tokenizer_.padding_side = "left"
    if tokenizer_.pad_token is None:
        if model_.generation_config.pad_token_id is not None:
            tokenizer_.pad_token_id = model_.generation_config.pad_token_id
        else:
            tokenizer_.pad_token_id = tokenizer_.eos_token_id
            model_.generation_config.pad_token_id = tokenizer_.eos_token_id
    model_.tokenizer = tokenizer_

    return model_, tokenizer_


def get_tensorboard_log_path(model_arch, bench_name, question_id, turn=0, time_stamp=None):
    path = os.path.join("tensorboard_log", model_arch, bench_name)
    log_name = f'{bench_name}_{question_id}_{turn}_{time_stamp}'
    return os.path.join(path, log_name)


def draw_freq(freq, title):
    plt.bar(freq.keys(), freq.values())
    plt.title(title)
    plt.show()


def draw_bin(freqs, title):
    plt.hist(freqs, bins=20)
    plt.title(title)
    plt.show()


def sample_questions(questions, sample_number):
    setup_seed(10)
    if sample_number == -1:
        sample_idx = list(range(0, len(questions)))
    elif sample_number == -2:
        sample_idx = [0, 1, 10, 11, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61, 70, 71]
    elif sample_number == -3:
        sample_idx = [0, 10, 20, 30, 40, 50, 60, 70]
    elif sample_number == 1:
        sample_idx = [0]
    else:
        sample_idx = random.sample(range(0, len(questions)), sample_number)
    sample_questions = []
    for i, idx in enumerate(sample_idx):
        sample_questions.append(questions[idx])

    return sample_idx, sample_questions

def extract_prune_questions(sampled_questions, bench_name):
    questions = []
    for question in sampled_questions:
        questions.append(extract_question(question, bench_name=bench_name))
    return questions


def start_cache_server(batch_size=1, rest_corpus_path=None, tokenizer_path=None):
    python_path = sys.executable
    if rest_corpus_path is not None and tokenizer_path is not None:
        cacher_server_process = subprocess.Popen(
            [python_path, "cache_server.py", str(batch_size), rest_corpus_path, tokenizer_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1)
    elif rest_corpus_path is None and tokenizer_path is None:
        cacher_server_process = subprocess.Popen([python_path, "cache_server.py", str(batch_size)],
                                                 stdout=subprocess.PIPE,
                                                 stderr=subprocess.PIPE, text=True, bufsize=1)
    else:
        raise RuntimeError('')
    print("Waiting for the cache server starting...")
    for line in cacher_server_process.stdout:
        if 'Tree cache server started' in line:
            print(line.strip() + " (PID: {})".format(cacher_server_process.pid))
            break







