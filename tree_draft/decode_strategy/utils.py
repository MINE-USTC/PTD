import asyncio
import os
import pickle
import struct
from multiprocessing import shared_memory

import matplotlib.pyplot as plt
import posix_ipc
import torch
import torch.nn.functional as F


def draw_freq(freq, title):
    plt.bar(freq.keys(), freq.values())
    plt.title(title)
    plt.show()


def receive_all_data(conn):
    data = b''
    while True:
        chunk = conn.recv(40960)
        if not chunk:
            break
        data += chunk
        if b"END" in data:  # special terminator
            break
    assert data[-3:] == b"END"
    return data[:-3]


def top_p_sampling(logits, p=0.9, k=50, temperature=1.0):
    logits = logits / temperature
    batch_size, seq_len, vocab_size = logits.shape
    sampled_tokens = []

    for t in range(seq_len):
        logits_t = logits[:, t, :]
        top_k_values, top_k_indices = torch.topk(logits_t, k, dim=-1)
        sorted_logits, sorted_indices = torch.sort(top_k_values, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cumulative_probs <= p
        mask = mask.float()
        logits_t = logits_t.gather(1, top_k_indices) * mask - (1 - mask) * float('inf')
        probs = F.softmax(logits_t, dim=-1)
        next_token = torch.multinomial(probs, 1)
        sampled_tokens.append(next_token.squeeze(1))

    sampled_tokens = torch.stack(sampled_tokens, dim=1)

    return sampled_tokens


def topk_sampling(logits, k=2):
    probs = torch.nn.functional.softmax(logits, dim=-1)

    top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)

    sampled_index = torch.multinomial(top_k_probs[0], 1).unsqueeze(0)
    return top_k_indices.gather(2, sampled_index)[0]


def top_k_random_sampling(logits, k=2):
    seq_len = logits.size(1)
    batch_size = logits.size(0)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    top_k_probs, top_k_indices = torch.topk(probs, k, dim=-1)
    random_idx = torch.randint(0, k, (batch_size, seq_len), device=logits.device)
    random_sample_idx = top_k_indices.gather(-1, random_idx.unsqueeze(-1))
    return random_sample_idx.squeeze()


async def async_to_list(tensor):
    loop = asyncio.get_event_loop()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    result = await loop.run_in_executor(None, tensor.tolist)
    return result


def read_shm(shm):
    data_size = struct.unpack('>I', bytes(shm.buf[:4]))[0]
    data = bytes(shm.buf[4:4 + data_size])
    obj = pickle.loads(data)
    return obj


def write_shm(shm, obj):
    data_size = len(obj)
    shm.buf[:4] = struct.pack('>I', data_size)
    shm.buf[4:4 + data_size] = obj


def filter_finished_sequences(input_ids, model_kwargs, mask):
    input_ids = input_ids[mask]
    model_kwargs["attention_mask"] = model_kwargs["attention_mask"][mask]
    model_kwargs['past_key_values'] = [
        (k[0][mask], k[1][mask]) for k in model_kwargs['past_key_values']
    ]
    return input_ids, model_kwargs


def remove_finished_cache_server(finished_batch_idx, request_shm, request_sem):
    retrieve_request = {"type": "reduce", "remove_idx": finished_batch_idx}
    write_shm(request_shm, pickle.dumps(retrieve_request))
    request_sem.release()


def update_guess_trees_with_retrieve_res(guess_trees, retrieve_res, unfinished_sequences, finished_batch_idx):
    if unfinished_sequences.shape[0] != len(retrieve_res):
        # Some sequences already finished but still got a retrieve result
        assert isinstance(finished_batch_idx, (int, list)) and finished_batch_idx != -1, \
            f"finished_batch_idx should be an int or a list of ints. Current finished_batch_idx: {finished_batch_idx}"
        finished_batch_idx = [finished_batch_idx] if isinstance(finished_batch_idx, int) else finished_batch_idx
        valid_indices = [i for i in range(len(retrieve_res)) if i not in finished_batch_idx]
        for j, i in enumerate(valid_indices):
            guess_trees[j] = retrieve_res[i]['guess_tree']
            if guess_trees[j] is not None:
                guess_trees[j].update_global_info()
    else:
        for i, res in enumerate(retrieve_res):
            guess_trees[i] = res['guess_tree']
            if guess_trees[i] is not None:
                guess_trees[i].update_global_info()


def update_guess_trees_with_retrieve_res_2(guess_trees, retrieve_res, unfinished_sequences, finished_batch_idx):
    retrieve_guess_trees = retrieve_res['guess_tree']

    if unfinished_sequences.shape[0] != len(retrieve_guess_trees):
        # Some sequences already finished but still got a retrieve result
        assert isinstance(finished_batch_idx, (int, list)) and finished_batch_idx != -1, \
            f"finished_batch_idx should be an int or a list of ints. Current finished_batch_idx: {finished_batch_idx}"
        finished_batch_idx = [finished_batch_idx] if isinstance(finished_batch_idx, int) else finished_batch_idx
        valid_indices = [i for i in range(len(retrieve_guess_trees)) if i not in finished_batch_idx]
        for j, i in enumerate(valid_indices):
            guess_trees[j] = retrieve_guess_trees[i]
            if guess_trees[j] is not None:
                guess_trees[j].update_global_info()
    else:
        for i, res in enumerate(retrieve_guess_trees):
            guess_trees[i] = res
            if guess_trees[i] is not None:
                guess_trees[i].update_global_info()

    return guess_trees


def init_ipc_and_reset_cache():
    guess_sem = posix_ipc.Semaphore("/guess_res_sem")
    request_sem = posix_ipc.Semaphore("/request_sem")
    while guess_sem.value != 0:
        guess_sem.acquire()
    assert guess_sem.value == 0
    while request_sem.value != 0:
        request_sem.acquire()
    assert request_sem.value == 0

    guess_shm = shared_memory.SharedMemory(name=f'guess_tree')
    request_shm = shared_memory.SharedMemory(name=f'request')
    reset_cache_request = {'type': 'reset_cache'}

    write_shm(request_shm, pickle.dumps(reset_cache_request))
    request_sem.release()
    return guess_sem, request_sem, guess_shm, request_shm
