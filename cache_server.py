import shutil
import sys
import copy
import json
import os
import pickle
import socket
import struct
import time
import signal
import atexit
from multiprocessing.shared_memory import SharedMemory
from os.path import isfile
from time import perf_counter
from typing import Union

import numpy as np
import posix_ipc
import torch
from fastchat.llm_judge.common import load_questions
from transformers import AutoTokenizer

from tree_draft.trees import DraftTree, GuessTree, Node, Tree, GuessTreeBatch
from datasets import load_dataset
from tqdm import tqdm

MAX_CHILD_NUM = int(os.environ.get('MAX_CHILD_NUM', -1))
MAX_DEPTH = int(os.environ.get('MAX_DEPTH', -1))

tree_config = json.load(open('configs/tree_config.json', 'r'))
if MAX_CHILD_NUM != -1:
    for i in range(0, len(tree_config['node_config']['max_child_num'])):
        tree_config['node_config']['max_child_num'][i] = MAX_CHILD_NUM
    tree_config['node_config']['min_child_num'][0] = MAX_CHILD_NUM
if MAX_DEPTH != -1:
    tree_config['max_depth'] = MAX_DEPTH

Node.load_config(tree_config['node_config'])
Tree.load_config(tree_config)
guess_tree_config = copy.deepcopy(tree_config)
# guess_tree_config['node_config']['max_child_num'][0] = 30  # f First level node limitation
# guess_tree_config['node_config']['min_child_num'][0] = 30  # f First level node limitation
MAX_N = 60

PLD_INI = int(os.environ.get('PLD_INI', 0))
REST_INI = int(os.environ.get('REST_INI', 0))
PTD = int(os.environ.get('PTD', 1))

class ContextCacheTree:
    def __init__(self, context_cache=None, max_val_len=None, gram_n=None, max_key_len=None, corpus_path=None,
                 tokenizer_path=None):
        self.max_val_len = 15 if max_val_len is None else max_val_len
        self.gram_n = 4 if gram_n is None else gram_n
        self.max_key_len = 1 if max_key_len is None else max_key_len
        self.cache = {}
        self.insert_count = 0
        self.remove_count = 0
        self.update_count = 0
        self.gram_count = 0
        self.all_ins_gram_count = 0
        self.inserted_inner_seqs = set()

        self.corpus_path = corpus_path
        self.tokenizer_path = tokenizer_path
        if self.corpus_path is not None and self.tokenizer_path is not None:
            self.tokenizer_name = os.path.basename(self.tokenizer_path)
            if isfile(self.corpus_path):
                self.corpus_cache_path = os.path.join(os.path.dirname(self.corpus_path),
                                                      f'rest/{self.tokenizer_name}/cache.pickle')
            else:
                self.corpus_cache_path = os.path.join(self.corpus_path, f'rest/{self.tokenizer_name}/cache.pickle')
        else:
            self.tokenizer_name = None
            self.corpus_cache_path = None

        self.c = 0
        if context_cache is not None:
            assert isinstance(context_cache, ContextCacheTree)
            self.cache = context_cache.cache.copy()

    def __str__(self):
        cache_len = len(self.cache)
        gram_num = sum([self.cache[key].tree.node_count for key in self.cache])
        avg_gram_num = gram_num / cache_len if cache_len > 0 else 0

        if len(self.cache) == 0:
            std_gram_num = 0
        else:
            std_gram_num = np.std([self.cache[key].tree.node_count for key in self.cache])

        return {
            "Cache Length": cache_len,
            "Gram Counts": gram_num,
            "Average gram len": avg_gram_num,
            "Std gram num": std_gram_num,
            "Total cache insert operation count": self.all_ins_gram_count,
            "Total update": self.update_count,
            "Total Removed": self.remove_count,
            "Total Inserted": self.insert_count,
            "(total insert - total remove)": self.insert_count - self.remove_count,
            "TOTAL GRAM COUNTS": self.gram_count
        }

    def __getitem__(self, item):
        return self.cache[item]

    def __setitem__(self, key, value):
        if isinstance(value, GuessTree):
            self.cache[key] = value
        else:
            raise RuntimeError(f'Unsupported data type: {type(value)}')

    def __contains__(self, item):
        return item in self.cache

    def __repr__(self):
        return self.__str__()

    def PLD_init_cache(self, prompt_tokens):
        """
        Initialize the whole cache with the given prompt token sequence(s).

        prompt_tokens can be:
        - 1D list[int]: single prompt
        - 2D list[list[int]]: batch of prompts
        - torch.Tensor: shape [L] or [B, L]
        """
        if isinstance(prompt_tokens, torch.Tensor):
            if prompt_tokens.dim() == 1:
                rows = [prompt_tokens.tolist()]
            elif prompt_tokens.dim() == 2:
                rows = prompt_tokens.tolist()
            else:
                raise ValueError(f"Unsupported prompt_tokens tensor shape: {prompt_tokens.shape}")
        elif isinstance(prompt_tokens, (list, tuple)):
            if len(prompt_tokens) == 0:
                return
            if isinstance(prompt_tokens[0], (list, tuple)):
                rows = [list(r) for r in prompt_tokens]
            else:
                rows = [list(prompt_tokens)]
        else:
            raise TypeError(f"Unsupported type for prompt_tokens: {type(prompt_tokens)}")

        for row in rows:
            if len(row) <= self.gram_n:
                continue

            for i in range(0, len(row) - self.gram_n):
                token_list = row[i: i + self.gram_n + 1]
                self.cache_insert_with_list(token_list)
        return

    def sequence_ini(self, row):
        assert isinstance(row, (list, tuple))
        if len(row) <= self.gram_n:
            return

        for i in range(0, len(row) - self.gram_n):
            token_list = row[i: i + self.gram_n + 1]
            self.cache_insert_with_list(token_list)

    def build_gsm(self, path, tokenizer_path):
        data = load_questions(path, None, None)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        for task in tqdm(data):
            input_list = tokenizer.encode(task['question'] + '\n' + task['answer'])
            self.sequence_ini(input_list)
        return

    def build_mbpp(self, path, tokenizer_path):
        data = load_questions(path, None, None)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        for line in tqdm(data[:900]):
            input_list = tokenizer.encode(line['text'] + '\n' + line['code'])
            self.sequence_ini(input_list)

        return

    def build_wmt20(self, path, tokenizer_path):
        data = load_dataset(path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        for task in tqdm(data['train']['translation']):
            en = task['en']
            zh = task['zh']
            input_list = tokenizer.encode(zh)
            self.sequence_ini(input_list)
        return

    def build_cnn_daily_mail(self, path, tokenizer_path):
        data = load_dataset(path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        max_len = min(len(data['train']), 10000)
        for i in tqdm(range(max_len)):
            input_list = tokenizer.encode(data['train'][i]['article'])
            self.sequence_ini(input_list)
        return

    def build_hagrid(self, path, tokenizer_path):

        data = load_dataset(path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        max_len = min(len(data['train']), 10000)

        for i in tqdm(range(max_len)):
            quotes = data['train']['quotes'][i]
            quote_sentences = ''
            for quote in quotes:
                text = quote['text']
                quote_sentences += text + '\n'

            query = data['train']['query'][i] + '\n'
            answer = ''
            for ans in data['train']['answers'][i]:
                answer += ans['answer']

            input_list = tokenizer.encode(query + quote_sentences + answer)
            self.sequence_ini(input_list)
        return

    def init_cache_by_corpus(self):
        assert self.tokenizer_path is not None
        assert self.corpus_path is not None
        self.cache = {}
        if os.path.exists(self.corpus_path) and os.path.exists(self.tokenizer_path):
            if 'wmt20' in self.corpus_path:
                build_func = self.build_wmt20
            elif 'cnn_dailymail' in self.corpus_path:
                build_func = self.build_cnn_daily_mail
            elif 'hagrid' in self.corpus_path:
                build_func = self.build_hagrid
            elif 'gsm' in self.corpus_path:
                build_func = self.build_gsm
            elif 'mbpp' in self.corpus_path:
                build_func = self.build_mbpp
            else:
                raise NotImplementedError(f"Unsupported corpus path: {self.corpus_path}")

            if os.path.exists(self.corpus_cache_path):
                print(f'Loading cache from {self.corpus_cache_path}')
                self.cache = pickle.load(open(self.corpus_cache_path, 'rb'))
            else:
                build_func(self.corpus_path, self.tokenizer_path)
                if not os.path.exists(os.path.dirname(self.corpus_cache_path)):
                    os.makedirs(os.path.dirname(self.corpus_cache_path), exist_ok=True)
                pickle.dump(self.cache, open(self.corpus_cache_path, "wb"))
        else:
            raise FileNotFoundError(f"{self.corpus_path} or {self.tokenizer_path}is not exist")

    def avg_info_per_step(self, all_steps):
        cache_len = len(self.cache)
        gram_num = sum([len(self.cache[key]) for key in self.cache])
        return (f"Average gram num per step: {gram_num / all_steps:.4f}\n"
                f"Average new keys per step:{cache_len / all_steps:.4f}\n")

    def stat_info(self):
        return self.__str__()

    def cache_insert_and_pop(self, sub_tree_root, max_n=-1):
        """
        Insert a sub_tree_root into the cache
        """
        if sub_tree_root is None:
            return
        assert isinstance(sub_tree_root, Node)
        if sub_tree_root.value in self.cache.keys():
            self.cache[sub_tree_root.value].merge_trees_with_draft_token(sub_tree_root)
            # self.cache[sub_tree_root.value].merge_trees_with_draft_map(sub_tree_root, draft_map)
        else:
            self.cache[sub_tree_root.value] = GuessTree(key=sub_tree_root.value, **guess_tree_config)
            self.cache[sub_tree_root.value].merge_trees_with_draft_token(sub_tree_root)

        popped_tree_root = self.cache[sub_tree_root.value].prune(max_n)
        if popped_tree_root is not None:
            self.cache_insert_and_pop(popped_tree_root, max_n)

    def cache_insert_with_list(self, token_list):
        key = token_list[0]
        remaining_tokens = token_list[1:]
        if key in self.cache.keys():
            self.cache[key].merge_trees_by_list(remaining_tokens)
        else:
            self.cache[key] = GuessTree(**guess_tree_config)
            self.cache[key].merge_trees_by_list(remaining_tokens)
        self.cache[key].prune(MAX_N)

    def retrieve(self, all_old_tokens):

        if isinstance(all_old_tokens, int):
            key = self.convert_key([all_old_tokens])
        elif isinstance(all_old_tokens, list):
            key = self.convert_key(all_old_tokens[-1:])

        if key in self.cache.keys() and self.cache[key].tree.node_count > 0:
            return self.cache[key]
        else:
            return None

    def get_cur_draft_results(self, grams, nodes_seqs, draft_tree, draft_map):
        new_results = []
        for i, gram in enumerate(grams):
            if len(nodes_seqs[i]) == 1:
                continue

            parent_node = nodes_seqs[i][-2]
            draft_token = draft_map[parent_node]
            draft_gram = gram[:-1] + (draft_token,)
            assert len(draft_gram) == len(gram)
            new_results.append(draft_gram)

        return new_results

    def get_cur_draft_token_with_map_2(self, grams, nodes_seqs, draft_tree, draft_map):
        new_results = []
        keys = []
        for i, gram in enumerate(grams):
            if len(nodes_seqs[i]) == 1:
                continue
            parent_node = nodes_seqs[i][-1]
            draft_token = draft_map[parent_node]
            draft_gram = gram[1:] + (draft_token,)
            assert len(draft_gram) == len(gram)
            new_results.append(draft_gram)
            keys.append(gram[0])
        return keys, new_results

    def get_cur_draft_token_with_map(self, grams, nodes_seqs, draft_map):
        new_results = []

        for i, gram in enumerate(grams):
            parent_node = nodes_seqs[i][-1]
            draft_token = draft_map[parent_node]
            if isinstance(draft_token, int):
                # draft_gram = gram + (draft_token,)
                new_results.append(gram + (draft_token,))
            elif isinstance(draft_token, list):
                for d in draft_token:
                    new_results.append(gram + (d,))
            else:
                raise RuntimeError(f'Unsupported data type: {type(draft_token)}')

        return new_results

    def get_tree_inner_grams(self, draft_tree):
        _res = {}
        for node in draft_tree.tree.dfs_node_seq:
            key = node.value
            grams, node_seqs = draft_tree.tree.serialize_children_dfs_fix_length(node, self.gram_n, return_node=True)
            for g in grams:
                if g not in _res:
                    _res[key] = []
                _res[key].append(g)
        return _res

    def get_draft_grams(self, draft_tree, draft_map):
        _res = {}
        for node in draft_tree.tree.dfs_node_seq:
            key = node.value
            grams, node_seqs = draft_tree.tree.serialize_children_dfs_fix_length(node, self.gram_n - 1,
                                                                                 return_node=True)
            draft_grams = self.get_cur_draft_token_with_map(grams, node_seqs, draft_map)
            if key not in _res:
                _res[key] = []
            _res[key] += draft_grams
        return _res

    def update_next_token_cache(self, next_token, draft_tree):

        for node in draft_tree.tree.dfs_node_seq[:]:
            # if node.value not in [2]:
            if node.value == next_token:
                self.cache_insert_and_pop(node, max_n=MAX_N)

        if next_token in self.cache:
            return self.cache[next_token]
        else:
            return None

    def update_other_token_cache(self, next_token, draft_tree):

        tmp_list = [x for x in draft_tree.tree.dfs_node_seq if x.value != next_token]
        for node in tmp_list:
            self.cache_insert_and_pop(node, max_n=MAX_N)

    async def update_cache_by_draft_tree(self, draft_tree: DraftTree):

        for node in draft_tree.tree.dfs_node_seq[:]:
            self.cache_insert_and_pop(node, max_n=MAX_N)

    def cache_stat(self):

        lengths = []
        for key, val in self.cache.items():
            lengths += [len(val)]
        all_count = sum(lengths)
        avg_len = sum(lengths) / len(lengths)
        std_len = np.std(lengths)
        cache_len = len(self.cache)

        return all_count, avg_len, std_len, cache_len

    def update_cache(self, next_token=None, draft: Union[DraftTree] = None,
                     draft_results=None):
        assert draft is not None

        if isinstance(draft, DraftTree):

            self.update_cache_by_draft_tree(next_token, draft)
        else:
            raise RuntimeError(f'Unsupported data type: {type(draft)}')

    def reduce_cache(self, keep_num):
        if keep_num is None:
            return
        else:
            res = {}
            if keep_num is not None:
                if keep_num == 0:
                    self.cache = {}
                    return
                i = -keep_num
            else:
                i = keep_num
            for key, val in self.cache.items():
                _ = len(val)
                res[key] = val[i:]
                self.remove_count += _ - len(res[key])
                self.gram_count -= _ - len(res[key])
            self.cache = res.copy()

    def flush_cache(self):
        self.cache = {}

    @staticmethod
    def convert_key(key):
        # convert the key to a hashable type
        if isinstance(key, int):
            key = key
        elif isinstance(key, list):
            key = tuple(key)
            if len(key) == 1:
                key = key[0]
            else:
                key = key
        elif isinstance(key, tuple):
            if len(key) == 1:
                key = key[0]
                key = key
        elif isinstance(key, torch.Tensor) or isinstance(key, np.ndarray):
            key = key.item()

        else:
            raise RuntimeError(f'Unsupported data type: {type(key)}')
        return key

    @staticmethod
    def slice_list(l, lengths, aux_sizes=[]):
        res = []
        p_size = sum(aux_sizes)
        for i in range(len(lengths)):
            if i == 0:
                t = []
                for j, aux_s in enumerate(aux_sizes):
                    t.append(l[sum(aux_sizes[:j + 1]) - 1])
                res.append(t)

            else:
                res.append(l[p_size + sum(lengths[1:i]):p_size + sum(lengths[1:i + 1])])
        for i, r in enumerate(res):
            assert len(r) == lengths[i]

        assert res[-1] == l[p_size + sum(lengths[1:-1]):]
        return res

    @staticmethod
    def extract_ngrams_fast(data, n):
        results = []
        for row in data:
            row_dict = {row[i]: tuple(row[i + 1:i + 1 + n]) for i in range(len(row) - n)}
            results.append(row_dict)
        return results

    @staticmethod
    def count_trie_nodes(sequences):
        trie = {}
        node_count = 0

        for seq in sequences:
            current_dict = trie
            for char in seq:
                if char not in current_dict:
                    current_dict[char] = {}
                    node_count += 1
                current_dict = current_dict[char]

        return node_count


class CacheServer:
    def __init__(self, corpus_path=None, tokenizer_path=None):
        self.corpus_path = corpus_path
        self.tokenizer_path = tokenizer_path
        self.guess_sem = None
        self.request_sem = None
        self.guess_shm = None
        self.request_shm = None
        self.guess_shm_size = 1024 * 1024
        self.request_shm_size = 1024 * 1024

        self.cache = ContextCacheTree(corpus_path=self.corpus_path, tokenizer_path=self.tokenizer_path)
        if self.corpus_path is not None and self.tokenizer_path is not None:
            self.cache.init_cache_by_corpus()
        elif self.corpus_path is None and self.tokenizer_path is None:
            pass
        else:
            raise RuntimeError(f'Unsupported data type: {type(corpus_path)}')
        self.set_shm_sem()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit, automatically clean up resources"""
        self.stop()
        return False

    def cleanup_existing_resources(self):
        """Clean up potentially existing old resources - silent mode"""
        # Clean up shared memory
        for shm_name in ["guess_tree", "request"]:
            try:
                shm = SharedMemory(name=shm_name)
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass  # Resource does not exist, no need to clean up
            except Exception:
                pass  # Silent failure, does not affect startup

        # Clean up semaphores
        for sem_name in ["/guess_res_sem", "/request_sem"]:
            try:
                sem = posix_ipc.Semaphore(sem_name)
                sem.unlink()
            except posix_ipc.ExistentialError:
                pass  # Resource does not exist, no need to clean up
            except Exception:
                pass  # Silent failure, does not affect startup

    def set_shm_sem(self):
        # First silently clean up potentially existing old resources
        self.cleanup_existing_resources()

        # Create shared memory
        try:
            self.guess_shm = SharedMemory(create=True, size=self.guess_shm_size, name="guess_tree")
        except FileExistsError:
            self.guess_shm = SharedMemory(name="guess_tree")
            self.guess_shm.close()
            self.guess_shm.unlink()
            self.guess_shm = SharedMemory(create=True, size=self.guess_shm_size, name="guess_tree")
            print('guess_shm exists', flush=True)
            print('create guess_shm', flush=True)

        try:
            self.request_shm = SharedMemory(create=True, size=self.request_shm_size, name="request")
        except FileExistsError:
            self.request_shm = SharedMemory(name="request")
            self.request_shm.close()
            self.request_shm.unlink()
            self.request_shm = SharedMemory(create=True, size=self.request_shm_size, name="request")
            print('request_shm exists', flush=True)
            print('create request_shm', flush=True)

        # Create semaphores
        try:
            self.guess_sem = posix_ipc.Semaphore("/guess_res_sem", flags=posix_ipc.O_CREX, initial_value=0)
        except posix_ipc.ExistentialError:
            self.guess_sem = posix_ipc.Semaphore("/guess_res_sem")
            print('guess_sem exists with value of:', self.guess_sem.value, flush=True)
            self.guess_sem.release()
            self.guess_sem.unlink()
            print('guess_sem unlinked', flush=True)
            self.guess_sem = posix_ipc.Semaphore("/guess_res_sem", flags=posix_ipc.O_CREX, initial_value=0)
            print('guess_sem created with value of:', self.guess_sem.value, flush=True)

        try:
            self.request_sem = posix_ipc.Semaphore("/request_sem", flags=posix_ipc.O_CREX, initial_value=0)
        except posix_ipc.ExistentialError:
            self.request_sem = posix_ipc.Semaphore("/request_sem")
            print('request_sem exists with value of:', self.request_sem.value, flush=True)
            self.request_sem.release()
            self.request_sem.unlink()
            print('request_sem unlinked', flush=True)
            self.request_sem = posix_ipc.Semaphore("/request_sem", flags=posix_ipc.O_CREX, initial_value=0)
            print('request_sem created with value of:', self.request_sem.value, flush=True)

    def write_shm_2(self, shm, obj):
        data_size = len(obj)
        shm.buf[:4] = struct.pack('>I', data_size)
        shm.buf[4:4 + data_size] = obj

    def get_data_in_buffer(self, conn):
        data = b''
        while True:
            chunk = conn.recv(40960)
            if not chunk:
                break
            data += chunk
            if b"END" == data[-3:]:  # special terminator
                break

        data = data[:-3]
        return data

    def update_next(self, next_token, draft_tree):
        t0 = perf_counter()
        guess_tree = self.cache.update_next_token_cache(next_token, draft_tree)
        if guess_tree is not None:
            guess_tree.guess_tree_attn_mask(torch.float16)

        obj = pickle.dumps(
            {'type': 'retrieve', 'guess_tree': guess_tree, 'update_cache_next_time': perf_counter() - t0})

        self.write_shm_2(self.guess_shm, obj)

        self.guess_sem.release()

    def handle_requests(self, request, conn):

        request_type = request.get('type')

        if request_type == 'update_cache_next_then_other':
            self.update_next(request['next_token'], request['draft'])
            self.cache.update_other_token_cache(next_token=request['next_token'], draft_tree=request['draft'])

        elif request_type == 'guess_tree_update':
            self.cache[request['key']] = request['guess_tree']

        elif request_type == 'retrieve':
            # print('retrieving cache')

            t0 = perf_counter()
            # clean_shm()

            key = request['all_old_tokens'][0]
            if key in self.cache.cache:
                guess_tree = self.cache.cache[key]
                guess_tree.guess_tree_attn_mask(torch.float16)
            else:
                guess_tree = None
            obj = pickle.dumps(
                {'type': 'retrieve', 'guess_tree': guess_tree, 'update_cache_next_time': perf_counter() - t0})
            self.write_shm_2(self.guess_shm, obj)
            self.guess_sem.release()

        elif request_type == 'reset_cache':
            print('resetting cache', flush=True)
            while self.guess_sem.value > 0:
                self.guess_sem.acquire()
            while self.request_sem.value > 0:
                self.request_sem.acquire()
            t0 = time.perf_counter()
            _corpus_path = self.cache.corpus_path
            _tokenizer_path = self.cache.tokenizer_path

            if PTD:
                # with PTD
                del self.cache.cache
                self.cache = ContextCacheTree(corpus_path=_corpus_path, tokenizer_path=_tokenizer_path)
                if PLD_INI:
                    # PLD + PTD
                    self.cache.PLD_init_cache(prompt_tokens=request['prompt_tokens'])

                elif REST_INI:
                    # REST + PTD
                    print('reset cache by reloading rest cache')
                    self.cache.init_cache_by_corpus()
            else:
                # WO PTD
                if PLD_INI:
                    # PLD
                    del self.cache.cache
                    self.cache = ContextCacheTree(corpus_path=_corpus_path, tokenizer_path=_tokenizer_path)
                    self.cache.PLD_init_cache(prompt_tokens=request['prompt_tokens'])
                elif REST_INI:
                    # REST
                    pass
                else:
                    # empty cache
                    pass

            ini_time = time.perf_counter() - t0
            obj = pickle.dumps(ini_time)
            self.write_shm_2(self.guess_shm, obj)
            self.guess_sem.release()

            print('cache resetted', flush=True)

        else:
            print('unknown request type', flush=True)

    def read_shm(self, shm):
        data_size = struct.unpack('>I', bytes(shm.buf[:4]))[0]
        data = bytes(shm.buf[4:4 + data_size])
        obj = pickle.loads(data)
        return obj

    def stop(self):
        """Safely clean up all resources"""
        # Clean up semaphores
        if hasattr(self, 'guess_sem') and self.guess_sem is not None:
            try:
                self.guess_sem.unlink()
            except Exception:
                pass  # Silent failure

        if hasattr(self, 'request_sem') and self.request_sem is not None:
            try:
                self.request_sem.unlink()
            except Exception:
                pass  # Silent failure

        # Clean up shared memory
        if hasattr(self, 'guess_shm') and self.guess_shm is not None:
            try:
                self.guess_shm.close()
                self.guess_shm.unlink()
            except Exception:
                pass  # Silent failure

        if hasattr(self, 'request_shm') and self.request_shm is not None:
            try:
                self.request_shm.close()
                self.request_shm.unlink()
            except Exception:
                pass  # Silent failure
        
        print('Cache server stopped', flush=True)

    def main(self):
        """Main loop with signal handling and resource cleanup"""
        # Register cleanup function on exit
        atexit.register(self.stop)

        # Register signal handlers
        def signal_handler(signum, frame):
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # kill command
        
        print('Tree cache server started', flush=True)
        try:
            while True:
                self.request_sem.acquire()
                request = self.read_shm(self.request_shm)
                self.handle_requests(request, None)
        except KeyboardInterrupt:
            pass  # Silent handling
        except Exception:
            pass  # Silent handling
        finally:
            self.stop()


class CacheServerBatch(CacheServer):
    def __init__(self, batch_size=1, corpus_path=None, tokenizer_path=None):
        super().__init__()
        self.batch_size = batch_size
        self.cache = [ContextCacheTree() for _ in range(batch_size)]
        self.batch_requests = []

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit, automatically clean up resources"""
        self.stop()
        return False

    def update_next_with_idx(self, lst_tokens, draft_tree, idx):
        t0 = perf_counter()
        guess_tree = self.cache[idx].update_next_token_cache(lst_tokens, draft_tree)
        if guess_tree is not None:
            guess_tree.guess_tree_attn_mask(torch.float16)
        retrieve_res = {'type': 'retrieve', 'guess_tree': guess_tree, 'update_cache_next_time': perf_counter() - t0}
        return retrieve_res

    def update_next_with_idx_and_retrieve(self, lst_tokens, draft_tree, idx):
        t0 = perf_counter()
        guess_tree = self.cache[idx].update_next_token_cache(lst_tokens, draft_tree)
        if guess_tree is not None:
            guess_tree.guess_tree_attn_mask(torch.float16)
            guess_tree.update_global_info()
        return guess_tree

    def retrieve(self, retrieve_keys):
        guess_trees = []
        for i, key in enumerate(retrieve_keys):
            if key in self.cache[i]:
                guess_tree = self.cache[i][key]
                guess_tree.guess_tree_attn_mask(torch.float16)
                guess_tree.update_global_info()
                guess_trees.append(guess_tree)
                # retrieve_res.append(
                #     {'type': 'retrieve', 'guess_tree': guess_tree, 'update_cache_next_time': perf_counter() - t0})
            else:
                guess_trees.append(None)
                # retrieve_res.append(
                #     {'type': 'retrieve', 'guess_tree': None, 'update_cache_next_time': perf_counter() - t0})
        return guess_trees
    def _update_single(self, batch_idx, tree):
        self.cache[batch_idx].update_cache_by_draft_tree(tree)

    def update_next_ignore_lst(self, trees):
        if trees is None:
            return

        for idx in range(len(trees)):
            self._update_single(idx, trees[idx])
            # self.cache[idx].update_cache_by_draft_tree(trees[idx])

    def update_nexts(self, lst_tokens, draft_trees):
        retrieve_list = []
        for i, lst_token in enumerate(lst_tokens):
            retrieve_list.append(self.update_next_with_idx_and_retrieve(lst_token, draft_trees[i], i))
        return retrieve_list

    def update_other_token_cache_batch(self, lst_tokens, draft_trees):
        for i, lst_token in enumerate(lst_tokens):
            self.cache[i].update_other_token_cache(next_token=lst_token, draft_tree=draft_trees[i])

    def update_by_ver_res(self, keys, guess_tree_next_tokens, guess_trees=None):

        for batch_idx, k in enumerate(keys):
            # all sequences
            if k in self.cache[batch_idx]:
                if guess_tree_next_tokens[batch_idx] is not None:
                    if self.cache[batch_idx][k].tree.node_count != len(guess_tree_next_tokens[batch_idx]):
                        # Tree was updated after last retrieve; node count may differ
                        self.cache[batch_idx][k].update_global_info()

                        if self.cache[batch_idx][k].tree.node_count == len(guess_tree_next_tokens[batch_idx]):
                            self.cache[batch_idx][k].expand_guess_tree_by_verify_res(guess_tree_next_tokens[batch_idx])
                            self._update_single(batch_idx, self.cache[batch_idx][k])

                        elif guess_trees[batch_idx] is not None:
                            if guess_trees[batch_idx].tree.node_count != len(guess_tree_next_tokens[batch_idx]):
                                guess_trees[batch_idx].update_global_info()
                            # guess_trees[batch_idx].expand_guess_tree_by_verify_res(guess_tree_next_tokens[batch_idx])

                            self.cache[batch_idx][k] = guess_trees[batch_idx].expand_guess_tree_by_verify_res(
                                guess_tree_next_tokens[batch_idx])
                            self._update_single(batch_idx, guess_trees[batch_idx])
                        else:
                            continue
                    else:
                        self.cache[batch_idx][k].expand_guess_tree_by_verify_res(guess_tree_next_tokens[batch_idx])
                        self._update_single(batch_idx, self.cache[batch_idx][k])

                else:
                    continue
            elif guess_trees[batch_idx] is not None:
                self.cache[batch_idx][k] = guess_trees[batch_idx].expand_guess_tree_by_verify_res(
                    guess_tree_next_tokens[batch_idx])
                self._update_single(batch_idx, guess_trees[batch_idx])
            else:
                continue

    def handle_requests(self, request, conn):
        request_type = request.get('type')
        if request_type == 'update_cache_next_then_other':
            retrieve_keys = request.get('lst_token', None)
            update_keys = request.get('update_keys', None)
            verify_token_lists = request.get('verify_token_lists', None)
            guess_trees = request.get('guess_trees', None)
            t0 = perf_counter()
            self.update_by_ver_res(update_keys, verify_token_lists, guess_trees)
            retrieve_list = self.update_nexts(retrieve_keys, request['draft'])
            retrieve_guess_trees = GuessTreeBatch(retrieve_list)
            retrieve_res = {'type': 'retrieve', 'guess_tree': retrieve_guess_trees,
                            'update_cache_next_time': perf_counter() - t0}
            retrieve_obj = pickle.dumps(retrieve_res)
            self.write_shm_2(self.guess_shm, retrieve_obj)
            self.guess_sem.release()
            self.update_other_token_cache_batch(retrieve_keys, request['draft'])

        elif request_type == 'retrieve':
            t0 = perf_counter()
            retrieve_keys = request['lst_token']
            update_keys = request['update_keys']
            verify_token_lists = request['verify_token_lists']
            update_guess_trees = request.get('guess_trees', None)
            retrieve_guess_trees = GuessTreeBatch(self.retrieve(retrieve_keys))
            retrieve_res = {'type': 'retrieve', 'guess_tree': retrieve_guess_trees,
                            'update_cache_next_time': perf_counter() - t0}
            obj = pickle.dumps(retrieve_res)
            self.update_by_ver_res(update_keys, verify_token_lists, update_guess_trees)
            self.write_shm_2(self.guess_shm, obj)
            self.guess_sem.release()

        elif request_type == 'reset_cache':
            print('resetting cache', flush=True)

            while self.guess_sem.value > 0:
                self.guess_sem.acquire()
            while self.request_sem.value > 0:
                self.request_sem.acquire()

            del self.cache

            self.cache = [ContextCacheTree() for i in range(self.batch_size)]
            print('cache reset', flush=True)

        elif request_type == 'reduce':
            remove_idx = request.get('remove_idx', None)
            if remove_idx is not None:
                if isinstance(remove_idx, int):
                    remove_idx = [remove_idx]
                elif isinstance(remove_idx, list):
                    pass
                else:
                    raise RuntimeError(f'Unsupported data type: {type(remove_idx)}')
                for i in sorted(remove_idx, reverse=True):
                    del self.cache[i]
        elif request_type == 'increase':
            target_num = request.get('target_num', None)
            if target_num is not None:
                if not isinstance(target_num, int):
                    raise RuntimeError(f'Unsupported data type: {type(target_num)}')
                if target_num > len(self.cache):
                    add_num = target_num - len(self.cache)
                    for i in range(add_num):
                        self.cache.append(ContextCacheTree())
            else:
                raise RuntimeError('add_num is None')
        else:
            print('unknown request type', flush=True)

    def main(self):
        """Main loop with signal handling and resource cleanup"""
        # Register cleanup function on exit
        atexit.register(self.stop)

        # Register signal handlers
        def signal_handler(signum, frame):
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # kill command
        
        print('Tree cache server started', flush=True)
        try:
            while True:
                self.request_sem.acquire()
                request = self.read_shm(self.request_shm)
                self.handle_requests(request, None)
        except KeyboardInterrupt:
            pass  # Silent handling
        except Exception:
            pass  # Silent handling
        finally:
            self.stop()


if __name__ == "__main__":
    args = sys.argv[1:]
    batch_size = int(args.pop(0))

    if len(args) < 2:
        print('cache init by empty')
        cache_server = CacheServerBatch(batch_size)
    else:
        corpus_path = args.pop(0)
        tokenizer_path = args.pop(0)
        print(f'cache init with: corpus_path: {corpus_path}, tokenizer_path: {tokenizer_path}')
        cache_server = CacheServerBatch(batch_size, corpus_path, tokenizer_path)
    
    cache_server.main()
