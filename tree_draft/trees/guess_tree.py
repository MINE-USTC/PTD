import random
import torch

from .node import Node
from .tree import Tree


class GuessTree:
    def __init__(self, key, guess_tup=[], root_value=-1, **kwargs):
        self.draft_map = {}
        self.gram_n = 4
        self.dfs = []
        self.tree = Tree(root_value, **kwargs)
        self.get_tree_from_guess_drafts(guess_tup)
        self.ignore_limit = True
        self.guess_map = {}
        self.key = key

    def update_global_info(self):
        self.tree.update_dfs()

    def clear(self):
        self.draft_map = {}
        self.all_tokens = None
        self.tree.root.remove_successor()
        self.tree.update_dfs()

    def get_tree_from_guess_drafts(self, guess_drafts):

        for draft in guess_drafts:
            cur_node = self.tree.root
            for token in draft:
                if token not in cur_node.children:
                    self.tree.add_child(cur_node, token, ignore_limit=self.ignore_limit)
                cur_node = cur_node.children[token]
        self.tree.update_dfs()
        return

    def refresh_draft_tree_tokens(self):
        self.tree.dfs_node_seq = []
        self.tree.traverse_DFS(self.tree.root, self.tree.get_dfs_node_seq)
        self.all_tokens = [node.value for node in self.tree.dfs_node_seq]
        self.tree.update_dfs()

    def guess_tree_attn_mask(self, dtype):
        self.tree.guess_attention(self.tree.root, dtype)

    def get_pos_ids(self, lst_id):
        pos_offset = lst_id - self.tree.root.depth
        pos_ids = []
        for node in self.tree.dfs_node_seq:
            pos_ids.append(node.depth + pos_offset)
        return pos_ids

    def prepare_ids(self, lst_id):
        self.refresh_draft_tree_tokens()
        pos_ids = self.get_pos_ids(lst_id)
        assert len(pos_ids) == len(self.all_tokens) == self.tree.node_count
        return pos_ids, self.all_tokens, self.tree.node_count

    def update_next_token(self, draft_tokens):
        if len(draft_tokens) != self.tree.node_count:
            breakpoint()
        assert len(draft_tokens) == self.tree.node_count
        for i, node in enumerate(self.tree.dfs_node_seq):
            node.next_token = draft_tokens[i]

    def expand_guess_tree_by_verify_res(self, verify_next_tokens):

        if verify_next_tokens is None:
            return self

        else:
            assert len(verify_next_tokens) == self.tree.node_count
            for i, node in enumerate(self.tree.dfs_node_seq):
                if i >= len(verify_next_tokens):
                    raise RuntimeError('error: verify_next_tokens length is less than the number of nodes in the tree.')
                node.add_child(verify_next_tokens[i], ignore_limit=self.ignore_limit)

        self.update_global_info()
        return self

    def update_child_probs(self, verify_probs):
        assert verify_probs.shape[0] == self.tree.node_count
        for i, node in enumerate(self.tree.dfs_node_seq):
            node.verify_probs = verify_probs[i, :]



    def dfs_longest_prefix_match(self, node, path, path_node, results, result_nodes):
        # only the right path from root
        path.append(node.value)
        path_node.append(node)

        if len(node.children) == 0:
            results.append(path + [node.next_token])
            result_nodes.append(path_node[:])
            return results, result_nodes

        if node.next_token in node.children:
            self.dfs_longest_prefix_match(node.children[node.next_token], path, path_node, results, result_nodes)
            path.pop()
            path_node.pop()
        else:
            results.append(path + [node.next_token])
            result_nodes.append(path_node[:])
            return results, result_nodes

        return results, result_nodes

    def dfs_logest_prefix_match_with_sample_2(self, node, node_idx, path, path_node, results, result_nodes,
                                              verify_probs, next_verify_prob):
        """
        DFS longest prefix match with sampling, but verify_probs are passed as parameters instead of stored in nodes.

        Args:
            node: Current tree node
            node_idx: Index of current node in dfs_node_seq
            path: Current token path
            path_node: Current node path
            results: Result token sequences
            result_nodes: Result node sequences
            next_verify_prob: Next token verification probability (as Python list or numpy array)
        """
        assert verify_probs.shape[0] == self.tree.node_count
        path.append(node.value)
        path_node.append(node)

        if node == self.tree.root:
            node_verify_probs = next_verify_prob
        else:
            node_verify_probs = verify_probs[node_idx, :]

        if len(node.children) == 0:
            # Leaf node: select token with highest probability
            verify_token = node_verify_probs.argmax().item()
            node.next_token = verify_token
            results.append(path + [verify_token])
            result_nodes.append(path_node[:])
            return results, result_nodes
        else:
            hit_flag = False
            next_list = list(node.children.keys())

            while len(next_list) > 0:

                verify_token = next_list.pop()
                verify_prob = node_verify_probs[verify_token]
                threshold = random.random()
                if verify_prob > threshold:
                    node.next_token = verify_token
                    # Find child node index in dfs_node_seq
                    child_node = node.children[verify_token]
                    child_idx = self.tree.dfs_node_seq.index(child_node)

                    self.dfs_logest_prefix_match_with_sample_2(
                        child_node, child_idx, path, path_node, results, result_nodes,
                        verify_probs, next_verify_prob
                    )
                    path.pop()
                    path_node.pop()
                    hit_flag = True
                    break
                else:
                    hit_flag = False
                    node_verify_probs[verify_token] = 0
                    node_verify_probs = node_verify_probs / node_verify_probs.sum()

            if not hit_flag:
                verify_token = node_verify_probs.argmax().item()
                node.next_token = verify_token
                results.append(path + [verify_token])
                result_nodes.append(path_node[:])
                path.pop()
                path_node.pop()
                return results, result_nodes

        return results, result_nodes


    def guess_hits_2(self):
        accept_idx = []
        accept_gram, accept_node = self.dfs_longest_prefix_match(self.tree.root, [], [], [], [])

        if len(accept_gram) == 0:
            accept_gram = [[self.tree.root.next_token]]
        else:
            assert len(accept_gram) == 1
            assert accept_gram[0][0] == self.tree.root.value
            accept_gram = [sublist[1:] for sublist in accept_gram]

        if len(accept_node) > 0:

            for node in accept_node[0]:
                if node.value == self.tree.root.value:
                    continue
                else:
                    accept_idx.append(self.tree.dfs_node_seq.index(node))

        return accept_gram[0], accept_idx

    def _guess_hits_sample_2(self, verify_probs, next_verify_prob):

        accept_idx = []  # for update kv cache
        accept_gram, accept_node = self.dfs_logest_prefix_match_with_sample_2(
            self.tree.root, 0, [], [], [], [],
            verify_probs, next_verify_prob
        )

        if len(accept_gram) == 0:
            accept_gram = [[self.tree.root.next_token]]
        else:
            assert len(accept_gram) == 1
            assert accept_gram[0][0] == self.tree.root.value
            accept_gram = [sublist[1:] for sublist in accept_gram]

        if len(accept_node) > 0:
            for node in accept_node[0]:
                if node.value == self.tree.root.value:
                    continue
                else:
                    accept_idx.append(self.tree.dfs_node_seq.index(node))

        return accept_gram[0], accept_idx



    def guess_hits(self, verify_token_list, first_guess):
        self.update_next_token(verify_token_list)
        self.tree.root.next_token = first_guess
        assert len(verify_token_list) == self.tree.node_count
        accept_seq, accept_index = self.guess_hits_2()
        max_hit = len(accept_seq) - 1
        return max_hit, accept_seq, accept_index


    def guess_hits_sample_2(self, verify_probs, next_verify_prob):

        accept_seq, accept_index = self._guess_hits_sample_2(verify_probs, next_verify_prob)
        max_hit = len(accept_seq) - 1
        return max_hit, accept_seq, accept_index


    def _merge_trees(self, node1, node2):

        stack = [(node1, node2)]

        while stack:
            current_node1, current_node2 = stack.pop()

            for child_value_2, child_node_2 in current_node2.children.items():
                if child_value_2 in current_node1.children:
                    stack.append((current_node1.children[child_value_2], child_node_2))
                else:
                    current_node1.add_child(child_node_2.value, ignore_limit=self.ignore_limit)
                    stack.append((current_node1.children[child_node_2.value], child_node_2))


    def merge_trees_by_list(self, token_list):
        """
        将一个 token 序列合并进当前 GuessTree。

        在 ContextCacheTree 中的使用方式是：
            key = token_list[0]
            remaining_tokens = token_list[1:]
            cache[key].merge_trees_by_list(remaining_tokens)

        因此这里的 token_list 表示：在某个 key 之后出现的后续 token 序列，
        我们在本树中从 root 出发依次插入这些 token，即：
            root -> token_list[0] -> token_list[1] -> ...

        参数：
            token_list: list[int]，可以为空；为空时不做任何事情。
        """
        # 类型和空列表检查
        if token_list is None or len(token_list) == 0:
            return

        assert isinstance(token_list, list), f"token_list should be list, got {type(token_list)}"

        # 从当前树的 root 开始向下走 / 插入
        cur_node = self.tree.root
        for tok in token_list:
            # 如果当前节点没有这个子节点，就新建一个
            if tok not in cur_node.children:
                # Node 支持 add_child，之前在 _merge_trees 里已经用过
                cur_node.add_child(tok, ignore_limit=True)
            # 继续往下走一层
            cur_node = cur_node.children[tok]

        # 更新 DFS 序等全局信息，保持和其它接口一致
        self.update_global_info()

    def merge_trees(self, other_tree_root):
        self._merge_trees(self.tree.root, other_tree_root)
        self.update_global_info()

    def _merge_trees_with_draft_token_iter(self, node1, node2):
        # merge node2 into node1, if node2 has draft_token, add it to node1
        stack = [(node1, node2)]
        while stack:
            current_node1, current_node2 = stack.pop()
            for child_value_2, child_node_2 in current_node2.children.items():
                if child_value_2 in current_node1.children:
                    stack.append((current_node1.children[child_value_2], child_node_2))
                else:
                    current_node1.add_child(child_node_2.value, ignore_limit=self.ignore_limit)
                    stack.append((current_node1.children[child_node_2.value], child_node_2))
            if hasattr(current_node2, 'draft_token'):
                if isinstance(current_node2.draft_token, int):
                    if current_node2.draft_token not in current_node1.children:
                        current_node1.add_child(current_node2.draft_token, ignore_limit=self.ignore_limit)
                elif isinstance(current_node2.draft_token, list):
                    for draft_token in current_node2.draft_token:
                        if draft_token not in current_node1.children:
                            current_node1.add_child(draft_token, ignore_limit=self.ignore_limit)
            else:
                pass



    def merge_trees_with_draft_token(self, other_tree_root):
        # merge other_tree_root into self.tree.root, if other_tree_root has draft_token, add it to self.tree.root
        self._merge_trees_with_draft_token_iter(self.tree.root, other_tree_root)
        self.update_global_info()

    def deepcopy(self, tree_config):
        new_tree = GuessTree(key=self.key, **tree_config)
        new_tree.tree.root.value = self.tree.root.value
        new_tree.merge_trees(self.tree.root)
        return new_tree

    def prune(self, max_n):

        if max_n < 0:
            return
        elif self.tree.node_count > max_n:
            self.tree.root.remove_child(next(iter(self.tree.root.children)))
        else:
            return





class GuessTreeBatch:
    def __init__(self, retrieve_trees=None, **kwargs):

        self.batch_size = len(retrieve_trees)
        self.guess_trees = []
        for retrieve_tree in retrieve_trees:
            if isinstance(retrieve_tree, GuessTree):
                self.guess_trees.append(retrieve_tree)
            elif retrieve_tree is None:
                self.guess_trees.append(None)
            else:
                raise TypeError(
                    f"Unsupported type {type(retrieve_tree)} for retrieve_trees. Expected GuessTree or dict.")
        self.guess_trees = retrieve_trees[:]
        self.sizes = self.get_node_counts()
        self.max_guess_count = max(self.sizes)

    def __getitem__(self, idx):
        return self.guess_trees[idx]

    def __len__(self):
        return len(self.guess_trees)

    def __setitem__(self, idx, value):
        self.guess_trees[idx] = value

    def __iter__(self):
        return iter(self.guess_trees)

    def remove(self, idx):
        if idx >= 0 and idx < len(self.guess_trees):
            popped_tree = self.guess_trees.pop(idx)
            self.batch_size -= 1
            self.sizes = self.get_node_counts()
            self.max_guess_count = max(self.sizes)
        else:
            raise IndexError(f"Index {idx} out of range for guess_trees with length {len(self.guess_trees)}.")

    def get_node_counts(self):
        return [tree.tree.node_count if tree is not None else 0 for tree in self.guess_trees]

    def guess_trees_attention_mask(self, dtype):
        for i in range(self.batch_size):
            if self.guess_trees[i] is not None:
                self.guess_trees[i].guess_tree_attn_mask(dtype=dtype)
            else:
                self.guess_trees[i] = None

    def get_verify_token_lists(self, argmax_next_token_list, outputs):
        verify_token_lists = []
        verify_probs = []

        for i in range(self.batch_size):

            if self.guess_trees[i] is not None and self.guess_trees[i].tree.node_count > 0:
                if -self.max_guess_count + self.sizes[i] == 0:
                    verify_token_lists.append(argmax_next_token_list[i][-self.max_guess_count:])
                    verify_probs.append(outputs.logits[i][-self.max_guess_count:])
                else:
                    verify_token_lists.append(
                        argmax_next_token_list[i][-self.max_guess_count:-self.max_guess_count + self.sizes[i]])
                    verify_probs.append(outputs.logits[i, -self.max_guess_count:-self.max_guess_count + self.sizes[i]])

            else:
                verify_token_lists.append(None)
                verify_probs.append(None)

        return verify_token_lists, verify_probs

    def guess_hits(self, verify_token_lists, next_tokens_int, eos_token_id, profile):
        hits = []
        max_hits = []
        accept_seq = [[] for _ in range(self.batch_size)]
        accept_index = [[] for _ in range(self.batch_size)]

        for i in range(self.batch_size):
            if self.guess_trees[i] is not None and self.guess_trees[i].tree.node_count > 0:
                _max_hit, _accept_seq, _accept_index = self.guess_trees[i].guess_hits(verify_token_lists[i],
                                                                                      next_tokens_int[i])
                for j, t in enumerate(_accept_seq):
                    if t in eos_token_id:
                        _accept_seq = _accept_seq[:j + 1]
                        _accept_index = _accept_index[:j + 1]
                        _max_hit = len(_accept_seq) - 1
                        break

                hits.append(_accept_seq[:])
                max_hits.append(_max_hit)
                accept_seq[i] = _accept_seq
                accept_index[i] = _accept_index

                profile.incremental_update('cache_hit_count', 1)
                profile.incremental_update('cdt_tokens_num', self.guess_trees[i].tree.node_count)
                profile.incremental_update('trie_node_count', self.guess_trees[i].tree.node_count)
                profile.append_list('hit_len_list', _max_hit)

            else:
                max_hits.append(0)
                hits.append([next_tokens_int[i]])

        return hits, max_hits, accept_seq, accept_index

    # def guess_hits_sample(self, verify_probs, next_verify_prob, eos_token_id, next_tokens_int, profile):
    #     hits = []
    #     max_hits = []
    #     accept_seq = [[] for _ in range(self.batch_size)]
    #     accept_index = [[] for _ in range(self.batch_size)]
    #
    #     for i in range(self.batch_size):
    #         if self.guess_trees[i] is not None and self.guess_trees[i].tree.node_count > 0:
    #             assert verify_probs[i] is not None
    #             _max_hit, _accept_seq, _accept_index = self.guess_trees[i].guess_hits_sample(verify_probs[i],
    #                                                                                          next_verify_prob[i])
    #             for j, t in enumerate(_accept_seq):
    #                 if t in eos_token_id:
    #                     _accept_seq = _accept_seq[:j + 1]
    #                     _accept_index = _accept_index[:j + 1]
    #                     _max_hit = len(_accept_seq) - 1
    #                     break
    #
    #             hits.append(_accept_seq[:])
    #             max_hits.append(_max_hit)
    #             accept_seq[i] = _accept_seq
    #             accept_index[i] = _accept_index
    #
    #             profile.incremental_update('cache_hit_count', 1)
    #             profile.incremental_update('cdt_tokens_num', self.guess_trees[i].tree.node_count)
    #             profile.incremental_update('trie_node_count', self.guess_trees[i].tree.node_count)
    #             profile.append_list('hit_len_list', _max_hit)
    #
    #         else:
    #             max_hits.append(0)
    #             hits.append([next_tokens_int[i]])
    #
    #     # 确保所有 guess_trees 的 verify_probs 都被清理（虽然每个 guess_hits_sample 内部已经清理，但这里再次确保）
    #     for i in range(self.batch_size):
    #         if self.guess_trees[i] is not None:
    #             self.guess_trees[i].clear_all_verify_probs()
    #
    #     return hits, max_hits, accept_seq, accept_index

    def guess_hits_sample(self, verify_probs, next_tokens_probs, next_tokens_int, eos_token_id, profile):
        """
        Sampling version of guess_hits for batch processing.
        Uses guess_hits_sample_2 which doesn't store tensors in tree nodes.

        Args:
            verify_probs: List of verify probabilities for each sequence (from logits_warper)
            next_tokens_probs: List of next token probabilities for each sequence (from logits_warper)
            next_tokens_int: List of next token integers for fallback
            outputs: Model outputs containing logits and all_ids
            logits_warper: Logits warper for sampling
            tokenizer: Tokenizer for padding side check
            prefill_size: Size of prefill
            prompt_sizes: Sizes of prompts for each sequence
            is_prefill: Whether in prefill mode
            eos_token_id: List of EOS token IDs
            profile: Profile object for metrics

        Returns:
            hits: List of accepted token sequences
            max_hits: List of max hit counts for each sequence
            accept_seq: List of accepted sequences
            accept_index: List of accepted indices
        """
        hits = []
        max_hits = []
        accept_seq = [[] for _ in range(self.batch_size)]
        accept_index = [[] for _ in range(self.batch_size)]

        for i in range(self.batch_size):
            if self.guess_trees[i] is not None and self.guess_trees[i].tree.node_count > 0:
                # Get next_probs for this sequence

                _max_hit, _accept_seq, _accept_index = self.guess_trees[i].guess_hits_sample_2(
                    verify_probs[i], next_tokens_probs[i]
                )
                # Check for eos_token
                for j, t in enumerate(_accept_seq):
                    if t in eos_token_id:
                        _accept_seq = _accept_seq[:j + 1]
                        _accept_index = _accept_index[:j + 1]
                        _max_hit = len(_accept_seq) - 1
                        break

                hits.append(_accept_seq[:])
                max_hits.append(_max_hit)
                accept_seq[i] = _accept_seq
                accept_index[i] = _accept_index

                profile.incremental_update('cache_hit_count', 1)
                profile.incremental_update('cdt_tokens_num', self.guess_trees[i].tree.node_count)
                profile.incremental_update('trie_node_count', self.guess_trees[i].tree.node_count)
                profile.append_list('hit_len_list', _max_hit)
            else:
                max_hits.append(0)
                hits.append([next_tokens_int[i]])

        return hits, max_hits, accept_seq, accept_index

    def update_guess_trees_with_retrieve_res(self, unfinished_sequences, finished_batch_idx):

        if unfinished_sequences.shape[0] != len(self.guess_trees):
            assert isinstance(finished_batch_idx, (int, list)) and finished_batch_idx != -1, \
                f"finished_batch_idx should be an int or a list of ints. Current finished_batch_idx: {finished_batch_idx}"

            finished_batch_idx = [finished_batch_idx] if isinstance(finished_batch_idx, int) else finished_batch_idx
            invalidate_indices = [i for i in range(len(self.guess_trees)) if i in finished_batch_idx]

            for ii in invalidate_indices[::-1]:
                self.remove(ii)
            for i in range(len(self.guess_trees)):
                if self.guess_trees[i] is not None:
                    self.guess_trees[i].update_global_info()

        else:
            for i in range(len(self.guess_trees)):
                if self.guess_trees[i] is not None:
                    self.guess_trees[i].update_global_info()

        return

    def detail_info(self):
        print(f"Draft Trees Size: {self.batch_size}\n")
        # info += "Draft Trees Details:\n"
        for i in range(self.batch_size):
            print(f"Draft Tree {i}:")
            # info += f"Draft Tree {i}:\n"
            self.guess_trees[i].tree.detail_info()
            # info += self.draft_trees[i].tree.__repr__() + "\n"
            print("\n")
        # return info
