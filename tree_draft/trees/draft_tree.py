import random

import torch

from .tree import Tree
from .utils import print_with_depth_format


class DraftTree:
    def __init__(self, all_tokens=None, **kwargs):
        self.kwargs = kwargs
        self.tree = Tree(-1, **kwargs)

        self.width = self.tree.root.max_child_num
        self.ini_max_depth = kwargs['ini_max_depth']
        self.sub_tree_num = kwargs.get('sub_tree_num', 0)

        if all_tokens is not None:
            self.init_only_several_nodes(token_pool=all_tokens)

        self.tree.update_dfs()
        self.dfs()
        self.max_key_len = 4
        self.attention = []
        self.remove_idx = 0

        self.top_p = 0.2
        self.top_k = 5
        self.GUESS_TREE_KEY = 0

    def __repr__(self):
        return self.tree

    def update_global_info(self):
        self.tree.update_dfs()

    def clear(self):
        self.all_tokens = None
        self.tree.root.remove_successor()
        self.tree.update_dfs()


    def re_ini(self, all_tokens):

        if all_tokens is not None:
            self.init_only_several_nodes(token_pool=all_tokens)
        else:
            raise RuntimeError('Please specify the tokens pool.')

    def init_only_several_nodes(self, token_pool, ignore_limit=False, tokenizer=None):
        all_old_tokens = token_pool[:]
        # Initialize the tree with a root node
        if self.ini_max_depth == 0:
            return

        for i in range(2):
            if len(self.tree.root.children) > 0:
                node_value = list(self.tree.root.children.keys())[0]
                while node_value in self.tree.root.children:
                    if len(all_old_tokens) == 0:
                        node_value = random.choice(list(range(0, tokenizer.vocab_size)))
                    else:
                        node_value = random.choice(all_old_tokens)
                        all_old_tokens.remove(node_value)

            else:
                node_value = random.choice(all_old_tokens)
                all_old_tokens.remove(node_value)

            cur_node, _ = self.tree.add_child(self.tree.root, node_value, ignore_limit)

            inner_length = random.randint(1, self.ini_max_depth)
            for j in range(1, inner_length):
                child_node, _ = self.tree.add_child(cur_node, random.choice(all_old_tokens), ignore_limit)
                cur_node = child_node
        return



    def get_pos_ids(self, lst_id):
        pos_offset = lst_id - self.tree.root.depth
        pos_ids = []
        first_level_children = list(self.tree.root.children.values())
        for node in self.tree.dfs_node_seq:
            p_node = node
            while p_node not in first_level_children:
                p_node = p_node.parent
            sub_tree_offset = first_level_children.index(p_node)
            pos_ids.append(node.depth + pos_offset + sub_tree_offset)
        return pos_ids

    def prepare_ids(self, lst_id):
        self.refresh_draft_tree_tokens()
        pos_ids = self.get_pos_ids(lst_id)
        assert len(pos_ids) == len(self.all_tokens) == self.tree.node_count
        return pos_ids, self.all_tokens, self.tree.node_count

    def refresh_draft_tree_tokens(self):
        self.tree.dfs_node_seq = []
        self.tree.traverse_DFS(self.tree.root, self.tree.get_dfs_node_seq)
        self.all_tokens = [node.value for node in self.tree.dfs_node_seq]
        self.tree.update_dfs()


    def update_draft_tree_inter_logit(self):
        if len(self.tree.dfs_node_seq) == self.tree.node_count:
            self.tree.update_sub_nodes_by_draft_token()
        else:
            raise ValueError('The length of draft tokens is not equal to the length of the draft tree nodes.')

    def draft_tree_prune_root_node(self):
        remove_child = list(self.tree.root.children.values())[0]
        self.tree.remove_child(self.tree.root, remove_child)

    def get_top_p_tokens(self, next_tokens_scores, top_p):
        assert len(next_tokens_scores.shape) == 1

        sorted_probs, sorted_indices = torch.sort(next_tokens_scores, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        indices = (cumulative_probs > top_p).int().argmax().item()
        top_p_tokens = sorted_indices[:indices + 1].tolist()
        return top_p_tokens

    def get_top_k_tokens(self, next_tokens_scores, top_k):
        top_k_values, top_k_indices = torch.topk(next_tokens_scores, top_k)
        top_k_tokens = top_k_indices.tolist()

        return top_k_tokens

    def prune_exceed_child_by_order(self):
        for i, node in enumerate(self.tree.dfs_node_seq):
            while len(node.children) > node.max_child_num:
                self.tree.remove_child_by_order(node)

    def prune_exceed_child_by_prob(self):

        for i, node in enumerate(self.tree.dfs_node_seq):
            while len(node.children) > node.max_child_num:
                self.tree.remove_min_probs_child(node)

        self.tree.update_dfs()

    def prune_exceed_child_by_order(self):
        for i, node in enumerate(self.tree.dfs_node_seq):
            while len(node.children) > node.max_child_num:
                self.tree.remove_child_by_order(node)


    def update_sub_tree_root(self, next_token):
        if len(self.tree.root.children) > self.tree.root.min_child_num:
            self.draft_tree_prune_root_node()
        if len(self.tree.root.children) < self.tree.root.min_child_num:

            first_level_nodes = list(self.tree.root.children.values())
            if len(first_level_nodes) > 0:
                last_node = first_level_nodes[-1]
                if isinstance(last_node.draft_token, list):
                    new_child_value = last_node.draft_token[0]
                elif isinstance(last_node.draft_token, int):
                    new_child_value = last_node.draft_token
                else:
                    raise RuntimeError('The draft token is not a valid type.')
                if len(first_level_nodes) < self.tree.root.max_child_num:
                    if new_child_value in last_node.children:
                        self.tree.remove_child(last_node, new_child_value)

            else:
                new_child_value = next_token

            self.tree.add_child(self.tree.root, new_child_value)
        self.tree.update_dfs()

    def remove_eos_sub_trees(self, eos_token=2):

        for node in self.tree.dfs_node_seq[:]:
            if node.value == eos_token:
                sub_tree_node = self.tree.allocate_sub_tree_d(node, 4)
                if sub_tree_node in sub_tree_node.parent.children.values():
                    self.tree.remove_child(sub_tree_node.parent, sub_tree_node)

    def prune_depth(self):
        self.prune_d()
        self.remove_eos_sub_trees()


    def prune_d(self):

        self.tree.go_deeper_for_only_exceed_sub_trees_keep_struct()
        self.prune_exceed_child_by_order()
        self.tree.update_dfs()



    def get_all_dfs_seq(self):
        return self.tree.get_all_seqs()

    def dfs(self, visit_func=print_with_depth_format):
        return self.tree.traverse_DFS(self.tree.root, visit_func)

    def draft_tree_attn_mask(self, dtype=torch.bfloat16, device=None):
        self.tree.build_attention_matrix_DFS_2(self.tree.root, dtype, device)

    def get_draft_map(self, draft_tokens):

        draft_map = {}
        for i, node in enumerate(self.tree.dfs_node_seq):
            draft_map[node] = draft_tokens[i]
        return draft_map

    def get_draft_map_by_node_probs(self, method='top_p', top_p=None, top_k=None):
        draft_map = {}
        top_p = self.top_p if top_p is None else top_p
        top_k = self.top_k if top_k is None else top_k

        for node in self.tree.dfs_node_seq:
            probs = node.draft_probs
            if method == 'top_p':
                top_p_tokens = self.get_top_p_tokens(probs, top_p)
                draft_map[node] = top_p_tokens
            if method == 'top_k':
                top_k_tokens = self.get_top_k_tokens(probs, top_k)
                draft_map[node] = top_k_tokens
            if method == 'argmax':
                draft_map[node] = torch.argmax(probs).item()
            if method == 'mix':
                top_p_tokens = self.get_top_p_tokens(probs, top_p)
                mix_tokens = top_p_tokens[:top_k]
                draft_map[node] = mix_tokens

        return draft_map

    def update_draft_token_argmax(self, next_tokens):
        """
        update draft tokens of the draft tree for updating
        """
        for i, node in enumerate(self.tree.dfs_node_seq):
            node.draft_token = next_tokens[i]

    def copy_draft_tree(self):
        new_draft_tree = DraftTree(**self.kwargs)
        new_draft_tree.tree.tree_depth = self.tree.tree_depth
        new_draft_tree.tree.max_depth = self.tree.max_depth
        new_draft_tree.tree = self.tree.copy_tree()
        # new_draft_tree.all_tokens = self.all_tokens.copy()
        # new_draft_tree.draft_map = self.draft_map.copy()
        return new_draft_tree



class DraftTrees:
    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.batch_size = kwargs['batch_size']
        self.draft_trees = [DraftTree(**kwargs) for _ in range(self.batch_size)]
        self.sizes = [self.draft_trees[i].tree.node_count for i in range(self.batch_size)]

    def __getitem__(self, idx):
        return self.draft_trees[idx]

    def update_count(self):
        self.sizes = [self.draft_trees[i].tree.node_count for i in range(self.batch_size)]

    def clear(self):
        for i in range(self.batch_size):
            self.draft_trees[i].clear()

    def re_ini(self, all_old_tokens):
        for i in range(self.batch_size):
            self.draft_trees[i].re_ini(all_old_tokens[i])

    def update_dfs(self):
        for i in range(self.batch_size):
            self.draft_trees[i].tree.update_dfs()

    def update_draft_tree_attn_mask(self, dtype=torch.bfloat16, device=None):
        for i in range(self.batch_size):
            self.draft_trees[i].draft_tree_attn_mask(dtype, device)

    def assert_node_count(self):
        for i in range(self.batch_size):
            try:
                assert self.draft_trees[i].tree.node_count == len(self.draft_trees[
                                                                      i].tree.dfs_node_seq), f"Draft tree {i} node count mismatch: {self.draft_trees[i].tree.node_count} != {len(self.draft_trees[i].tree.dfs_node_seq)}"
            except AssertionError as e:
                print(
                    f"Draft tree {i} node count mismatch: {self.draft_trees[i].tree.node_count} != {len(self.draft_trees[i].tree.dfs_node_seq)}")
                raise AssertionError(e)

    def prepare_ids(self, lst_id):
        draft_pos_list = []
        all_draft = []
        draft_attn_size = []

        for i in range(self.batch_size):
            try:
                pos_ids, all_tokens, node_count = self.draft_trees[i].prepare_ids(lst_id[i])
            except IndexError as e:
                print(f"Error in preparing IDs for draft tree {i}: {e}")
                raise IndexError(e)
            draft_pos_list.append(pos_ids)
            all_draft.append(all_tokens)
            draft_attn_size.append(node_count)
        return draft_pos_list, all_draft, draft_attn_size

    def refresh_draft_tree_tokens(self):
        for i in range(self.batch_size):
            self.draft_trees[i].refresh_draft_tree_tokens()

    def get_draft_counts(self):
        return [t.tree.node_count for t in self.draft_trees]

    def get_draft_count(self):
        return sum([t.tree.node_count for t in self.draft_trees])

    def split_by_sizes(self, lst, sizes):
        result = []
        i = 0
        for size in sizes:
            result.append(lst[i:i + size])
            i += size
        return result

    def update_draft_tree_argmax(self, draft_tokens_list):
        for i in range(self.batch_size):
            self.draft_trees[i].update_draft_token_argmax(draft_tokens_list[i])

    def copy_draft_tree(self):
        new_draft_trees = DraftTrees(**self.kwargs)
        for i in range(self.batch_size):
            new_draft_trees.draft_trees[i] = self.draft_trees[i].copy_draft_tree()
        return new_draft_trees

    def update_draft_tree_inter_logit(self):
        for i in range(self.batch_size):
            self.draft_trees[i].update_draft_tree_inter_logit()

    def update_sub_tree_root(self, next_tokens):
        for i in range(self.batch_size):
            self.draft_trees[i].update_sub_tree_root(next_tokens[i])

    def prune_depth(self):
        for i in range(self.batch_size):
            self.draft_trees[i].prune_depth()

    def draft_tree_attn_mask(self, dtype=torch.bfloat16, device=None):
        for i in range(self.batch_size):
            self.draft_trees[i].draft_tree_attn_mask(dtype, device)

    def detail_info(self):
        print(f"Draft Trees Size: {self.batch_size}\n")
        # info += "Draft Trees Details:\n"
        for i in range(self.batch_size):
            print(f"Draft Tree {i}:")
            # info += f"Draft Tree {i}:\n"
            self.draft_trees[i].tree.detail_info()
            # info += self.draft_trees[i].tree.__repr__() + "\n"
            print("\n")
        # return info

    def reduce(self, idx):
        """
        Reduce the draft tree at the specified index.
        """
        if isinstance(idx, int):
            idx = [idx]
        elif isinstance(idx, list):
            pass
        else:
            raise TypeError("Index must be int or list")

        for i in sorted(idx, reverse=True):
            if i < 0 or i >= self.batch_size:
                breakpoint()
                raise IndexError(f"Index {i} is out of range for draft trees of size {self.batch_size}.")
            else:
                del self.draft_trees[i]
                self.batch_size -= 1

    def update_global_info(self):
        """
        Update global information for all draft trees.
        """
        for i in range(self.batch_size):
            self.draft_trees[i].update_global_info()
