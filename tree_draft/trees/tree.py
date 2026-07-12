import collections
import json
import random
from collections import deque

import torch
from loguru import logger

from .node import Node


class Tree:
    config = None

    @classmethod
    def load_config(cls, config: [dict, str]):
        if isinstance(config, dict):
            cls.config = config
        elif isinstance(config, str):
            with open(config, 'r') as file:
                cls.config = json.load(file)

    def __init__(self, root_value=-1, **kwargs):
        self.root = Node(root_value, depth=0)
        self.exclude_root = kwargs['exclude_root']
        self.update_sub_nodes = kwargs['node_config']['update_sub_nodes']
        self.dfs_node_seq, self.bfs_node_seq = [], []
        self.attn_mask = None
        self.dfs_depth = []
        self.max_depth = kwargs['max_depth'] + self.root.depth
        self.gram_len = 4
        self.tree_depth = 1
        self.node_count = 0
        self.node_count_2 = 0
        self.s = ''

    def __str__(self):
        return self.s

    def __repr__(self):
        info = ('Root Value: ' + str(self.root.value) + '\n' +
                'Max depth: ' + str(self.max_depth) + '\n' +
                'Current depth: ' + str(self.tree_depth) + '\n' +
                'Number of nodes: ' + str(self.node_count) + '\n' +
                'Tree structure: ' + '\n' + self.s)
        return info

    def detail_info(self):
        """
        Print detailed information about the tree.
        """
        print("Tree Detailed Information:")
        print(f"Root Value: {self.root.value}")
        print(f"Max Depth: {self.max_depth}")
        print(f"Current Depth: {self.tree_depth}")
        print(f"Number of Nodes: {self.node_count}")
        print("Tree Structure:")
        self.print_tree()

    def add_child(self, parent_node: Node, child: [int, Node], ignore_limit=False):
        if isinstance(child, int):
            child_value = child
            if parent_node is not None:
                if parent_node.depth - self.root.depth >= self.max_depth:
                    logger.trace(
                        f'The depth of the parent node: {parent_node.value} exceeds the '
                        f'maximum depth: {self.max_depth} for child node:{child_value}.')

                child, node_change_count = parent_node.add_child(child_value, ignore_limit=ignore_limit)
                self.tree_depth = max(self.tree_depth, child.depth - self.root.depth)
                self.node_count += node_change_count
                return child, node_change_count
            else:
                return None, 0

        elif isinstance(child, Node):
            if parent_node is not None:
                if parent_node.depth - self.root.depth >= self.max_depth:
                    logger.trace(
                        f'The depth of the parent node: {parent_node.value} exceeds the '
                        f'maximum depth: {self.max_depth} for child node:{child.value}.')

                child, node_change_count = parent_node.add_child(child, ignore_limit=ignore_limit)
                self.tree_depth = max(self.tree_depth, child.depth - self.root.depth)
                self.node_count += node_change_count
                return child, node_change_count
            else:
                return None, 0
        else:
            raise RuntimeError(f'Unsupported data type: {type(child)}')

    def remove_child_by_order(self, node):
        removed_child = list(node.children.keys())[-1]
        node.remove_child(removed_child)

    def insert_child_by_order(self, parent_node: Node, child: [Node], order=0):
        children = list(parent_node.children.keys())
        order = order % len(children)

        child_items = list(parent_node.children.items())
        child_items.pop(order)
        child_items.insert(order, (child.value, child))
        parent_node.children.clear()
        parent_node.children.update(child_items)
        parent_node.children[child.value].parent = parent_node
        self.update_dfs()

    def add_sub_tree(self, parent_node: Node, sub_tree_root: Node):

        if parent_node is None:
            parent_node = self.root

        sub_tree_root_val = sub_tree_root.value
        parent_node.children[sub_tree_root_val] = sub_tree_root
        sub_tree_root.parent = parent_node

        return parent_node

    def remove_after_value(self, value, count):
        if value in self.dfs_node_seq:
            index = self.dfs_node_seq.index(value)
            del self.dfs_node_seq[index: index + count]

    def remove_child(self, parent_node: Node, child_node):
        parent_node, remove_count = parent_node.remove_child(child_node)
        self.node_count -= remove_count

        self.remove_after_value(child_node, remove_count)
        return parent_node

    def remove_min_probs_child(self, node):
        if node.draft_probs is None:
            raise RuntimeError('The draft freqs of the node is None.')
        else:
            child_tokens = list(node.children.keys())
            child_probs = node.draft_probs[child_tokens]
            min_prob_tokens = child_tokens[torch.argmin(child_probs).item()]
            self.remove_child(node, min_prob_tokens)

    def remove_child_by_order(self, node):
        removed_child = list(node.children.keys())[-1]
        node.remove_child(removed_child)

    def removed_child_get_depth(self, parent_node, child_node):
        parent_node, removed_count, removed_max_depth = parent_node.remove_node_and_get_removed_depth(child_node)
        self.node_count -= removed_count
        if removed_max_depth == self.tree_depth + self.root.depth:
            self.get_dfs_node_seq(self.root)
        return parent_node, removed_max_depth

    def get_predecessors(self, node):

        predecessors = []
        current = node
        while current is not None:
            predecessors.append(current)
            current = current.parent
        predecessors.reverse()
        return predecessors


    def get_node_str(self, node):
        self.s += node.depth * '\t' + str(node.value) + '\n'

    def get_dfs_node_seq(self, node):
        self.dfs_node_seq.append(node)

    def get_bfs_node_seq(self, node):
        self.bfs_node_seq.append(node)


    def traverse_children_DFS(self, nodes: list, visit_func):
        for node in nodes:
            visit_func(node)
            self.traverse_children_DFS(list(node.children.values()), visit_func)


    def traverse_DFS(self, nodes, visit_func):
        if nodes is None:
            start_nodes = list(self.root.children.values()) if self.exclude_root else [self.root]
        elif isinstance(nodes, Node):
            if self.exclude_root and self.root == nodes:
                start_nodes = list(nodes.children.values())
            else:
                start_nodes = [nodes]
        elif isinstance(nodes, list):
            assert isinstance(nodes[0], Node)
            start_nodes = nodes
        elif isinstance(nodes, collections.abc.ValuesView):
            start_nodes = list(nodes)
        else:
            raise RuntimeError(f'Unsupported data type: {type(nodes)}')
        self.traverse_children_DFS(start_nodes, visit_func)

    def traverse_BFS(self, nodes, visit_func):
        if nodes is None:
            start_nodes = list(self.root.children.values()) if self.exclude_root else [self.root]
        elif isinstance(nodes, Node):
            if nodes == self.root and self.exclude_root:
                start_nodes = list(nodes.children.values())
            else:
                start_nodes = [nodes]
        elif isinstance(nodes, list):
            start_nodes = nodes
        elif isinstance(nodes, collections.abc.ValuesView):
            start_nodes = list(nodes)
        else:
            raise RuntimeError(f'Unsupported data type: {type(nodes)}')

        queue = deque(start_nodes)
        while queue:
            current_node = queue.popleft()
            visit_func(current_node)
            for child in current_node.children.values():
                queue.append(child)

    def print_node(self, node):
        s = node.depth * '\t' + (f"{str(node.value)} (depth:{node.depth}, "
                                 f"max_child_num: {node.max_child_num}, "
                                 f"child_depth: {node.child_depth}")
        if hasattr(node, 'next_token'):
            s += f", next_token: {node.next_token}"
        if hasattr(node, 'draft_token'):
            s += f", draft_token: {node.draft_token}"
        s += ')'
        print(s)

    def print_tree(self):
        self.s = ''
        self.traverse_DFS(self.root, self.print_node)
        print(self.s)

    def update_depth(self):
        for node in self.dfs_node_seq:
            node.depth = node.parent.depth + 1

    def update_dfs(self):
        self.dfs_node_seq, self.bfs_node_seq = [], []
        self.s = ''

        self.traverse_DFS(self.root, self.get_dfs_node_seq)

        self.update_depth()
        self.update_child_nums()

        self.dfs_depth = [node.depth for node in self.dfs_node_seq]
        if len(self.dfs_depth) == 0:
            self.tree_depth = 0
        else:
            self.tree_depth = max([d - self.root.depth for d in self.dfs_depth])
        self.node_count = len(self.dfs_node_seq)

        self.update_child_depth()

        self.node_count_2 = len(self.dfs_node_seq)

    def guess_attention(self, node: Node = None, dtype=torch.bfloat16):
        if node is None:
            seq = self.dfs_node_seq
        else:
            start_node = node
            seq = []
            self.traverse_DFS(start_node, seq.append)

        n = len(seq)
        self.attn_mask = torch.full(
            (n, n), fill_value=torch.finfo(dtype).min, dtype=dtype)
        node_to_index = {node: i for i, node in enumerate(seq)}

        for node in seq:
            child_index = node_to_index[node]
            current = node
            while current is not None:
                if current == self.root and self.exclude_root:
                    break
                parent_index = node_to_index[current]
                self.attn_mask[child_index][parent_index] = 0
                current = current.parent

    def build_attention_matrix_DFS_2(self, node: Node = None, dtype=torch.bfloat16, device=None):

        n = len(self.dfs_node_seq)
        if device is not None:
            self.attn_mask = torch.full((n, n), fill_value=torch.finfo(dtype).min, dtype=dtype, device=device)
        else:
            self.attn_mask = torch.full((n, n), fill_value=torch.finfo(dtype).min, dtype=dtype)

        node_to_index = {node: i for i, node in enumerate(self.dfs_node_seq)}

        depths = [node.depth for node in self.dfs_node_seq]
        paths = [[] for _ in range(n)]
        first_level_children_indices = []

        stack = [self.root]
        if len(self.root.children) == 0:
            return
        pre_first_level_node = list(self.root.children.values())[0]

        while stack:
            current_node = stack.pop()

            if current_node.parent is not None:
                idx = node_to_index[current_node]

                if current_node.parent == self.root and self.exclude_root:
                    first_level_children_indices.append(idx)

                if current_node.parent == self.root:
                    paths[idx] = paths[node_to_index[pre_first_level_node]] + [idx]
                    pre_first_level_node = current_node
                else:
                    paths[idx] = paths[node_to_index[current_node.parent]] + [idx]

            for child in reversed(current_node.children.values()):
                stack.append(child)

        for i in range(n):
            self.attn_mask[i, paths[i]] = 0


    def update_sub_nodes_by_draft_token(self, ignore_limit=False):

        for node in self.dfs_node_seq:
            _, node_change_count = node.add_child(node.draft_token, ignore_limit=ignore_limit)
            if not self.update_sub_nodes:
                assert node_change_count >= 0

        return



    def get_cur_sub_tree_max_depth(self, node):
        cur_node = node
        while cur_node.parent is not self.root:
            cur_node = cur_node.parent

        return cur_node.child_depth


    def serialize_subtree_dfs(self, start_node, depth_limit=None):
        def dfs(current_node, path, all_sequences, current_depth=0):
            if current_node is None:
                return
            if current_node == self.root and self.exclude_root:
                new_path = path
            else:
                new_path = path + [current_node.value]

            if len(new_path) <= depth_limit:
                all_sequences.append(new_path)
            else:
                return

            for child in current_node.children.values():
                dfs(child, new_path, all_sequences, current_depth + 1)

        sequences = []
        if isinstance(start_node, Node):
            dfs(start_node, [], sequences)
        elif isinstance(start_node, list):
            for node in start_node:
                dfs(node, [], sequences)
        elif isinstance(start_node, dict):
            for node in start_node.values():
                dfs(node, [], sequences)
        return sequences

    def dfs_for_fixed_len_seq(self, current_node, path, all_sequences, fixed_length, seq_obj='value'):
        if current_node is None:
            return

        if current_node == self.root and self.exclude_root:
            new_path = path
        else:
            if seq_obj == 'value':
                new_path = path + [current_node.value]
            elif seq_obj == 'node':
                new_path = path + [current_node]
            else:
                raise RuntimeError(f'Unsupported seq_obj: {seq_obj}')

        if len(new_path) == fixed_length:
            all_sequences.append(tuple(new_path))
            return
        for child in current_node.children.values():
            self.dfs_for_fixed_len_seq(child, new_path, all_sequences, fixed_length, seq_obj)

    def serialize_subtree_dfs_fixed_length(self, start_node, fixed_length, seq_object='value'):

        sequences = []

        if isinstance(start_node, Node):
            self.dfs_for_fixed_len_seq(start_node, [], sequences, fixed_length, seq_object)
        elif isinstance(start_node, list):
            for node in start_node:
                assert isinstance(node, Node)
                self.dfs_for_fixed_len_seq(node, [], sequences, fixed_length, seq_object)
        elif isinstance(start_node, dict):
            for node in start_node.values():
                self.dfs_for_fixed_len_seq(node, [], sequences, fixed_length, seq_object)

        return sequences

    def serialize_children_dfs_fix_length(self, start_node, depth_limit=None, return_node=False):

        if depth_limit <= 0:
            if return_node:
                return [], []
            else:
                return []
        res = []
        for child_node in start_node.children.values():
            sequences = self.serialize_subtree_dfs_fixed_length(child_node, depth_limit)
            res += sequences

        if return_node:
            node_res = []
            for child_node in start_node.children.values():
                sequences = self.serialize_subtree_dfs_fixed_length(child_node, depth_limit, seq_object='node')
                node_res += sequences
            # judge whether the length of the node_res is equal to the length of the res
            assert len(node_res) == len(res)
            # assert the length of each element in the node_res is equal to the length of the res
            for i in range(len(res)):
                assert len(node_res[i]) == len(res[i])
            return res, node_res
        else:
            return res

    def remove_sibling_and_inherit(self, start_node, keep_value):
        # self.update_global_info()
        if keep_value not in start_node.children.keys():
            raise RuntimeError(f'The value {keep_value} is not in the children of the parent node {start_node.value}.')

        assert keep_value in start_node.children.keys()
        children_vals = list(start_node.children.keys())
        for child_val in children_vals:
            if child_val == keep_value:
                continue
            else:
                self.remove_child(start_node, child_val)
        return start_node.children[keep_value]

    def inherit_by_FIFO(self, parent: Node, cur_node: Node, inherit_node, ignore_limit=False):
        new_node = self.remove_sibling_and_inherit(cur_node, inherit_node)
        cur_node.remove_successor()
        parent.remove_child(cur_node)
        assert len(parent.children) < parent.max_child_num
        # self.add_sub_tree(parent, new_node)
        parent.add_child(new_node, ignore_limit=ignore_limit)

    def inherit_by_FIFO_keep_dup(self, parent: Node, cur_node: Node, inherit_node, ignore_limit=False):
        new_node = self.remove_sibling_and_inherit(cur_node, inherit_node)
        new_value = new_node.value + random.random()
        new_node.value = new_value
        cur_node.remove_successor()
        parent.remove_child(cur_node)
        assert len(parent.children) < parent.max_child_num
        parent.add_child(new_node, child_key=new_value, ignore_limit=ignore_limit)

    def inherit_keep_sub_tree(self, parent: Node, cur_node, keep_value):
        new_node = self.remove_sibling_and_inherit(cur_node, keep_value)
        parent.children.pop(cur_node.value)
        new_node.parent = parent
        parent.children[new_node.value] = new_node

    def inherit_keep_sub_tree_keep_node_val(self, parent: Node, cur_node, keep_value):
        children_values = parent.children.keys()
        grand_children_values = cur_node.children.keys()


    def go_deeper_for_only_exceed_sub_trees(self):
        first_level_children = list(self.root.children.values()).copy()
        for node in first_level_children:
            if node.child_depth >= self.max_depth:
                self.inherit_by_FIFO(self.root, node, list(node.children.keys())[0])
                # self.inherit_by_FIFO_keep_dup(self.root, node, list(node.children.keys())[0])
            else:
                pass
        self.update_dfs()
        assert all([node.child_depth <= self.max_depth for node in self.root.children.values()])

    def get_inherit_node_val_by_FIFO(self, first_level_children_val, first_level_node):
        grand_children_val = list(first_level_node.children.keys())
        try:
            inherit_node_value = grand_children_val.pop(0)
        except IndexError:
            print('IndexError')
        while inherit_node_value in first_level_children_val and len(grand_children_val):
            inherit_node_value = grand_children_val.pop(0)

        return inherit_node_value


    def get_inherit_node_val(self, first_level_children_val, first_level_node):
        return self.get_inherit_node_val_by_FIFO(first_level_children_val, first_level_node)

    def go_deeper_for_only_exceed_sub_trees_keep_struct(self):
        first_level_children = list(self.root.children.values()).copy()
        for i, node in enumerate(first_level_children):
            if node.child_depth >= self.max_depth:

                first_level_children_val = list(self.root.children.keys()).copy()
                inherit_node_value = self.get_inherit_node_val(first_level_children_val, node)
                if inherit_node_value in first_level_children_val:
                    self.remove_child(self.root, node)
                else:
                    self.inherit_keep_sub_tree(self.root, node, inherit_node_value)
            else:
                pass
        self.update_dfs()



    def go_deeper_for_all_sub_trees_keep_struct(self):
        first_level_children = list(self.root.children.values()).copy()
        if self.tree_depth > self.max_depth:
            for node in first_level_children:
                first_level_children_val = list(self.root.children.keys()).copy()
                if len(node.children) > 0:
                    inherit_node_value = self.get_inherit_node_val(first_level_children_val, node)
                    if inherit_node_value in first_level_children_val:
                        self.remove_child(self.root, node)
                    else:
                        self.inherit_keep_sub_tree(self.root, node, inherit_node_value)
            self.update_dfs()


    def prune_exceed_subtree(self):
        first_level_children = list(self.root.children.values()).copy()
        for node in first_level_children:
            if node.child_depth >= self.max_depth:
                self.remove_child(self.root, node)
            else:
                pass
        self.update_dfs()

    def remove_first_level_by_order(self):
        first_level_children = list(self.root.children.values()).copy()
        for node in first_level_children:
            self.remove_child(self.root, node)

    def update_child_nums(self):
        for node in self.dfs_node_seq:
            if node.depth <= self.max_depth:
                node.max_child_num = Node.config['max_child_num'][node.depth]
                node.min_child_num = Node.config['min_child_num'][node.depth]
            else:
                node.max_child_num = Node.config['default_max_child_num']
                node.min_child_num = Node.config['default_min_child_num']

    def remove_exceed_child(self):
        for node in self.dfs_node_seq:
            while len(node.children) > node.max_child_num:
                latest_child = list(node.children.keys())[-1]
                _, remove_count = node.remove_child(latest_child)
                self.node_count -= remove_count
        return

    def update_successor_num(self, start_node=None):
        if start_node is None:
            self.root.get_all_child_num()
        else:
            start_node.get_all_child_num()

    def update_child_depth(self, start_node=None):
        if start_node is None:
            self.root.get_max_depth()
        else:
            start_node.get_max_depth()


    def allocate_sub_tree(self, cur_node):
        while cur_node.parent != self.root:
            cur_node = cur_node.parent

        return cur_node

    def allocate_sub_tree_d(self, cur_node, depth):
        i = 0
        while i < depth:
            if cur_node.parent == self.root:
                break
            cur_node = cur_node.parent
            i += 1
            if cur_node.parent == self.root:
                break

        return cur_node

    def allocate_sub_root(self, cur_node):
        while cur_node.parent != self.root:
            cur_node = cur_node.parent
        return cur_node

    def get_all_seqs(self):

        def dfs(node, path, all_path):
            if node is None:
                return

            if node == self.root and self.exclude_root:
                new_path = path
            else:
                new_path = path + [node.value]
            if not node.children:
                all_path.append(new_path)
            for child in node.children.values():
                dfs(child, new_path, all_path)

        all_paths = []
        dfs(self.root, [], all_paths)
        return all_paths

    def copy_tree(self):

        def copy_node(node):

            if node is None:
                return None

            new_node = Node(node.value, depth=node.depth)
            if hasattr(node, 'draft_token'):
                new_node.draft_token = node.draft_token

            for child in node.children.values():
                copied_child = copy_node(child)
                copied_child.parent = new_node

                new_node.children[copied_child.value] = copied_child

            return new_node

        new_tree = Tree(self.root.value,
                        exclude_root=self.exclude_root,
                        node_config=self.config['node_config'],
                        max_depth=self.max_depth - self.root.depth)

        new_tree.root = copy_node(self.root)

        new_tree.update_dfs()

        return new_tree
