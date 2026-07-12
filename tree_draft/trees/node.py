import json
import random
from typing import Union

from loguru import logger


class Node:
    config = None

    @classmethod
    def load_config(cls, config: [dict, str]):
        if isinstance(config, dict):
            cls.config = config
        elif isinstance(config, str):
            with open(config, 'r') as file:
                cls.config = json.load(file)

        assert cls.config is not None
        assert 'max_child_num' in cls.config
        assert 'min_child_num' in cls.config
        assert 'update_sub_nodes' in cls.config
        assert len(cls.config['max_child_num']) == len(cls.config['min_child_num'])

    def __init__(self, value, parent=None, depth=1, **node_config):
        self._value = value

        self.children = {}  # type: dict[int, Node]
        self.parent = parent
        self.depth = depth
        self.draft_probs = None
        self.freq = 1

        if self.depth > len(Node.config['max_child_num']) - 1:
            self.max_child_num = Node.config['default_max_child_num']
            self.min_child_num = Node.config['default_min_child_num']
        else:
            self.max_child_num = node_config.get('max_child_num', Node.config['max_child_num'][depth])
            self.min_child_num = node_config.get('min_child_num', Node.config['min_child_num'][depth])

        self.successors_num = 0
        self.child_depth = 0

    @property
    def value(self):
        return int(self._value)

    def reset(self):
        self.children = {}
        self.successors_num = 0
        self.child_depth = 0

    @property
    def org_value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value

    def __add_child(self, child):
        if isinstance(child, int):
            child_node = Node(child, parent=self, depth=self.depth + 1)
            self.children[child] = child_node
        elif isinstance(child, Node):
            child.depth = self.depth + 1
            child.parent = self
            child_node = child
            self.children[child.value] = child
        else:
            raise RuntimeError(f'Unsupported child type: {type(child)}')

        return child_node

    def __add_child_with_child_key(self, child, child_key):
        if isinstance(child, int):
            child_node = Node(child, parent=self, depth=self.depth + 1)
        elif isinstance(child, Node):
            child.depth = self.depth + 1
            child.parent = self
            child_node = child
        else:
            raise RuntimeError(f'Unsupported child type: {type(child)}')
        child_node.value = child_key
        self.children[child_key] = child_node
        return child_node

    def remove_successor(self):
        self.children = {}
        # self.children_order = []
        self.successors_num = 0

    def get_all_child_num(self):
        self.successors_num = len(self.children)
        for child in self.children.values():
            self.successors_num += child.get_all_child_num()
        return self.successors_num

    def get_max_depth(self):
        # Hop count from this node to deepest leaf
        if len(self.children) == 0:
            self.child_depth = 0
            return 0
        else:
            self.child_depth = 1 + max(child.get_max_depth() for child in self.children.values())
            return self.child_depth

    def _add_child(self, child_value, ignore_limit=False):
        assert child_value not in self.children or len(self.children) == 1
        node_change_num = 0

        if Node.config['ignore_exceed'] or ignore_limit:
            pass
        else:

            if len(self.children) >= self.max_child_num:
                if Node.config['update_sub_nodes']:
                    removed_node_val = list(self.children.keys())[0]
                    _, remove_count = self.remove_child(removed_node_val)
                    logger.trace(f'removed {remove_count} nodes from the node with root of: {removed_node_val}.')
                    node_change_num -= remove_count
                else:
                    logger.trace('The number of the children of the parent node has reached the maximum. add 0 child.')
                    node_change_num = 0
                    return self, node_change_num
            else:
                pass
        child_node = self.__add_child(child_value)
        node_change_num += 1
        return child_node, node_change_num

    def _add_child_Node(self, child_node, ignore_limit=False):
        assert child_node.value not in self.children or len(self.children) == 1
        node_change_num = 0
        if Node.config['ignore_exceed'] or ignore_limit:
            pass
        else:
            if len(self.children) >= self.max_child_num:
                if Node.config['update_sub_nodes']:
                    remove_node_val = list(self.children.keys())[0]
                    _, remove_node_count = self.remove_child(remove_node_val)
                    node_change_num -= remove_node_count
                else:
                    node_change_num = 0
                    return self, node_change_num
            else:
                pass
        self.__add_child(child_node)
        node_change_num += 1
        return child_node, node_change_num

    def _add_child_with_key(self, child, ignore_limit=False, spec_child_node_key=None):
        assert spec_child_node_key not in self.children or len(self.children) == 1
        node_change_num = 0

        if Node.config['ignore_exceed'] or ignore_limit:
            pass
        else:

            if len(self.children) >= self.max_child_num:
                if Node.config['update_sub_nodes']:
                    removed_node_val = list(self.children.keys())[0]
                    _, remove_count = self.remove_child(removed_node_val)
                    logger.trace(f'removed {remove_count} nodes from the node with root of: {removed_node_val}.')
                    node_change_num -= remove_count
                else:
                    logger.trace(
                        'The number of the children of the parent node has reached the maximum. add 0 child.')
                    node_change_num = 0
                    return self, node_change_num
            else:
                pass

        child_node = self.__add_child_with_child_key(child, spec_child_node_key)

        node_change_num += 1
        return child_node, node_change_num

    def update_for_exist(self, child_value):
        assert child_value in self.children
        exist_node = self.children.pop(child_value)
        self.children[child_value] = exist_node

    def add_child_by_val(self, child_value: int, ignore_limit=False):

        if child_value in self.children:
            if len(self.children) == 1:
                return self.children[child_value], 0
            elif Node.config['update_sub_nodes']:
                logger.trace(f'The child node {child_value} already exists in the parent node: {self.value}.')
                self.update_for_exist(child_value)
                return self.children[child_value], 0
            else:
                return self.children[child_value], 0
        else:
            return self._add_child(child_value, ignore_limit)

    def add_child_by_node(self, child: "Node", ignore_limit=False, spec_child_node_key=None):
        assert isinstance(child, Node)
        if spec_child_node_key is None:
            if child.value in self.children:
                if len(self.children) == 1:
                    return self.children[child.value], 0
                elif Node.config['update_sub_nodes']:
                    logger.trace(f'The child node {child.value} already exists in the parent node: {self.value}.')
                    self.update_for_exist(child.value)
                    # remove to the first
                    return self.children[child.value], 0
                else:
                    return self.children[child.value], 0
            else:
                return self._add_child(child.value, ignore_limit)
        else:
            if spec_child_node_key in self.children:
                while spec_child_node_key in self.children:
                    spec_child_node_key = int(spec_child_node_key) + random.random()
            child.value = spec_child_node_key
            return self._add_child_with_key(child, spec_child_node_key=spec_child_node_key)

    def add_child_recursive(self, child: "Node", ignore_limit=False):
        assert isinstance(child, Node)
        total_add_count = 0

        # Add the child node
        added_child, add_count = self.add_child_by_node(child, ignore_limit)
        total_add_count += add_count

        for grandchild_value, grandchild_node in child.children.items():
            _, _add_count = added_child.add_child_recursive(grandchild_node, ignore_limit)
            total_add_count += _add_count

        return added_child, total_add_count

    def add_children(self, children: list, ignore_limit=False):
        change_num = 0
        child_node_list = []
        for child in children:
            if isinstance(child, int):
                _child = Node(child, parent=self, depth=self.depth + 1)
                child_node_list.append(_child)
                _, _change_num = self.add_child_by_node(_child, ignore_limit)
            elif isinstance(child, Node):
                child_node_list.append(_child)
                _, _change_num = self.add_child_recursive(child, ignore_limit)
            else:
                raise RuntimeError(f'Unsupported child type: {type(child)}')
            change_num += _change_num
        return child_node_list, change_num

    def add_child_with_key(self, child, child_key, ignore_limit=False):
        assert isinstance(child, Node)
        total_add_count = 0

        # Add the child node
        added_child, add_count = self.add_child_by_node(child, ignore_limit=ignore_limit, spec_child_node_key=child_key)
        total_add_count += add_count

        # Recursively add all children of the child node
        for grandchild_value, grandchild_node in child.children.items():
            _, _add_count = added_child.add_child_recursive(grandchild_node, ignore_limit)
            total_add_count += _add_count

        return added_child, total_add_count

    def add_child(self, child, child_key=None, ignore_limit=False):
        if child_key is None:
            if isinstance(child, int):
                return self.add_child_by_val(child, ignore_limit)
            elif isinstance(child, Node):
                return self.add_child_recursive(child, ignore_limit)
            elif isinstance(child, list):
                return self.add_children(child, ignore_limit)
        else:
            assert isinstance(child, Node)
            return self.add_child_with_key(child, child_key, ignore_limit)

    def remove_child(self, child: Union[int, float, "Node"]):
        if isinstance(child, int):
            child_value = child
        elif isinstance(child, Node):
            child_value = child.org_value
        else:
            raise RuntimeError(f'Unsupported child type: {type(child)}')

        remove_count = 0
        if child_value in self.children:

            child_node = self.children[child_value]
            children = list(child_node.children.keys())
            for grandchild in children:
                n, rc = child_node.remove_child(grandchild)
                remove_count += rc
            del self.children[child_value]
            remove_count += 1
            return self, remove_count
        else:
            raise RuntimeError(f'The child node: {child_value} does not exist in the parent node: {self.value}.')

    def remove_node_and_get_removed_depth(self, child: Union[int, "Node"]):
        if type(child) != int:
            child_value = child.value
        else:
            child_value = child
        remove_count = 0
        max_remove_depth = self.depth

        if child_value in self.children:

            child_node = self.children[child_value]
            children = list(child_node.children.keys())
            for grandchild in children:
                n, rc, _remove_depth = child_node.remove_node_and_get_removed_depth(grandchild)
                max_remove_depth = max(_remove_depth, max_remove_depth)
                remove_count += rc
            del self.children[child_value]
            remove_count += 1
            return self, remove_count, max_remove_depth
        else:
            logger.warning(f'The child node: {child_value} does not exist in the parent node: {self.value}.')
            return self, remove_count, 0

    # deepcopy method
    def __deepcopy__(self, memodict={}):
        depth = memodict['depth'] + 1 if 'depth' in memodict else 0
        parent = memodict['parent'] if 'parent' in memodict else None
        new_node = Node(self.value, parent=parent, depth=depth)
        memodict['depth'] = depth
        memodict['parent'] = new_node
        new_node.children = {k: v.__deepcopy__(memodict) for k, v in self.children.items()}
        return new_node

    def print_node(self):
        s = self.depth * '\t' + f"{str(self.value)} (depth:{self.depth}, max_child_num: {self.max_child_num}, child_depth: {self.child_depth}"
        if hasattr(self, 'draft_token'):
            s += f", next_token: {self.draft_token}"
        if hasattr(self, 'next_token'):
            s += f", next_token: {self.next_token}"
        print(s)

    def print_info(self):
        self.print_node()
        for child in self.children.values():
            child.print_info()
