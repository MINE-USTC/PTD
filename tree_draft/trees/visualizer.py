"""
Tree 可视化模块

使用 networkx 和 matplotlib 来可视化 Tree 结构，用于调试和观察 Tree 的变化。
"""

import os
from typing import Optional, Dict
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from loguru import logger

from .node import Node
from .tree import Tree


class TreeVisualizer:
    """Tree 可视化器"""

    def __init__(self, output_dir: str = "tree_visualizations",
                 figsize: tuple = (12, 8),
                 node_size: int = 1000,
                 font_size: int = 10):
        """
        初始化可视化器

        Args:
            output_dir: 输出目录
            figsize: 图片大小
            node_size: 节点大小
            font_size: 字体大小
        """
        self.output_dir = output_dir
        self.figsize = figsize
        self.node_size = node_size
        self.font_size = font_size

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

    def _build_networkx_graph(self, tree: Tree, tokenizer=None) -> nx.DiGraph:
        """
        将 Tree 转换为 networkx 图

        Args:
            tree: Tree 对象
            tokenizer: tokenizer 实例，用于将 token ID 解码为自然语言文本

        Returns:
            tuple: (networkx.DiGraph, node_to_id 映射)
        """
        G = nx.DiGraph()

        # 添加所有节点
        node_to_id = {}
        id_counter = 0

        # 收集所有节点（使用 DFS 序列）
        all_nodes = tree.dfs_node_seq if tree.dfs_node_seq else []

        # 如果没有 DFS 序列，从根节点开始遍历
        if not all_nodes:
            if not tree.exclude_root:
                all_nodes = [tree.root]
            else:
                all_nodes = list(tree.root.children.values())

        # 为每个节点分配唯一 ID
        for node in all_nodes:
            if node not in node_to_id:
                node_to_id[node] = id_counter
                id_counter += 1

        # 添加根节点（如果需要）
        if not tree.exclude_root:
            if tree.root not in node_to_id:
                node_to_id[tree.root] = id_counter
                id_counter += 1

        # 添加节点到 networkx 图
        for node, node_id in node_to_id.items():
            node_label = self._format_node_label(node, tokenizer=tokenizer)
            node_attrs = {
                'label': node_label,
                'value': str(node.value),
                'depth': node.depth,
                'node_id': node_id,
                'num_children': len(node.children),
            }

            # 添加 draft_token 信息（如果有）
            if hasattr(node, 'draft_token') and node.draft_token is not None:
                node_attrs['draft_token'] = str(node.draft_token)

            # 添加 next_token 信息（如果有）
            if hasattr(node, 'next_token') and node.next_token is not None:
                node_attrs['next_token'] = str(node.next_token)

            G.add_node(node_id, **node_attrs)

        # 添加边（Tree 结构只有一个父节点）
        for node, node_id in node_to_id.items():
            # 添加从父节点到当前节点的边
            if hasattr(node, 'parent') and node.parent is not None:
                if node.parent in node_to_id:
                    parent_id = node_to_id[node.parent]
                    G.add_edge(parent_id, node_id)

            # 添加从当前节点到子节点的边
            for child in node.children.values():
                if child in node_to_id:
                    child_id = node_to_id[child]
                    G.add_edge(node_id, child_id)

        return G, node_to_id

    def _format_node_label(self, node: Node, tokenizer=None) -> str:
        """
        格式化节点标签

        Args:
            node: 树节点
            tokenizer: tokenizer 实例，用于将 token ID 解码为自然语言文本

        Returns:
            str: 格式化后的标签
        """
        label = ""
        # 如果提供了 tokenizer，尝试解码 token 为自然语言文本
        if tokenizer is not None:
            try:
                # 解码单个 token
                token_text = tokenizer.decode([node.value], skip_special_tokens=False)
                # 清理文本（移除特殊字符，限制长度）
                token_text = token_text.strip()
                # 如果文本太长，截断并添加省略号
                if len(token_text) > 20:
                    token_text = token_text[:17] + "..."
                # 如果文本包含换行符，替换为空格
                token_text = token_text.replace('\n', ' ').replace('\r', ' ')
                label += f"{token_text}"
            except Exception as e:
                # 如果解码失败，静默忽略（不显示文本）
                logger.debug(f"Failed to decode token {node.value}: {e}")

        return label

    def visualize(self, tree: Tree,
                  title: Optional[str] = None,
                  filename: Optional[str] = None,
                  layout: str = 'hierarchical',
                  show_labels: bool = True,
                  color_by_depth: bool = True,
                  show: bool = False,
                  tokenizer=None) -> Optional[str]:
        """
        可视化 Tree

        Args:
            tree: Tree 对象
            title: 图标题
            filename: 保存的文件名（不含路径），如果为 None 且 show=False，则自动生成
            layout: 布局方式 ('hierarchical', 'spring', 'circular')
            show_labels: 是否显示标签
            color_by_depth: 是否按深度着色
            show: 是否直接显示图形（如果为 True，则显示而不保存）
            tokenizer: tokenizer 实例，用于将 token ID 解码为自然语言文本

        Returns:
            Optional[str]: 如果保存文件，返回文件路径；如果直接显示，返回 None
        """
        # 构建 networkx 图
        G, node_to_id = self._build_networkx_graph(tree, tokenizer=tokenizer)

        if len(G.nodes()) == 0:
            logger.warning("Tree is empty, skipping visualization")
            return None

        # 创建图形
        fig, ax = plt.subplots(figsize=self.figsize)

        # 选择布局
        if layout == 'hierarchical':
            try:
                pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
            except Exception:
                # 如果没有 graphviz，使用分层布局
                pos = self._hierarchical_layout(G, tree, node_to_id)
        elif layout == 'spring':
            pos = nx.spring_layout(G, k=2, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G)

        # 准备节点颜色
        if color_by_depth:
            node_colors = []
            max_depth = max(1, tree.tree_depth + 1)
            for node_id in G.nodes():
                depth = G.nodes[node_id]['depth']
                # 使用颜色映射：深度越深，颜色越深
                node_colors.append(plt.cm.get_cmap('viridis')(depth / max_depth))
        else:
            node_colors = ['lightblue'] * len(G.nodes())

        # 绘制边
        nx.draw_networkx_edges(G, pos, ax=ax,
                               arrows=True,
                               arrowsize=20,
                               edge_color='gray',
                               alpha=0.6,
                               arrowstyle='->')

        # 绘制节点
        nx.draw_networkx_nodes(G, pos, ax=ax,
                               node_color=node_colors,
                               node_size=self.node_size,
                               alpha=0.9)

        # 绘制标签
        if show_labels:
            labels = {node_id: G.nodes[node_id]['label'] for node_id in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, ax=ax,
                                    font_size=self.font_size,
                                    font_weight='bold')

        # 设置标题
        if title is None:
            title = f"Tree Visualization\nNodes: {tree.node_count}"
        ax.set_title(title, fontsize=14, fontweight='bold')

        # 添加图例
        if color_by_depth:
            depth_legend = mpatches.Patch(color='gray', label='Color indicates depth')
            ax.legend(handles=[depth_legend], loc='upper right')

        ax.axis('off')
        plt.tight_layout()

        # 根据 show 参数决定是显示还是保存
        if show:
            # 直接显示图形
            plt.show()
            return None
        else:
            # 保存文件
            if filename is None:
                filename = f"tree_{tree.node_count}_nodes.png"

            filepath = os.path.join(self.output_dir, filename)
            # 确保文件所在目录存在（如果 filename 包含子目录）
            file_dir = os.path.dirname(filepath)
            if file_dir:
                os.makedirs(file_dir, exist_ok=True)

            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()

            logger.info(f"Tree visualization saved to {filepath}")
            return filepath

    def _hierarchical_layout(self, G: nx.DiGraph, tree: Tree, node_to_id: Dict) -> Dict:
        """
        创建分层布局（不使用 graphviz）

        Args:
            G: networkx 图
            tree: Tree 对象
            node_to_id: 节点到 ID 的映射

        Returns:
            Dict: 位置字典
        """
        pos = {}

        # 按深度分组节点
        depth_groups = {}
        for node_id in G.nodes():
            depth = G.nodes[node_id]['depth']
            if depth not in depth_groups:
                depth_groups[depth] = []
            depth_groups[depth].append(node_id)

        # 为每一层分配位置
        max_depth = max(depth_groups.keys()) if depth_groups else 0
        y_spacing = 2.0
        x_spacing = 1.5

        for depth, nodes in depth_groups.items():
            y = (max_depth - depth) * y_spacing
            num_nodes = len(nodes)

            # 水平分布节点
            if num_nodes == 1:
                x_positions = [0]
            else:
                x_positions = [i * x_spacing - (num_nodes - 1) * x_spacing / 2
                               for i in range(num_nodes)]

            for i, node_id in enumerate(nodes):
                pos[node_id] = (x_positions[i], y)

        return pos

    def visualize_sequence(self, trees: list,
                           step_names: Optional[list] = None,
                           base_filename: str = "tree_step",
                           tokenizer=None,
                           **kwargs) -> list:
        """
        可视化一系列 Tree（用于观察 Tree 的变化）

        Args:
            trees: Tree 对象列表
            step_names: 步骤名称列表
            base_filename: 基础文件名
            tokenizer: tokenizer 实例，用于将 token ID 解码为自然语言文本
            **kwargs: 传递给 visualize 的其他参数

        Returns:
            list: 保存的文件路径列表
        """
        filepaths = []

        for i, tree in enumerate(trees):
            step_name = step_names[i] if step_names and i < len(step_names) else f"step_{i}"
            filename = f"{base_filename}_{i:04d}_{step_name}.png"
            title = f"Tree at {step_name}\nNodes: {tree.node_count}"

            filepath = self.visualize(tree, title=title, filename=filename, tokenizer=tokenizer, **kwargs)
            if filepath:
                filepaths.append(filepath)

        return filepaths
