"""
Common functions shared across PTD (Parallel Tree Decoding) implementations.
This module contains shared utilities for Llama, Qwen, and Qwen3 PTD models.
"""
import torch
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask as _expand_mask

from ..inference_profile import InferProfile, ProfileKeys
from ..trees.draft_tree import DraftTree, DraftTrees


def set_lower_triangular_true_efficient(matrix, start_row, start_col, end_row, end_col, set_value=0):
    """
    高效地将矩阵的下三角区域设置为指定值

    Args:
        matrix: 要修改的矩阵
        start_row: 起始行索引
        start_col: 起始列索引
        end_row: 结束行索引（不包含）
        end_col: 结束列索引（不包含）
        set_value: 要设置的值，默认为 0

    Returns:
        torch.Tensor: 修改后的矩阵
    """
    matrix_shape = matrix.shape
    assert 0 <= start_row < matrix_shape[0]
    assert 0 <= end_row <= matrix_shape[0]
    assert 0 <= start_col < matrix_shape[1]
    assert 0 <= end_col <= matrix_shape[1]
    rows, cols = torch.arange(matrix_shape[0]).unsqueeze(1), torch.arange(matrix_shape[1])

    mask = ((cols <= rows) & (rows >= start_row) & (rows < end_row) & (cols >= start_col) & (cols < end_col)).to(
        matrix.device)

    matrix.masked_fill_(mask, set_value)

    return matrix


def set_lower_triangular_true_efficient_batch(matrix, start_row, start_col, end_row, end_col, set_value=0):
    """
    高效地将批次矩阵的下三角区域设置为指定值

    Args:
        matrix: 要修改的批次矩阵，形状为 [batch_size, rows, cols]
        start_row: 起始行索引
        start_col: 起始列索引
        end_row: 结束行索引（不包含）
        end_col: 结束列索引（不包含）
        set_value: 要设置的值，默认为 0

    Returns:
        torch.Tensor: 修改后的批次矩阵
    """
    batch_size, rows, cols = matrix.shape
    assert 0 <= start_row < rows
    assert 0 <= end_row <= rows
    assert 0 <= start_col < cols
    assert 0 <= end_col <= cols

    row_indices = torch.arange(rows).unsqueeze(1).to(matrix.device)  # 行索引
    col_indices = torch.arange(cols).to(matrix.device)  # 列索引

    mask = ((col_indices <= row_indices) &
            (row_indices >= start_row) & (row_indices < end_row) &
            (col_indices >= start_col) & (col_indices < end_col)).unsqueeze(0).expand(batch_size, -1, -1)

    matrix.masked_fill_(mask, set_value)

    return matrix


def prepare_guess_tree_mask(guess_tree, dtype, tgt_len, mask):
    """
    为 guess tree 准备注意力掩码

    Args:
        guess_tree: GuessTree 对象
        dtype: 掩码的数据类型
        tgt_len: 目标长度
        mask: 基础掩码矩阵

    Returns:
        torch.Tensor: 更新后的掩码矩阵
    """
    guess_count = guess_tree.tree.node_count
    pre_idx = tgt_len - guess_count
    if guess_count > 0:
        mask[:, 0] = 0
        mask[pre_idx:, pre_idx:] = guess_tree.tree.attn_mask
    return mask


def prepare_guess_tree_mask_pad(guess_tree, dtype, mask, start_idx):
    """
    为 guess tree 准备注意力掩码（支持 padding）

    Args:
        guess_tree: GuessTree 对象，可能为 None
        dtype: 掩码的数据类型
        mask: 基础掩码矩阵
        start_idx: guess tree 在掩码中的起始索引

    Returns:
        torch.Tensor: 更新后的掩码矩阵
    """
    guess_count = guess_tree.tree.node_count if guess_tree is not None else 0
    if guess_count > 0:
        mask[:, 0] = 0
        mask[start_idx:start_idx + guess_count, start_idx:start_idx + guess_count] = guess_tree.tree.attn_mask
    return mask


def prepare_draft_and_guess_tree_causal_mask(draft_tree, guess_tree, is_prefill,
                                             device, input_ids_shape, dtype,
                                             past_key_value_length, prompt_size):
    """
    为 draft tree 和 guess tree 准备因果注意力掩码（单批次模式）

    该函数构建一个因果注意力掩码，支持 draft tree 和 guess tree 的特殊结构。
    在 prefill 阶段和 decode 阶段的掩码构建逻辑不同。

    Args:
        draft_tree: DraftTree 对象
        guess_tree: GuessTree 对象，可能为 None
        is_prefill: 是否为 prefill 阶段
        device: 设备
        input_ids_shape: 输入形状 (batch_size, seq_len)
        dtype: 掩码的数据类型
        past_key_value_length: 过去 key-value 的长度
        prompt_size: prompt 的大小（prefill 阶段使用）

    Returns:
        torch.Tensor: 形状为 [batch_size, 1, tgt_len, tgt_len + past_key_value_length] 的掩码
    """
    guess_count = guess_tree.tree.node_count if guess_tree is not None else 0
    draft_count = draft_tree.tree.node_count if draft_tree is not None else 0
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device, dtype=dtype)

    if is_prefill:
        if prompt_size == -1:
            prompt_size = tgt_len
        assert guess_count == 0
        draft_tree.tree.update_dfs()
        draft_tree.refresh_draft_tree_tokens()

        mask = set_lower_triangular_true_efficient(mask, 0, 0, prompt_size - draft_count,
                                                   prompt_size - draft_count)
        if draft_count > 0:
            mask[-draft_count:, 0:prompt_size - draft_count] = 0
            mask[-draft_count:, -draft_count:] = draft_tree.tree.attn_mask
        assert mask.shape[0] == prompt_size
    else:
        pre_idx = 1

        if draft_count + pre_idx != tgt_len - guess_count:
            raise RuntimeError('tree node count is not consistent with the target length')

        mask[:, 0] = 0

        mask[pre_idx:draft_count + pre_idx, pre_idx:draft_count + pre_idx] = draft_tree.tree.attn_mask
        assert mask.shape[0] == pre_idx + draft_count + guess_count
    if guess_count > 0:
        mask = prepare_guess_tree_mask(guess_tree, dtype, tgt_len, mask)

    if past_key_value_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_value_length, dtype=dtype, device=device), mask], dim=-1)

    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_value_length)


def combine_attn_mask_for_trees(trees, mask, start_idx):
    """
    将多个树的注意力掩码合并到单个掩码中

    该函数用于批次模式，将多个 DraftTree 或 DraftGraph 的注意力掩码合并到批次掩码中。
    每个树/图的掩码会被放置在掩码矩阵的指定位置。

    Args:
        trees: 包含多个 DraftTree 或 DraftGraph 的列表，每个树/图包含注意力掩码
        mask: 基础掩码矩阵，形状为 [batch_size, seq_len, seq_len]
        start_idx: 树/图掩码在掩码矩阵中的起始索引

    Returns:
        torch.Tensor: 合并后的注意力掩码
    """
    for i, tree in enumerate(trees):
        if tree is not None:
            # 支持 DraftTree 和 DraftGraph（两者都有 tree 属性）
            # DraftTree.tree 指向 Tree，DraftGraph.tree 指向 Graph（DAG）
            graph_or_tree = tree.tree

            mask[i][start_idx:start_idx + graph_or_tree.node_count, 0:start_idx] = 0
            mask[i][
                start_idx:start_idx + graph_or_tree.node_count,
                start_idx:start_idx + graph_or_tree.node_count,
            ] = graph_or_tree.attn_mask
    return mask


def prepare_draft_and_guess_tree_causal_mask_batch(draft_tree, guess_tree, is_prefill, device, input_ids_shape,
                                                   dtype, past_key_value_length=0, prompt_size=-1):
    """
    为 draft tree 和 guess tree 准备因果注意力掩码（批次模式）

    该函数构建一个因果注意力掩码，支持多个序列的并行处理。
    与单批次模式不同，该函数处理 DraftTrees 和 GuessTree 列表。

    Args:
        draft_tree: DraftTrees 对象（包含多个 DraftTree）
        guess_tree: GuessTree 列表，每个元素可能为 None
        is_prefill: 是否为 prefill 阶段
        device: 设备
        input_ids_shape: 输入形状 (batch_size, seq_len)
        dtype: 掩码的数据类型
        past_key_value_length: 过去 key-value 的长度，默认为 0
        prompt_size: prompt 的大小，默认为 -1（自动推断）

    Returns:
        torch.Tensor: 形状为 [batch_size, 1, tgt_len, tgt_len + past_key_value_length] 的掩码
    """
    draft_counts = draft_tree.get_draft_counts()
    guess_counts = [gt.tree.node_count if gt is not None else 0 for gt in guess_tree]
    draft_count = max(draft_counts)
    guess_count = max(guess_counts)
    pre_idx = 1

    bsz, tgt_len = input_ids_shape
    mask = torch.full((bsz, tgt_len, tgt_len), torch.finfo(dtype).min, device=device, dtype=dtype)

    if is_prefill:
        if prompt_size == -1:
            prompt_size = tgt_len
        draft_tree.update_dfs()
        draft_tree.refresh_draft_tree_tokens()

        mask = set_lower_triangular_true_efficient_batch(mask, 0, 0, prompt_size - draft_count,
                                                         prompt_size - draft_count)

        if draft_count > 0:
            start_idx = tgt_len - draft_count
            mask = combine_attn_mask_for_trees(draft_tree.draft_trees, mask, start_idx)
    else:
        mask[:, :, 0] = 0
        for i in range(bsz):
            if draft_counts[i] > 0:
                mask[i, pre_idx:pre_idx + draft_counts[i], pre_idx:pre_idx + draft_counts[i]] = draft_tree.draft_trees[
                    i].tree.attn_mask

    if guess_count > 0:
        for i in range(bsz):
            mask[i] = prepare_guess_tree_mask_pad(guess_tree[i], dtype, mask[i], start_idx=pre_idx + draft_count)

    if past_key_value_length > 0:
        mask = torch.cat([torch.zeros(bsz, tgt_len, past_key_value_length, dtype=dtype, device=device), mask], dim=-1)

    return mask[:, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_value_length)


def build_causal_attention_mask(attn_mask_1d):
    """
    从 1D 注意力掩码构建因果注意力掩码

    Args:
        attn_mask_1d: 1D 注意力掩码，形状为 [batch_size, seq_len]

    Returns:
        torch.Tensor: 因果注意力掩码，形状为 [batch_size, seq_len, seq_len]
    """
    B, L = attn_mask_1d.shape
    base = attn_mask_1d[:, :, None] * attn_mask_1d[:, None, :]
    causal = torch.tril(torch.ones(L, L, device=attn_mask_1d.device)).unsqueeze(0)
    return base * causal  # shape: (B, L, L)


def prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length, others,
                                   auto_select_batch_mode=True):
    """
    准备 decoder 注意力掩码，支持自动选择批次模式

    该函数根据 draft_branches 的类型自动选择使用批次模式或非批次模式。
    如果 auto_select_batch_mode 为 True 且 draft_branches 是 DraftTrees 类型，
    则使用批次模式；否则使用单批次模式。

    Args:
        self: 模型实例
        attention_mask: 基础注意力掩码
        input_shape: 输入形状
        inputs_embeds: 输入的嵌入向量
        past_key_values_length: 过去 key-value 的长度
        others: 元组 (is_prefill, guess_branches, draft_branches)
        auto_select_batch_mode: 是否自动选择批次模式，默认为 True

    Returns:
        tuple: (combined_attention_mask, mask_profile)
            - combined_attention_mask: 组合后的注意力掩码
            - mask_profile: 性能分析结果
    """
    mask_profile = InferProfile()
    combined_attention_mask = None
    is_prefill, guess_branches, draft_branches = others

    if input_shape[-1] > 1:
        if auto_select_batch_mode and (
                isinstance(draft_branches, DraftTrees) or isinstance(draft_branches, DraftGraphs)):
            combined_attention_mask = prepare_draft_and_guess_tree_causal_mask_batch(
                draft_branches, guess_branches, is_prefill,
                inputs_embeds.device, input_shape, inputs_embeds.dtype,
                past_key_values_length, prompt_size=-1)
        else:
            combined_attention_mask = prepare_draft_and_guess_tree_causal_mask(
                draft_branches, guess_branches, is_prefill,
                inputs_embeds.device, input_shape, inputs_embeds.dtype,
                past_key_values_length, prompt_size=-1)

    if attention_mask is not None:
        expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
            inputs_embeds.device)
        if auto_select_batch_mode and (
                isinstance(draft_branches, DraftTrees) or isinstance(draft_branches, DraftGraphs)):
            combined_attention_mask = torch.minimum(combined_attention_mask,
                                                    expanded_attn_mask) if combined_attention_mask is not None else expanded_attn_mask
        else:
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
    return combined_attention_mask, mask_profile


def prepare_decoder_attention_mask_batch(self, attention_mask, input_shape, input_embeds, past_key_values_length,
                                         others):
    """
    准备 decoder 注意力掩码（批次模式）

    该函数专门用于批次模式，处理多个序列的并行注意力掩码构建。

    Args:
        self: 模型实例
        attention_mask: 基础注意力掩码
        input_shape: 输入形状
        input_embeds: 输入的嵌入向量
        past_key_values_length: 过去 key-value 的长度
        others: 元组 (is_prefill, guess_branches, draft_branches)

    Returns:
        tuple: (combined_attention_mask, mask_profile)
            - combined_attention_mask: 组合后的注意力掩码
            - mask_profile: 性能分析结果
    """
    mask_profile = InferProfile()
    combined_attention_mask = None
    is_prefill, guess, draft = others
    if input_shape[-1] > 1:
        combined_attention_mask = prepare_draft_and_guess_tree_causal_mask_batch(draft, guess, is_prefill,
                                                                                 input_embeds.device,
                                                                                 input_shape, input_embeds.dtype,
                                                                                 past_key_values_length,
                                                                                 prompt_size=-1)
    if attention_mask is not None:
        expanded_attn_mask = _expand_mask(attention_mask, input_embeds.dtype, tgt_len=input_shape[-1]).to(
            input_embeds.device)
        combined_attention_mask = torch.minimum(combined_attention_mask,
                                                expanded_attn_mask) if combined_attention_mask is not None else expanded_attn_mask

    return combined_attention_mask, mask_profile
