import time
from time import perf_counter
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from loguru import logger
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import BaseModelOutputWithPast, CausalLMOutputWithPast

from ...inference_profile import InferProfile, ProfileKeys
from ...trees.draft_tree import DraftTree, DraftTrees
from ...trees.guess_tree import GuessTree
from ..ptd_common import *


def prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length, others):
    """
    Qwen3 模型的 decoder 注意力掩码准备函数（包装器）

    该函数是 ptd_common.prepare_decoder_attention_mask 的包装器，
    自动启用批次模式选择功能。

    Args:
        self: Qwen3Model 实例
        attention_mask: 基础注意力掩码
        input_shape: 输入形状
        inputs_embeds: 输入的嵌入向量
        past_key_values_length: 过去 key-value 的长度
        others: 元组 (is_prefill, guess_branches, draft_branches)

    Returns:
        tuple: (combined_attention_mask, mask_profile)
    """
    return prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length, others,
        auto_select_batch_mode=True
    )


def QwenModelPTDForward(self,
                        input_ids: torch.LongTensor = None,
                        attention_mask: Optional[torch.Tensor] = None,
                        position_ids: Optional[torch.LongTensor] = None,
                        past_key_values: Optional[List[torch.FloatTensor]] = None,
                        inputs_embeds: Optional[torch.FloatTensor] = None,
                        use_cache: Optional[bool] = None,
                        output_attentions: Optional[bool] = None,
                        output_hidden_states: Optional[bool] = None,
                        return_dict: Optional[bool] = None,
                        is_prefill: bool = False,
                        cache_position=None,
                        draft_branches=None,
                        guess_branches=None,
                        **kwargs,
                        ):
    """
    Qwen3 模型的 PTD (Parallel Tree Decoding) forward 方法

    这是 Tree Draft 方法中 Qwen3Model 的核心 forward 函数，支持并行处理
    draft tree 和 guess tree 中的多个 tokens。

    Args:
        self: Qwen3Model 实例
        input_ids: 输入 token IDs
        attention_mask: 注意力掩码
        position_ids: 位置 IDs
        past_key_values: 缓存的 key-value 对
        inputs_embeds: 输入的嵌入向量
        use_cache: 是否使用缓存
        output_attentions: 是否输出注意力权重
        output_hidden_states: 是否输出隐藏状态
        return_dict: 是否以字典形式返回
        is_prefill: 是否为 prefill 阶段
        cache_position: 缓存位置
        draft_branches: draft tree 分支（DraftTree 或 DraftTrees）
        guess_branches: guess tree 分支（GuessTree 列表）
        **kwargs: 其他参数

    Returns:
        tuple: (BaseModelOutputWithPast, InferProfile)
            - BaseModelOutputWithPast: 模型输出
            - InferProfile: 性能分析结果
    """
    profile_LlamaModelPTD_forward = InferProfile()

    # if branch_cdt_tokens is None:
    #     branch_cdt_tokens = []

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")
    seq_length_with_past = seq_length
    past_key_values_length = 0

    if use_cache:
        use_legacy_cache = not isinstance(past_key_values, Cache)
        if use_legacy_cache:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        past_key_values_length = past_key_values.get_seq_length()

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
    # if cache_position is None:
    #     past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
    #     cache_position = torch.arange(
    #         past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
    #     )
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
        )

    with profile_LlamaModelPTD_forward.timer(ProfileKeys.ATTENTION_MASK_TIME):
        attention_mask, mask_profile = self.prepare_decoder_attention_mask(attention_mask, input_ids.shape,
                                                                           inputs_embeds,
                                                                           past_key_values_length,
                                                                           (is_prefill, guess_branches, draft_branches))
    profile_LlamaModelPTD_forward.incremental_updates(mask_profile)

    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(  # type: ignore
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # next_decoder_cache = None
    with profile_LlamaModelPTD_forward.timer(ProfileKeys.LAYER_FORWARD_TIME):
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        # for decoder_layer in self.layers:
        #     if output_hidden_states:
        #         all_hidden_states += (hidden_states,)
        #
        #     if self.gradient_checkpointing and self.training:
        #         layer_outputs = self._gradient_checkpointing_func(
        #             decoder_layer.__call__,
        #             hidden_states,
        #             attention_mask,
        #             position_ids,
        #             past_key_values,
        #             output_attentions,
        #             use_cache,
        #         )
        #     else:
        #         layer_outputs = decoder_layer(
        #             hidden_states,
        #             attention_mask=attention_mask,
        #             position_ids=position_ids,
        #             past_key_value=past_key_values,
        #             output_attentions=output_attentions,
        #             use_cache=use_cache,
        #             position_embeddings=position_embeddings,
        #
        #         )

        # hidden_states = layer_outputs[0]

        # if use_cache:
        #     next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        # if output_attentions:
        #     all_self_attns += (layer_outputs[1],)
        # torch.cuda.synchronize()
        hidden_states = self.norm(hidden_states)
    return (BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        # attentions=all_self_attns,
    ), profile_LlamaModelPTD_forward)
    # hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    # if output_hidden_states:
    #     all_hidden_states += (hidden_states,)

    # next_cache = None
    # if use_cache:
    #     next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
    # if not return_dict:
    #     return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    # return (BaseModelOutputWithPast(
    #     last_hidden_state=hidden_states,
    #     past_key_values=past_key_values,
    #     hidden_states=all_hidden_states,
    #     attentions=all_self_attns,
    #
    # ), profile_LlamaModelPTD_forward)


def PTDfroward(self,
               input_ids: torch.LongTensor = None,
               draft: Union[DraftTree, DraftTrees] = None,
               guess: Tuple = None,
               lst_idx=None,
               attention_mask: Optional[torch.Tensor] = None,
               position_ids: Optional[torch.LongTensor] = None,
               past_key_values: Optional[List[torch.FloatTensor]] = None,
               inputs_embeds: Optional[torch.FloatTensor] = None,
               labels: Optional[torch.LongTensor] = None,
               use_cache: Optional[bool] = None,
               output_attentions: Optional[bool] = None,
               output_hidden_states: Optional[bool] = None,
               return_dict: Optional[bool] = None,
               cache_position: Optional[int] = None,
               **kwargs
               ):
    """
    Qwen3 模型的 PTD (Parallel Tree Decoding) forward 方法

    这是 Tree Draft 方法中 Qwen3ForCausalLM 的核心 forward 函数。
    该方法会：
    1. 准备 draft tree 和 guess tree 的 tokens、position IDs 和 attention masks
    2. 将这些 tokens 与输入 tokens 拼接
    3. 调用模型进行前向传播
    4. 提取并组织输出 logits（next_logits, draft_logits, verify_logits）

    Args:
        self: Qwen3ForCausalLM 实例
        input_ids: 输入 token IDs
        draft: draft tree（DraftTree 或 DraftTrees）
        guess: guess tree 列表
        lst_idx: 最后一个 token 的索引
        attention_mask: 注意力掩码
        position_ids: 位置 IDs
        past_key_values: 缓存的 key-value 对
        inputs_embeds: 输入的嵌入向量
        labels: 标签（推理模式下应为 None）
        use_cache: 是否使用缓存
        output_attentions: 是否输出注意力权重
        output_hidden_states: 是否输出隐藏状态
        return_dict: 是否以字典形式返回
        cache_position: 缓存位置
        **kwargs: 其他参数

    Returns:
        tuple: (CausalLMOutputWithPast, InferProfile)
            - CausalLMOutputWithPast: 模型输出，包含以下额外属性：
                - kvcache_len: KV cache 的长度
                - step_len: 当前步的序列长度
                - next_logits: 下一个 token 的 logits
                - draft_logits: draft tokens 的 logits
                - verify_logits: guess tokens 的 logits（用于验证）
                - all_ids: 所有输入 token IDs
            - InferProfile: 性能分析结果
    """
    guess_tree = guess
    forward_profile = InferProfile()
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    assert labels is None, " Inference Mode "

    if past_key_values[0][0] is not None:
        past_size = past_key_values[0][0].size(2)
    else:
        past_size = 0

    is_prefill = True if past_key_values[0][0] is None else False

    prefill_size = input_ids.size(1)
    if not is_prefill:
        assert prefill_size == 1
    for layer in self.model.layers:
        layer.self_attn.cur_len = prefill_size

    batch_indices = torch.arange(position_ids.size(0), dtype=torch.long, device=input_ids.device)

    if lst_idx is not None:
        lst_id = position_ids[batch_indices, lst_idx].tolist()
    else:
        lst_id = position_ids[:, -1].tolist()

    with forward_profile.timer(ProfileKeys.PREPARE_IDS_TIME):
        draft_pos_list, all_draft, draft_attn_size = draft.prepare_ids(lst_id)

    guess_pos_list = []
    all_guess = []
    guess_attn_size = []
    for i, gt in enumerate(guess_tree):
        if gt is not None:
            assert isinstance(gt, GuessTree)
            g_p, a_g, g_a_s = gt.prepare_ids(lst_id[i])
            guess_pos_list.append(g_p)
            all_guess.append(a_g)
            guess_attn_size.append(g_a_s)
        else:
            guess_pos_list.append([])
            all_guess.append([])
            guess_attn_size.append(0)

    pad_draft_tokens = pad_sequence(
        [torch.tensor(d, dtype=input_ids.dtype, device=input_ids.device) for d in all_draft], batch_first=True,
        padding_value=self.tokenizer.pad_token_id)
    pad_guess_tokens = pad_sequence(
        [torch.tensor(g, dtype=input_ids.dtype, device=input_ids.device) for g in all_guess], batch_first=True,
        padding_value=self.tokenizer.pad_token_id)
    pad_draft_pos = pad_sequence(
        [torch.tensor(d, dtype=input_ids.dtype, device=input_ids.device) for d in draft_pos_list], batch_first=True,
        padding_value=1)
    pad_guess_pos = pad_sequence(
        [torch.tensor(g, dtype=input_ids.dtype, device=input_ids.device) for g in guess_pos_list], batch_first=True,
        padding_value=1)
    pad_draft_attn = pad_sequence(
        [torch.ones(ds, dtype=input_ids.dtype, device=input_ids.device) for ds in draft_attn_size], batch_first=True,
        padding_value=0)
    pad_guess_attn = pad_sequence(
        [torch.ones(gs, dtype=input_ids.dtype, device=input_ids.device) for gs in guess_attn_size], batch_first=True,
        padding_value=0)

    input_ids = torch.cat([input_ids, pad_draft_tokens, pad_guess_tokens], dim=1)
    position_ids = torch.cat([position_ids, pad_draft_pos, pad_guess_pos], dim=1)
    attention_mask = torch.cat([attention_mask, pad_draft_attn, pad_guess_attn], dim=1)

    step_len = attention_mask.size(1)

    is_prefill = True if past_key_values[0][0] is None else False
    with forward_profile.timer(ProfileKeys.LLAMA_MODEL_FORWARD_TIME):
        outputs, llama_model_PTD_forward_profile = self.model.QwenModelPTDForward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            is_prefill=is_prefill,
            draft_branches=draft,
            guess_branches=guess_tree
        )
        # torch.cuda.synchronize()
    forward_profile.incremental_updates(llama_model_PTD_forward_profile)
    hidden_states = outputs[0]
    t0 = perf_counter()

    logits = self.lm_head(hidden_states)

    logits = logits.float()

    loss = None

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    ret = CausalLMOutputWithPast(
        loss=loss,
        logits=logits.to(input_ids.device),
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
    ret.kvcache_len = prefill_size + past_size  # this forward step decoded token length and all_decoded size
    ret.step_len = step_len  # length of this forward step, include the length of the kv cache

    cdt_lens = [t.tree.node_count if t is not None else 0 for t in guess_tree]
    dft_lens = [t.tree.node_count if t is not None else 0 for t in draft.draft_trees]
    cdt_len = max(cdt_lens)
    dft_len = max(dft_lens)

    ret.next_logits = ret.logits[:, prefill_size - 1, :].to(input_ids.device)

    if isinstance(draft, DraftTree):
        if cdt_len > 0:
            ret.verify_logits = ret.logits[:, -cdt_len:, :].to(input_ids.device)
        ret.draft_logits = ret.logits[:, prefill_size:draft.tree.node_count + prefill_size, :].to(
            input_ids.device)
    elif isinstance(draft, DraftTrees):
        if cdt_len > 0:
            ret.verify_logits = ret.logits[:, -cdt_len:, :].to(input_ids.device)
        ret.draft_logits = ret.logits[:, prefill_size:prefill_size + dft_len, :].to(
            input_ids.device)
    else:
        raise RuntimeError(f'Unsupported draft branches:{type(draft)}')
    # torch.cuda.synchronize()
    ret.all_ids = input_ids
    forward_profile.incremental_update(ProfileKeys.POSTPROCESS_FORWARD_TIME, perf_counter() - t0)
    return ret, forward_profile
