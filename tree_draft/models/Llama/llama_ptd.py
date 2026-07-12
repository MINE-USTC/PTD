import time
from time import perf_counter
from typing import List, Optional, Tuple

import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask as _expand_mask
from transformers.models.llama.modeling_llama import BaseModelOutputWithPast, CausalLMOutputWithPast

from ...inference_profile import InferProfile, ProfileKeys
from ...trees import *

from ..ptd_common import *

def LlamaModelPTDforward(
        self,
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
        draft_branches=None,
        guess_branches=None,
) -> Union[Tuple, BaseModelOutputWithPast]:
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
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None
    t0 = time.perf_counter()
    with profile_LlamaModelPTD_forward.timer(ProfileKeys.LAYER_FORWARD_TIME):
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                hidden_states = decoder_layer.forward(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    position_embeddings=position_embeddings,
                )

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    # next_cache = None
    # if use_cache:
    #     next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
    # if not return_dict:
    #     return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

    return (BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    ), profile_LlamaModelPTD_forward)


def PTDforward(
        self,
        input_ids: torch.LongTensor = None,
        draft: Union[DraftTree] = None,
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
        cache_position=None,
        **kwargs
) -> Union[Tuple, CausalLMOutputWithPast]:
    guess_tree = guess
    forward_profile = InferProfile()
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    assert labels is None, " Inference Mode "
    # assert input_ids.size(0) == 1, " single batch only "

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

    # lst_id = position_ids[0][-1].item()

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
    t0 = time.perf_counter()
    with forward_profile.timer(ProfileKeys.LLAMA_MODEL_FORWARD_TIME):

        outputs, llama_model_PTD_forward_profile = self.model.LlamaModelPTDforward(
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
    if self.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
        logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
        logits = torch.cat(logits, dim=-1)
    else:
        logits = self.lm_head(hidden_states)

    logits = logits.float()

    loss = None
    if labels is not None:  # train
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

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

    # cdt_len = guess_tree.tree.node_count if guess_tree is not None else 0
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
