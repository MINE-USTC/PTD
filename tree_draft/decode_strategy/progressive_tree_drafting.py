import time
from time import perf_counter
from typing import List, Optional

import torch.distributed as dist
from fastchat.data.split_long_conversation import tokenizer
from torch.nn.utils.rnn import pad_sequence
from transformers import LogitsProcessorList, DynamicCache
from transformers.generation import validate_stopping_criteria
from transformers.generation.utils import GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput
from transformers.generation.utils import StoppingCriteriaList, GreedySearchOutput

from .utils import *
from ..inference_profile import InferProfile, ProfileKeys
from ..trees import *


def _configure_tree(self, batch_size: int, tree_config: dict) -> None:
    if not hasattr(self, 'trees'):
        Node.load_config(tree_config['node_config'])
        Tree.load_config(tree_config)
        self.draft_trees = DraftTrees(batch_size=batch_size, **tree_config)
        logger.success(f"Draft tree config: {tree_config}")


def _get_next_tokens(outputs, method: str):
    if method == 'argmax':
        return outputs.logits.argmax(dim=-1)
    elif method == 'sample':
        return torch.multinomial(torch.softmax(outputs.logits, dim=-1)[0], 1)
    elif method == 'topk':
        return topk_sampling(outputs.logits, 8)
    elif method == 'topp':
        return top_p_sampling(outputs.logits, 0.1)
    else:
        raise ValueError(f"Invalid global update method: {method}")


def _generate_new_attention_mask(batch_size, model_kwargs, pad_token_size, max_hits, unfinished_sequences):
    new_attn_mask = []
    for i in range(batch_size):
        attention_mask = model_kwargs["attention_mask"][i]
        assert unfinished_sequences[i]
        new_attn_mask.append(torch.cat((
            attention_mask,
            torch.zeros(pad_token_size[i], device=attention_mask.device, dtype=attention_mask.dtype),
            torch.ones(max_hits[i], device=attention_mask.device, dtype=attention_mask.dtype)),
        ))
    return new_attn_mask


def remove_pad_tokens(all_old_tokens, pad_token_id):
    """
    all_old_tokens: List[List[int]] or List[torch.Tensor]
    Returns: new list with all pad_token_id removed per sequence.
    """
    cleaned = []
    for seq in all_old_tokens:
        if isinstance(seq, torch.Tensor):
            seq = seq.tolist()
        cleaned.append([t for t in seq if t != pad_token_id])
    return cleaned


def _update_past_key_values(outputs, batch_size, max_hit, pad_token_size, accept_index):
    if isinstance(outputs.past_key_values, DynamicCache):
        num_layers = len(outputs.past_key_values.key_cache)

        for layer_idx in range(num_layers):
            k = outputs.past_key_values.key_cache[layer_idx]
            v = outputs.past_key_values.value_cache[layer_idx]

            assert outputs.step_len == k.size(2)

            if max_hit > 0:
                head_dim, embed_dim = k.size(1), k.size(-1)
                total_len = outputs.kvcache_len + max_hit

                padded_k = torch.zeros(batch_size, head_dim, max_hit, embed_dim, device=k.device, dtype=k.dtype)
                padded_v = torch.zeros(batch_size, head_dim, max_hit, embed_dim, device=v.device, dtype=v.dtype)

                for i in range(batch_size):
                    accept_len = accept_index[i].__len__()
                    pad_len = pad_token_size[i]

                    padded_k[i, :, pad_len:pad_len + accept_len, :] = k[i, :, accept_index[i], :]
                    padded_v[i, :, pad_len:pad_len + accept_len, :] = v[i, :, accept_index[i], :]

                k[:, :, outputs.kvcache_len:total_len, :] = padded_k
                v[:, :, outputs.kvcache_len:total_len, :] = padded_v

            outputs.past_key_values.key_cache[layer_idx] = k[:, :, :outputs.kvcache_len + max_hit, :]
            outputs.past_key_values.value_cache[layer_idx] = v[:, :, :outputs.kvcache_len + max_hit, :]

        if hasattr(outputs.past_key_values, "_seen_tokens"):
            outputs.past_key_values._seen_tokens = outputs.kvcache_len + max_hit
    else:
        past_key_values = []
        for idx, (k, v) in enumerate(outputs.past_key_values):
            assert outputs.step_len == k.size(2)

            if max_hit > 0:
                head_dim, embed_dim = k.size(1), k.size(-1)
                total_len = outputs.kvcache_len + max_hit

                padded_k = torch.zeros(batch_size, head_dim, max_hit, embed_dim, device=k.device, dtype=k.dtype)
                padded_v = torch.zeros(batch_size, head_dim, max_hit, embed_dim, device=v.device, dtype=v.dtype)

                for i in range(batch_size):
                    accept_len = accept_index[i].__len__()
                    pad_len = pad_token_size[i]

                    padded_k[i, :, pad_len:pad_len + accept_len, :] = k[i, :, accept_index[i], :]
                    padded_v[i, :, pad_len:pad_len + accept_len, :] = v[i, :, accept_index[i], :]

                k[:, :, outputs.kvcache_len:total_len, :] = padded_k
                v[:, :, outputs.kvcache_len:total_len, :] = padded_v

            past_key_values.append((
                k[:, :, :outputs.kvcache_len + max_hit, :],
                v[:, :, :outputs.kvcache_len + max_hit, :]
            ))

        outputs.past_key_values = past_key_values


def tree_draft_greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        **model_kwargs,
) -> Union[GreedySearchOutput, torch.LongTensor]:
    """
    CUSTOM forward with temperature-controlled sampling strategy.
    temperature=0: greedy (argmax)
    temperature>0: sampling
    """
    generation_config = model_kwargs.get("generation_config", None)
    temperature = generation_config.temperature if generation_config is not None else 0.0
    do_sample = temperature > 0.0
    if logits_warper is None:
        logits_warper = model_kwargs.pop("logits_warper", None)
    if logits_warper is None and do_sample:
        # Try to get logits_warper from model's method (standard transformers approach)
        if hasattr(self, "_get_logits_warper"):
            logits_warper = self._get_logits_warper(generation_config)
        else:
            # Fallback: create empty LogitsProcessorList if not available
            logits_warper = LogitsProcessorList()
    elif logits_warper is None:
        logits_warper = LogitsProcessorList()
    decode_method = "greedy" if temperature == 0 else f"sample(temp={temperature})"
    logger.success(f'Inference with tree draft {decode_method} decode method.')
    t_begin = perf_counter()
    profile = InferProfile()
    update_cache_process_thread = None
    setup_seed(10)
    batch_size = input_ids.shape[0]

    with profile.timer(ProfileKeys.PREPARE_TIME):
        guess_sem, request_sem, guess_shm, request_shm = init_ipc_and_reset_cache()
        # Corpus cache (REST, PLD, etc.) not supported yet; see ICLR branch for reference
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.tokenizer.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        assert self.tokenizer.pad_token_id == self.generation_config.pad_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        this_peer_finished = False
        all_old_tokens = input_ids.tolist()
        init_len = input_ids.shape[1]

        tree_config = getattr(self, 'tree_config', None)
        if tree_config is None:
            raise RuntimeError(
                "model.tree_config is not set. When using draft mode, base_tree_draft() must be called first "
                "so that tree_config is set on the model."
            )
        _configure_tree(self, batch_size, tree_config)
        self.draft_trees.clear()
        self.draft_trees.re_ini(remove_pad_tokens(all_old_tokens, pad_token_id))
        steps = 0
        is_prefill = True
        prompt_sizes = (input_ids != pad_token_id).sum(dim=1)
        init = self.tokenizer.batch_decode(all_old_tokens, skip_special_tokens=True,
                                           spaces_between_special_tokens=False, clean_up_tokenization_spaces=True, )
    update_cache_request = None
    retrieve_request = None
    pre_lst_tokens = [None for _ in range(batch_size)]

    while guess_sem.value > 0:
        guess_sem.acquire()

    res_all_old_tokens = [None for _ in range(batch_size)]
    org_idx = list(range(batch_size))
    finished_batch_idx = -1

    def _prepare_inputs():
        """Build model inputs for current step; set cache_position on first round."""
        nonlocal model_kwargs, is_prefill, input_ids, prompt_sizes
        with profile.timer(ProfileKeys.PREPARE_INPUT_TIME):
            sequence_len = input_ids.shape[1]
            if is_prefill:
                cache_position = torch.arange(sequence_len, dtype=torch.long).to(input_ids.device)
                model_kwargs['cache_position'] = cache_position

            model_inputs_inner = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            if not is_prefill:
                model_inputs_inner['input_ids'] = input_ids[:, -1:]
        return model_inputs_inner

    def _sync_guess_trees():
        """Sync/build guess_trees, including reading shared-memory results."""
        nonlocal retrieve_request, update_cache_request, guess_sem, guess_shm, unfinished_sequences, finished_batch_idx, batch_size
        if retrieve_request is not None or update_cache_request is not None:
            guess_sem.acquire()
            retrieve_res = read_shm(guess_shm)
            guess_trees_inner = retrieve_res['guess_tree']
            guess_trees_inner.update_guess_trees_with_retrieve_res(unfinished_sequences, finished_batch_idx)
            update_cache_request = None
            retrieve_request = None
        else:
            guess_trees_inner = GuessTreeBatch(retrieve_trees=[None for _ in range(batch_size)])
        return guess_trees_inner

    def _update_finished_sequences(next_tokens, hits, max_hits):
        """Handle finalization and trimming of finished sequences."""
        nonlocal unfinished_sequences, res_all_old_tokens, org_idx, all_old_tokens, pre_lst_tokens, batch_size, input_ids, model_kwargs, this_peer_finished
        nonlocal retrieve_request, update_cache_request, finished_batch_idx
        if eos_token_id_tensor is None:
            return
        pre_unfinished_sequences = unfinished_sequences.clone()
        unfinished_sequences = torch.tensor([
            0 if any(h in eos_token_id for h in hit_list) else unfinished_sequences[i]
            for i, hit_list in enumerate(hits)
        ], device=unfinished_sequences.device)
        finished_sequences = pre_unfinished_sequences - unfinished_sequences
        finished_sequences += (input_ids != pad_token_id).sum(dim=1) >= stopping_criteria.max_length

        if sum(finished_sequences) > 0:
            batch_size -= sum(finished_sequences)
            mask = torch.ones(input_ids.shape[0], device=next_tokens.device, dtype=torch.bool)
            mask[finished_sequences == 1] = False
            finished_batch_idx_local = (finished_sequences == 1).nonzero(as_tuple=False).squeeze().tolist()
            finished_batch_idx = finished_batch_idx_local
            input_ids_filtered, model_kwargs_filtered = filter_finished_sequences(input_ids, model_kwargs, mask)
            input_ids = input_ids_filtered
            model_kwargs = model_kwargs_filtered

            _request = {"type": "reduce", "remove_idx": finished_batch_idx_local}
            write_shm(request_shm, pickle.dumps(_request))
            request_sem.release()

            if not isinstance(finished_batch_idx_local, list):
                finished_batch_idx_local = [finished_batch_idx_local]
            for f_idx in finished_batch_idx_local[::-1]:
                res_all_old_tokens[org_idx.pop(f_idx)] = all_old_tokens.pop(f_idx)
                pre_lst_tokens.pop(f_idx)
            self.draft_trees.reduce(finished_batch_idx_local)
            this_peer_finished = len(all_old_tokens) == 0 or unfinished_sequences.max() == 0
            if not this_peer_finished:
                unfinished_sequences = unfinished_sequences[mask]
        else:
            if unfinished_sequences.max() == 0:  # all finished
                this_peer_finished = True

    def _finalize_iteration_timing(iter_start_ts, forward_end_ts):
        profile.incremental_update(ProfileKeys.ITER_TIME, perf_counter() - iter_start_ts)
        profile.incremental_update(ProfileKeys.AFTER_FORWARD_TIME, perf_counter() - forward_end_ts)

    setup_seed(10)
    while True:
        ti = perf_counter()
        # print(f'steps:{steps}')
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic all ows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                if guess_sem.value == 1:
                    guess_sem.acquire()
                break
        model_inputs = _prepare_inputs()

        assert return_dict_in_generate == False

        with profile.timer(ProfileKeys.UPDATE_GLOBAL_INFO_TIME):
            if is_prefill:
                self.draft_trees.update_dfs()
                self.draft_trees.update_draft_tree_attn_mask(torch.bfloat16)
            self.draft_trees.assert_node_count()

        with profile.timer(ProfileKeys.BEFORE_FORWARD_TIME):
            guess_trees = _sync_guess_trees()
            self.draft_trees.update_count()
            draft_counts = [self.draft_trees[i].tree.node_count if self.draft_trees[i] is not None else 0 for i in
                            range(batch_size)]

        profile.incremental_update(ProfileKeys.AUX_TOKENS_NUM, sum(draft_counts))
        profile.incremental_update(ProfileKeys.CDT_TOKENS_NUM, sum(guess_trees.sizes))

        with profile.timer(ProfileKeys.FORWARD_TIME):
            lst_idx = prompt_sizes - 1 if is_prefill else None
            outputs, forward_profile = self.PTDforward(draft=self.draft_trees, guess=guess_trees,
                                                       output_attentions=output_attentions,
                                                       output_hidden_states=output_hidden_states, return_dict=True,
                                                       lst_idx=lst_idx,
                                                       **model_inputs)
            torch.cuda.synchronize()

        profile.incremental_update(ProfileKeys.SEQUENCE_FORWARD_COUNT, batch_size)
        profile.incremental_update(ProfileKeys.FORWARD_COUNT, 1)
        steps += 1
        profile.incremental_updates(forward_profile)
        ta = perf_counter()
        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        t0 = perf_counter()
        prefill_size = model_inputs['input_ids'].size(1)
        argmax_next_token = outputs.logits.argmax(dim=-1)
        argmax_next_token_list = argmax_next_token.tolist()
        outputs_probs = torch.softmax(outputs.logits, dim=-1)

        # Determine next token based on temperature
        if temperature == 0:
            # Greedy: use argmax
            if self.tokenizer.padding_side == 'left':
                next_tokens = argmax_next_token[
                    torch.arange(batch_size), prefill_size - 1]
            else:
                next_tokens = argmax_next_token[
                    torch.arange(batch_size), (prompt_sizes - 1) if is_prefill else (prefill_size - 1)]
        else:
            # Sampling: use logits_warper and multinomial
            if logits_warper is None:
                raise ValueError("logits_warper must be provided when temperature > 0")

            # Get next logits for each sequence in the batch
            next_logits_list = []
            for i in range(batch_size):
                if self.tokenizer.padding_side == 'left':
                    idx = prefill_size - 1
                else:
                    idx = (prompt_sizes[i] - 1) if is_prefill else (prefill_size - 1)
                next_logits_list.append(outputs.logits[i, idx])

            # Apply warper and sample
            next_tokens_list = []
            for i in range(batch_size):
                warped_logits = logits_warper(outputs.all_ids[i:i + 1], next_logits_list[i].unsqueeze(0))
                next_probs = torch.softmax(warped_logits, dim=-1)
                sampled_token = torch.multinomial(next_probs, num_samples=1)
                next_tokens_list.append(sampled_token.squeeze())
            next_tokens = torch.stack(next_tokens_list)
        next_tokens_int = next_tokens.tolist()
        tree_update_method = self.tree_config.get('tree_update_method', 'argmax')
        all_next_tokens = _get_next_tokens(outputs, tree_update_method)
        all_next_list = all_next_tokens.tolist()
        profile.incremental_update(ProfileKeys.OUTPUT_ARGMAX_LIST_TIME, perf_counter() - t0)

        with profile.timer(ProfileKeys.PROCESS_OUTPUT_TIME):
            draft_counts = [self.draft_trees[i].tree.node_count for i in range(batch_size)]
            draft_tokens_list = [all_next_list[i][prefill_size: prefill_size + draft_counts[i]] for i in
                                 range(batch_size)]

            # Process verify tokens based on temperature
            if temperature == 0:
                # Greedy mode: use argmax
                verify_token_lists, verify_probs = guess_trees.get_verify_token_lists(argmax_next_token_list, outputs)
            else:
                # Sampling mode: apply logits_warper to verify_logits
                verify_token_lists = []
                verify_probs = []
                for i in range(batch_size):
                    if guess_trees[i] is not None and guess_trees.sizes[i] > 0:
                        # Apply warper to verify logits
                        verify_logits = logits_warper(outputs.all_ids[i:i + 1], outputs.verify_logits[i])
                        verify_prob = torch.softmax(verify_logits, dim=-1)
                        verify_probs.append(verify_prob)
                        verify_token_lists.append(verify_logits.argmax(dim=-1).tolist())
                    else:
                        verify_probs.append(None)
                        verify_token_lists.append([])

        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        with profile.timer(ProfileKeys.COPY_DRAFT_TREE_TIME):
            self.draft_trees.update_draft_tree_argmax(draft_tokens_list)
            copy_draft_tree = self.draft_trees.copy_draft_tree()
        if temperature == 0:
            # Greedy mode
            hits, max_hits, accept_seq, accept_index = guess_trees.guess_hits(verify_token_lists, next_tokens_int,
                                                                              eos_token_id, profile)
        else:
            # Sampling mode: use the unified guess_hits_sample method
            # print(steps)
            # hits, max_hits, accept_seq, accept_index = guess_trees.guess_hits_sample(verify_probs, next_probs,
            #                                                                          eos_token_id, next_tokens_int,
            #                                                                          profile)
            hits, max_hits, accept_seq, accept_index = guess_trees.guess_hits_sample(
                verify_probs=verify_probs,
                next_tokens_probs=next_probs,
                next_tokens_int=next_tokens_int,
                eos_token_id=eos_token_id,
                profile=profile,
            )

        lst_token = [hits[i][max_hits[i]] for i in range(batch_size)]

        for i in range(batch_size):
            for hh in range(max_hits[i] + 1):
                assert unfinished_sequences[i]
                all_old_tokens[i].append(hits[i][hh])

        accept_index = [accept_index[i][:max_hits[i]] for i in range(batch_size)]
        offset = outputs.kvcache_len + max(draft_counts)
        accept_index = [[x + offset for x in accept_index[i]] for i in range(batch_size)]

        with profile.timer(ProfileKeys.UPDATE_CACHE_TIME):
            tree_depths = [_.tree.tree_depth for _ in copy_draft_tree]

            if max(tree_depths) >= copy_draft_tree[0].tree.max_depth:
                update_cache_request = {"type": "update_cache_next_then_other",
                                        'lst_token': lst_token,
                                        'all_old_tokens': all_old_tokens,
                                        "update_keys": pre_lst_tokens,
                                        "draft": copy_draft_tree,
                                        'verify_token_lists': verify_token_lists,
                                        'guess_trees': guess_trees}

                write_shm(request_shm, pickle.dumps(update_cache_request))
                request_sem.release()
            else:
                retrieve_request = {"type": "retrieve",
                                    "lst_token": lst_token,
                                    'all_old_tokens': all_old_tokens,
                                    'update_keys': pre_lst_tokens,
                                    'verify_token_lists': verify_token_lists,
                                    'guess_trees': guess_trees}

                write_shm(request_shm, pickle.dumps(retrieve_request))
                request_sem.release()

        pre_lst_tokens = lst_token

        with profile.timer(ProfileKeys.UPDATE_MASK_TIME):
            max_hit = max(max_hits)
            pad_token_size = [max_hit - mh for mh in max_hits]
            new_attn_mask = _generate_new_attention_mask(batch_size, model_kwargs, pad_token_size, max_hits,
                                                         unfinished_sequences)
            if len(new_attn_mask) > 0:
                model_kwargs["attention_mask"] = torch.stack(new_attn_mask, dim=0)

        with profile.timer(ProfileKeys.UPDATE_DRAFT_TREE_INTER_LOGIT_TIME):
            self.draft_trees.update_draft_tree_inter_logit()

        with profile.timer(ProfileKeys.UPDATE_SUB_TREE_ROOT_TIME):
            self.draft_trees.update_sub_tree_root(next_tokens_int)

        with profile.timer(ProfileKeys.PRUNE_DEPTH_TIME):
            self.draft_trees.prune_depth()

        with profile.timer(ProfileKeys.DRAFT_ATTN_TIME):
            self.draft_trees.draft_tree_attn_mask(torch.bfloat16)

        with profile.timer(ProfileKeys.UPDATE_KV_TIME):
            _update_past_key_values(outputs, batch_size, max_hit, pad_token_size, accept_index)

        with profile.timer(ProfileKeys.UPDATE_INPUT_TIME):
            new_ids_list = []
            for i in range(batch_size):
                assert unfinished_sequences[i]
                new_ids_list.append(
                    torch.tensor(
                        [pad_token_id for _ in range(pad_token_size[i])] + hits[i][:max_hits[i] + 1],
                        device=next_tokens.device, dtype=next_tokens.dtype
                    ))
            new_ids = torch.stack(new_ids_list, dim=0)
            input_ids = torch.cat((input_ids, new_ids), dim=-1)

        with profile.timer(ProfileKeys.MODEL_KWARGS_UPDATE_TIME):
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

        _update_finished_sequences(next_tokens, hits, max_hits)

        if this_peer_finished and not synced_gpus:
            if retrieve_request is not None or update_cache_request is not None:
                while guess_sem.value > 0:
                    guess_sem.acquire()
                assert guess_sem.value == 0
            break

        is_prefill = False
        _finalize_iteration_timing(ti, ta)

    with profile.timer(ProfileKeys.POST_PROCESS_TIME):
        prompt_sizes = prompt_sizes.tolist()
        input_ids = pad_sequence([torch.tensor(lst) for lst in res_all_old_tokens], batch_first=True,
                                 padding_value=pad_token_id)
        gen_count = sum((input_ids.ne(pad_token_id).sum(dim=1) - torch.tensor(prompt_sizes)).tolist())
        profile.set_profile(ProfileKeys.OVERALL_GEN, gen_count)
        gap_time = time.perf_counter() - t_begin
        profile.set_profile(ProfileKeys.DECODING_TIME, gap_time)
        profile.append_list(ProfileKeys.TP, gen_count / gap_time)
    
    logger.debug("\n" + profile.__str__())
    profile.log_profile(logger, "Current Summary")

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GreedySearchEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
            )
        else:
            return GreedySearchDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
    else:
        return (input_ids, steps, profile)
