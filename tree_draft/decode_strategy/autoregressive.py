"""
Unified autoregressive decode strategy.
Temperature controls greedy vs sampling:
- temperature ≈ 0: greedy (argmax)
- temperature > 0: sampling (multinomial)
"""
from time import perf_counter
from typing import List, Optional, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers.generation import validate_stopping_criteria
from transformers.generation.utils import (
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
    GenerateNonBeamOutput,
    LogitsProcessorList,
    StoppingCriteriaList,
    GreedySearchOutput
)

from ..inference_profile import InferProfile, ProfileKeys


def autoregressive_decode(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        temperature: float = 0.0,
        do_sample: bool = None,
        enable_batch_processing: bool = True,
        **model_kwargs,
) -> Union[GreedySearchOutput, GenerateNonBeamOutput, torch.LongTensor]:
    """
    Unified autoregressive decode.

    Args:
        temperature: 0 or near 0 -> greedy; > 0 -> sampling.
        do_sample: If None, inferred from temperature.
        enable_batch_processing: Dynamic removal of finished sequences.
        Other args: standard transformers generate params.

    Returns:
        Generated sequences and metadata.
    """
    if do_sample is None:
        do_sample = temperature > 1e-4

    mode = "Sampling" if do_sample else "Greedy"
    print(f"{mode} decoding (temperature={temperature:.2f}) with profiling code")

    profile = InferProfile()
    t_begin = perf_counter()

    with profile.timer(ProfileKeys.PREPARE_TIME):
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        if max_length is not None:
            import warnings
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)

        if do_sample:
            logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()

        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id

        if pad_token_id is None and eos_token_id is not None:
            if isinstance(eos_token_id, list):
                pad_token_id = eos_token_id[0]
            else:
                pad_token_id = eos_token_id

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None

        # 输出配置
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

        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        batch_size = input_ids.shape[0]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

        prompt_sizes = input_ids.shape[1]

    this_peer_finished = False

    if enable_batch_processing and not do_sample:
        res_all_old_tokens = [None for _ in range(batch_size)]
        org_idx = list(range(batch_size))
        finished_order = []

    if do_sample and streamer is None and batch_size == 1:
        prev_decoded_len = 0
    else:
        prev_decoded_len = None

    s = 0

    while True:
        s += 1
        iter_start = perf_counter()

        if synced_gpus:
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            if this_peer_finished_flag.item() == 0.0:
                break

        with profile.timer(ProfileKeys.BEFORE_FORWARD_TIME):
            sequence_len = input_ids.shape[1]
            if s == 1:
                cache_position = torch.arange(sequence_len, dtype=torch.long).to(input_ids.device)
                model_kwargs['cache_position'] = cache_position

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        with profile.timer(ProfileKeys.FORWARD_TIME):
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            torch.cuda.synchronize()

        current_batch_size = input_ids.shape[0]
        profile.incremental_update(ProfileKeys.FORWARD_COUNT, current_batch_size)
        profile.incremental_update(ProfileKeys.SEQUENCE_FORWARD_COUNT, current_batch_size)
        profile.incremental_update(ProfileKeys.AUX_TOKENS_NUM, 0)
        profile.incremental_update(ProfileKeys.CDT_TOKENS_NUM, current_batch_size)

        after_forward_start = perf_counter()
        if synced_gpus and this_peer_finished:
            continue

        next_token_logits = outputs.logits[:, -1, :]

        next_token_scores = logits_processor(input_ids, next_token_logits)
        if do_sample:
            next_token_scores = logits_warper(input_ids, next_token_scores)

        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)
            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        with profile.timer(ProfileKeys.OUTPUT_ARGMAX_LIST_TIME):
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

        # Pad finished sequences
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        # Streaming output (sampling + single sequence only)
        if prev_decoded_len is not None:
            with profile.timer(ProfileKeys.STRING_TIME):
                token_list = input_ids[0].tolist()
                all_str = self.tokenizer.decode(
                    token_list,
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
                print(all_str[prev_decoded_len:], flush=True, end="")
                prev_decoded_len = len(all_str)

        if streamer is not None:
            streamer.put(next_tokens.cpu())

        with profile.timer(ProfileKeys.MODEL_KWARGS_UPDATE_TIME):
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

        if eos_token_id_tensor is not None:
            pre_unfinished_sequences = unfinished_sequences.clone()
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            if enable_batch_processing and not do_sample:
                finished_sequences = pre_unfinished_sequences - unfinished_sequences
                real_lengths = (input_ids != pad_token_id).sum(dim=1)
                finished_sequences = finished_sequences + (real_lengths >= stopping_criteria.max_length).long()

                if sum(finished_sequences) > 0:
                    current_batch_size -= sum(finished_sequences).item()
                    mask = torch.ones(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
                    mask[finished_sequences == 1] = False

                    finished_batch_idx = (finished_sequences == 1).nonzero(as_tuple=False).squeeze().tolist()
                    if isinstance(finished_batch_idx, int):
                        finished_batch_idx = [finished_batch_idx]

                    for f_idx in finished_batch_idx[::-1]:
                        global_finished_idx = org_idx.pop(f_idx)
                        finished_order.append(global_finished_idx)
                        res_all_old_tokens[global_finished_idx] = input_ids[f_idx]

                    input_ids = input_ids[mask]
                    model_kwargs['attention_mask'] = model_kwargs['attention_mask'][mask]

                    tmp_kv = []
                    for idx, kv in enumerate(model_kwargs['past_key_values']):
                        tmp_kv.append((kv[0][mask], kv[1][mask]))
                    model_kwargs['past_key_values'] = tmp_kv

                    if input_ids.shape[0] == 0 or unfinished_sequences.max() == 0:
                        this_peer_finished = True
                    else:
                        unfinished_sequences = unfinished_sequences[mask]
                else:
                    if unfinished_sequences.max() == 0:
                        this_peer_finished = True
            else:
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

        profile.incremental_update(ProfileKeys.AFTER_FORWARD_TIME, perf_counter() - after_forward_start)
        profile.incremental_update(ProfileKeys.ITER_TIME, perf_counter() - iter_start)

    if streamer is not None:
        streamer.end()

    if enable_batch_processing and not do_sample and res_all_old_tokens[0] is not None:
        input_ids = pad_sequence(res_all_old_tokens, batch_first=True, padding_value=pad_token_id)

    with profile.timer(ProfileKeys.POST_PROCESS_TIME):
        gen_count = (input_ids.ne(pad_token_id).sum(dim=1) - prompt_sizes).sum().item()
        profile.set_profile(ProfileKeys.OVERALL_GEN, gen_count)
        gap_time = perf_counter() - t_begin
        profile.set_profile(ProfileKeys.DECODING_TIME, gap_time)
        profile.append_list(ProfileKeys.TP, gen_count / gap_time if gap_time > 0 else 0)

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return (input_ids, s, profile)


# Legacy aliases
def greedy_search(self, *args, **kwargs):
    """Greedy search (temperature=0)."""
    kwargs['temperature'] = 0.0
    kwargs['do_sample'] = False
    return autoregressive_decode(self, *args, **kwargs)


def sample(self, *args, **kwargs):
    """Sampling decode (temperature > 0)."""
    if 'temperature' not in kwargs:
        kwargs['temperature'] = 1.0
    kwargs['do_sample'] = True
    return autoregressive_decode(self, *args, **kwargs)
