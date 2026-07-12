from statistics import stdev
from time import perf_counter
from contextlib import contextmanager
from typing import Optional
from threading import local
import functools


# ============================================================================
# Profile key constants (avoid typos, IDE autocomplete)
# ============================================================================


class ProfileKeys:
    """Profile key constants."""
    # Time metrics
    BEFORE_FORWARD_TIME = 'before_forward_time'
    FORWARD_TIME = 'forward_time'
    AFTER_FORWARD_TIME = 'after_forward_time'
    PREPARE_INPUT_TIME = 'prepare_input_time'
    RETRIEVE_SEND_TIME = 'retrieve_send_time'
    UPDATE_GLOBAL_INFO_TIME = 'update_global_info_time'
    SEM_ACQUIRE_TIME = 'sem_acquire_time'
    CACHE_RETRIEVE_TIME_DECODE = 'cache_retrieve_time_decode'
    CACHE_RETRIEVE_TIME = 'cache_retrieve_time'
    PREPARE_IDS_TIME = 'prepare_ids_time'
    LLAMA_MODEL_FORWARD_TIME = 'LlamaModel_forward_time'
    ATTENTION_MASK_TIME = 'attention_mask_time'
    LAYER_FORWARD_TIME = 'layer_forward_time'
    POSTPROCESS_FORWARD_TIME = 'postprocess_forward_time'
    PROCESS_OUTPUT_TIME = 'process_output_time'
    OUTPUT_ARGMAX_TIME = 'output_argmax_time'
    OUTPUT_ARGMAX_LIST_TIME = 'output_argmax_list_time'
    UPDATE_CACHE_TIME = 'update_cache_time'
    UPDATE_CACHE_NEXT_TIME = 'update_cache_next_time'
    UPDATE_CACHE_OTHER_TIME = 'update_cache_other_time'
    UPDATE_MASK_TIME = 'update_mask_time'
    COPY_DRAFT_TREE_TIME = 'copy_draft_tree_time'
    UPDATE_DRAFT_TREE_INTER_LOGIT_TIME = 'update_draft_tree_inter_logit_time'
    UPDATE_SUB_TREE_ROOT_TIME = 'update_sub_tree_root_time'
    PRUNE_DEPTH_TIME = 'prune_depth_time'
    DRAFT_ATTN_TIME = 'draft_attn_time'
    VERIFY_TIME = 'verify_time'
    UPDATE_KV_TIME = 'update_kv_time'
    STRING_TIME = 'string_time'
    UPDATE_INPUT_TIME = 'update_input_time'
    MODEL_KWARGS_UPDATE_TIME = 'model_kwargs_update_time'
    ITER_TIME = 'iter_time'
    DECODING_TIME = 'decoding_time'
    PREPARE_TIME = 'prepare_time'
    POST_PROCESS_TIME = 'post_process_time'

    # Count metrics
    AUX_TOKENS_NUM = 'aux_tokens_num'
    CDT_TOKENS_NUM = 'cdt_tokens_num'
    FORWARD_COUNT = 'forward_count'
    SEQUENCE_FORWARD_COUNT = 'sequence_forward_count'
    CACHE_HIT_COUNT = 'cache_hit_count'
    GEN_COUNT = 'gen_count'
    OVERALL_GEN = 'overall_gen'

    # List metrics
    TP = 'tp'
    HIT_LEN_LIST = 'hit_len_list'
    CDT_TOKEN_NUM_LIST = 'cdt_token_num_list'
    TRIE_NODE_COUNT_LIST = 'trie_node_count_list'
    TRIE_NODE_COUNT = 'trie_node_count'


# Thread-local storage for current profile
_thread_local = local()


def get_current_profile() -> Optional['InferProfile']:
    """Return the current thread's profile instance."""
    return getattr(_thread_local, 'profile', None)


def set_current_profile(profile: Optional['InferProfile']):
    """Set the current thread's profile instance."""
    _thread_local.profile = profile


# ============================================================================
# BaseProfile and InferProfile (backward compatible)
# ============================================================================
class BaseProfile:
    def __init__(self, profile_keys=None):
        profile_keys = [] if profile_keys is None else profile_keys
        self.pf = {}
        for key in profile_keys:
            self.pf[key] = 0

    def update_profile(self, key, value):
        if key in self.pf:
            self.pf[key] = value
        else:
            print(f"Key {key} does not exist in the profile.")

    def set_profile(self, key, value):
        self.pf[key] = value

    def get_profile(self, key):
        if key in self.pf:
            return self.pf[key]
        else:
            return 0

    def incremental_update(self, key, value):
        if key in self.pf:
            # Type check: avoid += on list
            if isinstance(self.pf[key], list):
                if isinstance(value, list):
                    self.pf[key].extend(value)
                else:
                    self.pf[key].append(value)
            else:
                self.pf[key] += value
        else:
            self.pf[key] = value

    def append_list(self, key, new_value):
        if key in self.pf:
            self.pf[key].append(new_value)
        else:
            self.pf[key] = [new_value]

    def stat_list(self, key):
        assert key in self.pf
        assert isinstance(self.pf[key], list)
        freq = {}
        for val in self.pf[key]:
            freq[val] = freq.get(val, 0) + 1
        return freq

    def incremental_updates(self, new_profile):
        if isinstance(new_profile, dict):
            for key, val in new_profile.items():
                if isinstance(val, list):
                    if key in self.pf:
                        if isinstance(self.pf[key], list):
                            self.pf[key].extend(val)
                        else:
                            self.pf[key] = [self.pf[key]] + val if self.pf[key] != 0 else val
                    else:
                        self.pf[key] = val.copy() if isinstance(val, list) else val
                else:
                    self.incremental_update(key, val)
        elif isinstance(new_profile, InferProfile) or "InferProfile" in str(type(new_profile)):
            for key, val in new_profile.pf.items():
                if isinstance(val, list):
                    if key in self.pf:
                        if isinstance(self.pf[key], list):
                            self.pf[key].extend(val)
                        else:
                            self.pf[key] = [self.pf[key]] + val if self.pf[key] != 0 else val
                    else:
                        self.pf[key] = val.copy() if isinstance(val, list) else val
                else:
                    self.incremental_update(key, val)
        elif new_profile is None:
            return
        else:
            raise RuntimeError('Unsupported type of new_profile:', type(new_profile))

    def __str__(self):
        s = '\n'
        for key, val in self.pf.items():
            s += f'{key}\t\t{val}\n'
        return s

    def output(self, keys):
        res = '\n'
        for key in keys:
            res += f'{key}\t\t{self.get_profile(key)}\n'
        return res

    # ========================================================================
    # Context manager for auto timing
    # ========================================================================
    @contextmanager
    def timer(self, key: str):
        """Context manager: time block and add to key.

        Example:
            with profile.timer(ProfileKeys.FORWARD_TIME):
                outputs = model(inputs)
        """
        start = perf_counter()
        try:
            yield
        finally:
            self.incremental_update(key, perf_counter() - start)

    def time_it(self, key: str):
        """Decorator: time function execution.

        Example:
            @profile.time_it(ProfileKeys.PREPARE_INPUT_TIME)
            def prepare_inputs():
                ...
        """

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with self.timer(key):
                    return func(*args, **kwargs)

            return wrapper

        return decorator


class InferProfile(BaseProfile):
    def __init__(self, profile_keys=None):
        super().__init__(profile_keys)

        # Init all profile keys (backward compat: if no keys given, init all)
        keys = ProfileKeys()
        all_keys = [attr for attr in dir(keys) if not attr.startswith('_') and attr.isupper()]
        for key_name in all_keys:
            key_value = getattr(keys, key_name)
            if key_value not in self.pf:
                if 'LIST' in key_name or key_value == ProfileKeys.TP:
                    self.pf[key_value] = []
                else:
                    self.pf[key_value] = 0

        self.average_profile_keys = [
            ProfileKeys.AUX_TOKENS_NUM, ProfileKeys.CDT_TOKENS_NUM,
            ProfileKeys.DECODING_TIME, ProfileKeys.ITER_TIME,
            ProfileKeys.BEFORE_FORWARD_TIME, ProfileKeys.PREPARE_INPUT_TIME,
            ProfileKeys.RETRIEVE_SEND_TIME, ProfileKeys.UPDATE_GLOBAL_INFO_TIME,
            ProfileKeys.SEM_ACQUIRE_TIME, ProfileKeys.CACHE_RETRIEVE_TIME_DECODE,
            ProfileKeys.CACHE_RETRIEVE_TIME, ProfileKeys.FORWARD_TIME,
            ProfileKeys.PREPARE_IDS_TIME, ProfileKeys.LLAMA_MODEL_FORWARD_TIME,
            ProfileKeys.ATTENTION_MASK_TIME, ProfileKeys.LAYER_FORWARD_TIME,
            ProfileKeys.POSTPROCESS_FORWARD_TIME, ProfileKeys.AFTER_FORWARD_TIME,
            ProfileKeys.PROCESS_OUTPUT_TIME, ProfileKeys.OUTPUT_ARGMAX_TIME,
            ProfileKeys.OUTPUT_ARGMAX_LIST_TIME, ProfileKeys.UPDATE_CACHE_TIME,
            ProfileKeys.UPDATE_CACHE_NEXT_TIME, ProfileKeys.UPDATE_CACHE_OTHER_TIME,
            ProfileKeys.UPDATE_MASK_TIME, ProfileKeys.COPY_DRAFT_TREE_TIME,
            ProfileKeys.UPDATE_DRAFT_TREE_INTER_LOGIT_TIME,
            ProfileKeys.UPDATE_SUB_TREE_ROOT_TIME, ProfileKeys.PRUNE_DEPTH_TIME,
            ProfileKeys.DRAFT_ATTN_TIME, ProfileKeys.VERIFY_TIME,
            ProfileKeys.UPDATE_KV_TIME, ProfileKeys.STRING_TIME,
            ProfileKeys.UPDATE_INPUT_TIME, ProfileKeys.MODEL_KWARGS_UPDATE_TIME
        ]

        self.key_average_time_keys = [
            ProfileKeys.BEFORE_FORWARD_TIME, ProfileKeys.FORWARD_TIME,
            ProfileKeys.AFTER_FORWARD_TIME
        ]

    # ========================================================================
    # Context manager: set global profile
    # ========================================================================
    def __enter__(self):
        set_current_profile(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_current_profile(None)

    def log_model_and_seq_forward(self):
        info = '\n' + '------' * 5 + 'Profile Per Model Forward' + '------' * 5 + '\n'
        forward_count = self.get_profile(ProfileKeys.FORWARD_COUNT)
        overall_gen = self.get_profile(ProfileKeys.OVERALL_GEN)
        aux_tokens = self.get_profile(ProfileKeys.AUX_TOKENS_NUM)
        cdt_tokens = self.get_profile(ProfileKeys.CDT_TOKENS_NUM)

        info += (f"MODEL FORWARD STEP: {forward_count}\n"
                 f"AVG DECODING EFFICIENCY PER MODEL STEP: "
                 f"{(overall_gen / forward_count if forward_count else 0):.4f}\n"
                 f"AVERAGE AUX TOKENS NUMBER PER MODEL FORWARD STEP: "
                 f"{(aux_tokens / forward_count if forward_count else 0):.4f}\n"
                 f"AVERAGE CANDIDATE TOKENS NUMBER PER MODEL FORWARD STEP: "
                 f"{(cdt_tokens / forward_count if forward_count else 0):.4f}\n")

        seq_forward_count = self.get_profile(ProfileKeys.SEQUENCE_FORWARD_COUNT)
        info += ('\n' + '------' * 5 + 'Profile Per Sequence Forward' + '------' * 5 + '\n'
                                                                                       f"SEQUENCE FORWARD STEP: {seq_forward_count}\n"
                                                                                       f"AVG DECODING EFFICIENCY PER SEQUENCE STEP: "
                                                                                       f"{(overall_gen / seq_forward_count if seq_forward_count else 0):.4f}\n"
                                                                                       f"AVERAGE AUX TOKENS NUMBER PER SEQUENCE FORWARD STEP: "
                                                                                       f"{(aux_tokens / seq_forward_count if seq_forward_count else 0):.4f}\n"
                                                                                       f"AVERAGE CANDIDATE TOKENS NUMBER PER SEQUENCE FORWARD STEP: "
                                                                                       f"{(cdt_tokens / seq_forward_count if seq_forward_count else 0):.4f}\n")

        return info

    def log_profile(self, logger, stage="FINAL PROFILE"):
        tp_list = self.get_profile(ProfileKeys.TP)
        overall_gen = self.get_profile(ProfileKeys.OVERALL_GEN)
        decoding_time = self.get_profile(ProfileKeys.DECODING_TIME)
        cache_hit_count = self.get_profile(ProfileKeys.CACHE_HIT_COUNT)
        seq_forward_count = self.get_profile(ProfileKeys.SEQUENCE_FORWARD_COUNT)
        hit_len_list = self.get_profile(ProfileKeys.HIT_LEN_LIST)
        cdt_tokens = self.get_profile(ProfileKeys.CDT_TOKENS_NUM)

        info = '\n' + '==========' * 5 + f'{stage}' + '==========' * 5 + '\n'
        info += (f"AVERAGE THROUGHPUT1: {(sum(tp_list) / len(tp_list) if tp_list else 0):.4f} "
                 f"({stdev(tp_list) if len(tp_list) > 1 else 0:.4f})\n"
                 f"AVERAGE THROUGHPUT2: {(overall_gen / decoding_time if decoding_time else 0):.4f}\n"
                 f"OVERALL GEN: {overall_gen}\n")
        info += f"Cache hit count: {cache_hit_count}\n"
        info += f"Cache hit ratio: {(cache_hit_count / seq_forward_count if seq_forward_count else 0):.4f}\n"
        info += f"Avg cache accept len per hit: {(1 + sum(hit_len_list) / cache_hit_count if cache_hit_count else 0):.4f}\n"
        info += f"Avg cache token count per hit: {(cdt_tokens / cache_hit_count if cache_hit_count else 0):.4f}\n"
        info += self.log_model_and_seq_forward()
        logger.success(info)

    def log_profile_time(self, logger):
        logger.success('\n' + '==========' * 5 + 'RUN TIME PROFILE' + '==========' * 5)
        run_time_s = ''
        forward_count = self.get_profile(ProfileKeys.FORWARD_COUNT)
        for key in self.key_average_time_keys:
            val = self.get_profile(key) / forward_count if forward_count else 0
            if isinstance(val, float):
                run_time_s += f'{key}: {val:.6f}\n'
            else:
                run_time_s += f'{key}: {val}\n'
        logger.success(run_time_s)

    def log_profile_average(self, logger):
        logger.debug(self)
        logger.debug('\n=================average profile==============')
        s = '\n'
        forward_count = self.get_profile(ProfileKeys.FORWARD_COUNT)
        for k in self.average_profile_keys:
            v = self.get_profile(k) / forward_count if forward_count else 0
            s += f'{k}\t\t{v:.6f}\n'
        logger.debug(s)


# ============================================================================
# Helpers: use current profile in functions
# ============================================================================
@contextmanager
def profile_timer(key: str, profile: Optional[InferProfile] = None):
    """Time a block using current thread profile or given profile.

    Example:
        with profile_timer(ProfileKeys.FORWARD_TIME):
            outputs = model(inputs)
        with profile_timer(ProfileKeys.FORWARD_TIME, my_profile):
            outputs = model(inputs)
    """
    prof = profile or get_current_profile()
    if prof is None:
        yield
        return

    start = perf_counter()
    try:
        yield
    finally:
        prof.incremental_update(key, perf_counter() - start)


def profile_timed(key: str, profile: Optional[InferProfile] = None):
    """Decorator: time function with current thread profile.

    Example:
        @profile_timed(ProfileKeys.PREPARE_INPUT_TIME)
        def prepare_inputs():
            ...
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            prof = profile or get_current_profile()
            if prof is None:
                return func(*args, **kwargs)

            with prof.timer(key):
                return func(*args, **kwargs)

        return wrapper

    return decorator
