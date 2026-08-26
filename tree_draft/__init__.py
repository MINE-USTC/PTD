"""Progressive Tree Drafting package.

Public helpers are imported lazily so lightweight commands do not require the
full GPU inference stack at import time.
"""


def _utils_function(name):
    from . import utils
    return getattr(utils, name)


def augment_all(*args, **kwargs):
    return _utils_function("augment_all")(*args, **kwargs)


def augment_generate(*args, **kwargs):
    return _utils_function("augment_generate")(*args, **kwargs)


def augment_llama(*args, **kwargs):
    return _utils_function("augment_llama")(*args, **kwargs)


def ini_model(*args, **kwargs):
    return _utils_function("ini_model")(*args, **kwargs)


def load_model(*args, **kwargs):
    return _utils_function("load_model")(*args, **kwargs)


def get_inference_id(*args, **kwargs):
    return _utils_function("get_inference_id")(*args, **kwargs)


def start_cache_server(*args, **kwargs):
    return _utils_function("start_cache_server")(*args, **kwargs)


__all__ = [
    "augment_all",
    "augment_generate",
    "augment_llama",
    "ini_model",
    "load_model",
    "get_inference_id",
    "start_cache_server",
]
