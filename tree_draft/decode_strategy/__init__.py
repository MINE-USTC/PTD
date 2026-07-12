# Unified autoregressive decode strategy
from .autoregressive import autoregressive_decode, greedy_search, sample

# Tree Draft decode strategy
from .progressive_tree_drafting import *

# Legacy (deprecated): use autoregressive_decode
# from .greedy_autoregressive import greedy_search  # merged into autoregressive.py
# from .sampe_autoregressive import sample  # merged into autoregressive.py
