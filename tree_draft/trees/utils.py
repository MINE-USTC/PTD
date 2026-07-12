import random

import numpy as np
import torch


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def print_with_depth_format(node):
    print('\t' * node.depth + str(node.value))
