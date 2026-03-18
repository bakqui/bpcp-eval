import os
import random

import numpy as np
import torch


class AttributeDict(dict):
    def __getattr__(self, key):
        try:
            value = self[key]
            # auto-convert nested dicts to DotDict
            if isinstance(value, dict) and not isinstance(value, AttributeDict):
                value = AttributeDict(value)
                self[key] = value
            return value
        except KeyError:
            raise AttributeError(f"No such attribute: {key}")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]


def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
