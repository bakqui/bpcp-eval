import random
from collections import deque


class DualBuffer:
    def __init__(self, max_unlabeled=64, max_labeled=8):
        self.unlabeled = deque(maxlen=max_unlabeled)
        self.labeled   = deque(maxlen=max_labeled)

    def add(self, x, y=None):
        if y is None:
            self.unlabeled.append(x)
        else:
            self.labeled.append((x, y))

    def sample(self, batch_size: int, labeled_ratio: float):
        n_l = int(batch_size * labeled_ratio)
        n_u = batch_size - n_l
        lab = random.sample(self.labeled,   min(len(self.labeled),   n_l)) if self.labeled   else []
        unl = random.sample(self.unlabeled, min(len(self.unlabeled), n_u)) if self.unlabeled else []
        if lab:
            xs, ys = zip(*lab)
        else:
            xs, ys = [], []
        return unl, list(xs), list(ys)
