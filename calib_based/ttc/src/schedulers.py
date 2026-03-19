import math


class WarmupCosineScheduler:
    def __init__(self, optimizer, total_steps, warmup_steps=0, min_lr=0.0):
        self.optimizer = optimizer
        self.total_steps = max(1, total_steps)
        self.warmup_steps = max(0, warmup_steps)
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step = 0

    def step(self):
        self._step += 1
        t = self._step
        for i, pg in enumerate(self.optimizer.param_groups):
            base = self.base_lrs[i]
            if t <= self.warmup_steps and self.warmup_steps > 0:
                lr = base * t / self.warmup_steps
            else:
                tw = max(1, self.total_steps - self.warmup_steps)
                tt = min(t - self.warmup_steps, tw)
                cos = 0.5 * (1 + math.cos(math.pi * tt / tw))
                lr = self.min_lr + (base - self.min_lr) * cos
            pg["lr"] = lr

    @property
    def last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]
