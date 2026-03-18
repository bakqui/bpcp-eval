import torch.optim as optim


# Adopted from https://github.com/facebookresearch/XLM
class AdamInverseSqrtWithWarmup(optim.Adam):
    """
    Decay the LR based on the inverse square root of the update number.
    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (`warmup-init-lr`) until the configured
    learning rate (`lr`). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.
    During warmup:
        lrs = torch.linspace(warmup_init_lr, lr, warmup_updates)
        lr = lrs[update_num]
    After warmup:
        lr = decay_factor / sqrt(update_num)
    where
        decay_factor = lr * sqrt(warmup_updates)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, warmup_updates=4000, warmup_init_lr=1e-7,
                 exp_factor=0.5):
        super().__init__(
            params,
            lr=warmup_init_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )

        # linearly warmup for the first warmup_updates
        self.warmup_updates = warmup_updates
        self.warmup_init_lr = warmup_init_lr
        warmup_end_lr = lr
        self.lr_step = (warmup_end_lr - warmup_init_lr) / warmup_updates

        # then, decay prop. to the inverse square root of the update number
        self.exp_factor = exp_factor
        self.decay_factor = warmup_end_lr * warmup_updates ** self.exp_factor

        # total number of updates
        for param_group in self.param_groups:
            param_group['num_updates'] = 0

    def get_lr_for_step(self, num_updates):
        if num_updates < self.warmup_updates:
            return self.warmup_init_lr + num_updates * self.lr_step
        else:
            return self.decay_factor * (num_updates ** -self.exp_factor)

    def step(self, closure=None):
        super().step(closure)
        for param_group in self.param_groups:
            param_group['num_updates'] += 1
            param_group['lr'] = self.get_lr_for_step(param_group['num_updates'])


def get_optimizer(model, params):

    parameters = model

    opt_name = params.optimizer
    learning_rate = params.learning_rate

    if opt_name == 'sgd':
        weight_decay = params.weight_decay
        momentum = params.momentum
        optimizer = optim.SGD(parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif opt_name == 'adam':
        weight_decay = params.weight_decay
        optimizer = optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif opt_name == 'adamw':
        weight_decay = params.weight_decay
        optimizer = optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    elif opt_name == 'adam_warmup':
        weight_decay = params.weight_decay
        warmup_updates = params.warmup_updates
        optimizer = AdamInverseSqrtWithWarmup(parameters, lr=learning_rate, weight_decay=weight_decay, warmup_updates=warmup_updates)
    else:
        raise Exception('@@set proper optimizer')

    return optimizer
