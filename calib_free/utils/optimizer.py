from typing import Dict, Optional, Sequence

import torch
import torch.optim as optim


def add_weight_decay(
    model: torch.nn.Module,
    weight_decay: float = 1e-5,
    skip_list: Sequence = (),
):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay},
    ]


def build_optimizer(
    train_config: dict,
    model: Optional[torch.nn.Module] = None,
    param_groups: Optional[Sequence[Dict[str, torch.Tensor]]] = None,
) -> optim.Optimizer:
    opt_name = train_config['optimizer']
    lr = train_config['lr']
    weight_decay = train_config['weight_decay']
    kwargs = train_config.get('optimizer_kwargs', {})
    if param_groups is None:
        assert model is not None, "Model must be provided if param_groups is None"
        param_groups = add_weight_decay(model, weight_decay=weight_decay)
    if opt_name == "sgd":
        momentum = kwargs.get('momentum', 0)
        return optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif opt_name == "adamw":
        betas = kwargs.get('betas', (0.9, 0.999))
        if isinstance(betas, list):
            betas = tuple(betas)
        eps = kwargs.get('eps', 1e-8)
        return optim.AdamW(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")
