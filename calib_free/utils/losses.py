import torch.nn as nn


def build_loss_fn(config: dict) -> nn.Module:
    loss_name = config['name']

    if loss_name == "mse":
        loss_fn = nn.MSELoss()
    elif loss_name == "l1":
        loss_fn = nn.L1Loss()
    else:
        raise ValueError(f"Invalid loss name: {loss_name}")
    return loss_fn
