import os
import warnings

import torch
import torch.nn.functional as F
import torch_optimizer as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .schedulers import WarmupCosineScheduler


def shrinkage_loss(pred, target, p=1.0, loc=0.1, speed=10.0):
    diff = torch.abs(pred - target - loc)
    num = torch.pow(diff, p)
    den = num + (speed ** p)
    return (num / den).mean()


def _make_opt_and_sched(
    model,
    cfg,
    mode,
    steps_per_epoch,
):
    # Training hyperparameters
    if mode == "ssl":
        base_lr = cfg.get("lr_ssl_init", 2.516152722130999e-3)
        wd = cfg.get("wd_ssl_init", 1.2775200014634934e-4)
        warm = cfg.get("warmup_pct_init", 0.1)
        total_epochs = cfg.get("epochs_ssl_init", 10)
    else:
        base_lr = cfg.get("lr_finetune", 5.70373305632515e-4)
        wd = cfg.get("wd_finetune", 1.5399401149371588e-3)
        warm = cfg.get("warmup_pct_ft", 0.1)
        total_epochs = cfg.get("epochs_finetune", 10)
    betas = cfg.get("betas", (0.9, 0.999))
    if isinstance(betas, list):
        betas = tuple(betas)
    eps = cfg.get("eps", 1.0e-6)
    min_lr_factor = cfg.get("min_lr_factor", 0.01)

    if cfg.get("use_lamb", False):
        opt = optim.Lamb(
            model.parameters(),
            lr=base_lr,
            betas=betas,
            eps=eps,
            weight_decay=wd,
        )
    else:
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=base_lr,
            betas=betas,
            eps=eps,
            weight_decay=wd,
        )

    total_steps = max(1, steps_per_epoch * total_epochs)
    warmup_steps = int(warm * total_steps)
    min_lr = base_lr * min_lr_factor
    sched = WarmupCosineScheduler(
        opt,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr=min_lr,
    )

    return opt, sched


def train_ssl_init(model, dataset, cfg):
    device = cfg.get("device", "cuda")

    # Training hyperparameters
    batch_size = cfg.get("batch_size", 256)
    total_epochs = cfg.get("epochs_ssl_init", 10)
    patch_size = cfg.get("patch_size", 30)
    patch_stride = cfg.get("patch_stride", 15)
    grad_clip_norm = cfg.get("grad_clip_norm", 1.0)

    savedir = cfg.get("savedir", "")
    if not savedir:
        warnings.warn("savedir is not set, using default directory (./exp_dump)")
        savedir = "./exp_dump"
    os.makedirs(savedir, exist_ok=True)

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.get("workers", 16),
        pin_memory=True,
        drop_last=True,
    )
    steps_per_epoch = len(dl)
    opt, sched = _make_opt_and_sched(
        model,
        cfg,
        mode="ssl",
        steps_per_epoch=steps_per_epoch,
    )
    model.train()

    for epoch in range(total_epochs):
        pbar = tqdm(dl, desc=f"SSL init {epoch + 1}/{total_epochs}")
        for b in pbar:
            x = b["x"].to(device)
            recon, _ = model(x)
            tgt = model.unfold_1d(x, patch_size, patch_stride)
            loss = F.mse_loss(recon, tgt)

            opt.zero_grad()
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip_norm,
                )
            opt.step()
            sched.step()
            pbar.set_postfix(ssl=float(loss), lr=sched.last_lr[0])

    torch.save(
        {"state_dict": model.state_dict(), "config": dict(cfg)},
        os.path.join(savedir, "ttc_phase_I.ckpt"),
    )
    print(f"Saved: {os.path.join(savedir, 'ttc_phase_I.ckpt')}")


def train_joint(model, dataset, cfg):
    device = cfg.get("device", "cuda")

    # Training hyperparameters
    batch_size = cfg.get("batch_size", 256)
    total_epochs = cfg.get("epochs_finetune", 10)
    patch_size = cfg.get("patch_size", 30)
    patch_stride = cfg.get("patch_stride", 15)
    shrink_p = cfg.get("shrink_p_ft", 1.0)
    shrink_loc = cfg.get("shrink_loc_ft", 0.1)
    shrink_speed = cfg.get("shrink_speed_ft", 10.0)
    ssl_weight = cfg.get("ssl_weight", 1.0)
    sl_weight = cfg.get("sl_weight", 1.0)
    grad_clip_norm = cfg.get("grad_clip_norm", 1.0)

    savedir = cfg.get("savedir", "")
    if not savedir:
        warnings.warn("savedir is not set, using default directory (./exp_dump)")
        savedir = "./exp_dump"
    os.makedirs(savedir, exist_ok=True)

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.get("workers", 16),
        pin_memory=True,
        drop_last=True,
    )
    steps_per_epoch = len(dl)
    opt, sched = _make_opt_and_sched(
        model,
        cfg,
        mode="ft",
        steps_per_epoch=steps_per_epoch,
    )
    model.train()

    for epoch in range(total_epochs):
        pbar = tqdm(dl, desc=f"Finetune {epoch + 1}/{total_epochs}")
        for b in pbar:
            x = b["x"].to(device)
            y = b["y"].to(device)
            recon, bp = model(x)
            tgt = model.unfold_1d(x, patch_size, patch_stride)

            loss_ssl = F.mse_loss(recon, tgt)
            loss_sl  = shrinkage_loss(
                bp,
                y,
                p=shrink_p,
                loc=shrink_loc,
                speed=shrink_speed,
            )
            loss = ssl_weight * loss_ssl + sl_weight * loss_sl

            opt.zero_grad()
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    grad_clip_norm,
                )
            opt.step()
            sched.step()
            pbar.set_postfix(sl=float(loss_sl), ssl=float(loss_ssl), lr=sched.last_lr[0])

    torch.save(
        {"state_dict": model.state_dict(), "config": dict(cfg)},
        os.path.join(savedir, "ttc_phase_II.ckpt"),
    )
    print(f"Saved: {os.path.join(savedir, 'ttc_phase_II.ckpt')}")
