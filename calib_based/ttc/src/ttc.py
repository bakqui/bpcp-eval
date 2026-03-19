
import os
import shutil
import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data.dual_buffer import DualBuffer


def shrinkage_loss(pred, target, p=1.0, loc=0.1, speed=10.0):
    diff = torch.abs(pred - target - loc)
    num = torch.pow(diff, p)
    den = num + (speed ** p)
    return (num / den).mean()


def test_time_calibration_subjectwise(
    model,
    test_ds,
    cfg,
    ckpt_state: Optional[dict] = None,
):
    """
    Subject-wise TTC for PulseDB
    - For each subject, reset model to base weights
    - For each sample in temporal order:
        * decide if it's 'labeled' (calibration sample)
        * update dual buffers (B_unlabel, B_label)
        * construct batch from buffers with ratio r
        * update model using BOTH:
            - reconstruction MSE loss on unlabeled batch
            - shrinkage loss on labeled batch (test-phase params)
        * then predict current sample x and store pred/gt
    - Saves:
        - pred.npy: (N,2) predictions for all samples
        - gt.npy:   (N,2) ground truth for all samples
        - is_labeled.npy: (N,) boolean mask (True for calibration samples)
    """
    device = cfg.get("device")

    model.to(device)
    if ckpt_state is None:
        base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        base_state = ckpt_state

    # TTC hyperparameters - mini-batch
    batch_size = cfg.get("ttc_batch_size", 32)
    labeled_ratio = cfg.get("sampling_ratio", 0.25)  # e.g., 0.25 → 8 labeled, 24 unlabeled
    buffer_sizes = cfg.get("buffer_sizes", {})
    dual_buffer_max_unlabeled = buffer_sizes.get("unlabeled", 64)
    dual_buffer_max_labeled = buffer_sizes.get("labeled", 8)

    # TTC hyperparameters - optimizer
    lr = cfg.get("ttc_lr", 1e-3)
    momentum = cfg.get("ttc_momentum", 0.9)
    wd = cfg.get("ttc_weight_decay", 1e-6)
    n_update_per_batch = cfg.get("n_update_per_batch", 5)

    # TTC hyperparameters - patchify
    patch_size = cfg.get("patch_size", 30)
    patch_stride = cfg.get("patch_stride", 15)

    # TTC hyperparameters - shrinkage loss
    p_test = cfg.get("shrink_p_ttc", 2.5)
    loc_test = cfg.get("shrink_loc_ttc", 0.0)
    speed_test = cfg.get("shrink_speed_ttc", 10.0)

    # TTC hyperparameters - loss weights
    ssl_w = cfg.get("ssl_weight_ttc", 1.0)
    sl_w = cfg.get("sl_weight_ttc",  1.0)

    # Save directory
    savedir = cfg.get("savedir", "")
    if not savedir:
        warnings.warn("savedir is not set, using default directory (./exp_dump)")
        savedir = "./exp_dump"
    os.makedirs(savedir, exist_ok=True)

    # Initialize prediction, ground truth, and labeled mask
    N = len(test_ds)
    preds = torch.zeros((N, 2), dtype=torch.float32, device=device)
    gts = torch.zeros((N, 2), dtype=torch.float32, device=device)
    is_labeled = torch.zeros((N,), dtype=torch.bool, device=device)

    # ----- subject-wise loop -----
    for caseid, group in tqdm(
        test_ds.df_group.items(),
        desc="Subject-wise TTC",
        leave=True,
        position=0,
    ):
        idxs = group.index.to_list()

        # reset model for this subject
        model.load_state_dict(base_state, strict=True)

        opt = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=wd,
        )
        buffer = DualBuffer(
            max_unlabeled=dual_buffer_max_unlabeled,
            max_labeled=dual_buffer_max_labeled,
        )

        # for temporary saving local results
        local_savedir = os.path.join(savedir, "temp", str(caseid))
        os.makedirs(local_savedir, exist_ok=True)
        local_preds = torch.zeros((len(idxs), 2), dtype=torch.float32, device=device)
        local_gts = torch.zeros((len(idxs), 2), dtype=torch.float32, device=device)
        local_is_labeled = torch.zeros((len(idxs),), dtype=torch.bool, device=device)

        # ---- per-sample stream ----
        for local_idx, global_idx in tqdm(
            enumerate(idxs),
            total=len(idxs),
            desc=f"Case {caseid}",
            leave=False,
            position=1,
        ):
            model.train()

            # get full ground-truth (we use it for eval, but only some steps for calibration)
            item = test_ds[global_idx]
            x = item["x"].unsqueeze(0).to(device)
            y_full = item["y"].unsqueeze(0).to(device)

            cpd = item["cpd"].item()

            use_label = (cpd == 1)
            if local_idx == 0:
                use_label = True

            if use_label:
                buffer.add(x, y_full)
            else:
                buffer.add(x, None)

            # construct batch from current buffers (even if small)
            unl, xs, ys = buffer.sample(batch_size, labeled_ratio)

            x_unl = torch.cat(unl, dim=0) if unl else None
            x_lab = torch.cat(xs, dim=0) if xs else None
            y_lab = torch.cat(ys, dim=0) if ys else None

            # ----- adaptation step (gradients ON) -----
            if (x_unl is not None) or (x_lab is not None):
                for _ in range(n_update_per_batch):
                    opt.zero_grad()
                    total_loss = 0.0

                    # SSL branch: reconstruction on unlabeled
                    if x_unl is not None:
                        recon_u, _ = model(x_unl)
                        tgt_u = model.unfold_1d(x_unl, patch_size, patch_stride)
                        loss_ssl = F.mse_loss(recon_u, tgt_u)
                        total_loss = total_loss + ssl_w * loss_ssl

                    # SL branch: shrinkage on labeled
                    if (x_lab is not None) and (y_lab is not None):
                        _, bp_lab = model(x_lab)
                        loss_sl = shrinkage_loss(
                            bp_lab,
                            y_lab,
                            p=p_test,
                            loc=loc_test,
                            speed=speed_test,
                        )
                        total_loss = total_loss + sl_w * loss_sl

                    if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad:
                        total_loss.backward()
                        opt.step()

            # ----- predict current sample AFTER adaptation -----
            with torch.no_grad():
                model.eval()
                _, bp_pred = model(x)  # (1,2)

            preds[global_idx] = bp_pred.squeeze(0)
            gts[global_idx] = y_full.squeeze(0)
            is_labeled[global_idx] = use_label

            local_preds[local_idx] = bp_pred.squeeze(0)
            local_gts[local_idx] = y_full.squeeze(0)
            local_is_labeled[local_idx] = use_label

        # temporary save local results
        np.save(os.path.join(local_savedir, "pred.npy"), local_preds.cpu().numpy())
        np.save(os.path.join(local_savedir, "gt.npy"), local_gts.cpu().numpy())
        np.save(os.path.join(local_savedir, "is_labeled.npy"), local_is_labeled.cpu().numpy())

    # save final results
    np.save(os.path.join(savedir, "pred.npy"), preds.cpu().numpy())
    np.save(os.path.join(savedir, "gt.npy"), gts.cpu().numpy())
    np.save(os.path.join(savedir, "is_labeled.npy"), is_labeled.cpu().numpy())
    print("Saved pred.npy, gt.npy, is_labeled.npy (subject-wise TTC)")

    # if successfully completed, delete temporary files
    if os.path.exists(os.path.join(savedir, "temp")):
        shutil.rmtree(os.path.join(savedir, "temp"))
