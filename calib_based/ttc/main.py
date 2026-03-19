import argparse
import os

import torch
import yaml

from src.data.dataset import PulseDBDataset
from src.models.ttc_model import TTCModel
from src.ttc import test_time_calibration_subjectwise
from src.utils import set_seed, AttributeDict


def read_config(config_path):
    """Read configuration from a file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} does not exist.")
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = AttributeDict(config)
    return config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--config_path",
        type=str,
        default="./configs/config.yaml",
    )
    args = parser.parse_args()
    params = read_config(args.config_path)
    return params


def main(params):
    set_seed(params.seed)
    device = params.device

    model = TTCModel(
        in_ch=params.input_channels,
        embed_dim=params.embed_dim,
        patch_size=params.patch_size,
        patch_stride=params.patch_stride,
        depth=params.depth,
        num_heads=params.num_heads,
    ).to(device)

    train_ds = PulseDBDataset(
        params,
        index_filename=params.train_index_filename
    )

    if not params.get("eval_only", False):
        from src.train import train_ssl_init, train_joint

        # ----- Phase 1: SSL init -----
        print("=== Phase 1: SSL initialization ===")
        train_ssl_init(model, train_ds, params)

        # ----- Phase 2: SL+SSL fine-tune -----
        print("=== Phase 2: SL+SSL fine-tuning ===")
        train_joint(model, train_ds, params)

        savedir = params.get("savedir", None)
        if savedir is None:
            savedir = "./exp_dump"
        ckpt_path = os.path.join(savedir, "ttc_phase_II.ckpt")
    else:
        ckpt_path = params.get("ckpt_path", None)
        if ckpt_path is None:
            raise ValueError("Missing ckpt_path in config. Set path to trained TTC checkpoint.")

    if any(
        [
            params.get(key, None) is None
            for key in ["sbp_mean", "sbp_std", "dbp_mean", "dbp_std"]
        ]
    ):
        sbp_mean = train_ds.sbp_mean
        sbp_std = train_ds.sbp_std
        dbp_mean = train_ds.dbp_mean
        dbp_std = train_ds.dbp_std
    else:
        sbp_mean = params.sbp_mean
        sbp_std = params.sbp_std
        dbp_mean = params.dbp_mean
        dbp_std = params.dbp_std

    test_ds = PulseDBDataset(
        params,
        index_filename=params.test_index_filename,
        sbp_mean=sbp_mean,
        dbp_mean=dbp_mean,
        sbp_std=sbp_std,
        dbp_std=dbp_std,
    )

    # ----- Test-time calibration -----
    print("=== Phase 3: Test-time calibration ===")
    # Simulate a sequential stream; replace with your real deployment stream
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])

    test_time_calibration_subjectwise(
        model=model,
        test_ds=test_ds,
        cfg=params,
    )

if __name__ == "__main__":
    params = parse_args()
    main(params)
