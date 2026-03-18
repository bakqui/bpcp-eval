import argparse
import os
import sys
import warnings

import pandas as pd
import torch
import yaml

from src.data.loader import fetch_dataloader
from src.trainer import Trainer
from src.evaluator import Evaluator
from src.utils import init_exp, AttributeDict
from src.models import build_model


warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


def read_config(config_path):
    """Read configuration from a file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} does not exist.")
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = AttributeDict(config)
    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibration-based PPG-to-BP estimation with PPG2BP-Net"
    )
    parser.add_argument(
        '--config_path',
        type=str,
        default='./configs/dummy.yaml',
        help='Path to the configuration file',
    )
    args = parser.parse_args()
    params = read_config(args.config_path)
    return params


def main(params):

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(params.GPU)

    # Initialize experiments
    logger, dump_path = init_exp(params)
    model = build_model(params, logger)

    train_dataloader = fetch_dataloader(params, mode="train")

    train_dataset = train_dataloader.dataset
    sbp_mean = train_dataset.sbp_mean
    sbp_std = train_dataset.sbp_std
    dbp_mean = train_dataset.dbp_mean
    dbp_std = train_dataset.dbp_std

    valid_dataloder = fetch_dataloader(
        params,
        mode='valid',
        sbp_mean=sbp_mean,
        sbp_std=sbp_std,
        dbp_mean=dbp_mean,
        dbp_std=dbp_std
    )
    test_dataloader = fetch_dataloader(
        params,
        mode='test',
        sbp_mean=sbp_mean,
        sbp_std=sbp_std,
        dbp_mean=dbp_mean,
        dbp_std=dbp_std
    )

    trainer = Trainer(params, model, logger, train_dataloader)
    evaluator = Evaluator(params, model, valid_dataloder, test_dataloader, logger)

    if params.eval_only:
        valid_mae, test_mae, test_pred, test_label = evaluator.evaluate_loader()
        save_data = {
            'test_pred': test_pred,
            'test_label': test_label,
        }
        torch.save(save_data, os.path.join(dump_path, 'best_model.pth'))
        sys.exit()

    best_valid_mae = 1e12
    best_test_mae = 1e12

    # Number of required iterations for each evaluation
    num_iters = params.num_iters

    for epoch in range(params.n_epochs):
        logger.info(f"=======================Epoch: {epoch}=======================")
        total_losses = []
        while trainer.n_iter < num_iters:
            loss = trainer.train_step()
            total_losses.append(loss)
        trainer.n_iter = 0
        if epoch < params.skip_val:
            continue
        logger.info(f"==================End of Epoch: {epoch}====================")

        valid_mae, test_mae, test_pred, test_label = evaluator.evaluate_loader()

        if valid_mae < best_valid_mae:
            best_valid_mae = valid_mae

            best_test_on_valid = test_mae
            logger.info(f"Saving the best model in {dump_path}")

            save_data = {
                'params': vars(params),
                'model': model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
                'test_pred': test_pred,
                'test_label': test_label,
            }
            torch.save(save_data, os.path.join(dump_path, 'best_model.pth'))

        if test_mae < best_test_mae:
            best_test_mae = test_mae
            logger.info(f"Saving the best model in {dump_path}")
            save_data = {
                'params': vars(params),
                'model': model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
                'test_pred': test_pred,
                'test_label': test_label,
            }
            torch.save(save_data, os.path.join(dump_path, 'test_best_model.pth'))

        save_data = {
            'model': model.state_dict(),
            'optimizer': trainer.optimizer.state_dict(),
        }

        torch.save(save_data, os.path.join(dump_path, 'current_model.pth'))
        logger.info(
            f"Current best valid mae: {best_valid_mae}, log saved in: {dump_path}"
        )
        logger.info(
            f"Current best test mae based on valid: {best_test_on_valid}, log saved in: {dump_path}"
        )

if __name__ ==  "__main__":
    params = parse_args()
    main(params)
