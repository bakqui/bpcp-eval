"""Calibration-free PPG-based BP estimation training and evaluation
"""
import argparse
import datetime
import json
import math
import os
import random
import sys
import time
from typing import Dict, Optional, Tuple, Union

import mergedeep
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import yaml
from torch.utils.tensorboard import SummaryWriter

import models
import utils.misc as misc
from utils.losses import build_loss_fn
from utils.lr_sched import adjust_learning_rate
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.optimizer import build_optimizer
from utils.perf_metrics import build_metric_fn, is_best_metric
from utils.dataset import build_dataset, get_dataloader


def parse() -> dict:
    parser = argparse.ArgumentParser(
        description="Calibration-free PPG-based BP estimation training and evaluation",
    )

    parser.add_argument(
        '-f',
        '--config_path',
        dest='config_path',
        required=True,
        type=str,
        metavar='FILE',
        help='YAML config file path',
    )
    parser.add_argument(
        '-o',
        '--override_yaml',
        dest='override_yaml',
        default=None,
        type=str,
        metavar='FILE',
        help='YAML config file path to override',
    )
    parser.add_argument(
        '--output_dir',
        default="",
        type=str,
        metavar='DIR',
        help='path where to save',
    )
    parser.add_argument(
        '--exp_name',
        default="",
        type=str,
        help='experiment name',
    )
    parser.add_argument(
        '--resume',
        default="",
        type=str,
        metavar='PATH',
        help='resume from checkpoint',
    )
    parser.add_argument(
        '--start_epoch',
        default=0,
        type=int,
        metavar='N',
        help='start epoch',
    )
    parser.add_argument(
        '--encoder_path',
        default="",
        type=str,
        metavar='PATH',
        help='pretrained encoder checkpoint',
    )

    args = parser.parse_args()
    with open(os.path.realpath(args.config_path), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.override_yaml:
        with open(os.path.realpath(args.override_yaml), 'r') as f:
            override_config = yaml.load(f, Loader=yaml.FullLoader)
        config = mergedeep.merge(config, override_config)

    for k, v in vars(args).items():
        if v:
            config[k] = v

    return config


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler: NativeScaler,
    train_config: dict,
    use_amp: bool = True,
    log_writer: Optional[SummaryWriter] = None,
) -> Dict[str, float]:

    print_freq = train_config.get('print_freq', 20)
    accum_iter = train_config.get('accum_iter', 1)
    max_norm = train_config.get('max_norm', None)

    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        'lr',
        misc.SmoothedValue(window_size=1, fmt='{value:.6f}')
    )
    header = 'Epoch: [{}]'.format(epoch)
    if log_writer is not None:
        print(f'log_dir: {log_writer.log_dir}')

    optimizer.zero_grad()

    for data_iter_step, samples in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            adjust_learning_rate(
                optimizer,
                data_iter_step / len(data_loader) + epoch,
                train_config,
            )

        ppgs = samples['input'].to(device, non_blocking=True)
        labels = samples['label'].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(ppgs)
            loss = criterion(outputs, labels)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        max_lr = 0.
        for group in optimizer.param_groups:
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int(
                (epoch + data_iter_step / len(data_loader)) * 1000
            )
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    metric_fn: torch.nn.Module,
    use_amp=True,
) -> Tuple[Dict[str, float], Dict[str, float], torch.Tensor]:
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    outputs_total = []
    for samples in metric_logger.log_every(data_loader, 10, header):
        ppgs = samples['input'].to(device, non_blocking=True)
        labels = samples['label'].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(ppgs)
            loss = criterion(outputs, labels)

        outputs = misc.concat_all_gather(outputs)
        labels = misc.concat_all_gather(labels)
        metric_fn.update(outputs, labels)
        metric_logger.meters['loss'].update(loss.item(), n=ppgs.size(0))
        outputs_total.append(outputs.cpu())

    metric_logger.synchronize_between_processes()
    valid_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    metrics = metric_fn.compute()
    if isinstance(metrics, dict):  # MetricCollection
        metrics = {k: v.item() for k, v in metrics.items()}
    else:
        metrics = {metric_fn.__class__.__name__: metrics.item()}
    metric_str = "  ".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
    metric_str = f"{metric_str} loss: {metric_logger.loss.global_avg:.3f}"
    print(f"* {metric_str}")
    outputs = torch.cat(outputs_total, dim=0)
    metric_fn.reset()
    return valid_stats, metrics, outputs


def run(config):
    # 1) Set environmental variables and random seeds
    # - set random seeds for reproducibility
    # - set logging directory
    print(f'job dir: {os.path.dirname(os.path.realpath(__file__))}')
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))

    device = torch.device(config['device'])

    seed = config.get('seed', 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    cudnn.benchmark = False

    if misc.is_main_process() and config['output_dir']:
        output_dir = os.path.join(config['output_dir'], config['exp_name'])
        os.makedirs(output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=output_dir)
    else:
        output_dir = None
        log_writer = None

    # 2) Load data for eval task and create dataloaders
    # - load train/val/test data and labels based on eval config
    # - create dataloaders for train/val/test sets
    dataset_train = build_dataset(config['dataset'], split='train')
    dataset_valid = build_dataset(config['dataset'], split='valid')

    data_loader_train = get_dataloader(
        dataset_train,
        is_distributed=config['ddp']['distributed'],
        mode='train',
        **config['dataloader'],
    )
    data_loader_valid = get_dataloader(
        dataset_valid,
        is_distributed=config['ddp']['distributed'],
        dist_eval=config['train']['dist_eval'],
        mode='eval',
        **config['dataloader'],
    )

    # 3) Load pretrained PulsePPG model
    # - import model architecture based on eval config
    # - create model instance
    model_name = config['model_name']
    model = models.__dict__[model_name](**config['model'])

    if config['mode'] != "scratch":
        checkpoint = torch.load(config['encoder_path'], map_location='cpu')
        print(f"Load pre-trained checkpoint from: {config['encoder_path']}")
        if 'model' in checkpoint:
            checkpoint_model = checkpoint['model']
        elif 'net' in checkpoint:
            checkpoint_model = checkpoint['net']
        else:
            raise ValueError("Checkpoint does not contain 'model' or 'net' keys")
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Remove key {k} from pre-trained checkpoint")
                del checkpoint_model[k]
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        assert set(msg.missing_keys).issubset({'head.weight', 'head.bias'})

    if config['mode'] == "linear":
        for _, p in model.named_parameters():
            p.requires_grad = False
        for _, p in model.head.named_parameters():
            p.requires_grad = True

    model.to(device)
    model_without_ddp = model

    if config['ddp']['distributed']:
        if config['ddp']['sync_bn']:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config['ddp']['gpu']])
        model_without_ddp = model.module

    eff_batch_size = config['dataloader']['batch_size'] * config['train']['accum_iter'] * misc.get_world_size()
    if config['train']['lr'] is None:
        config['train']['lr'] = config['train']['blr'] * eff_batch_size / 256
    print(f"base lr: {config['train']['lr'] * 256 / eff_batch_size}")
    print(f"actual lr: {config['train']['lr']}")
    print(f"accumulate grad iterations: {config['train']['accum_iter']}")
    print(f"effective batch size: {eff_batch_size}")
    if config['mode'] == "linear":
        optimizer = build_optimizer(config['train'], model_without_ddp.head)
    else:
        optimizer = build_optimizer(config['train'], model_without_ddp)

    criterion = build_loss_fn(config['loss'])
    loss_scaler = NativeScaler()

    best_loss = float('inf')
    metric_fn, best_metrics = build_metric_fn(config['metric'])
    metric_fn.to(device)

    misc.load_model(config, model_without_ddp, optimizer, loss_scaler)

    # Start training
    use_amp = config.get('use_amp', True)
    print(f"Start training for {config['train']['epochs']} epochs")
    start_time = time.time()
    for epoch in range(config['start_epoch'], config['train']['epochs']):
        if config['ddp']['distributed']:
            data_loader_train.sampler.set_epoch(epoch)
        if config['mode'] == "linear":
            model.eval()
        else:
            model.train()
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            config['train'],
            use_amp,
            log_writer,
        )

        valid_stats, metrics, _ = evaluate(
            model,
            criterion,
            data_loader_valid,
            device,
            metric_fn,
            use_amp=use_amp,
        )
        curr_loss = valid_stats['loss']
        if output_dir and curr_loss < best_loss:
            best_loss = curr_loss
            misc.save_model(
                config,
                os.path.join(output_dir, 'best-loss.pth'),
                epoch,
                model_without_ddp,
                optimizer,
                loss_scaler,
                metrics={'loss': curr_loss, **metrics},
            )
        for metric_name, metric_class in metric_fn.items():
            curr_metric = metrics[metric_name]
            print(f"{metric_name}: {curr_metric:.3f}")
            if output_dir and is_best_metric(
                metric_class,
                best_metrics[metric_name],
                curr_metric,
            ):
                best_metrics[metric_name] = curr_metric
                misc.save_model(
                    config,
                    os.path.join(output_dir, f'best-{metric_name}.pth'),
                    epoch,
                    model_without_ddp,
                    optimizer,
                    loss_scaler,
                    metrics={'loss': valid_stats['loss'], **metrics},
                )
            print(f"Best {metric_name}: {best_metrics[metric_name]:.3f}")

        if log_writer is not None:
            log_writer.add_scalar('perf/valid_loss', curr_loss, epoch)
            for metric_name, curr_metric in metrics.items():
                log_writer.add_scalar(f'perf/{metric_name}', curr_metric, epoch)

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'valid_{k}': v for k, v in valid_stats.items()},
            **metrics,
            'epoch': epoch,
        }

        if output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(output_dir, 'log.txt'), mode='a', encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + '\n')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")

    if log_writer is not None:
        log_writer.close()

    # Start test
    if config.get('test', False) and misc.is_main_process():
        # turn off ddp for testing
        if config['ddp']['distributed']:
            torch.distributed.destroy_process_group()

        dataset_test = build_dataset(config['dataset'], split='test')
        data_loader_test = get_dataloader(
            dataset_test,
            mode='eval',
            **config['dataloader'],
        )

        model = models.__dict__[model_name](**config['model'])

        target_metric = config.get('test', {}).get('target_metric', 'loss')
        checkpoint_path = os.path.join(output_dir, f'best-{target_metric}.pth')
        assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"Load trained checkpoint from: {checkpoint_path}")
        checkpoint_model = checkpoint['model']
        model.load_state_dict(checkpoint_model)

        model.to(device)

        test_stats, metrics, outputs = evaluate(
            model,
            criterion,
            data_loader_test,
            device,
            metric_fn,
            use_amp=use_amp,
        )
        print(f"Test loss: {test_stats['loss']:.3f}")
        for metric_name, metric in metrics.items():
            print(f"{metric_name}: {metric:.3f}")

        if output_dir:
            metrics['loss'] = test_stats['loss']
            metric_df = pd.DataFrame([metrics])
            metric_df.to_csv(
                os.path.join(output_dir, 'test_metrics.csv'),
                index=False,
                float_format='%.4f',
            )
            np.save(os.path.join(output_dir, 'test_outputs.npy'), outputs.numpy())

        print('Done!')


if __name__ == "__main__":
    args = parse()
    run(args)
