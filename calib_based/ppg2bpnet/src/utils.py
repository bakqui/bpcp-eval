import logging
import os
import random
import string
import subprocess
import sys
import time
from datetime import timedelta

import numpy as np
import torch


class LogFormatter():
    '''
    Ported from https://github.com/facebookresearch/XLM/blob/cd281d32612d145c6742b4d3f048f80df8669c30/xlm/logger.py#L13
    '''
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (
            record.levelname,
            time.strftime('%x %X'),
            timedelta(seconds=elapsed_seconds)
        )
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''


class AttributeDict(dict):
    def __getattr__(self, key):
        try:
            value = self[key]
            # auto-convert nested dicts to DotDict
            if isinstance(value, dict) and not isinstance(value, AttributeDict):
                value = AttributeDict(value)
                self[key] = value
            return value
        except KeyError:
            raise AttributeError(f"No such attribute: {key}")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]


def create_logger(dump_path):
    '''
    Creating a logger
    '''

    logger = logging.getLogger()

    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(filename=os.path.join(dump_path, "log.txt"))

    formatter = LogFormatter()

    stream_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.DEBUG)

    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def check_params(config):
    '''
    Checking parameters and
    printing the error message if it is not valid
    '''
    assert isinstance(config.learning_rate, float), \
        'learning_rate should be float value.'
    assert isinstance(config.n_epochs, int), \
        'n_epochs should be int value.'
    assert isinstance(config.batch_size, int), \
        'batch_size should be int value.'
    assert len(config.optimizer) > 0, \
        'optimizer must not be None'
    assert isinstance(config.warmup_updates, int), \
        'warmup_updates should be int value.'
    assert isinstance(config.weight_decay, float), \
        'weight_decay should be float value.'
    assert len(config.data_path) > 0, \
        'data_path must not be None'
    assert len(config.dump_path) > 0, \
        'dump_path must not be None'
    assert len(config.exp_name) > 0, \
        'exp_name must not be None'
    return


def init_exp(params):
    dump_path = create_dump_path(params)
    logger = create_logger(dump_path)
    set_seed(params.seed)
    logger.info("===============Initializing logger===============")
    logger.info("\n".join("%s: %s" % (k, str(v))
                          for k, v in sorted(vars(params).items())))
    logger.info("The experiment logs will be stored in %s" % dump_path)
    logger.info("python {}".format(sys.argv[0]))
    return logger, dump_path


def get_random_exp_trial_name():
    '''
    Create random exp_trial_name
    '''
    exp_trial_name = ''.join(random.choice(string.ascii_letters) for i in range(8))
    return exp_trial_name


def create_dump_path(params):
    '''
    Creating dump_path with the
    random exp trial_names
    '''
    dump_path = os.path.join(params.dump_path, params.exp_name)
    if not os.path.isdir(dump_path):
        subprocess.Popen("mkdir -p %s" % dump_path, shell=True).wait()

    if params.exp_trial_name is None:
        trial_names = os.listdir(dump_path)
        while True:
            exp_trial_name = get_random_exp_trial_name()
            if exp_trial_name not in trial_names:
                break
    else:
        exp_trial_name = params.exp_trial_name

    dump_path = os.path.join(dump_path, exp_trial_name)
    if not os.path.isdir(dump_path):
        subprocess.Popen("mkdir -p %s" % dump_path, shell=True).wait()

    return dump_path


def set_seed(seed):
    # Set the random seed for reproducible experiments
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def to_device(args, device):
    return {k:None if v is None else v.to(device) for k,v in args.items()}


def make_labels(batch: dict, keys=("sbp_tar", "dbp_tar")) -> torch.Tensor:
    return torch.stack([batch[k] for k in keys], dim=-1)
