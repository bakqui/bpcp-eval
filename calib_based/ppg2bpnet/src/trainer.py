import os
import numpy as np

import torch

from .optimizer import get_optimizer
from .utils import to_device, make_labels


class Trainer:
    def __init__(self, params, model, logger, dataloader):
        self.params = params
        self.logger = logger

        if params.GPU is not None:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = model.to(self.device)
        self.loss_fn = torch.nn.MSELoss().to(self.device)
        self.set_optimizers()
        self.scaler = torch.cuda.amp.GradScaler()

        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self.n_iter = 0
        self.n_total_iter = 0

        self.status = {
            params.model_name: [],
        }

    def print_status(self):
        if self.n_total_iter % 10 != 0:
            return

        lr = self.optimizer.param_groups[0]['lr']

        model_state = ' || '.join([
            '{}: {:7.4f}'.format(k, np.mean(v)) for k, v in self.status.items()
            if type(v) is list and len(v) > 0
        ])

        print_state = f"Iter: {self.n_total_iter} || lr: {lr:.4e} || "

        self.logger.info(print_state + model_state)

        for k in self.status.keys():
            if type(self.status[k]) is list:
                del self.status[k][:]

    def set_optimizers(self):
        self.optimizer = get_optimizer(self.model.parameters(), self.params)

        if os.path.isfile(self.params.reload_path) and self.params.reload_optimizer:
            reloaded_model = torch.load(self.params.reload_path)
            self.optimizer.load_state_dict(reloaded_model['optimizer'])
            self.logger.info("========Reloaded optimizer from {}".format(self.params.reload_path))

    def train_step(self):
        self.model.train()

        try:
            train_batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            train_batch = next(self.iterator)
        train_batch = to_device(train_batch, self.device)

        self.optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.params.use_amp):
            y_pred = self.model(train_batch)

        label_bp = make_labels(train_batch)
        label_bp = label_bp.to(y_pred.dtype)

        sbp_gt = label_bp[:, 0]
        dbp_gt = label_bp[:, 1]
        sbp_pred = y_pred[:, 0]
        dbp_pred = y_pred[:, 1]
        sbp_loss = self.loss_fn(sbp_pred, sbp_gt)
        dbp_loss = self.loss_fn(dbp_pred, dbp_gt)
        loss = sbp_loss + dbp_loss

        if self.params.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        
        self.status[self.params.model_name].append(loss.item())

        self.n_iter += 1
        self.n_total_iter += 1
        self.print_status()

        return loss.item()
