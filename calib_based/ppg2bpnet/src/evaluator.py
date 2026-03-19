import numpy as np
import torch
from tqdm import tqdm

from .metrics import metrics
from .utils import to_device, make_labels


class Evaluator(object):
    def __init__(self, params, model, valid_dataloder, test_dataloder, logger):

        self.model = model

        if params.GPU is not None:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.dataloader = [valid_dataloder, test_dataloder]
        self.ds = {}
        self.trial_ds = {}

        self.params = params
        self.metrics = metrics
        self.logger = logger

        self.status = {
            'sbp_mae': [],
            'dbp_mae': [],
            'sbp_mse': [],
            'dbp_mse': [],
            'sbp_rmse': [],
            'dbp_rmse': [],
            'sbp_r2score': [],
            'dbp_r2score': [],
            'sbp_mean_error': [],
            'dbp_mean_error': [],
            'sbp_std_error': [],
            'dbp_std_error': [],
        }
        self.status_stat = {
            'mean_mae': [],
            'std_mae': [],
        }

        self.loss_fn = torch.nn.MSELoss().to(self.device)

    def print_status(self, mode):
        self.logger.info("\n".join(mode+"_%s: %s" % (k, str(v[0])) \
                    for k, v in sorted(dict(self.status).items())))

        for k in self.status.keys():
            if type(self.status[k]) is list:
                del self.status[k][:]

    def denormalize(self, pred, label, dataloader):
        scale_sbp_mean = dataloader.dataset.sbp_mean
        scale_sbp_std = dataloader.dataset.sbp_std
        scale_dbp_mean = dataloader.dataset.dbp_mean
        scale_dbp_std = dataloader.dataset.dbp_std
        pred[:, 0] = pred[:, 0] * scale_sbp_std + scale_sbp_mean
        pred[:, 1] = pred[:, 1] * scale_dbp_std + scale_dbp_mean
        label[:, 0] = label[:, 0] * scale_sbp_std + scale_sbp_mean
        label[:, 1] = label[:, 1] * scale_dbp_std + scale_dbp_mean
        return pred, label

    def evaluate_loader(self, test_only=False):

        # set model to evaluation mode
        self.model.eval()

        if test_only:
            split = ['test']
            dataloader = self.dataloader[1:2]
        else:
            split = ['valid', 'test']
            dataloader = self.dataloader

        for i, dataloader in enumerate(dataloader):

            total_losses = []
            label, pred = [], []

            # compute metrics over the dataset
            for batch in tqdm(dataloader):

                batch = to_device(batch, self.device)

                with torch.no_grad():
                    y_pred = self.model(batch)
                    label_data = make_labels(batch)
                    loss = self.loss_fn(y_pred, label_data)

                total_losses.append(loss.item())
                pred.append(y_pred.cpu().detach())
                label.append(label_data.cpu())

            pred = np.asarray(torch.cat(pred).cpu().detach().numpy())
            label = np.asarray(torch.cat(label).cpu())

            pred, label = self.denormalize(pred, label, dataloader)

            sbp_mae = self.metrics['mae'](pred[:, 0], label[:, 0])
            dbp_mae = self.metrics['mae'](pred[:, 1], label[:, 1])

            sbp_mse = self.metrics['mse'](pred[:, 0], label[:, 0])
            dbp_mse = self.metrics['mse'](pred[:, 1], label[:, 1])

            sbp_rmse = self.metrics['rmse'](pred[:, 0], label[:, 0])
            dbp_rmse = self.metrics['rmse'](pred[:, 1], label[:, 1])
            sbp_r2score = self.metrics['r2score'](pred[:, 0], label[:, 0])
            dbp_r2score = self.metrics['r2score'](pred[:, 1], label[:, 1])
            sbp_mean_error = self.metrics['mean_error'](pred[:, 0], label[:, 0])
            dbp_mean_error = self.metrics['mean_error'](pred[:, 1], label[:, 1])
            sbp_std_error = self.metrics['std_error'](pred[:, 0], label[:, 0])
            dbp_std_error = self.metrics['std_error'](pred[:, 1], label[:, 1])

            self.status['sbp_mae'].append(sbp_mae)
            self.status['dbp_mae'].append(dbp_mae)
            self.status['sbp_mse'].append(sbp_mse)
            self.status['dbp_mse'].append(dbp_mse)
            self.status['sbp_rmse'].append(sbp_rmse)
            self.status['dbp_rmse'].append(dbp_rmse)
            self.status['sbp_r2score'].append(sbp_r2score)
            self.status['dbp_r2score'].append(dbp_r2score)
            self.status['sbp_mean_error'].append(sbp_mean_error)
            self.status['dbp_mean_error'].append(dbp_mean_error)
            self.status['sbp_std_error'].append(sbp_std_error)
            self.status['dbp_std_error'].append(dbp_std_error)

            if split[i] == 'valid':
                valid_mae = sbp_mae

            if split[i] == 'test':
                test_mae = sbp_mae
                test_pred = pred
                test_label = label

            self.print_status(split[i])

        if test_only:
            return test_mae, test_pred, test_label
        else:
            return valid_mae, test_mae, test_pred, test_label
