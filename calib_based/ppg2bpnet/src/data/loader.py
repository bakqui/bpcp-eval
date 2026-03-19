from typing import Literal

from torch.utils.data import DataLoader

from .dataset import PulseDBDataset


def fetch_dataloader(
    params,
    mode: Literal['train', 'valid', 'test'],
    sbp_mean=None,
    sbp_std=None,
    dbp_mean=None,
    dbp_std=None,
):
    if mode == 'train':
        index_filename = params.train_index_filename
        dataset = PulseDBDataset(params, index_filename=index_filename)
    elif mode in ['valid', 'test']:
        index_filename = params.get(f'{mode}_index_filename')
        dataset = PulseDBDataset(
            params,
            index_filename=index_filename,
            sbp_mean=sbp_mean,
            sbp_std=sbp_std,
            dbp_mean=dbp_mean,
            dbp_std=dbp_std,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=params.batch_size,
        shuffle=(mode == 'train'),
        pin_memory=True,
        persistent_workers=(params.num_workers > 0),
        num_workers=params.num_workers,
    )
    return dataloader
