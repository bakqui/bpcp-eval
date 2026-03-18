import os
import pickle as pkl
import random
from collections import defaultdict
from typing import Dict, Iterable, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, Sampler

import utils.transforms as T
from utils.transforms import get_transforms_from_config
from utils.misc import get_rank, get_world_size


class PulseDBDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        fpaths: Iterable,
        labels: Optional[Iterable] = None,
        fs_list: Optional[Iterable] = None,
        case_ids: Optional[Iterable] = None,
        segment_ids: Optional[Iterable] = None,
        is_unstable: Optional[Iterable] = None,
        init_fs_list: Optional[Iterable] = None,
        init_sbps: Optional[Iterable] = None,
        init_dbps: Optional[Iterable] = None,
        target_fs: Optional[int] = None,
        target_length: Optional[int] = None,
        transform: Optional[object] = None,
        label_transform: Optional[object] = None,
    ):
        self.root_dir = root_dir
        self.fpaths = [os.path.join(root_dir, fpath) for fpath in fpaths]
        self.labels = labels
        self.fs_list = fs_list
        self.case_ids = case_ids
        self.segment_ids = segment_ids
        self.is_unstable = is_unstable
        self.init_fpaths = [os.path.join(root_dir, fpath) for fpath in init_fs_list] if init_fs_list is not None else None
        self.init_sbps = init_sbps
        self.init_dbps = init_dbps
        self._check_dataset()
        if case_ids is not None:
            self.case_mapper = LabelEncoder().fit(case_ids)

        if fs_list is not None:
            self.resample = T.Resample(target_fs=target_fs)
        elif target_length is not None:
            self.resample = T.Resample(target_length=target_length)

        self.transform = transform
        self.label_transform = label_transform

    def _check_dataset(self):
        fpath_not_pkl = [f for f in self.fpaths if not f.endswith('.pkl')]
        assert len(fpath_not_pkl) == 0, \
            f"Some files do not have .pkl extension. (e.g. {fpath_not_pkl[0]}...)"
        assert all([os.path.exists(fpath) for fpath in self.fpaths]), \
            f"Some files do not exist. (e.g. {self.fpaths[0]}...)"
        if self.labels is not None:
            assert len(self.fpaths) == len(self.labels), \
                "The number of filenames and labels are different."
        if self.fs_list is not None:
            assert len(self.fpaths) == len(self.fs_list), \
                "The number of filenames and fs_list are different."
        if self.case_ids is not None:
            assert len(self.fpaths) == len(self.case_ids), \
                "The number of filenames and case_ids are different."
        if self.segment_ids is not None:
            assert len(self.fpaths) == len(self.segment_ids), \
                "The number of filenames and segment_ids are different."
        if self.is_unstable is not None:
            assert len(self.fpaths) == len(self.is_unstable), \
                "The number of filenames and is_unstable flags are different."
        init_fpath_not_pkl = [f for f in self.init_fpaths if not f.endswith('.pkl')] if self.init_fpaths is not None else []
        assert len(init_fpath_not_pkl) == 0, \
            f"Some calibration PPG files do not have .pkl extension. (e.g. {init_fpath_not_pkl[0]}...)"
        assert all([os.path.exists(fpath) for fpath in self.init_fpaths]) if self.init_fpaths is not None else True, \
            f"Some calibration PPG files do not exist. (e.g. {self.init_fpaths[0]}...)"
        if self.init_fpaths is not None:
            assert len(self.fpaths) == len(self.init_fpaths), \
                "The number of filenames and init_fpaths are different."
        if self.init_sbps is not None:
            assert len(self.fpaths) == len(self.init_sbps), \
                "The number of filenames and init_sbps are different."
        if self.init_dbps is not None:
            assert len(self.fpaths) == len(self.init_dbps), \
                "The number of filenames and init_dbps are different."

    def __len__(self):
        return len(self.fpaths)

    def _process_signal(self, x: np.ndarray, fs: Optional[int] = None) -> np.ndarray:
        # Resample signal
        if self.resample is not None:
            x = self.resample(x, fs)

        if self.transform is not None:
            x = self.transform(x)

        return x

    def _load_signal(self, fpath: str) -> np.ndarray:
        with open(fpath, 'rb') as f:
            data = pkl.load(f)
        assert isinstance(data, dict), "Data should be a dictionary."
        assert 'PPG_Raw' in data, "PPG_Raw key not found in the data dictionary."

        ppg = data['PPG_Raw']
        ppg = ppg[np.newaxis, :]  # (1, siglen)

        return ppg

    def __getitem__(self, idx) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        fpath = self.fpaths[idx]
        fs = self.fs_list[idx] if self.fs_list is not None else None

        ppg = self._load_signal(fpath)
        ppg = self._process_signal(ppg, fs)

        sample = {"input": ppg}

        if self.labels is not None:
            label = self.labels[idx]
            if self.label_transform is not None:
                label = self.label_transform(label)
            sample["label"] = label

        # Add optional metadata (string type)
        if self.case_ids is not None:
            case_id = self.case_ids[idx]
            sample["case_id"] = torch.tensor(
                self.case_mapper.transform([case_id])[0], dtype=torch.long
            )
        if self.segment_ids is not None:
            segment_id = self.segment_ids[idx]
            sample["segment_id"] = torch.tensor(segment_id, dtype=torch.long)

        if self.is_unstable is not None:
            is_unstable = self.is_unstable[idx]
            sample["is_unstable"] = torch.tensor(is_unstable, dtype=torch.float32)

        if self.init_fpaths is not None:
            init_fpath = self.init_fpaths[idx]
            init_ppg = self._load_signal(init_fpath)
            init_ppg = self._process_signal(init_ppg, fs)
            sample["init_input"] = init_ppg

        if self.init_sbps is not None:
            init_sbp = self.init_sbps[idx]
            sample["init_sbp"] = torch.tensor([init_sbp], dtype=torch.float32)

        if self.init_dbps is not None:
            init_dbp = self.init_dbps[idx]
            sample["init_dbp"] = torch.tensor([init_dbp], dtype=torch.float32)

        return sample


class PulseDBSegmentBalancedSampler(Sampler):
    def __init__(
        self,
        case_ids: Iterable,
        segment_ids: Iterable,
        samples_per_epoch: int = None,
    ):
        """
        Args:
            df: DataFrame with columns [Subject ID, Case ID, Segment ID, PPG filename]
            samples_per_epoch: Number of samples per epoch (default: len(df))
            seed: Random seed for reproducibility
        """
        self.samples_per_epoch = samples_per_epoch or len(case_ids)

        # Build (case, segment) -> indices mapping
        # Note: Segment ID is NOT unique across cases
        self.segment_to_indices = defaultdict(list)
        for idx, (case_id, segment_id) in enumerate(zip(case_ids, segment_ids)):
            # Use (Subject ID, Case ID, Segment ID) as unique key
            segment_key = (case_id, segment_id)
            self.segment_to_indices[segment_key].append(idx)

        self.segments = list(self.segment_to_indices.keys())

    def __iter__(self):
        indices = []

        for _ in range(self.samples_per_epoch):
            # Step 1: Uniformly sample a segment (across all cases)
            segment_key = random.choice(self.segments)

            # Step 2: Uniformly sample within the segment
            segment_indices = self.segment_to_indices[segment_key]
            idx = random.choice(segment_indices)

            indices.append(idx)
        
        return iter(indices)
    
    def __len__(self):
        return self.samples_per_epoch


def build_dataset(dataset_cfg: dict, split: str) -> Dataset:
    """Load train, validation, or test dataloaders.
    """
    waveform_dir = os.path.realpath(dataset_cfg["waveform_dir"])
    index_dir = os.path.realpath(dataset_cfg["index_dir"])
    df_name = dataset_cfg.get(f"{split}_index", None)
    assert df_name is not None, f"{split}_index is not defined in the config."
    if os.path.splitext(df_name)[1] == ".csv":
        df = pd.read_csv(os.path.join(index_dir, df_name))
    elif os.path.splitext(df_name)[1] == ".pkl":
        df = pd.read_pickle(os.path.join(index_dir, df_name))
    elif os.path.splitext(df_name)[1] == ".parquet":
        df = pd.read_parquet(os.path.join(index_dir, df_name))
    else:
        raise ValueError(f"Unknown file extension: {os.path.splitext(df_name)[1]}")

    fpath_col = dataset_cfg.get("fpath_col", "FILE_PATH")
    label_col = dataset_cfg.get("label_col", None)
    fs_col = dataset_cfg.get("fs_col", None)
    case_id_col = dataset_cfg.get("case_id_col", None)
    segment_id_col = dataset_cfg.get("segment_id_col", None)
    is_unstable_col = dataset_cfg.get("is_unstable_col", None)
    init_fpath_col = dataset_cfg.get("init_fpath_col", None)
    init_sbp_col = dataset_cfg.get("init_sbp_col", None)
    init_dbp_col = dataset_cfg.get("init_dbp_col", None)
    target_fs = dataset_cfg.get("target_fs", None)
    target_length = dataset_cfg.get("target_length", None)

    fpaths = df[fpath_col].tolist()
    labels = df[label_col].astype(float).values if label_col is not None else None
    fs_list = df[fs_col].astype(int).tolist() if fs_col is not None else None
    case_ids = df[case_id_col].tolist() if case_id_col is not None else None
    segment_ids = df[segment_id_col].tolist() if segment_id_col is not None else None
    is_unstable = df[is_unstable_col].astype(bool).tolist() if is_unstable_col is not None else None
    init_fpaths = df[init_fpath_col].tolist() if init_fpath_col is not None else None
    init_sbps = df[init_sbp_col].astype(float).values if init_sbp_col is not None else None
    init_dbps = df[init_dbp_col].astype(float).values if init_dbp_col is not None else None

    transforms = []
    if split == "train":
        transforms_cfg = dataset_cfg.get("train_transforms", [])
        if init_sbps is not None:
            # normalize by initial SBP during training
            init_sbps_mean, init_sbps_std = init_sbps.mean(), init_sbps.std()
            init_sbps = np.divide(
                init_sbps - init_sbps_mean,
                init_sbps_std,
                out=np.zeros_like(init_sbps),
                where=init_sbps_std != 0,
            )
        if init_dbps is not None:
            # normalize by initial DBP during training
            init_dbps_mean, init_dbps_std = init_dbps.mean(), init_dbps.std()
            init_dbps = np.divide(
                init_dbps - init_dbps_mean,
                init_dbps_std,
                out=np.zeros_like(init_dbps),
                where=init_dbps_std != 0,
            )
    else:
        transforms_cfg = dataset_cfg.get("eval_transforms", [])
        if init_sbps is not None:
            # normalize by initial SBP during evaluation
            init_sbps_mean, init_sbps_std = dataset_cfg.get("init_sbp_mean", 0), dataset_cfg.get("init_sbp_std", 1)
            init_sbps_mean, init_sbps_std = np.array(init_sbps_mean), np.array(init_sbps_std)
            init_sbps = np.divide(
                init_sbps - init_sbps_mean,
                init_sbps_std,
                out=np.zeros_like(init_sbps),
                where=init_sbps_std != 0,
            )
        if init_dbps is not None:
            # normalize by initial DBP during evaluation
            init_dbps_mean, init_dbps_std = dataset_cfg.get("init_dbp_mean", 0), dataset_cfg.get("init_dbp_std", 1)
            init_dbps_mean, init_dbps_std = np.array(init_dbps_mean), np.array(init_dbps_std)
            init_dbps = np.divide(
                init_dbps - init_dbps_mean,
                init_dbps_std,
                out=np.zeros_like(init_dbps),
                where=init_dbps_std != 0,
            )
    transforms += get_transforms_from_config(transforms_cfg)
    transform = T.Compose(transforms + [T.ToTensor()])

    if labels is not None:
        label_transform_cfg = dataset_cfg.get("label_transforms", [])
        label_transforms = get_transforms_from_config(label_transform_cfg)
        label_transform = T.Compose(
            label_transforms + [T.ToTensor(dtype=dataset_cfg["label_dtype"])]
        )
    else:
        label_transform = None

    dataset = PulseDBDataset(
        waveform_dir,
        fpaths=fpaths,
        labels=labels,
        fs_list=fs_list,
        case_ids=case_ids,
        segment_ids=segment_ids,
        is_unstable=is_unstable,
        init_fs_list=init_fpaths,
        init_sbps=init_sbps,
        init_dbps=init_dbps,
        target_fs=target_fs,
        target_length=target_length,
        transform=transform,
        label_transform=label_transform,
    )

    return dataset


def get_dataloader(
    dataset: Dataset,
    is_distributed: bool = False,
    dist_eval: bool = False,
    mode: Literal["train", "eval"] = "train",
    segment_balanced: bool = False,
    samples_per_epoch: Optional[int] = None,
    **kwargs
):
    is_train = mode == "train"
    if is_distributed and (is_train or dist_eval):
        num_tasks = get_world_size()
        global_rank = get_rank()
        if not is_train and len(dataset) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')
        # shuffle=True to reduce monitor bias even if it is for validation.
        # https://github.com/facebookresearch/mae/blob/main/main_finetune.py#L189
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=num_tasks,
            rank=global_rank,
            shuffle=True
        )
    elif is_train:
        if segment_balanced:
            assert hasattr(dataset, 'case_ids') and hasattr(dataset, 'segment_ids'), \
                "Dataset must have case_ids and segment_ids for segment balanced sampling."
            sampler = PulseDBSegmentBalancedSampler(
                case_ids=dataset.case_ids,
                segment_ids=dataset.segment_ids,
                samples_per_epoch=samples_per_epoch,
            )
        else:
            sampler = torch.utils.data.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    return DataLoader(
        dataset,
        sampler=sampler,
        drop_last=is_train,
        **kwargs
    )
