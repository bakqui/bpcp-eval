import os

import pandas as pd
import torch
from torch.utils.data import Dataset

from .transforms import build_transforms


def _standardize(df, sbp_mean, dbp_mean, sbp_std, dbp_std, column_names=['sbp', 'dbp']):
    sbp_col, dbp_col = column_names
    df[sbp_col] = (df[sbp_col] - sbp_mean) / sbp_std
    df[dbp_col] = (df[dbp_col] - dbp_mean) / dbp_std
    return df


class PulseDBDataset(Dataset):
    def __init__(
        self,
        params,
        index_filename: str,
        sbp_mean=None,
        sbp_std=None,
        dbp_mean=None,
        dbp_std=None,
    ):
        self.params = params
        self.index_data_path = params.index_data_path
        self.waveform_data_path = params.waveform_data_path
        self.df = pd.read_parquet(os.path.join(self.index_data_path, index_filename))
        self.cal_col = params.cal_column
        self.case_id_col = params.case_id_column
        self.segment_id_col = params.segment_id_column
        self.sig_filename_col = params.sig_filename_column
        self.sbp_col = params.sbp_column
        self.dbp_col = params.dbp_column

        self.df = self.df.sort_values(
            by=[self.case_id_col, self.segment_id_col]
        ).reset_index(drop=True)

        self.sbp_mean = sbp_mean if sbp_mean is not None else self.df[self.sbp_col].mean()
        self.sbp_std = sbp_std if sbp_std is not None else self.df[self.sbp_col].std()
        self.dbp_mean = dbp_mean if dbp_mean is not None else self.df[self.dbp_col].mean()
        self.dbp_std = dbp_std if dbp_std is not None else self.df[self.dbp_col].std()

        # Standardize the BP values using the stored instance variables
        self.df = _standardize(
            self.df,
            sbp_mean=self.sbp_mean,
            dbp_mean=self.dbp_mean,
            sbp_std=self.sbp_std,
            dbp_std=self.dbp_std,
            column_names=[self.sbp_col, self.dbp_col],
        )

        self.df_group = {caseid: group for caseid, group in self.df.groupby(self.case_id_col)}
        self.transform = build_transforms(params)

    def get_ppg(self, file):
        """Loads one PPG pickle file."""
        path = os.path.join(self.params.waveform_data_path, file)
        ppg = pd.read_pickle(path)['PPG_Raw']
        if len(ppg.shape) == 1:
            ppg = ppg.reshape(1, -1)  # Reshape to (1, N) if it's a single channel
        ppg = self.transform(ppg)
        return ppg

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        caseid = row[self.case_id_col]
        file = row[self.sig_filename_col]
        sbp = row[self.sbp_col]
        dbp = row[self.dbp_col]

        global_idx = row.name
        cal_df = self.df_group[caseid]
        cal_row = cal_df.loc[:global_idx][cal_df.loc[:global_idx, self.cal_col] == 1].iloc[-1]
        cal_file = cal_row[self.sig_filename_col]
        cal_sbp = cal_row[self.sbp_col]
        cal_dbp = cal_row[self.dbp_col]

        cal_ppg = self.get_ppg(cal_file)
        ppg = self.get_ppg(file)
        data = {
            'ppg_cal': cal_ppg,
            'sbp_cal': torch.tensor(cal_sbp, dtype=torch.float32),
            'dbp_cal': torch.tensor(cal_dbp, dtype=torch.float32),
            'ppg_tar': ppg,
            'sbp_tar': torch.tensor(sbp, dtype=torch.float32),
            'dbp_tar': torch.tensor(dbp, dtype=torch.float32),
        }
        return data
