import torch
import torch.nn as nn


# Shared 1D-CNN feature extractor
class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, output_dim=8):
        super().__init__()
        self.cnn_net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            
            nn.AdaptiveAvgPool1d(1),
            nn.Dropout(0.3)
        )
        self.fcl = nn.Sequential(
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(), 
        )

    def forward(self, x):
        x = self.cnn_net(x).squeeze(-1)
        x = self.fcl(x)
        return x


# MLP for calibration SBP/DBP values
class CalibrationMLP(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=8, output_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


# Full PPG2BP-Net
class PPG2BPNet(nn.Module):
    def __init__(self, params, hidden_dim=8):
        super().__init__()

        fc_dim = hidden_dim*5

        self.cnn = CNNFeatureExtractor(output_dim=hidden_dim)
        self.calib_sbp = CalibrationMLP(output_dim=hidden_dim)
        self.calib_dbp = CalibrationMLP(output_dim=hidden_dim)
        
        # Final fusion layer
        self.fc = nn.Sequential(
            nn.Linear(fc_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        target_ppg = x['ppg_tar']
        calib_ppg = x['ppg_cal']

        calib_sbp = x['sbp_cal'].unsqueeze(-1)
        calib_dbp = x['dbp_cal'].unsqueeze(-1)

        f_target = self.cnn(target_ppg)
        f_calib  = self.cnn(calib_ppg)
        f_diff   = torch.abs(f_calib - f_target)

        f_sbp = self.calib_sbp(calib_sbp)
        f_dbp = self.calib_dbp(calib_dbp)

        # Concatenate features
        fused = torch.cat([f_dbp, f_sbp, f_calib, f_diff, f_target], dim=-1)
        out = self.fc(fused)
        return out
