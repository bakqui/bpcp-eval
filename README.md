# bpcp-eval

This repository contains the implementation used in our paper titled "Change Point-Aware Evaluation and Re-Calibration of PPG-Based Blood Pressure Estimation".

## Table of Contents
- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Run Experiments](#run-experiments)
  - [Calibration-Free Estimation](#calibration-free-estimation)
  - [Calibration-Based Estimation](#calibration-based-estimation)

## Overview

<div align="center">
  <img src="assets/example.png" alt="Framework overview" width="90%">
</div>

Non-invasive continuous blood pressure (BP) estimation from photoplethysmography (PPG) is promising, but standard aggregate metrics can hide large errors during rapid BP fluctuations. In this work, we introduce a change point-aware evaluation framework that detects abrupt distribution shifts in BP trajectories and measures model performance specifically around those unstable periods. We show that both calibration-free and periodically calibrated approaches can degrade substantially near change points, and we further propose targeted re-calibration triggered by detected change points to improve robustness without changing model architecture.

<div align="center">
  <img src="assets/overview.png" alt="Framework overview" width="80%">
</div>

## Data Preparation

Expected data layout:

```text
data/pulsedb/
|-- waveform/
|   `-- <source>/<case_id>/<case_id>_<segment_id>.pkl
`-- index/
    |-- train.parquet
    |-- valid.parquet
    `-- test.parquet
```

### Waveform segment format

Each waveform file is a pickled dictionary containing `PPG_Raw`, 1D array of length `1250` (10 seconds at 125 Hz)

Index files are expected to include file path and label columns (configured in each YAML file), e.g. `filepath`, `sbp`, `dbp`, `caseid`, and segment/calibration-related columns.

## Run Experiments

### Calibration-Free Estimation

Run SBP experiment:

```bash
cd calib_free
bash run.sh -f configs/sbp.yaml
```

Run DBP experiment:

```bash
cd calib_free
bash run.sh -f configs/dbp.yaml
```

### Calibration-Based Estimation

Run calibration-based baseline (`PPG2BPNet`):

```bash
cd calib_based/ppg2bpnet
python main.py --config_path configs/config.yaml
```

Run test-time calibration module (`TTC`):

```bash
cd calib_based/ttc
python main.py -f configs/config.yaml
```

Before running, update each config file with your local dataset paths and index column names.

