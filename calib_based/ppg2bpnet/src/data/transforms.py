# Copyright 2024 ST-MEM paper authors. <https://github.com/bakqui/ST-MEM>

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.signal import butter, resample, sosfiltfilt, square


__all__ = [
    'Resample',
    'RandomCrop',
    'CenterCrop',
    'MovingWindowCrop',
    'NCrop',
    'ButterworthFilter',
    'HighpassFilter',
    'LowpassFilter',
    'Standardize',
    'MinMaxScale',
    'YFlip',
    'RandomMask',
    'Cutout',
    'RandomShift',
    'SineNoise',
    'SquareNoise',
    'WhiteNoise',
    'RandomPartialSineNoise',
    'RandomPartialSquareNoise',
    'RandomPartialWhiteNoise',
    'RandomApply',
    'Compose',
    'ToTensor',
    'RandAugment',
    'build_transforms'
]

"""Preprocessing1
"""
class Resample:
    """Resample the input sequence.
    """
    def __init__(self,
                 target_length: Optional[int] = None,
                 target_fs: Optional[int] = None) -> None:
        self.target_length = target_length
        self.target_fs = target_fs

    def __call__(self, x: np.ndarray, fs: Optional[int] = None) -> np.ndarray:
        if fs and self.target_fs and fs != self.target_fs:
            x = resample(x, int(x.shape[1] * self.target_fs / fs), axis=1)
        elif self.target_length and x.shape[1] != self.target_length:
            x = resample(x, self.target_length, axis=1)
        return x

class RandomCrop:
    """Crop randomly the input sequence.
    """
    def __init__(self, crop_length: int) -> None:
        self.crop_length = crop_length

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.crop_length > x.shape[1]:
            raise ValueError(f"crop_length must be smaller than the length of x ({x.shape[1]}).")
        start_idx = np.random.randint(0, x.shape[1] - self.crop_length + 1)
        return x[:, start_idx:start_idx + self.crop_length]

class CenterCrop:
    """Crop the input sequence at the center.
    """
    def __init__(self, crop_length: int) -> None:
        self.crop_length = crop_length

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.crop_length > x.shape[1]:
            raise ValueError(f"crop_length must be smaller than the length of x ({x.shape[1]}).")
        start_idx = (x.shape[1] - self.crop_length) // 2
        return x[:, start_idx:start_idx + self.crop_length]

class MovingWindowCrop:
    """Crop the input sequence with a moving window.
    """
    def __init__(self, crop_length: int, crop_stride: int) -> None:
        self.crop_length = crop_length
        self.crop_stride = crop_stride

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.crop_length > x.shape[1]:
            raise ValueError(f"crop_length must be smaller than the length of x ({x.shape[1]}).")
        start_idx = np.arange(0, x.shape[1] - self.crop_length + 1, self.crop_stride)
        return np.stack([x[:, i:i + self.crop_length] for i in start_idx], axis=0)

class NCrop:
    """Crop the input sequence to N segments with equally spaced intervals.
    """
    def __init__(self, crop_length: int, num_segments: int) -> None:
        self.crop_length = crop_length
        self.num_segments = num_segments

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.crop_length > x.shape[1]:
            raise ValueError(f"crop_length must be smaller than the length of x ({x.shape[1]}).")
        start_idx = np.arange(start=0,
                              stop=x.shape[1] - self.crop_length + 1,
                              step=(x.shape[1] - self.crop_length) // (self.num_segments - 1))
        return np.stack([x[:, i:i + self.crop_length] for i in start_idx], axis=0)

class ButterworthFilter:
    """Apply SOS filter to the input sequence.
    """
    def __init__(self,
                 fs: int,
                 cutoff: float,
                 order: int = 5,
                 btype: str = 'highpass') -> None:
        self.sos = butter(order, cutoff, btype=btype, fs=fs, output='sos')

    def __call__(self, x):
        return sosfiltfilt(self.sos, x)

class HighpassFilter(ButterworthFilter):
    """Apply highpass filter to the input sequence.
    """
    def __init__(self, fs: int, cutoff: float, order: int = 5) -> None:
        super(HighpassFilter, self).__init__(fs, cutoff, order, btype='highpass')

class LowpassFilter(ButterworthFilter):
    """Apply lowpass filter to the input sequence.
    """
    def __init__(self, fs: int, cutoff: float, order: int = 5) -> None:
        super(LowpassFilter, self).__init__(fs, cutoff, order, btype='lowpass')

class Standardize:
    """Standardize the input sequence.
    """
    def __init__(self, axis: Union[int, Tuple[int, ...], List[int]] = (-1, -2)) -> None:
        if isinstance(axis, list):
            axis = tuple(axis)
        self.axis = axis

    def __call__(self, x: np.ndarray) -> np.ndarray:
        loc = np.mean(x, axis=self.axis, keepdims=True)
        scale = np.std(x, axis=self.axis, keepdims=True)
        # Set rst = 0 if std = 0
        return np.divide(x - loc, scale, out=np.zeros_like(x), where=scale != 0)

class MinMaxScale:
    def __init__(
        self,
        min_val: Optional[Any] = None,
        max_val: Optional[Any] = None,
    ) -> None:
        self.min_val = np.array(min_val) if min_val is not None else None
        self.max_val = np.array(max_val) if max_val is not None else None

    def __call__(self, y: np.ndarray) -> np.ndarray:
        min_val = self.min_val if self.min_val is not None else np.min(y)
        max_val = self.max_val if self.max_val is not None else np.max(y)
        return (y - min_val) / (max_val - min_val)

"""Augmentations
"""
class _BaseAugment:
    """Base class for augmentations.
    """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _set_level(self, level: int, max_level: int = 10, **kwargs) -> None:
        pass

class YFlip(_BaseAugment):
    """Flip the signal along the y-axis.
    """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return -x

"""Group 2: Signal manipulation
"""
class _Mask(_BaseAugment):
    """Base class for signal masking.
    """
    def __init__(self, mask_ratio: float = 0.3) -> None:
        self.mask_ratio = mask_ratio

    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _set_level(self, level: int, max_level: int = 10) -> None:
        self.mask_ratio = level / max_level * 0.3

class RandomMask(_Mask):
    """Randomly mask the input sequence.
    """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        rst = x.copy()
        count = np.random.randint(0, int(x.shape[-1] * self.mask_ratio))
        indices = np.random.choice(x.shape[-1], (1, count), replace=False)
        rst[:, indices] = 0
        return rst

class Cutout(_Mask):
    """Cutout the input sequence.
    """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        rst = x.copy()
        count = int(np.random.uniform(0, self.mask_ratio) * x.shape[-1])
        start_idx = np.random.randint(0, x.shape[-1] - count)
        rst[:, start_idx:start_idx + count] = 0
        return rst

class RandomShift(_Mask):
    """Randomly shift (left or right) the input sequence and pad zeros.
    """
    def __call__(self, x: np.ndarray) -> np.ndarray:
        rst = x.copy()
        direction = np.random.choice([-1, 1])
        sig_len = x.shape[-1]
        shift = int(np.random.uniform(0, self.mask_ratio) * sig_len)
        if direction == 1:
            rst[:, shift:] = rst[:, :sig_len - shift]
            rst[:, :shift] = 0
        else:
            rst[:, :sig_len - shift] = rst[:, shift:]
            rst[:, sig_len - shift:] = 0
        return rst

"""Group 3: Noise manipulation
"""
class _Noise(_BaseAugment):
    """Base class for noise manipulation.
    """
    def __init__(self, amplitude: float = 0.3, freq: float = 0.5) -> None:
        self.amplitude = amplitude
        self.freq = freq

    def _get_noise(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, x: np.ndarray) -> np.ndarray:
        noise = self._get_noise(x)
        return x + noise

    def _set_level(self, level: int, max_level: int = 10) -> None:
        level = level / max_level
        self.amplitude = level * 0.3
        self.freq = 0.5 / level

class SineNoise(_Noise):
    """Add sine noise to the input sequence.
    """
    def _get_noise(self, x: np.ndarray) -> np.ndarray:
        t = np.expand_dims(np.arange(x.shape[-1]) / x.shape[-1], axis=0)
        return self.amplitude * np.sin(2 * np.pi * t / self.freq)

class SquareNoise(_Noise):
    """Add square noise to the input sequence.
    """
    def _get_noise(self, x: np.ndarray) -> np.ndarray:
        t = np.expand_dims(np.arange(x.shape[-1]) / x.shape[-1], axis=0)
        return self.amplitude * square(2 * np.pi * t / self.freq)

class WhiteNoise(_Noise):
    """Add white noise to the input sequence.
    """
    def _get_noise(self, x: np.ndarray) -> np.ndarray:
        return self.amplitude * np.random.randn(*x.shape)

class _RandomPartialNoise(_Noise):
    """Base class for adding noise to the random part of the input sequence.
    """
    def __init__(self, amplitude: float = 0.3, freq: float = 0.5, ratio: float = 0.3) -> None:
        super(_RandomPartialNoise, self).__init__(amplitude, freq)
        self.ratio = ratio

    def _get_partial_noise(self, x: np.ndarray) -> np.ndarray:
        noise = self._get_noise(x)
        count = int(np.random.uniform(0, self.ratio) * x.shape[-1])
        start_idx = np.random.randint(0, x.shape[-1] - count)
        partial_noise = np.zeros_like(x)
        partial_noise[:, start_idx:start_idx + count] = noise[:, :count]
        return partial_noise

    def __call__(self, x: np.ndarray) -> np.ndarray:
        noise = self._get_partial_noise(x)
        return x + noise

    def _set_level(self, level: int, max_level: int = 10) -> None:
        super(_RandomPartialNoise, self)._set_level(level, max_level)
        self.ratio = level / max_level * 0.3

class RandomPartialSineNoise(_RandomPartialNoise, SineNoise):
    """Add sine noise to the random part of the input sequence.
    """

class RandomPartialSquareNoise(_RandomPartialNoise, SquareNoise):
    """Add square noise to the random part of the input sequence.
    """

class RandomPartialWhiteNoise(_RandomPartialNoise, WhiteNoise):
    """Add white noise to the random part of the input sequence.
    """

"""Etc
"""
class RandomApply:
    """Apply randomly the given transform.
    """
    def __init__(self, transform: _BaseAugment, prob: float = 0.5) -> None:
        self.transform = transform
        self.prob = prob

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.prob:
            x = self.transform(x)
        return x

class Compose:
    """Compose several transforms together.
    """
    def __init__(self, transforms: List[Any]) -> None:
        self.transforms = transforms

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            x = transform(x)
        return x

class ToTensor:
    """Convert ndarrays in sample to Tensors.
    """
    _DTYPES = {
        "float": torch.float32,
        "double": torch.float64,
        "int": torch.int32,
        "long": torch.int64,
    }

    def __init__(self, dtype: Union[str, torch.dtype] = torch.float32) -> None:
        if isinstance(dtype, str):
            assert dtype in self._DTYPES, f"Invalid dtype: {dtype}"
            dtype = self._DTYPES[dtype]
        self.dtype = dtype

    def __call__(self, x: Any) -> torch.Tensor:
        return torch.tensor(x, dtype=self.dtype)


"""Random augmentation
"""
class RandAugment:
    """RandAugment: Practical automated data augmentation with a reduced search space.
        ref: https://arxiv.org/abs/1909.13719
    """
    def __init__(self,
                 ops: list,
                 level: int = 10,
                 num_layers: int = 2,
                 prob: float = 0.5,
                 ) -> None:
        self.ops = []
        for op in ops:
            if hasattr(op, '_set_level'):
                op._set_level(level=level)
            self.ops.append(RandomApply(op, prob=prob))
        self.num_layers = num_layers
        self.prob = prob

    def __call__(self, x: np.ndarray) -> np.ndarray:
        ops = np.random.choice(self.ops, self.num_layers, replace=False)
        for op in ops:
            x = op(x)
        return x

def substitute_vars(value, variables):
    if isinstance(value, str):
        for k, v in variables.items():
            value = value.replace(f"${{{k}}}", str(v))
        try:
            return eval(value, {}, variables)
        except:
            return value
    elif isinstance(value, list):
        return [substitute_vars(v, variables) for v in value]
    elif isinstance(value, dict):
        return {k: substitute_vars(v, variables) for k, v in value.items()}
    else:
        return value

def build_transforms(cfg):
    variables = {
        "crop_second": cfg["crop_second"],
        "fs": cfg["fs"],
        "resample_fs": cfg["resample_fs"]
    }

    transforms = []
    for t_cfg in cfg["transform"]:
        name = t_cfg["name"]
        params = substitute_vars(t_cfg.get("params", {}), variables)

        # globals()에서 클래스 가져오기
        cls = globals()[name]
        transforms.append(cls(**params))

    # Compose도 globals()에서
    Compose_cls = globals()["Compose"]
    return Compose_cls(transforms)
