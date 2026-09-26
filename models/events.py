"""Event-file readers and spike binning (replaces slayerSNN.io).

Follows slayerSNN's conventions so the input spike tensors match the ones the
networks were trained on:
- polarity is shifted so its minimum in the sample is 0;
- timestamps (ms) are binned with numpy's round-half-to-even: bin = round(t / Ts);
- events outside the tensor are dropped; a bin holding any event is 1/Ts ("OR").
"""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Events:
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    t_ms: np.ndarray

    def __post_init__(self):
        self.x = _as_int(self.x)
        self.y = _as_int(self.y)
        self.p = _as_int(self.p)
        self.t_ms = np.asarray(self.t_ms)
        if len(self.p):
            self.p = self.p - self.p.min()

    def to_spike_tensor(self, shape, sampling_time):
        """Binary spike tensor of shape (polarity, height, width, time bins)."""
        tensor = torch.zeros(shape)
        t = np.round(self.t_ms / sampling_time).astype(int)
        x, y, p = self.x, self.y, self.p
        valid = ((p >= 0) & (p < shape[0]) & (y >= 0) & (y < shape[1])
                 & (x >= 0) & (x < shape[2]) & (t >= 0) & (t < shape[3]))
        index = tuple(torch.from_numpy(a[valid].astype(np.int64)) for a in (p, y, x, t))
        tensor[index] = 1 / sampling_time
        return tensor


def _as_int(values):
    values = np.asarray(values)
    return values if np.issubdtype(values.dtype, np.integer) else values.astype(int)


def read_nmnist_bin(path):
    """N-MNIST / N-Caltech101 binary format: 5 bytes per event holding
    x (8 bits), y (8 bits), polarity (1 bit) and a 23-bit timestamp in us."""
    raw = np.fromfile(path, dtype=np.uint8).astype(np.int64)
    x, y = raw[0::5], raw[1::5]
    p = raw[2::5] >> 7
    t_us = ((raw[2::5] << 16) | (raw[3::5] << 8) | raw[4::5]) & 0x7FFFFF
    return Events(x, y, p, t_us / 1000)


def read_npy_events(path, time_unit=1e-3):
    """Numpy event array with columns x, y, p, t; t * time_unit is seconds."""
    events = np.load(path)
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError(f"{path}: expected an (n_events, 4) x/y/p/t array")
    return Events(events[:, 0], events[:, 1], events[:, 2], events[:, 3] * time_unit * 1e3)
