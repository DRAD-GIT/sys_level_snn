"""Plain-PyTorch SRM spiking layers, replacing the slayerSNN (SLAYER) framework.

Implements inference with the same numerical behaviour as slayerSNN's
``spikeLayer`` (``snn.layer``), which the pretrained checkpoints were trained
with. No CUDA extension: runs on CPU and GPU. Only the forward pass is
provided (no surrogate gradients), so this is for evaluation, not training.

Tensors are ``[batch, channels, height, width, time]``, as in SLAYER.

Neuron model (SRM with alpha kernels), for each neuron and time step t:
  psp:    u[t] = Ts * sum_i srmKernel[i] * input[t - i]
  spike:  s[t] = 1/Ts if u[t] >= theta, else 0; each spike adds refKernel to
          u[t], u[t+1], ... (the refractory response starts at the spike step).

The kernels are regenerated from the neuron parameters, and are also stored
in the checkpoints as ``slayer.srmKernel`` / ``slayer.refKernel``.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F


def alpha_kernel(tau, t_sample, ts, mult=1.0, epsilon=0.01):
    """Sampled alpha kernel mult * t/tau * exp(1 - t/tau), truncated once it
    has decayed below epsilon after its peak (SLAYER's convention)."""
    values = []
    for t in np.arange(0, t_sample, ts):
        value = mult * t / tau * math.exp(1 - t / tau)
        if abs(value) < epsilon and t > tau:
            break
        values.append(value)
    return torch.tensor(values, dtype=torch.float32)


def _triple(value, last):
    """int or (h, w) -> (h, w, last), for 2-D ops applied per time step."""
    if isinstance(value, int):
        return (value, value, last)
    if len(value) == 2:
        return (value[0], value[1], last)
    raise ValueError(f"expected int or pair, got {value!r}")


class Conv(torch.nn.Conv3d):
    """2-D convolution applied independently at every time step."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, weight_scale=1):
        super().__init__(in_channels, out_channels, _triple(kernel_size, 1),
                         _triple(stride, 1), _triple(padding, 0), _triple(dilation, 1),
                         groups, bias=False)
        if weight_scale != 1:  # only affects the initialization used for training
            self.weight = torch.nn.Parameter(weight_scale * self.weight)


class Dense(torch.nn.Conv3d):
    """Fully connected layer per time step. in_features is an int, or a
    (width, height[, channels]) tuple whose kernel spans the whole input."""

    def __init__(self, in_features, out_features, weight_scale=1):
        if isinstance(in_features, int):
            kernel, in_channels = (1, 1, 1), in_features
        elif len(in_features) in (2, 3):
            kernel = (in_features[1], in_features[0], 1)
            in_channels = in_features[2] if len(in_features) == 3 else 1
        else:
            raise ValueError(f"in_features must be int or 2/3-tuple, got {in_features!r}")
        super().__init__(in_channels, out_features, kernel, bias=False)
        if weight_scale != 1:
            self.weight = torch.nn.Parameter(weight_scale * self.weight)


class Pool(torch.nn.Conv3d):
    """SLAYER's pooling: a per-channel sum over each window, scaled by a fixed
    weight of 1.1 * theta, whose output feeds a spiking neuron."""

    def __init__(self, theta, kernel_size, stride=None, padding=0, dilation=1):
        kernel = _triple(kernel_size, 1)
        super().__init__(1, 1, kernel, kernel if stride is None else _triple(stride, 1),
                         _triple(padding, 0), _triple(dilation, 1), bias=False)
        self.weight = torch.nn.Parameter(torch.full_like(self.weight, 1.1 * theta),
                                         requires_grad=False)

    def forward(self, x):
        # SLAYER pads a non-divisible height/width with (size % kernel) zero
        # rows/columns (not up to the next multiple), then pools all channels
        # as one tall image. Kept as-is so outputs match the trained networks.
        kh, kw = self.weight.shape[2], self.weight.shape[3]
        if x.shape[2] % kh:
            x = F.pad(x, (0, 0, 0, 0, 0, x.shape[2] % kh))
        if x.shape[3] % kw:
            x = F.pad(x, (0, 0, 0, x.shape[3] % kw))
        n, c, h, w, t = x.shape
        out = F.conv3d(x.reshape(n, 1, c * h, w, t), self.weight, None,
                       self.stride, self.padding, self.dilation)
        return out.reshape(n, c, -1, out.shape[3], out.shape[4])


class Dropout(torch.nn.Dropout3d):
    """Dropout that drops a neuron for all time steps (identity in eval mode)."""

    def forward(self, x):
        shape = x.shape
        return F.dropout3d(x.reshape(shape[0], -1, 1, 1, shape[-1]),
                           self.p, self.training, self.inplace).reshape(shape)


class SRMLayer(torch.nn.Module):
    """Drop-in replacement for ``slayerSNN.layer(neuron, simulation)``.

    Holds the neuron kernels and provides psp/spike plus the conv, dense,
    pool and dropout factories, with SLAYER's method names so the model
    definitions are unchanged.
    """

    def __init__(self, neuron, simulation):
        super().__init__()
        self.neuron = dict(neuron)
        self.simulation = dict(simulation)
        if self.neuron.get("type", "SRMALPHA") != "SRMALPHA":
            raise ValueError(f"Unsupported neuron type {self.neuron['type']!r}")
        self.theta = float(self.neuron["theta"])
        self.ts = float(self.simulation["Ts"])
        t_sample = self.simulation["tSample"]
        self.register_buffer("srmKernel", alpha_kernel(
            self.neuron["tauSr"], t_sample, self.ts))
        self.register_buffer("refKernel", alpha_kernel(
            self.neuron["tauRef"], t_sample, self.ts,
            mult=-self.neuron["scaleRef"] * self.theta))

    # Layer factories (same signatures and default weight scales as SLAYER).
    def conv(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
             dilation=1, groups=1, weightScale=100):
        return Conv(in_channels, out_channels, kernel_size, stride, padding,
                    dilation, groups, weightScale)

    def dense(self, in_features, out_features, weightScale=10):
        return Dense(in_features, out_features, weightScale)

    def pool(self, kernel_size, stride=None, padding=0, dilation=1):
        return Pool(self.theta, kernel_size, stride, padding, dilation)

    def dropout(self, p=0.5, inplace=False):
        return Dropout(p, inplace)

    def psp(self, spikes):
        """Causal filtering of every neuron's signal with srmKernel, times Ts.

        Accumulates taps in the same order as SLAYER's CUDA kernel (i = 0, 1,
        ...) to stay as close as possible to its float32 rounding.
        """
        kernel = self.srmKernel
        n_steps = spikes.shape[-1]
        out = torch.zeros_like(spikes)
        for i in range(min(len(kernel), n_steps)):
            out[..., i:] += spikes[..., :n_steps - i] * kernel[i]
        return out * self.ts

    def spike(self, membrane):
        """Threshold the membrane potential over time, adding the refractory
        response after every spike. Returns spikes of amplitude 1/Ts; the
        input tensor is left unchanged."""
        shape = membrane.shape
        n_steps = shape[-1]
        # Time-major copy so each step is a contiguous row of all neurons.
        u = membrane.reshape(-1, n_steps).t().clone(memory_format=torch.contiguous_format)
        spikes = torch.zeros_like(u)
        ref = self.refKernel.to(u.dtype)
        for t in range(n_steps):
            fired = u[t] >= self.theta
            if fired.any():
                spikes[t] = fired.to(u.dtype) / self.ts
                span = min(len(ref), n_steps - t)
                u[t:t + span] += ref[:span, None] * fired.to(u.dtype)
        return spikes.t().reshape(shape)
