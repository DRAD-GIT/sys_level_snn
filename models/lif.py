"""Leaky integrate-and-fire (LIF) spiking layers, trainable with surrogate
gradients.

Same interface as srm.SRMLayer (conv, dense, pool, dropout, psp, spike), so
the evaluation pipeline (probes, quantization, recording, hardware engine)
treats a LIF network like the SLAYER-style ones: every weighted layer is a
Conv3d applied independently at each time step, and tensors are
``[batch, channels, height, width, time]``.

Neuron model, for each neuron and time step t (inputs x[t] = the layer's
weighted sum plus bias, i.e. the current from the crossbar):
  v[t] = beta * v[t-1] + x[t]
  s[t] = 1 if v[t] >= theta, else 0
  v[t] = v[t] - theta * s[t]          (soft reset: subtract the threshold)
The backward pass replaces the step's derivative with a surrogate,
d s / d v = alpha / (2 (1 + (pi/2 alpha (v - theta))^2)) (arctan surrogate).

Pooling is a max over each window, so pooled spikes stay binary (a 0/1
input at every weighted layer, as the CIM hardware needs).
"""
import math

import torch

from .srm import Dropout, _triple


class _SpikeFn(torch.autograd.Function):
    """Heaviside forward, arctan surrogate backward."""

    @staticmethod
    def forward(ctx, v_minus_theta, alpha):
        ctx.save_for_backward(v_minus_theta)
        ctx.alpha = alpha
        return (v_minus_theta >= 0).to(v_minus_theta.dtype)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        surrogate = alpha / 2 / (1 + (math.pi / 2 * alpha * x) ** 2)
        return grad * surrogate, None


class Conv(torch.nn.Conv3d):
    """2-D convolution applied independently at every time step (with bias:
    the folded BatchNorm's shift, a constant current into each neuron)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__(in_channels, out_channels, _triple(kernel_size, 1),
                         _triple(stride, 1), _triple(padding, 0), bias=bias)


class Dense(torch.nn.Conv3d):
    """Fully connected layer per time step; in_features is an int, or a
    (width, height, channels) tuple whose kernel spans the whole input."""

    def __init__(self, in_features, out_features, bias=True):
        if isinstance(in_features, int):
            kernel, in_channels = (1, 1, 1), in_features
        else:
            kernel, in_channels = (in_features[1], in_features[0], 1), in_features[2]
        super().__init__(in_channels, out_features, kernel, bias=bias)


class MaxPool(torch.nn.MaxPool3d):
    """Spatial max pooling per time step: binary spikes stay binary."""

    def __init__(self, kernel_size, stride=None):
        kernel = _triple(kernel_size, 1)
        super().__init__(kernel, kernel if stride is None else _triple(stride, 1))


class LIFLayer(torch.nn.Module):
    """LIF neurons and layer factories (srm.SRMLayer's interface).

    neuron: {"theta": threshold, "beta": membrane decay per time step,
    "alpha": surrogate sharpness}. psp() is the identity (the LIF state is
    integrated in spike()).
    """

    def __init__(self, neuron, simulation):
        super().__init__()
        self.neuron = dict(neuron)
        self.simulation = dict(simulation)
        if self.neuron.get("type", "LIF") != "LIF":
            raise ValueError(f"LIFLayer needs neuron type LIF, got {self.neuron['type']!r}")
        self.theta = float(self.neuron["theta"])
        self.beta = float(self.neuron["beta"])
        self.alpha = float(self.neuron.get("alpha", 2.0))

    def conv(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        return Conv(in_channels, out_channels, kernel_size, stride, padding, bias)

    def dense(self, in_features, out_features, bias=True):
        return Dense(in_features, out_features, bias)

    def pool(self, kernel_size, stride=None):
        return MaxPool(kernel_size, stride)

    def dropout(self, p=0.5, inplace=False):
        return Dropout(p, inplace)

    @staticmethod
    def psp(current):
        return current

    def spike(self, current):
        """Integrate the input current over time (last axis) and fire."""
        v = torch.zeros_like(current[..., 0])
        out = []
        for t in range(current.shape[-1]):
            v = self.beta * v + current[..., t]
            s = _SpikeFn.apply(v - self.theta, self.alpha)
            v = v - self.theta * s
            out.append(s)
        return torch.stack(out, dim=-1)


def fold_batchnorm(conv, bn):
    """Weight and bias of `conv` followed by the BatchNorm `bn` (eval mode),
    as one layer: w' = w * g / s, b' = (b - mean) * g / s + beta, with
    s = sqrt(var + eps), g = bn.weight."""
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    weight = conv.weight * scale.reshape(-1, *([1] * (conv.weight.dim() - 1)))
    bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    return weight.detach(), ((bias - bn.running_mean) * scale + bn.bias).detach()


def rate_code(images, n_steps, generator=None):
    """Bernoulli rate coding: [batch, C, H, W] intensities in [0, 1] ->
    [batch, C, H, W, n_steps] spikes, each 1 with probability = intensity."""
    probs = images.unsqueeze(-1).expand(*images.shape, n_steps)
    return torch.bernoulli(probs, generator=generator)


__all__ = ["LIFLayer", "Conv", "Dense", "MaxPool", "fold_batchnorm", "rate_code"]
