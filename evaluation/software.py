"""Software classification metrics (replaces slayerSNN's predict/loss/stats)."""
import numpy as np
import torch


def predict_class(output):
    """Class with the most output spikes over the whole simulation."""
    counts = torch.sum(output, 4, keepdim=True).cpu()
    return torch.max(counts.reshape(counts.shape[0], -1), 1)[1]


def num_spikes_loss(output, target, net_params, psp):
    """SLAYER's NumSpikes loss (reporting only): spike-count error inside the
    target region, plus psp-filtered spikes outside it, squared and summed.

    target is the one-hot [batch, classes, 1, 1, 1] tensor; psp is the
    network's psp function (the same srmKernel filter).
    """
    error_desc = net_params["training"]["error"]
    ts = net_params["simulation"]["Ts"]
    region, counts = error_desc["tgtSpikeRegion"], error_desc["tgtSpikeCount"]
    start = int(np.rint(region["start"] / ts))
    stop = int(np.rint(region["stop"] / ts))

    actual = torch.sum(output[..., start:stop], 4, keepdim=True) * ts
    desired = torch.where(target == 1, float(counts[True]), float(counts[False])).to(output)
    count_error = (actual - desired) / (stop - start)
    in_region = torch.zeros_like(output)
    in_region[..., start:stop] = 1

    error = psp(output - in_region * output) + count_error * in_region
    return 0.5 * torch.sum(error ** 2) * ts


class TestStats:
    """Running accuracy and mean loss over the evaluated samples."""

    def __init__(self):
        self.correct = 0
        self.samples = 0
        self.loss_sum = 0.0

    def update(self, predicted, labels, loss):
        self.correct += torch.sum(predicted == labels).item()
        self.samples += len(labels)
        self.loss_sum += loss

    @property
    def accuracy(self):
        return 100 * self.correct / self.samples

    @property
    def loss(self):
        return self.loss_sum / self.samples
