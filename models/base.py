"""Shared base classes for the SNN models and datasets, plus the ModelSpec
that tells the evaluation pipeline how to run each model."""
import os
from dataclasses import dataclass

import numpy as np
import torch
import yaml

from .srm import SRMLayer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_params(path):
    """Network parameter YAML (simulation, neuron, training) as a dict."""
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


class NDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, samples_file: str, sampling_time: int, sample_length: int):
        self.path = data_path
        self.samples = np.loadtxt(samples_file, dtype='str')
        self.sampling_time = sampling_time
        self.n_time_bins = int(sample_length / sampling_time)

    def __len__(self):
        return self.samples.shape[0]


class NNetwork(torch.nn.Module):
    """Base network: ``self.slayer`` provides the spiking layers.

    backend defaults to the plain-PyTorch SRMLayer. Passing slayerSNN's
    ``snn.layer`` builds the same network on the original SLAYER framework;
    that is only used to verify SRMLayer (tools/export_slayer_reference.py).
    """

    def __init__(self, net_params: dict, backend=None):
        super(NNetwork, self).__init__()

        layer = backend or SRMLayer
        self.slayer = layer(net_params['neuron'], net_params['simulation'])


@dataclass(frozen=True)
class ModelSpec:
    """Everything the evaluation pipeline needs to know about one model.

    layers: names of the weighted conv/dense layers, in forward order. Only
        these layers are mapped onto CIM hardware.
    checkpoint / params_yaml: paths relative to the repository root.
    """
    name: str
    display_name: str
    network_class: type
    dataset_class: type
    checkpoint: str
    params_yaml: str
    layers: tuple[str, ...]
    max_batch_size: int | None = None

    def path(self, relative: str) -> str:
        return os.path.join(REPO_ROOT, relative)
