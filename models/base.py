"""Shared base classes for the SNN models and datasets, plus the ModelSpec
that tells the evaluation pipeline how to run each model."""
import os
from dataclasses import dataclass

import numpy as np
import torch
import slayerSNN as snn  # type: ignore

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class NDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, samples_file: str, sampling_time: int, sample_length: int):
        self.path = data_path
        self.samples = np.loadtxt(samples_file, dtype='str')
        self.sampling_time = sampling_time
        self.n_time_bins = int(sample_length / sampling_time)

    def __len__(self):
        return self.samples.shape[0]


class NNetwork(torch.nn.Module):
    def __init__(self, net_params: snn.params):
        super(NNetwork, self).__init__()

        self.slayer = snn.layer(net_params['neuron'], net_params['simulation'])


@dataclass(frozen=True)
class ModelSpec:
    """Everything the evaluation pipeline needs to know about one model.

    layers: (module name, padding) of each weighted conv/dense layer, in
        forward order. Only these layers are mapped onto CIM hardware.
    checkpoint / params_yaml: paths relative to the repository root.
    std_quantization: True clips the quantization range to mean +/- k*std;
        False uses the weights' min/max.
    """
    name: str
    display_name: str
    dataset_class: type
    checkpoint: str
    params_yaml: str
    layers: tuple[tuple[str, int], ...]
    max_batch_size: int | None = None
    std_quantization: bool = False

    def path(self, relative: str) -> str:
        return os.path.join(REPO_ROOT, relative)
