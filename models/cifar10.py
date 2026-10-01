"""CIFAR-10: a VGG-11 spiking network with LIF neurons, on rate-coded images.

Every weighted layer receives binary spikes, so all of them (including the
first) map onto CIM crossbars: each 32x32 RGB image is rate-coded into
tSample binary frames (pixel intensity = spike probability per time step),
with a fixed random seed per test image so recordings are reproducible.

VGG-11 for CIFAR-10 (the network Han et al., TCAS-I 2022, use on CIFAR-10):
conv 64 - M - 128 - M - 256 - 256 - M - 512 - 512 - M - 512 - 512 - M - fc 10,
3x3 convolutions (padding 1), 2x2 max pooling, LIF neurons after every
weighted layer; the class is the output neuron with the most spikes.

Trained with BatchNorm after each convolution (tools/train_cifar10.py); the
checkpoint has it folded into the convolutions' weights and biases, so the
evaluated network is weighted layers, LIF neurons and max pooling only. The
bias is a constant current into each neuron.
"""
import os
import pickle

import numpy as np
import torch

from .base import ModelSpec, NNetwork
from .lif import LIFLayer, rate_code

VGG11 = (64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M")


def read_batches(paths):
    """CIFAR-10 python batches -> (uint8 images [N, 3, 32, 32], int64 labels)."""
    images, labels = [], []
    for path in paths:
        with open(path, "rb") as file:
            batch = pickle.load(file, encoding="bytes")
        images.append(np.asarray(batch[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32))
        labels.append(np.asarray(batch[b"labels"], dtype=np.int64))
    return torch.from_numpy(np.concatenate(images)), torch.from_numpy(np.concatenate(labels))


class CIFAR10Dataset(torch.utils.data.Dataset):
    """The CIFAR-10 test set, rate-coded. samples_file is the batch file
    (test_batch); data_path is unused (the batch holds the images)."""

    seed = 2024   # rate-coding seed: image i uses seed + i

    def __init__(self, data_path, samples_file, sampling_time, sample_length):
        self.images, self.labels = read_batches([samples_file])
        self.n_time_bins = int(sample_length / sampling_time)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(self.seed + int(index))
        image = self.images[index].float().div_(255)
        spikes_in = rate_code(image[None], self.n_time_bins, generator)[0]
        label = int(self.labels[index])
        desired = torch.zeros((10, 1, 1, 1))
        desired[label, ...] = 1
        return int(index), spikes_in, desired, label


class VGG11Network(NNetwork):
    """VGG-11 SNN. batchnorm=True adds a BatchNorm after every convolution
    (training); the evaluated checkpoint has it folded (batchnorm=False).
    Weighted layers: SC1 ... SC8 (convolutions), SF1 (classifier)."""

    def __init__(self, net_params: dict, do_enable=False, backend=None, batchnorm=False):
        super().__init__(net_params, backend or LIFLayer)
        self.batchnorm = batchnorm
        self.plan, channels, conv = [], 3, 0
        for item in VGG11:
            if item == "M":
                self.plan.append("pool")
                continue
            conv += 1
            setattr(self, f"SC{conv}", self.slayer.conv(channels, item, 3, padding=1,
                                                         bias=not batchnorm))
            if batchnorm:
                setattr(self, f"BN{conv}", torch.nn.BatchNorm3d(item))
            self.plan.append(f"SC{conv}")
            channels = item
        self.pool = self.slayer.pool(2)
        self.SF1 = self.slayer.dense(channels, 10)
        self.SD = self.slayer.dropout(0.1 if do_enable else 0.0)

    def forward(self, s_in):
        s = s_in
        for step in self.plan:
            if step == "pool":
                s = self.pool(s)
                continue
            current = getattr(self, step)(s)
            if self.batchnorm:
                current = getattr(self, "BN" + step[2:])(current)
            s = self.SD(self.slayer.spike(self.slayer.psp(current)))
        return self.slayer.spike(self.slayer.psp(self.SF1(s)))   # 10 x 1 x 1

    def folded_state_dict(self):
        """State dict of the same network with every BatchNorm folded into its
        convolution (for VGG11Network(batchnorm=False))."""
        from .lif import fold_batchnorm
        if not self.batchnorm:
            return self.state_dict()
        state = {k: v for k, v in self.state_dict().items()
                 if not k.startswith("BN") and not k.startswith("SC")}
        for step in self.plan:
            if step != "pool":
                weight, bias = fold_batchnorm(getattr(self, step), getattr(self, "BN" + step[2:]))
                state[f"{step}.weight"], state[f"{step}.bias"] = weight, bias
        return state


SPEC = ModelSpec(
    name="cifar10",
    display_name="CIFAR-10",
    network_class=VGG11Network,
    dataset_class=CIFAR10Dataset,
    checkpoint="pretrained/cifar10_vgg11.pth",
    params_yaml="models/cifar10.yaml",
    layers=("SC1", "SC2", "SC3", "SC4", "SC5", "SC6", "SC7", "SC8", "SF1"),
    batch_size=100,
    dataset_folders=("cifar-10-batches",),
    sources=("models/lif.py", "models/cifar10.py"),
)
