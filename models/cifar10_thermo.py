"""CIFAR-10 VGG-11 SNN on thermometer-coded images (models/cifar10.py's
network and test set; the encoding and time steps are in
models/cifar10_thermo.yaml).

Each pixel channel becomes `levels` binary input channels (3 x 8 = 24), on
where the intensity exceeds (i+1)/(levels+1), the same at every time step:
deterministic binary spikes into the first layer (216 crossbar rows instead
of rate coding's 27), so fewer time steps are needed than with rate coding.
"""
from .base import ModelSpec
from .cifar10 import CIFAR10Dataset, VGG11Network

SPEC = ModelSpec(
    name="cifar10_thermo",
    display_name="CIFAR-10 (thermometer)",
    network_class=VGG11Network,
    dataset_class=CIFAR10Dataset,
    checkpoint="pretrained/cifar10_vgg11_thermo.pth",
    params_yaml="models/cifar10_thermo.yaml",
    layers=("SC1", "SC2", "SC3", "SC4", "SC5", "SC6", "SC7", "SC8", "SF1"),
    batch_size=100,
    dataset_folders=("cifar-10-batches",),
    sources=("models/lif.py", "models/cifar10.py", "models/cifar10_thermo.py"),
)
