"""SNN model definitions. Each model module exposes a ``SPEC`` (see base.ModelSpec).

To add a model: write its network/dataset classes in a new module, give it a
SPEC, and add it to MODEL_MODULES below.
"""
import importlib
import os
import re

import torch
from torch.utils.data import DataLoader, Subset

from .base import load_params

MODEL_MODULES = {"nmnist": "models.nmnist", "gesture": "models.gesture", "cifar10": "models.cifar10",
                 "cifar10_thermo": "models.cifar10_thermo", "gesture16": "models.gesture16"}


def load_tensors(path):
    """torch.load for files holding only tensors, safely (weights_only) where
    PyTorch supports it (>= 1.13)."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # older PyTorch: no weights_only argument
        return torch.load(path, map_location="cpu")


def get_spec(name):
    if name not in MODEL_MODULES:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(MODEL_MODULES)}")
    return importlib.import_module(MODEL_MODULES[name]).SPEC


def load_pretrained(spec, device="cpu", backend=None):
    """Build the network from its YAML parameters and load the trained weights.

    Checkpoints are plain tensor files (loaded with weights_only=True). SRM
    checkpoints also store the neuron kernels the network was trained with;
    a mismatch with the kernels generated from the YAML means the YAML was
    edited.
    """
    params = load_params(spec.path(spec.params_yaml))
    net = spec.network_class(params, backend=backend)
    checkpoint = load_tensors(spec.path(spec.checkpoint))
    state = checkpoint["state_dict"]
    for name in ("srmKernel", "refKernel"):
        if not hasattr(net.slayer, name):   # LIF networks have no kernels
            continue
        generated, trained = getattr(net.slayer, name).cpu(), state[f"slayer.{name}"]
        if generated.shape != trained.shape or not torch.allclose(generated, trained):
            raise ValueError(f"{spec.params_yaml}: neuron parameters do not reproduce the "
                             f"trained {name}; restore the original neuron/simulation values")
    net.load_state_dict(state)
    return net.to(device)


def _plain(name):
    """Folder name without case and separators: "N_MNIST" -> "nmnist"."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def find_dataset(spec, data_dir):
    """The model's dataset folder: data_dir itself if its name starts with one
    of spec.dataset_folders, else the one folder inside data_dir that does.
    Names are compared ignoring case, hyphens, underscores and spaces."""
    if not data_dir:
        raise ValueError("set DATASET_DIR in run.py (or pass --data) to the folder holding "
                         f"the {spec.display_name} dataset")
    data_dir = os.path.expanduser(data_dir)

    def matches(name):
        return _plain(name).startswith(tuple(_plain(p) for p in spec.dataset_folders))
    if matches(os.path.basename(os.path.normpath(data_dir))):
        return data_dir
    found = sorted(name for name in os.listdir(data_dir)
                   if matches(name) and os.path.isdir(os.path.join(data_dir, name)))
    if len(found) != 1:
        raise FileNotFoundError(
            f"{data_dir}: expected one folder starting with {' / '.join(spec.dataset_folders)} "
            f"for {spec.display_name}, found {found or 'none'}")
    return os.path.join(data_dir, found[0])


ALL_SAMPLES = -1


def sample_limit(value):
    """A sample count as given by the user: None or -1 = the whole test set
    (returned as None), a positive integer = the first that many samples."""
    if value is None or value == ALL_SAMPLES:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"samples must be a positive integer, or -1 for all, got {value!r}")
    return value


def samples_argument(text):
    """argparse type for --samples: -1 = all (None), else a positive count."""
    import argparse
    try:
        return sample_limit(int(text))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a positive number of samples, or -1 for all, got {text!r}") from None


def positive_argument(text):
    """argparse type for counts that must be positive (--every, --parallel)."""
    import argparse
    try:
        value = int(text)
    except ValueError:
        value = 0
    if value <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}")
    return value


def bits_argument(text):
    """argparse type for weight bit widths: 'float' -> None, else an integer
    of at least 2 (signed codes)."""
    import argparse
    if text == "float":
        return None
    try:
        bits = int(text)
    except ValueError:
        bits = 0
    if bits < 2:
        raise argparse.ArgumentTypeError(f"expected 'float' or a bit width >= 2, got {text!r}")
    return bits


def test_loader(spec, net_params, data_dir, max_samples=None, parallel=None, num_workers=4):
    """The test set (its first max_samples, if given; None or -1 = all) in
    groups of `parallel` samples evaluated at once (default: spec.batch_size)."""
    max_samples = sample_limit(max_samples)
    if parallel is not None and parallel <= 0:
        raise ValueError("parallel must be positive")
    dataset = test_dataset(spec, net_params, data_dir)
    if max_samples is not None:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    return DataLoader(dataset, batch_size=parallel or spec.batch_size, shuffle=False,
                      num_workers=num_workers)


def test_dataset(spec, net_params, data_dir):
    """The model's test set; the YAML's dataset paths are relative to its
    dataset folder (see find_dataset)."""
    root = find_dataset(spec, data_dir)
    paths = net_params["training"]["path"]
    extra = {"encoding": net_params["encoding"]} if "encoding" in net_params else {}
    return spec.dataset_class(
        data_path=os.path.join(root, paths["dir_test"]),
        samples_file=os.path.join(root, paths["list_test"]),
        sampling_time=net_params["simulation"]["Ts"],
        sample_length=net_params["simulation"]["tSample"],
        **extra,
    )
