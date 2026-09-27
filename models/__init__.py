"""SNN model definitions. Each model module exposes a ``SPEC`` (see base.ModelSpec).

To add a model: write its network/dataset classes in a new module, give it a
SPEC, and add it to MODEL_MODULES below.
"""
import importlib
import os

import torch

from .base import load_params

MODEL_MODULES = {"nmnist": "models.nmnist", "gesture": "models.gesture"}


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

    Checkpoints are plain tensor files (loaded with weights_only=True). They
    also store the neuron kernels the network was trained with; a mismatch
    with the kernels generated from the YAML means the YAML was edited.
    """
    params = load_params(spec.path(spec.params_yaml))
    net = spec.network_class(params, backend=backend)
    checkpoint = load_tensors(spec.path(spec.checkpoint))
    state = checkpoint["state_dict"]
    for name in ("srmKernel", "refKernel"):
        generated, trained = getattr(net.slayer, name).cpu(), state[f"slayer.{name}"]
        if generated.shape != trained.shape or not torch.allclose(generated, trained):
            raise ValueError(f"{spec.params_yaml}: neuron parameters do not reproduce the "
                             f"trained {name}; restore the original neuron/simulation values")
    net.load_state_dict(state)
    return net.to(device)


def find_dataset(spec, data_dir):
    """The model's dataset folder: data_dir itself if its name starts with one
    of spec.dataset_folders, else the one folder inside data_dir that does."""
    if not data_dir:
        raise ValueError("set DATASET_DIR in run.py (or pass --data) to the folder holding "
                         f"the {spec.display_name} dataset")
    data_dir = os.path.expanduser(data_dir)

    def matches(name):
        return name.lower().startswith(tuple(p.lower() for p in spec.dataset_folders))
    if matches(os.path.basename(os.path.normpath(data_dir))):
        return data_dir
    found = sorted(name for name in os.listdir(data_dir)
                   if matches(name) and os.path.isdir(os.path.join(data_dir, name)))
    if len(found) != 1:
        raise FileNotFoundError(
            f"{data_dir}: expected one folder starting with {' / '.join(spec.dataset_folders)} "
            f"for {spec.display_name}, found {found or 'none'}")
    return os.path.join(data_dir, found[0])


def test_dataset(spec, net_params, data_dir):
    """The model's test set; the YAML's dataset paths are relative to its
    dataset folder (see find_dataset)."""
    root = find_dataset(spec, data_dir)
    paths = net_params["training"]["path"]
    return spec.dataset_class(
        data_path=os.path.join(root, paths["dir_test"]),
        samples_file=os.path.join(root, paths["list_test"]),
        sampling_time=net_params["simulation"]["Ts"],
        sample_length=net_params["simulation"]["tSample"],
    )
