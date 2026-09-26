"""SNN model definitions. Each model module exposes a ``SPEC`` (see base.ModelSpec).

To add a model: write its network/dataset classes in a new module, give it a
SPEC, and add it to MODEL_MODULES below.
"""
import importlib
import os

import torch

from .base import REPO_ROOT, load_params

MODEL_MODULES = {"nmnist": "models.nmnist", "gesture": "models.gesture"}


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
    checkpoint = torch.load(spec.path(spec.checkpoint), map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    for name in ("srmKernel", "refKernel"):
        generated, trained = getattr(net.slayer, name).cpu(), state[f"slayer.{name}"]
        if generated.shape != trained.shape or not torch.allclose(generated, trained):
            raise ValueError(f"{spec.params_yaml}: neuron parameters do not reproduce the "
                             f"trained {name}; restore the original neuron/simulation values")
    net.load_state_dict(state)
    return net.to(device)


def test_dataset(spec, net_params):
    """The model's test set, with YAML dataset paths resolved from the repo root."""
    paths = net_params["training"]["path"]
    return spec.dataset_class(
        data_path=os.path.join(REPO_ROOT, paths["dir_test"]),
        samples_file=os.path.join(REPO_ROOT, paths["list_test"]),
        sampling_time=net_params["simulation"]["Ts"],
        sample_length=net_params["simulation"]["tSample"],
    )
