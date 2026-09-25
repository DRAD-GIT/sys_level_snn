"""SNN model definitions. Each model module exposes a ``SPEC`` (see base.ModelSpec).

To add a model: write its network/dataset classes in a new module, give it a
SPEC, and add it to MODEL_MODULES below.
"""
import importlib
import sys
import types

MODEL_MODULES = {"nmnist": "models.nmnist", "gesture": "models.gesture"}

# The pretrained checkpoints are full pickled nn.Modules that were saved when
# these classes lived in demo/nets/. Pickle stores that import path, so alias
# the old module names to the current ones while loading.
LEGACY_PICKLE_MODULES = {"demo.nets.nmnist": "models.nmnist",
                         "demo.nets.gesture": "models.gesture"}


def get_spec(name):
    if name not in MODEL_MODULES:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(MODEL_MODULES)}")
    return importlib.import_module(MODEL_MODULES[name]).SPEC


def load_pretrained(spec, device="cpu"):
    """Load a trusted checkpoint (torch.load(weights_only=False) runs pickle code)."""
    import torch

    for legacy, current in LEGACY_PICKLE_MODULES.items():
        # Unpickling also imports the parent packages ("demo", "demo.nets").
        parts = legacy.split(".")
        for i in range(1, len(parts)):
            parent = ".".join(parts[:i])
            sys.modules.setdefault(parent, types.ModuleType(parent))
        sys.modules.setdefault(legacy, importlib.import_module(current))
    return torch.load(spec.path(spec.checkpoint), map_location=device, weights_only=False)
