"""Convert legacy pickled slayerSNN checkpoints into plain weight files.

The original checkpoints were whole pickled nn.Modules that need slayerSNN
(and the old demo/nets/ import paths) to unpickle. This reads them with
stand-in classes, so neither is needed, and writes
{"state_dict", "neuron", "simulation"} files loadable with weights_only=True.

    python tools/convert_checkpoints.py OLD.pt MODEL_NAME [OUT.pth]

The originals are in git history, e.g.
    git show 7f020a7:pretrained/nmnist_lenet.pt > /tmp/nmnist_lenet.pt
    python tools/convert_checkpoints.py /tmp/nmnist_lenet.pt nmnist
Only convert trusted files: this still runs the pickle.
"""
import os
import pickle
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models  # noqa: E402


class _SpikeLayer(torch.nn.Module):
    pass


class _Conv(torch.nn.Conv3d):
    pass


class _Dropout(torch.nn.Dropout3d):
    pass


class _Network(torch.nn.Module):
    pass


STAND_INS = {
    ("slayerSNN.slayer", "spikeLayer"): _SpikeLayer,
    ("slayerSNN.slayer", "_convLayer"): _Conv,
    ("slayerSNN.slayer", "_denseLayer"): _Conv,
    ("slayerSNN.slayer", "_poolLayer"): _Conv,
    ("slayerSNN.slayer", "_dropoutLayer"): _Dropout,
}


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in STAND_INS:
            return STAND_INS[module, name]
        if module.startswith(("slayerSNN", "demo.nets")):
            if module.startswith("demo.nets"):
                return _Network
            raise pickle.UnpicklingError(f"No stand-in for {module}.{name}")
        return super().find_class(module, name)


_pickle_shim = types.ModuleType("slayer_pickle_shim")
_pickle_shim.__dict__.update({k: getattr(pickle, k) for k in dir(pickle) if not k.startswith("__")})
_pickle_shim.Unpickler = _Unpickler


def convert(source, model_name, target=None):
    spec = models.get_spec(model_name)
    target = target or spec.path(spec.checkpoint)
    legacy = torch.load(source, map_location="cpu", weights_only=False, pickle_module=_pickle_shim)
    state = {k: v.detach().clone() for k, v in legacy.state_dict().items()}
    neuron, simulation = dict(legacy.slayer.neuron), dict(legacy.slayer.simulation)

    params = models.load_params(spec.path(spec.params_yaml))
    for key in ("theta", "tauSr", "tauRef", "scaleRef"):
        if float(params["neuron"][key]) != float(neuron[key]):
            raise ValueError(f"neuron {key}: checkpoint {neuron[key]} != YAML {params['neuron'][key]}")
    for key in ("Ts", "tSample"):
        if float(params["simulation"][key]) != float(simulation[key]):
            raise ValueError(f"simulation {key}: checkpoint {simulation[key]} != YAML {params['simulation'][key]}")

    torch.save({"format": 1, "model": model_name, "state_dict": state,
                "neuron": neuron, "simulation": simulation}, target)
    # Round trip: the new network must accept every tensor, unchanged.
    net = models.load_pretrained(spec)
    for key, value in net.state_dict().items():
        if not torch.equal(value, state[key]):
            raise AssertionError(f"{key} changed during conversion")
    print(f"{source} -> {target}: {len(state)} tensors")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    convert(*sys.argv[1:])
