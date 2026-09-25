"""Shared helpers for recording and comparing SLAYER reference runs.

A reference file (reference/<model>_slayer.pt) holds, for the first N test
samples, the input spikes read by slayerSNN's own readers and by
models/events.py, the input of every profiled layer, and the output spikes,
all recorded while the original slayerSNN framework ran the network.
Spike tensors are stored bit-packed with their shape and spike amplitude.
"""
import numpy as np
import torch

FORMAT = 1


def pack(tensor):
    """Bit-pack a spike tensor whose entries are all 0 or one amplitude."""
    tensor = tensor.detach().cpu()
    nonzero = tensor[tensor != 0]
    amplitude = nonzero[0].item() if len(nonzero) else 1.0
    if len(nonzero) and not torch.all(nonzero == amplitude):
        raise ValueError("tensor is not a binary spike tensor")
    bits = np.packbits((tensor != 0).numpy().reshape(-1))
    return {"shape": list(tensor.shape), "amplitude": amplitude,
            "bits": torch.from_numpy(bits)}


def unpack(packed):
    count = int(np.prod(packed["shape"]))
    bits = np.unpackbits(packed["bits"].numpy(), count=count)
    return torch.from_numpy(bits.astype(np.float32) * packed["amplitude"]).reshape(packed["shape"])


def record_layer_inputs(net, layer_names):
    """Forward hooks storing each named layer's input; returns (store, handles)."""
    store = {}
    handles = [getattr(net, name).register_forward_hook(
        lambda _m, inputs, _o, name=name: store.__setitem__(name, inputs[0].detach()))
        for name in layer_names]
    return store, handles
