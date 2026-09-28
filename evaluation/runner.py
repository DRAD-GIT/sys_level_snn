"""Software SNN inference with per-layer CIM hardware evaluation.

Runs the pretrained network on the test set and feeds each profiled layer's
input spikes to hardware.evaluate_layer for every architecture. Architectures
are grouped by weight precision: each group runs the network with the weights
quantized as that hardware stores them, so accuracy and spike activity match
the hardware. Hardware evaluation does not simulate analog nonidealities.
"""
import copy

import torch
from torch.utils.data import DataLoader

import models
from evaluation.probes import LayerProbe
from evaluation.software import TestStats, num_spikes_loss, predict_class
from hardware import evaluate_layer, quantize_weights


def quantized_network(net, layer_names, bits):
    """A copy of `net` whose weighted layers compute with weights quantized to
    `bits` (symmetric uniform per layer; None keeps them), and each layer's
    stored values: integer codes, or the float weights for bits=None."""
    net = copy.deepcopy(net)
    codes = {}
    with torch.no_grad():
        for name in layer_names:
            module = getattr(net, name)
            codes[name], scale = quantize_weights(module.weight.detach(), bits)
            if bits is not None:
                module.weight.copy_(codes[name] * scale)
    return net, codes


class _PrecisionGroup:
    """A copy of the network with weights quantized to `bits`, the integer
    codes (or floats) the hardware stores, and its evaluation state."""

    def __init__(self, net, layer_names, bits, architectures):
        self.net, codes = quantized_network(net, layer_names, bits)
        self.architectures = architectures
        self.codes = {name: c[..., 0].cpu() for name, c in codes.items()}
        self.stride_padding = {name: (getattr(self.net, name).stride[0],
                                      getattr(self.net, name).padding[0]) for name in layer_names}
        self.probe = LayerProbe(self.net, layer_names)
        self.stats = TestStats()
        self.costs = {arch.name: {} for arch in architectures}


def evaluate(model, architectures, *, data_dir, batch_size=1, max_batches=None, num_workers=4,
             log=print):
    """Evaluate `architectures` (hardware.Architecture) on `model`.

    data_dir: the dataset folder, or a folder holding it (models.find_dataset).
    Returns {architecture name: (software accuracy %, {layer: LayerCost})}.
    """
    if batch_size <= 0 or (max_batches is not None and max_batches <= 0):
        raise ValueError("batch_size and max_batches must be positive")
    names = [a.name for a in architectures]
    if not architectures or len(set(names)) != len(names):
        raise ValueError("define at least one architecture, with unique names")

    spec = models.get_spec(model)
    batch_size = min(batch_size, spec.max_batch_size or batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = models.load_pretrained(spec, device).eval()
    params = models.load_params(spec.path(spec.params_yaml))
    loader = DataLoader(models.test_dataset(spec, params, data_dir), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)

    by_bits = {}
    for arch in architectures:
        by_bits.setdefault(arch.precision.weight_bits, []).append(arch)
    groups = [_PrecisionGroup(net, spec.layers, bits, archs) for bits, archs in by_bits.items()]
    log(f"{spec.display_name}: evaluating {', '.join(names)}")

    for batch_index, (_, spikes, target, label) in enumerate(loader):
        if max_batches is not None and batch_index == max_batches:
            break
        spikes, target = spikes.to(device), target.to(device)
        for group in groups:
            group.probe.clear()
            with torch.no_grad():
                output = group.net(spikes)
            loss = num_spikes_loss(output, target, params, group.net.slayer.psp).item()
            group.stats.update(predict_class(output), label, loss)
            for name in spec.layers:
                layer_input = group.probe.inputs[name].cpu()
                stride, padding = group.stride_padding[name]
                shared = {}  # spike activity, reused across architectures
                for arch in group.architectures:
                    cost = evaluate_layer(arch, layer_input, group.codes[name], stride=stride,
                                          padding=padding, activity_cache=shared,
                                          output_spikes=group.probe.output_spikes.get(name, 0))
                    costs = group.costs[arch.name]
                    if name in costs:
                        costs[name].add(cost)
                    else:
                        costs[name] = cost
        if (batch_index + 1) % 10 == 0 or batch_index == 0:
            log(f"batch {batch_index + 1}: accuracy " + ", ".join(
                f"{'float' if bits is None else f'{bits}-bit'} {g.stats.accuracy:.2f}%"
                for bits, g in zip(by_bits, groups)))

    results = {}
    for group in groups:
        group.probe.remove()
        for arch in group.architectures:
            results[arch.name] = (group.stats.accuracy, group.costs[arch.name])
    return results


def accuracy_sweep(model, bit_widths, *, data_dir, batch_size=None, max_batches=None,
                   num_workers=4, log=print):
    """Test accuracy (%) of `model` with its weights quantized to each of
    `bit_widths` (None = the trained float weights). The test set is read once;
    every batch runs through all quantized networks. No hardware evaluation.
    Returns {bits: accuracy}."""
    spec = models.get_spec(model)
    batch_size = min(batch_size or 32, spec.max_batch_size or batch_size or 32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = models.load_pretrained(spec, device).eval()
    params = models.load_params(spec.path(spec.params_yaml))
    loader = DataLoader(models.test_dataset(spec, params, data_dir), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)
    nets = {bits: quantized_network(net, spec.layers, bits)[0] for bits in bit_widths}
    correct, total = dict.fromkeys(bit_widths, 0), 0
    log(f"{spec.display_name}: accuracy with weights of {', '.join(_label(b) for b in bit_widths)}")
    with torch.no_grad():
        for batch_index, (_, spikes, _, label) in enumerate(loader):
            if max_batches is not None and batch_index == max_batches:
                break
            spikes = spikes.to(device)
            for bits, quantized in nets.items():
                correct[bits] += int((predict_class(quantized(spikes)) == label).sum())
            total += len(label)
            if total % 1000 < len(label):
                log(f"  {total} samples: " + ", ".join(
                    f"{_label(b)} {100 * c / total:.2f}%" for b, c in correct.items()))
    return {bits: 100 * c / total for bits, c in correct.items()}


def _label(bits):
    return "float" if bits is None else f"{bits}-bit"
