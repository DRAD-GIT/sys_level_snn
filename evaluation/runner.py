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


def quantized_network(net, layer_names, bits, scaling="max"):
    """A copy of `net` whose weighted layers compute with quantized weights
    (symmetric uniform per layer, range from `scaling`, see
    hardware.quantize_weights), and each layer's stored values: integer codes,
    or the float weights where unquantized.

    bits: one bit width (None = float) for every layer in layer_names, or
    {layer: bits} per layer (layers not listed stay float).
    """
    per_layer = bits if isinstance(bits, dict) else dict.fromkeys(layer_names, bits)
    unknown = set(per_layer) - set(layer_names)
    if unknown:
        raise ValueError(f"unknown layers {sorted(unknown)}; the weighted layers are {list(layer_names)}")
    net = copy.deepcopy(net)
    codes = {}
    with torch.no_grad():
        for name in layer_names:
            module = getattr(net, name)
            layer_bits = per_layer.get(name)
            codes[name], scale = quantize_weights(module.weight.detach(), layer_bits, scaling)
            if layer_bits is not None:
                module.weight.copy_(codes[name] * scale)
    return net, codes


class _PrecisionGroup:
    """A copy of the network with weights quantized to `bits` with `scaling`,
    the integer codes (or floats) the hardware stores, and its evaluation state."""

    def __init__(self, net, layer_names, bits, scaling, architectures):
        self.net, codes = quantized_network(net, layer_names, bits, scaling)
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

    by_precision = {}  # (weight bits, scaling): architectures storing weights that way
    for arch in architectures:
        by_precision.setdefault(_precision_key(arch.precision.weight_bits,
                                               arch.precision.weight_scaling), []).append(arch)
    groups = [_PrecisionGroup(net, spec.layers, bits, scaling, archs)
              for (bits, scaling), archs in by_precision.items()]
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
                f"{_label(*key)} {g.stats.accuracy:.2f}%" for key, g in zip(by_precision, groups)))

    results = {}
    for group in groups:
        group.probe.remove()
        for arch in group.architectures:
            results[arch.name] = (group.stats.accuracy, group.costs[arch.name])
    return results


def accuracy_sweep(model, configs, *, data_dir, batch_size=None, max_batches=None,
                   num_workers=4, log_every=2, log=print):
    """Test accuracy (%) of `model` for each weight quantization in `configs`,
    {label: (bits, scaling)}, with bits and scaling as in quantized_network
    (bits=None: the trained float weights). The test set is read once; every
    batch runs through all configurations. No hardware evaluation. Progress is
    logged every `log_every` batches. Returns {label: accuracy}."""
    spec = models.get_spec(model)
    batch_size = min(batch_size or 32, spec.max_batch_size or batch_size or 32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = models.load_pretrained(spec, device).eval()
    params = models.load_params(spec.path(spec.params_yaml))
    loader = DataLoader(models.test_dataset(spec, params, data_dir), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers)
    nets = {label: quantized_network(net, spec.layers, bits, scaling)[0]
            for label, (bits, scaling) in configs.items()}
    correct, total = dict.fromkeys(configs, 0), 0
    log(f"{spec.display_name}: accuracy of {len(configs)} weight configurations")
    with torch.no_grad():
        for batch_index, (_, spikes, _, label) in enumerate(loader):
            if max_batches is not None and batch_index == max_batches:
                break
            spikes = spikes.to(device)
            for name, quantized in nets.items():
                correct[name] += int((predict_class(quantized(spikes)) == label).sum())
            total += len(label)
            if (batch_index + 1) % log_every == 0:
                log(f"  batch {batch_index + 1}, {total} samples: " + ", ".join(
                    f"{name} {100 * c / total:.2f}%" for name, c in correct.items()))
    return {name: 100 * c / total for name, c in correct.items()}


def _precision_key(bits, scaling):
    """(bits, scaling); float weights have no scaling."""
    return (None, None) if bits is None else (bits, scaling)


def _label(bits, scaling):
    return "float" if bits is None else f"{bits}-bit {scaling}"
