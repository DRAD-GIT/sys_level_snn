"""Software SNN inference and per-layer CIM hardware evaluation.

1. The network runs once over the test set per weight quantization; each
   weighted layer's input spikes and output spike counts, the predictions and
   the labels are saved as a recording (evaluation/recording.py). An existing
   recording that matches is reused, so inference is not repeated.
2. The hardware of every architecture is evaluated from the recording of its
   weight quantization: accuracy from the recorded predictions, costs from the
   recorded spikes (hardware.evaluate_layer). Analog nonidealities are not
   simulated.
"""
import torch

import models
from evaluation.recording import open_recording, precision_label, quantized_network, record, recording_path
from evaluation.software import predict_class
from hardware import evaluate_layer

__all__ = ["evaluate", "accuracy_sweep", "quantized_network"]


def evaluate(model, architectures, *, data_dir, recording_dir, max_samples=None, parallel=None,
             rerecord=False, full_outputs=False, num_workers=4, log_every=1000, log=print):
    """Evaluate `architectures` (hardware.Architecture) on `model`.

    data_dir: the dataset folder, or a folder holding it (models.find_dataset);
    only read when a recording has to be made.
    recording_dir: where recordings are kept; rerecord=True makes new ones.
    max_samples: evaluate only the first samples of the test set (None or -1 = all).
    parallel: samples processed at once: the batch when recording (default:
    the model's batch_size) and the samples per hardware evaluation step,
    whole recorded files (default: one recorded file per step).
    Progress is logged every `log_every` samples and at the end.
    Returns {architecture name: (software accuracy %, {layer: LayerCost})}.
    """
    names = [a.name for a in architectures]
    if not architectures or len(set(names)) != len(names):
        raise ValueError("define at least one architecture, with unique names")
    spec = models.get_spec(model)
    max_samples = models.sample_limit(max_samples)

    by_precision = {}  # (weight bits, scaling): architectures storing weights that way
    for arch in architectures:
        by_precision.setdefault(_precision_key(arch.mapping.weight_bits,
                                               arch.mapping.weight_scaling), []).append(arch)
    recordings = {key: None if rerecord else open_recording(recording_dir, model, *key, max_samples,
                                                            full_outputs)
                  for key in by_precision}
    missing = [key for key, found in recordings.items() if found is None]
    for key, found in recordings.items():
        if found is not None:
            log(f"{spec.display_name}: using the recorded forward pass {found.path}")
    if missing:
        for key in missing:
            log(f"{spec.display_name}: no matching recording "
                f"{recording_path(recording_dir, model, *key)} (missing, incomplete, or made "
                "with another checkpoint, YAML, code or quantization): recording it from the dataset")
        recordings.update(record(model, missing, data_dir=data_dir, recording_dir=recording_dir,
                                 max_samples=max_samples, parallel=parallel,
                                 full_outputs=full_outputs, num_workers=num_workers,
                                 log_every=log_every, log=log))

    net = models.load_pretrained(spec).eval()
    layers = {name: getattr(net, name) for name in spec.layers}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # spike activity
    results = {}
    for key, archs in by_precision.items():
        codes = {name: c[..., 0] for name, c in quantized_network(net, spec.layers, *key)[1].items()}
        costs = {arch.name: {} for arch in archs}
        correct = total = logged = 0
        log(f"{spec.display_name}: hardware of {', '.join(a.name for a in archs)} "
            f"({precision_label(*key)} weights) on {device}")
        for chunk in recordings[key].chunks(max_samples, device, parallel):
            correct += int((chunk.predictions == chunk.labels).sum())
            total += len(chunk)
            for name, module in layers.items():
                shared = {}  # spike activity, reused across architectures
                for arch in archs:
                    cost = evaluate_layer(arch, chunk.inputs[name], codes[name],
                                          stride=module.stride[0], padding=module.padding[0],
                                          activity_cache=shared,
                                          output_spikes=int(chunk.output_counts[name].sum()))
                    if name in costs[arch.name]:
                        costs[arch.name][name].add(cost)
                    else:
                        costs[arch.name][name] = cost
            if _passed(total, logged, log_every):
                log(f"  {total} samples evaluated")
                logged = total
        if total != logged:
            log(f"  {total} samples evaluated")
        for arch in archs:
            results[arch.name] = (100 * correct / total, costs[arch.name])
    return results


def accuracy_sweep(model, configs, *, data_dir, max_samples=None, parallel=None, num_workers=4,
                   log_every=1000, log=print):
    """Test accuracy (%) of `model` for each weight quantization in `configs`,
    {label: (bits, scaling)}, with bits and scaling as in quantized_network
    (bits=None: the trained float weights). The test set (its first
    max_samples, if given; None or -1 = all) is read once; every batch runs through all
    configurations, `parallel` samples at once (default: the model's
    batch_size). No hardware evaluation. The accuracy so far is logged every
    `log_every` samples and at the end. Returns {label: accuracy}."""
    spec = models.get_spec(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = models.load_pretrained(spec, device).eval()
    params = models.load_params(spec.path(spec.params_yaml))
    loader = models.test_loader(spec, params, data_dir, max_samples, parallel, num_workers)
    nets = {label: quantized_network(net, spec.layers, bits, scaling)[0]
            for label, (bits, scaling) in configs.items()}
    correct, total, logged = dict.fromkeys(configs, 0), 0, 0
    log(f"{spec.display_name}: accuracy of {len(configs)} weight configuration"
        f"{'' if len(configs) == 1 else 's'}, {loader.batch_size} samples in parallel")

    def progress():
        log(f"  {total} samples: " + ", ".join(
            f"{name} {100 * c / total:.2f}%" for name, c in correct.items()))
    with torch.no_grad():
        for _, spikes, _, label in loader:
            spikes = spikes.to(device)
            for name, quantized in nets.items():
                correct[name] += int((predict_class(quantized(spikes)) == label).sum())
            total += len(label)
            if _passed(total, logged, log_every):
                progress()
                logged = total
    if total != logged:
        progress()
    return {name: 100 * c / total for name, c in correct.items()}


def _passed(total, logged, every):
    """True when a multiple of `every` samples was passed since the last log."""
    return total // every > logged // every


def _precision_key(bits, scaling):
    """(bits, scaling); float weights have no scaling."""
    return (None, None) if bits is None else (bits, scaling)

