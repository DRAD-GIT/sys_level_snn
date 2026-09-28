"""Recorded forward passes: run the network once per weight quantization and
keep what the hardware evaluation needs, so it never reruns inference.

A recording (recordings/<model>_<bits>b_<scaling>/) holds, for every sample of
the test set (or its first samples):
  - the input spikes of every weighted layer, all channels, pixels and time
    bins, bit-packed (1 bit per entry, time innermost) and zlib-compressed;
  - the spikes each layer's LIF neurons emit: a count per sample (always),
    and with full_outputs the spike tensors themselves (same packing);
  - the predicted class and the label.
Files: meta.json (what was recorded, and a fingerprint of the checkpoint, the
neuron YAML, the SNN code and the quantization) and chunk_<first sample>.npz
(one batch each). A recording whose fingerprint no longer matches is recorded
again rather than reused.
"""
import copy
import hashlib
import json
import os
import shutil
import time

import numpy as np
import torch

import models
from evaluation.probes import LayerProbe
from evaluation.software import predict_class
from hardware import quantize_weights

FORMAT = 1
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SNN_SOURCES = ("models/srm.py", "models/events.py")


def quantized_network(net, layer_names, bits, scaling="std3"):
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


def precision_label(bits, scaling):
    return "float" if bits is None else f"{bits}b_{scaling}"


def recording_path(recording_dir, model, bits, scaling):
    return os.path.join(recording_dir, f"{model}_{precision_label(bits, scaling)}")


def fingerprint(spec, bits, scaling):
    """Identifies everything a recording depends on."""
    digest = hashlib.sha256(f"format {FORMAT}; {spec.name}; {bits}; {scaling}".encode())
    for path in (spec.path(spec.checkpoint), spec.path(spec.params_yaml),
                 *(os.path.join(_ROOT, p) for p in _SNN_SOURCES)):
        with open(path, "rb") as file:
            digest.update(file.read())
    return digest.hexdigest()


def pack(spikes):
    """Spike tensor -> (bit-packed uint8, shape, amplitude)."""
    values = spikes.detach().cpu()
    nonzero = values[values != 0]
    amplitude = float(nonzero[0]) if len(nonzero) else 1.0
    if len(nonzero) and not torch.all(nonzero == amplitude):
        raise ValueError("not a spike tensor: nonzero entries differ")
    return (np.packbits((values != 0).numpy().reshape(-1)), np.array(values.shape, dtype=np.int64),
            np.float32(amplitude))


def unpack(bits, shape, amplitude):
    count = int(np.prod(shape))
    spikes = np.unpackbits(bits, count=count).reshape(tuple(shape)).astype(np.float32)
    return torch.from_numpy(spikes * np.float32(amplitude))


class Chunk:
    """One recorded batch: labels, predictions, and per layer the input
    spikes, output spike counts per sample and (if recorded) output spikes."""

    def __init__(self, data, layers, samples):
        n = samples
        self.labels = torch.from_numpy(data["labels"][:n])
        self.predictions = torch.from_numpy(data["predictions"][:n])
        self.inputs = {name: unpack(data[f"in_{name}"], data[f"in_{name}_shape"],
                                    data[f"in_{name}_amplitude"])[:n] for name in layers}
        self.output_counts = {name: data[f"out_{name}_count"][:n] for name in layers}
        self.outputs = {name: unpack(data[f"out_{name}"], data[f"out_{name}_shape"],
                                     data[f"out_{name}_amplitude"])[:n]
                        for name in layers if f"out_{name}" in data}

    def __len__(self):
        return len(self.labels)


class Recording:
    def __init__(self, path):
        self.path = path
        with open(os.path.join(path, "meta.json"), encoding="utf-8") as file:
            self.meta = json.load(file)

    @property
    def samples(self):
        return self.meta["samples"]

    def chunks(self, max_samples=None):
        """The recorded batches in order, up to max_samples samples."""
        remaining = self.samples if max_samples is None else min(max_samples, self.samples)
        for entry in self.meta["chunks"]:
            if remaining <= 0:
                return
            with np.load(os.path.join(self.path, entry["file"])) as data:
                chunk = Chunk(data, self.meta["layers"], min(entry["samples"], remaining))
            remaining -= len(chunk)
            yield chunk


def open_recording(recording_dir, model, bits, scaling, max_samples=None, full_outputs=False):
    """An existing, matching recording with enough samples, else None."""
    path = recording_path(recording_dir, model, bits, scaling)
    try:
        recording = Recording(path)
    except (OSError, ValueError):
        return None
    meta = recording.meta
    enough = meta["complete"] if max_samples is None else (meta["complete"] or meta["samples"] >= max_samples)
    if (meta.get("format") != FORMAT or not meta.get("finished") or not enough
            or meta.get("fingerprint") != fingerprint(models.get_spec(model), bits, scaling)
            or (full_outputs and not meta.get("full_outputs"))):
        return None
    return recording


def record(model, precisions, *, data_dir, recording_dir, max_samples=None, parallel=None,
           full_outputs=False, num_workers=4, log_every=1000, log=print):
    """Run the test set (its first max_samples, if given) once through the
    network quantized as each of `precisions` [(bits, scaling), ...], all in
    one pass over the data, and save a recording for each.
    Returns {(bits, scaling): Recording}."""
    spec = models.get_spec(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = models.load_pretrained(spec, device).eval()
    params = models.load_params(spec.path(spec.params_yaml))
    loader = models.test_loader(spec, params, data_dir, max_samples, parallel, num_workers)
    groups = []
    for bits, scaling in precisions:
        path = recording_path(recording_dir, model, bits, scaling)
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path)
        quantized = quantized_network(net, spec.layers, bits, scaling)[0]
        groups.append({"key": (bits, scaling), "path": path, "net": quantized,
                       "probe": LayerProbe(quantized, spec.layers, keep_outputs=full_outputs),
                       "chunks": [], "correct": 0})
    log(f"{spec.display_name}: recording the forward pass for "
        f"{', '.join(precision_label(*g['key']) for g in groups)}; "
        f"{loader.batch_size} samples in parallel")

    total = logged = 0
    start = time.time()
    with torch.no_grad():
        for _, spikes, _, label in loader:
            spikes = spikes.to(device)
            for g in groups:
                g["probe"].clear()
                predicted = predict_class(g["net"](spikes))
                g["correct"] += int((predicted == label).sum())
                arrays = {"labels": label.numpy().astype(np.int64),
                          "predictions": predicted.numpy().astype(np.int64)}
                for name in spec.layers:
                    for prefix, tensor in (("in", g["probe"].inputs[name]),
                                           ("out", g["probe"].outputs.get(name))):
                        if tensor is not None:
                            bits_, shape, amplitude = pack(tensor)
                            arrays.update({f"{prefix}_{name}": bits_, f"{prefix}_{name}_shape": shape,
                                           f"{prefix}_{name}_amplitude": amplitude})
                    arrays[f"out_{name}_count"] = g["probe"].output_counts[name].numpy().astype(np.int64)
                file = f"chunk_{total:06d}.npz"
                np.savez_compressed(os.path.join(g["path"], file), **arrays)
                g["chunks"].append({"file": file, "samples": len(label)})
            total += len(label)
            if total // log_every > logged // log_every:
                log(f"  {total} samples recorded: accuracy " + ", ".join(
                    f"{precision_label(*g['key'])} {100 * g['correct'] / total:.2f}%" for g in groups))
                logged = total

    recordings = {}
    for g in groups:
        g["probe"].remove()
        bits, scaling = g["key"]
        meta = {"format": FORMAT, "model": model, "weight_bits": bits, "weight_scaling": scaling,
                "fingerprint": fingerprint(spec, bits, scaling), "layers": list(spec.layers),
                "samples": total, "complete": total == len(loader.dataset.dataset)
                if isinstance(loader.dataset, torch.utils.data.Subset) else True,
                "full_outputs": full_outputs, "accuracy": 100 * g["correct"] / total,
                "chunks": g["chunks"], "finished": True,
                "recorded": time.strftime("%Y-%m-%d %H:%M:%S"),
                "seconds": round(time.time() - start, 1)}
        with open(os.path.join(g["path"], "meta.json"), "w", encoding="utf-8") as file:
            json.dump(meta, file, indent=2)
        recordings[g["key"]] = Recording(g["path"])
        size = sum(os.path.getsize(os.path.join(g["path"], c["file"])) for c in g["chunks"])
        log(f"  saved {g['path']}: {total} samples, {size / 2 ** 20:.1f} MiB, "
            f"accuracy {meta['accuracy']:.2f}%")
    return recordings
