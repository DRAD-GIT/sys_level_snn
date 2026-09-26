"""Check the plain-PyTorch SRMLayer against a recorded slayerSNN run.

    python tools/compare_slayer_reference.py reference/nmnist_slayer.pt [--full]

Feeds the exact input spikes slayerSNN saw into the network built on
SRMLayer and reports, per profiled layer, how many spike entries differ and
the first time step where they do, the predicted classes, and the effect on
the hardware estimates. --full re-runs the whole test set (needs the dataset
in datasets/) and compares against slayerSNN's recorded predictions.

Identical spikes are expected almost everywhere. SLAYER's CUDA kernels and
PyTorch round float32 sums in different orders, so a membrane potential that
lands within rounding of the threshold can occasionally flip one spike.
"""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import models  # noqa: E402
from evaluation.probes import LayerProbe  # noqa: E402
from evaluation.software import predict_class  # noqa: E402
from hardware import evaluate_layer, quantize_weights  # noqa: E402
from tools.slayer_reference import FORMAT, unpack  # noqa: E402


def _hardware_energy(architectures, net, layer_names, inputs):
    """Network energy (nJ) per architecture for one set of layer inputs."""
    energy = {}
    for arch in architectures:
        total = 0.0
        for name in layer_names:
            module = getattr(net, name)
            codes, _ = quantize_weights(module.weight.detach()[..., 0], arch.precision.weight_bits)
            total += evaluate_layer(arch, inputs[name], codes, stride=module.stride[0],
                                    padding=module.padding[0]).energy_nj
        energy[arch.name] = total
    return energy


def compare(path, full=False, architectures=None, log=print):
    reference = torch.load(path, map_location="cpu", weights_only=True)
    if reference.get("format") != FORMAT:
        raise ValueError(f"{path}: unsupported reference format {reference.get('format')}")
    spec = models.get_spec(reference["model"])
    net = models.load_pretrained(spec).eval()
    layer_names = spec.layers
    probe = LayerProbe(net, layer_names)
    store = probe.inputs
    if architectures is None:
        import run
        architectures = run.ARCHITECTURES

    report = {"model": spec.name, "samples": len(reference["samples"]),
              "reader_mismatches": 0, "prediction_mismatches": 0,
              "layers": {name: {"differ": 0, "total": 0, "first_step": None} for name in layer_names},
              "output_differ": 0, "energy_ours": {}, "energy_slayer": {}}
    for sample in reference["samples"]:
        slayer_input = unpack(sample["slayer_input"])
        if not torch.equal(unpack(sample["our_input"]), slayer_input):
            report["reader_mismatches"] += 1
        probe.clear()
        with torch.no_grad():
            output = net(slayer_input[None])
        slayer_layers = {name: unpack(sample["layer_inputs"][name]) for name in layer_names}
        for name in layer_names:
            differ = store[name] != slayer_layers[name]
            entry = report["layers"][name]
            entry["differ"] += int(differ.sum())
            entry["total"] += differ.numel()
            if differ.any():
                step = int(differ.nonzero()[:, -1].min())
                entry["first_step"] = step if entry["first_step"] is None else min(entry["first_step"], step)
        report["output_differ"] += int((output != unpack(sample["output"])).sum())
        if int(predict_class(output)[0]) != sample["predicted"]:
            report["prediction_mismatches"] += 1
        for key, inputs in (("energy_ours", store), ("energy_slayer", slayer_layers)):
            for arch, value in _hardware_energy(architectures, net, layer_names, inputs).items():
                report[key][arch] = report[key].get(arch, 0.0) + value

    log(f"Model {spec.name}: {report['samples']} samples recorded on {reference.get('device')}")
    log(f"  input readers identical: {report['samples'] - report['reader_mismatches']}/{report['samples']}")
    for name, entry in report["layers"].items():
        fraction = entry["differ"] / entry["total"] if entry["total"] else 0.0
        log(f"  layer {name} input: {entry['differ']} of {entry['total']} entries differ "
            f"({fraction:.2e}); first differing step: {entry['first_step']}")
    log(f"  output spikes differing: {report['output_differ']}")
    log(f"  predicted class matches: {report['samples'] - report['prediction_mismatches']}/{report['samples']}")
    for arch in report["energy_ours"]:
        ours, theirs = report["energy_ours"][arch], report["energy_slayer"][arch]
        log(f"  {arch} energy from SRMLayer vs slayerSNN spikes: {ours:.6g} vs {theirs:.6g} nJ "
            f"({(ours - theirs) / theirs * 100 if theirs else 0.0:+.4f}%)")

    if full:
        recorded = reference.get("full")
        if not recorded:
            raise ValueError(f"{path} has no full-test-set predictions; export with --full")
        dataset = models.test_dataset(spec, models.load_params(spec.path(spec.params_yaml)))
        agree = correct = 0
        with torch.no_grad():
            for index, (theirs, label) in enumerate(zip(recorded["predictions"], recorded["labels"])):
                ours = int(predict_class(net(dataset[index][1][None]))[0])
                agree += ours == theirs
                correct += ours == label
        n = len(recorded["labels"])
        slayer_acc = 100 * sum(p == l for p, l in zip(recorded["predictions"], recorded["labels"])) / n
        report.update(full_agreement=agree / n, full_accuracy=100 * correct / n,
                      full_slayer_accuracy=slayer_acc)
        log(f"  full test set: SRMLayer {100 * correct / n:.2f}% vs slayerSNN {slayer_acc:.2f}% "
            f"accuracy; same prediction on {agree}/{n} samples")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare SRMLayer against a slayerSNN reference")
    parser.add_argument("reference", nargs="+")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    for reference_path in args.reference:
        compare(reference_path, full=args.full)
