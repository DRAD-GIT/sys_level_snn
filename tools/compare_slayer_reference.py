"""Check the plain-PyTorch SRMLayer against a recorded slayerSNN run.

    python tools/compare_slayer_reference.py reference/nmnist_slayer.pt [--full --data DIR]

Feeds the exact input spikes slayerSNN saw into the network built on
SRMLayer and reports, per profiled layer, how many spike entries differ and
the first time step where they do, the predicted classes, and the effect on
the hardware estimates. --full re-runs the whole test set (needs the dataset:
--data) and compares against slayerSNN's recorded predictions.

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


def _hardware_energy(architectures, net, layer_names, inputs, output_spikes):
    """Network energy (nJ) per architecture for one set of layer inputs and
    LIF output spike counts."""
    energy = {}
    for arch in architectures:
        total = 0.0
        for name in layer_names:
            module = getattr(net, name)
            codes, _ = quantize_weights(module.weight.detach()[..., 0].cpu(),
                                        arch.precision.weight_bits, arch.precision.weight_scaling)
            total += evaluate_layer(arch, inputs[name], codes, stride=module.stride[0],
                                    padding=module.padding[0],
                                    output_spikes=output_spikes.get(name, 0)).energy_nj
        energy[arch.name] = total
    return energy


def compare(path, full=False, data_dir=None, architectures=None, device=None, parallel=None,
            num_workers=4, log=print):
    """device: where SRMLayer runs; default the GPU if available (as run.py).
    parallel / num_workers: samples evaluated at once (default: the model's
    batch_size) and file-reading processes, for the full test set."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    reference = models.load_tensors(path)
    if reference.get("format") != FORMAT:
        raise ValueError(f"{path}: unsupported reference format {reference.get('format')}")
    spec = models.get_spec(reference["model"])
    net = models.load_pretrained(spec, device).eval()
    layer_names = spec.layers
    probe = LayerProbe(net, layer_names)
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
            output = net(slayer_input[None].to(device)).cpu()
        store = {name: probe.inputs[name].cpu() for name in layer_names}
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
        for key, inputs, spikes in (("energy_ours", store, probe.output_spikes),
                                    ("energy_slayer", slayer_layers, sample["output_spikes"])):
            for arch, value in _hardware_energy(architectures, net, layer_names, inputs,
                                                spikes).items():
                report[key][arch] = report[key].get(arch, 0.0) + value

    log(f"Model {spec.name}: {report['samples']} samples recorded on {reference.get('device')}; "
        f"SRMLayer on {device}")
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
        loader = models.test_loader(spec, models.load_params(spec.path(spec.params_yaml)), data_dir,
                                    parallel=parallel, num_workers=num_workers)
        agree = correct = done = 0
        n = len(recorded["labels"])
        with torch.no_grad():
            for _, spikes, _, _ in loader:
                for ours in predict_class(net(spikes.to(device))).tolist():
                    agree += ours == recorded["predictions"][done]
                    correct += ours == recorded["labels"][done]
                    done += 1
                    if done % 1000 == 0:
                        log(f"    {done}/{n} samples: same prediction on {agree}")
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
    parser.add_argument("--data", help="dataset folder, or a folder holding it (for --full)")
    parser.add_argument("--device", help="where SRMLayer runs (cpu / cuda); default cuda if available")
    parser.add_argument("--parallel", type=int,
                        help="samples evaluated at once for --full (default: N-MNIST 50, gesture 2)")
    args = parser.parse_args()
    for reference_path in args.reference:
        compare(reference_path, full=args.full, data_dir=args.data, device=args.device,
                parallel=args.parallel)
