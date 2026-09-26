"""Record a reference run of the ORIGINAL slayerSNN framework.

Run once on a machine that has slayerSNN (with its CUDA backend), a GPU and
the dataset in datasets/. It builds each network on slayerSNN (same weights,
via backend=snn.layer), runs the first N test samples, and saves everything
tools/compare_slayer_reference.py needs to check the plain-PyTorch SRMLayer:

    python tools/export_slayer_reference.py --model nmnist --samples 20
    python tools/export_slayer_reference.py --model gesture --samples 22 --full

--full also records slayerSNN's predicted class for every test sample, so the
comparison can check the full-test-set accuracy too (slow for gesture).
Copy the resulting reference/*.pt files back into the repository.
"""
import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import models  # noqa: E402
from evaluation.probes import LayerProbe  # noqa: E402
from tools.slayer_reference import FORMAT, pack  # noqa: E402


def slayer_input(snn, dataset, index):
    """The sample's input spikes, read with slayerSNN's own readers."""
    path = dataset.event_file(index)
    events = snn.io.read2Dspikes(path) if path.endswith(".bin") else snn.io.readNpSpikes(path)
    return events.toSpikeTensor(torch.zeros((*dataset.sensor_shape, dataset.n_time_bins)),
                                dataset.sampling_time)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=sorted(models.MODEL_MODULES))
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--full", action="store_true", help="also record predictions for the whole test set")
    parser.add_argument("--out", help="default: reference/<model>_slayer.pt")
    args = parser.parse_args()

    import slayerSNN as snn  # type: ignore

    device = torch.device("cuda")
    spec = models.get_spec(args.model)
    params = models.load_params(spec.path(spec.params_yaml))
    net = models.load_pretrained(spec, device, backend=snn.layer).eval()
    dataset = models.test_dataset(spec, params)
    probe = LayerProbe(net, spec.layers)

    samples = []
    with torch.no_grad():
        for index in range(min(args.samples, len(dataset))):
            _, ours, _, label = dataset[index]
            reference = slayer_input(snn, dataset, index)
            probe.clear()
            output = net(reference[None].to(device))
            samples.append({
                "index": index, "label": int(label),
                "slayer_input": pack(reference), "our_input": pack(ours),
                "layer_inputs": {name: pack(probe.inputs[name]) for name in spec.layers},
                "output": pack(output),
                "predicted": int(snn.predict.getClass(output)[0]),
            })
            print(f"sample {index}: label {label}, predicted {samples[-1]['predicted']}")

        full = None
        if args.full:
            predictions, labels = [], []
            for index in range(len(dataset)):
                output = net(slayer_input(snn, dataset, index)[None].to(device))
                predictions.append(int(snn.predict.getClass(output)[0]))
                labels.append(int(dataset[index][3]))
            correct = sum(p == l for p, l in zip(predictions, labels))
            print(f"slayerSNN full test accuracy: {100 * correct / len(labels):.2f}%")
            full = {"predictions": predictions, "labels": labels}

    out = args.out or os.path.join(ROOT, "reference", f"{args.model}_slayer.pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save({"format": FORMAT, "model": args.model, "device": torch.cuda.get_device_name(),
                "torch": str(torch.__version__), "samples": samples, "full": full}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
