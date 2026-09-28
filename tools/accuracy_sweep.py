"""Test accuracy of the pretrained networks with quantized weights.

    python tools/accuracy_sweep.py                          # both models, 2..8-bit and float
    python tools/accuracy_sweep.py --scaling max mse std3   # compare quantization ranges
    python tools/accuracy_sweep.py --model nmnist --layer-bits SF1=8 SF2=8   # mixed precision
    python tools/accuracy_sweep.py --model nmnist --sensitivity --bits 3 4   # one layer at a time

Weights are quantized symmetric-uniformly per layer (codes in
+/-(2^(b-1) - 1), as the hardware evaluation stores them), with the range set
by each --scaling: "max" (largest |weight|), "mse" (the clip with the least
squared error) or "std<k>" (k standard deviations). --layer-bits fixes the bit
width of the named layers (8, or float) while --bits sets the others.
--sensitivity quantizes one layer at a time, the rest staying float, to show
which layers limit the accuracy. Inputs stay binary spikes; only the software
network runs, no hardware estimation. Results are printed and saved to
logs/weight_quantization.csv.
"""
import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import models  # noqa: E402
import run  # noqa: E402
from evaluation.runner import accuracy_sweep  # noqa: E402
from hardware.architecture import scaling_std  # noqa: E402


def bit_width(text):
    return None if text == "float" else int(text)


def layer_bits(text):
    layer, _, bits = text.partition("=")
    if not layer or not bits:
        raise argparse.ArgumentTypeError(f"expected LAYER=BITS (e.g. SF2=8), got {text!r}")
    return layer, bit_width(bits)


def bits_label(bits):
    return "float" if bits is None else f"{bits}-bit"


def config_label(key):
    """Short progress label: "float", "4b std3" (all layers), "SC1 4b std3"."""
    quantized, bits, scaling = key
    if quantized == "float":
        return "float"
    return f"{'' if quantized == 'all' else quantized + ' '}{bits}b {scaling}"


def configurations(layers, args):
    """{(quantized layers, bits, scaling): (bits per layer, scaling)} for one model."""
    configs = {("float", None, None): (None, "max")} if None in args.bits else {}
    for scaling in args.scaling:
        for bits in args.bits:
            if bits is None:
                continue
            if args.sensitivity:
                for layer in layers:
                    configs[(layer, bits, scaling)] = ({layer: bits}, scaling)
            fixed = {layer: b for layer, b in args.layer_bits.items() if layer in layers}
            configs[("all", bits, scaling)] = (dict(dict.fromkeys(layers, bits), **fixed), scaling)
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", nargs="+", default=["nmnist", "gesture"],
                        choices=["nmnist", "gesture"])
    parser.add_argument("--bits", nargs="+", type=bit_width, default=[2, 3, 4, 5, 6, 8, None],
                        help="weight bit widths; 'float' = trained weights")
    parser.add_argument("--scaling", nargs="+", default=["std3"],
                        help="quantization ranges to compare: max, mse, std<k> (e.g. std3)")
    parser.add_argument("--layer-bits", nargs="+", type=layer_bits, default=[],
                        help="fixed bit widths for named layers, e.g. SF1=8 SF2=float")
    parser.add_argument("--sensitivity", action="store_true",
                        help="also quantize one layer at a time (the rest float)")
    parser.add_argument("--data", default=run.DATASET_DIR, help="dataset folder (default: run.py's)")
    parser.add_argument("-b", "--batch-size", type=int, help="default 32 (gesture: 2)")
    parser.add_argument("--batches", type=int, help="max batches (default: whole test set)")
    parser.add_argument("--every", type=int, default=2, help="print accuracy every N batches (default 2)")
    args = parser.parse_args()
    args.layer_bits = dict(args.layer_bits)

    # Reject bad options before reading any data.
    for scaling in args.scaling:
        scaling_std(scaling)
    all_layers = {layer for model in args.model for layer in models.get_spec(model).layers}
    unknown = set(args.layer_bits) - all_layers
    if unknown:
        parser.error(f"--layer-bits: no layer {sorted(unknown)} in {', '.join(args.model)} "
                     f"(layers: {sorted(all_layers)})")

    rows = []
    for model in args.model:
        layers = models.get_spec(model).layers
        configs = configurations(layers, args)
        accuracy = accuracy_sweep(model, {config_label(key): value
                                          for key, value in configs.items()},
                                  data_dir=args.data, batch_size=args.batch_size,
                                  max_batches=args.batches, log_every=args.every)
        result = {key: accuracy[config_label(key)] for key in configs}
        bit_widths = [b for b in args.bits if b is not None]
        fixed = {layer: b for layer, b in args.layer_bits.items() if layer in layers}
        reference = result.get(("float", None, None))
        print(f"\n{model} test accuracy (%)"
              + ("" if reference is None else f"; float weights: {reference:.2f}"))
        if fixed:
            print("  'all' = every layer at the column's bits, except "
                  + ", ".join(f"{layer} {bits_label(b)}" for layer, b in fixed.items()))
        for scaling in args.scaling:
            rows_shown = (list(layers) if args.sensitivity else []) + ["all"]
            print(f"  scaling {scaling}: quantized layer(s) x weight bits")
            print("    " + f"{'layer':<8}" + "".join(f"{bits_label(b):>9}" for b in bit_widths))
            for shown in rows_shown:
                print("    " + f"{shown:<8}" + "".join(
                    f"{result[(shown, b, scaling)]:9.2f}" for b in bit_widths))
        for (quantized, bits, scaling), value in result.items():
            rows.append({"model": model, "quantized_layers": quantized,
                         "weight_bits": bits_label(bits) if quantized != "float" else "float",
                         "scaling": scaling or "",
                         "fixed_layers": " ".join(f"{layer}={bits_label(b)}"
                                                  for layer, b in fixed.items())
                         if quantized == "all" else "",
                         "accuracy_percent": f"{value:.2f}"})
        print()

    os.makedirs(run.LOG_DIR, exist_ok=True)
    path = os.path.join(run.LOG_DIR, "weight_quantization.csv")
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
