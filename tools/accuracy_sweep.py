"""Test accuracy of the pretrained networks with quantized weights.

    python tools/accuracy_sweep.py                          # both models, 2..8-bit and float
    python tools/accuracy_sweep.py --scaling max mse std3   # compare quantization ranges
    python tools/accuracy_sweep.py --model gesture --bits 3 4 8 float --data /path/to/datasets

Weights of every weighted layer are quantized symmetric-uniformly per layer
(codes in +/-(2^(b-1) - 1), as the hardware evaluation stores them), with the
range set by each --scaling: "max" (largest |weight|), "mse" (the clip with
the least squared error) or "std<k>" (k standard deviations). Inputs stay
binary spikes. Only the software network runs, no hardware estimation.
Results are printed and saved to logs/weight_quantization.csv.
"""
import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import run  # noqa: E402
from evaluation.runner import accuracy_sweep  # noqa: E402
from hardware.architecture import scaling_std  # noqa: E402


def bit_width(text):
    return None if text == "float" else int(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", nargs="+", default=["nmnist", "gesture"],
                        choices=["nmnist", "gesture"])
    parser.add_argument("--bits", nargs="+", type=bit_width, default=[2, 3, 4, 5, 6, 8, None],
                        help="weight bit widths; 'float' = trained weights")
    parser.add_argument("--scaling", nargs="+", default=["max"],
                        help="quantization ranges to compare: max, mse, std<k> (e.g. std3)")
    parser.add_argument("--data", default=run.DATASET_DIR, help="dataset folder (default: run.py's)")
    parser.add_argument("-b", "--batch-size", type=int, help="default 32 (gesture: 2)")
    parser.add_argument("--batches", type=int, help="max batches (default: whole test set)")
    args = parser.parse_args()

    for scaling in args.scaling:
        scaling_std(scaling)   # reject unknown scalings before reading any data
    rows = []
    for model in args.model:
        accuracy = accuracy_sweep(model, args.bits, args.scaling, data_dir=args.data,
                                  batch_size=args.batch_size, max_batches=args.batches)
        print(f"\n{model} test accuracy (%)")
        print("  weights " + "".join(f"{s:>9}" for s in args.scaling))
        for bits in args.bits:
            if bits is None:
                print(f"  {'float':>7} {accuracy[(None, None)]:9.2f}")
                continue
            print(f"  {bits:>5}-b " + "".join(f"{accuracy[(bits, s)]:9.2f}" for s in args.scaling))
        for (bits, scaling), value in accuracy.items():
            rows.append({"model": model, "weight_bits": "float" if bits is None else bits,
                         "scaling": scaling or "", "accuracy_percent": f"{value:.2f}"})
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
