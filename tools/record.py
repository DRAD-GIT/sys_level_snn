"""Record the forward pass once, ahead of any hardware evaluation.

    python tools/record.py --data /path/to/datasets                 # both models, run.py's quantization
    python tools/record.py --model gesture --bits 6 --scaling std3 --data /path/to/datasets
    python tools/record.py --bits 4 6 8 --scaling std3 mse          # several quantizations, one pass

Runs each test set once per weight quantization and saves every weighted
layer's input spikes, output spike counts (--full-outputs: the output spikes
too), the predictions and the labels, compressed, under run.py's
RECORDING_DIR (or --recordings). run.py then evaluates hardware from these
without rerunning inference.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import models  # noqa: E402
import run  # noqa: E402
from evaluation.recording import record  # noqa: E402
from hardware.architecture import scaling_std  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", nargs="+", default=["nmnist", "gesture"],
                        choices=sorted(models.MODEL_MODULES))
    parser.add_argument("--bits", nargs="+", default=[run.WEIGHT_BITS], type=models.bits_argument,
                        help="weight bit widths (default: run.py's WEIGHT_BITS); 'float' = trained weights")
    parser.add_argument("--scaling", nargs="+", default=[run.WEIGHT_SCALING],
                        help="quantization ranges (default: run.py's WEIGHT_SCALING)")
    parser.add_argument("--data", default=run.DATASET_DIR, help="dataset folder (default: run.py's)")
    parser.add_argument("--recordings", default=run.RECORDING_DIR, help="where to save the recordings")
    parser.add_argument("--samples", type=models.samples_argument, default=None,
                        help="first N test samples to record; -1 = all (default: all)")
    parser.add_argument("--parallel", type=models.positive_argument, help="samples run at once (default: N-MNIST 50, gesture 2)")
    parser.add_argument("--full-outputs", action="store_true",
                        help="also save every layer's output spikes (not only their counts)")
    parser.add_argument("--every", type=models.positive_argument, default=1000, help="print progress every N samples")
    args = parser.parse_args()
    for scaling in args.scaling:
        scaling_std(scaling)
    precisions = list(dict.fromkeys((None, None) if bits is None else (bits, scaling)
                                    for bits in args.bits for scaling in args.scaling))
    for model in args.model:
        record(model, precisions, data_dir=args.data, recording_dir=args.recordings,
               max_samples=args.samples, parallel=args.parallel, full_outputs=args.full_outputs,
               log_every=args.every)


if __name__ == "__main__":
    main()
