"""Main script: pick the model, the hardware architectures and the metrics, then

    python run.py

Hardware is composed from one crossbar type (crossbars/, with its memory cells
given directly), the Mapping of the network onto it, and any Components you
define: each has a name, how many are installed and powered, optionally a
stage of the timeline it defines, when it draws current (start / end), and
its current / event energy. Command-line flags override the run settings for
one run.

Results: printed, and saved under logs/ (a JSON per architecture and the
comparison table logs/comparison_summary.csv).
"""
import argparse
import os

import crossbars
from evaluation.report import export, format_results
from evaluation.runner import evaluate
from hardware import Component, Mapping, compose

# ============================================================================
# RUN SETTINGS
# ============================================================================
MODEL = "nmnist"       # "nmnist" or "gesture"
MAX_SAMPLES = 100      # first test samples to evaluate; None = the full test set
PARALLEL = None        # samples evaluated at once; None = the model's default (N-MNIST 50, gesture 2)
# Folder holding the dataset folders (names starting with N-MNIST / Gesture),
# or the dataset folder itself.
DATASET_DIR = None     # e.g. "/data/neuromorphic"

# ============================================================================
# HARDWARE: architectures to evaluate (names must be unique)
# ============================================================================
VDD = 1.1
RRAM_1BIT_XBAR = compose(
    "rram_1bit_conv_xbar",
    # conv="parallel": one weight copy per output position (copies that fit
    # share a tile), so every analog LIF has its own columns; it cannot store
    # and restore its membrane potential to serve several pixels.
    Mapping(weight_bits=6, weight_scaling="std3", weight_encoding="twos_complement",
            conv="parallel"),
    [
        # Crossbar: 64x64 tiles of 1-bit RRAM, 0.2 V read made from VDD; its
        # "read" stage (5 ns) runs in every activation.
        crossbars.conv_xbar(cell_bits=1, r_on=20e3, r_off=200e3, rows=64, cols=64,
                            v_read=0.2, cell_supply_v=VDD, stage="read", stage_ns=5.0),
        Component("sl_ota", count="physical_columns",          # one per column,
                  on={"rule": "used_columns", "gated": True},  # on when its tile gets a spike
                  start="read", end="read",                    # powered during the read
                  supply_v=VDD, static_ua=10.0),
        Component("lif", count="outputs",                      # one per output neuron
                  stage="fire", stage_ns=2.0,                  # once per time bin, after the reads
                  supply_v=VDD, static_ua=10.0),               # powered during fire
    ],
)

ARCHITECTURES = [RRAM_1BIT_XBAR]

# ============================================================================
# METRICS: switch each reported metric on or off
# ============================================================================
METRICS = {
    "accuracy": True,      # software accuracy at the architecture's weight precision
    "energy": True,        # nJ per inference
    "latency": True,       # us per inference
    "power": True,         # mW, energy / latency
    "area": False,         # mm^2, installed components (needs component areas)
    "tops_per_w": True,    # dense-MAC efficiency (every input, every time bin)
    "pj_per_sop": True,    # energy per synaptic operation (per input spike x fan-out)
    "layers": True,        # per-layer results
    "components": True,    # per-component breakdown within each layer
}

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


def main():
    parser = argparse.ArgumentParser(description="SNN inference + CIM hardware metrics")
    parser.add_argument("--model", default=MODEL, choices=["nmnist", "gesture"])
    parser.add_argument("--data", default=DATASET_DIR, help="dataset folder (overrides DATASET_DIR)")
    parser.add_argument("--samples", type=int, default=MAX_SAMPLES,
                        help="first test samples to evaluate; -1 = the full test set")
    parser.add_argument("--every", type=int, default=1000,
                        help="print the accuracy so far every N samples (default 1000)")
    parser.add_argument("--parallel", type=int, default=PARALLEL,
                        help="samples evaluated at once (default: N-MNIST 50, gesture 2)")
    args = parser.parse_args()

    results = evaluate(args.model, ARCHITECTURES, data_dir=args.data,
                       max_samples=None if args.samples in (None, -1) else args.samples,
                       parallel=args.parallel, log_every=args.every)
    print(format_results(results, METRICS))
    for row in export(results, ARCHITECTURES, args.model, METRICS, LOG_DIR):
        print(f"saved {row['result_json']}")


if __name__ == "__main__":
    main()
