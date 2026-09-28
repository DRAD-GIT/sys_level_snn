"""Main script: pick the model, the hardware architectures and the metrics, then

    python run.py

Hardware is composed from one crossbar type (architectures/crossbars.py, with
its memory cells given directly) and any Stages and Components you define:
each component has a name, how many are installed, how many are powered,
when (stages or a window) and its current / event energy. Command-line flags
override the run settings for one run.

Results: printed, and saved under logs/ (a JSON per architecture and the
comparison table logs/comparison_summary.csv).
"""
import argparse
import os

from architectures import crossbars
from evaluation.report import export, format_results
from hardware import Component, Precision, Stage, compose
from evaluation.runner import evaluate

# ============================================================================
# RUN SETTINGS
# ============================================================================
MODEL = "nmnist"       # "nmnist" or "gesture"
BATCH_SIZE = 16        # samples per forward pass; gesture is capped at 2
MAX_BATCHES = 1        # None = the full test set (slow)
# Folder holding the dataset folders (names starting with N-MNIST / Gesture),
# or the dataset folder itself.
DATASET_DIR = None     # e.g. "/data/neuromorphic"

# ============================================================================
# HARDWARE: architectures to evaluate (names must be unique)
# ============================================================================
VDD = 1.1
RRAM_1BIT_XBAR = compose(
    "rram_1bit_conv_xbar",
    Precision(weight_bits=4, weight_encoding="twos_complement"),
    [
        # Crossbar: 64x64 tiles, 0.2 V read made from VDD, 5 ns read stage.
        crossbars.conv_xbar(cell_bits=1, r_on=20e3, r_off=200e3,   # 1-bit RRAM cells
                            rows=64, cols=64, v_read=0.2, read_ns=5.0,
                            cell_supply_v=VDD),
        Stage("fire", 2.0, level="timestep"),                  # once per time bin, after the reads
        Component("sl_ota", count="physical_columns",          # one per column,
                  on={"rule": "used_columns", "gated": True},  # on when its tile gets a spike
                  during="read", supply_v=VDD, static_ua=10.0),
        Component("lif", count="outputs", during="fire",       # one per output neuron
                  supply_v=VDD, static_ua=10.0),
    ],
    conv_mapping="sequential",                                 # or "parallel"
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
    parser.add_argument("-b", "--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--batches", type=int, default=MAX_BATCHES,
                        help="max batches to run; -1 = full test set")
    args = parser.parse_args()

    results = evaluate(args.model, ARCHITECTURES, data_dir=args.data, batch_size=args.batch_size,
                       max_batches=None if args.batches in (None, -1) else args.batches)
    print(format_results(results, METRICS))
    for row in export(results, ARCHITECTURES, args.model, METRICS, LOG_DIR):
        print(f"saved {row['result_json']}")


if __name__ == "__main__":
    main()
