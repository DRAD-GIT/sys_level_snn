"""Main script: pick the model, the hardware architectures and the metrics, then

    python run.py

Architectures are defined in architectures/ (one file each: crossbar,
precision, conv mapping, timeline stages, components). Copy a file to start a
new design, or build a hardware.Architecture right here and add it to
ARCHITECTURES. Command-line flags override the run settings for one run.

Results: printed, and saved under logs/ (a JSON per architecture and the
comparison table logs/comparison_summary.csv).
"""
import argparse
import os

from architectures import c3cim, conventional, rram_1bit
from evaluation.report import export, format_results
from evaluation.runner import evaluate

# ============================================================================
# RUN SETTINGS
# ============================================================================
MODEL = "nmnist"       # "nmnist" or "gesture"
BATCH_SIZE = 1         # gesture is capped at 2
MAX_BATCHES = 1        # None = the full test set (slow)

# ============================================================================
# HARDWARE: architectures to evaluate (names must be unique)
# ============================================================================
ARCHITECTURES = [rram_1bit.ARCH, conventional.ARCH, c3cim.ARCH]

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
    parser.add_argument("-b", "--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--batches", type=int, default=MAX_BATCHES,
                        help="max batches to run; -1 = full test set")
    args = parser.parse_args()

    results = evaluate(args.model, ARCHITECTURES, batch_size=args.batch_size,
                       max_batches=None if args.batches in (None, -1) else args.batches)
    print(format_results(results, METRICS))
    for row in export(results, ARCHITECTURES, args.model, METRICS, LOG_DIR):
        print(f"saved {row['result_json']}")


if __name__ == "__main__":
    main()
