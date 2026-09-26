"""Main script: pick the model, the hardware architectures and the metrics, then

    python run.py

Hardware is composed from building blocks (architectures/): a memory, one
crossbar type, any periphery blocks and a neuron. Swap or re-parameterize any
block here; add new ones to the architectures/ modules. Command-line flags
override the run settings for one run.

Results: printed, and saved under logs/ (a JSON per architecture and the
comparison table logs/comparison_summary.csv).
"""
import argparse
import os

from architectures import crossbars, designs, memories, neurons, periphery
from evaluation.report import export, format_results
from hardware import Precision, compose
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
RRAM_1BIT_XBAR = compose(
    "rram_1bit_conv_xbar",
    Precision(weight_bits=4, weight_encoding="twos_complement"),
    blocks=[
        crossbars.conv_xbar(memories.RRAM_1BIT, rows=64, cols=64, v_read=0.2, read_ns=5.0),
        periphery.source_line_ota(static_ua=10.0),            # powered only when spikes arrive
        periphery.slice_mirrors(),                             # binary-weighted slice sum
        neurons.lif_neuron(static_ua=10.0, fire_ns=2.0),       # comparator on for the fire step
    ],
    conv_mapping="sequential",                                 # or "parallel"
)

ARCHITECTURES = [RRAM_1BIT_XBAR, designs.conventional(), designs.c3cim()]

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
