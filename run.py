"""Main script. The flow, top to bottom:

  1. the model, its pretrained weights and the dataset;
  2. the weight quantization;
  3. the hardware: one crossbar type (crossbars/, with its memory cells given
     directly), then its Components: each has a name, how many are installed
     and powered, optionally its time (time_ns: a step of the timeline), when
     it draws current (when; default: during its step), and its current /
     event energy;
  4. the forward pass: every weighted layer's input spikes and output spikes,
     for every time bin and sample, recorded once per weight quantization
     under recordings/ and reused by later runs (--rerecord to redo);
  5. the metrics, estimated from the recording.

    python run.py

Command-line flags override the settings for one run. Results: printed, and
saved under logs/ (a JSON per architecture and logs/comparison_summary.csv).
"""
import argparse
import os

import crossbars
import models
from evaluation.report import export, format_results
from evaluation.runner import evaluate
from hardware import Component, Mapping, compose

ROOT = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# 1. MODEL, PRETRAINED WEIGHTS AND DATASET
# ============================================================================
MODEL = "nmnist"       # "nmnist" or "gesture" (pretrained weights in pretrained/)
# Folder holding the dataset folders (names starting with N-MNIST / Gesture),
# or the dataset folder itself.
DATASET_DIR = None     # e.g. "/data/neuromorphic"
MAX_SAMPLES = -1       # first test samples to evaluate; -1 = the full test set
PARALLEL = None        # samples processed at once; None = defaults (recording: N-MNIST 50,
                       # gesture 2; hardware evaluation: one recorded file per step)
# Recorded forward passes (layer inputs and outputs), reused across runs.
RECORDING_DIR = os.path.join(ROOT, "recordings")

# ============================================================================
# 2. WEIGHT QUANTIZATION
# ============================================================================
WEIGHT_BITS = 6        # None = float weights (analog encoding only)
WEIGHT_SCALING = "std3"  # "std<k>", "max" or "mse" (see README, Mapping)

# ============================================================================
# 3. HARDWARE: the crossbar first, then its components (names must be unique)
# ============================================================================
VDD = 1.1
RRAM_1BIT_XBAR = compose(
    "rram_1bit_conv_xbar",
    # conv="parallel": one weight copy per output position (copies that fit
    # share a tile), so every analog LIF has its own columns; it cannot store
    # and restore its membrane potential to serve several pixels.
    Mapping(weight_bits=WEIGHT_BITS, weight_scaling=WEIGHT_SCALING,
            weight_encoding="twos_complement", conv="parallel",
            columns="contiguous"),   # weight columns from column 0 on, filling one group after another
    [
        # Crossbar: 64x64 tiles of 1-bit RRAM, 0.2 V read made from VDD; its
        # read, the step "cells" (5 ns), runs in every activation.
        crossbars.conv_xbar(cell_bits=1, r_on=20e3, r_off=200e3, rows=64, cols=64,
                            v_read=0.2, cell_supply_v=1.1, time_ns=5.0,
                            when=("cells.start", "lif.end")),       # conducts until the LIF has fired
        Component("sl_ota", count="physical_columns",               # one per column,
                  powered={"rule": "used_columns"},                 # on for the used columns
                  when=("cells.start", "lif.end"),                  # from the read until the LIF has fired
                  supply_v=1.1, static_ua=10.0),
        Component("lif", count="outputs",                           # one per output neuron
                  time_ns=2.0,                                      # step: once per time bin, after the reads
                  supply_v=VDD, static_ua=10.0),                    # powered during its step
    ],
    # Descriptive specifications for the paper table (no effect on any cost);
    # cell precision, R_High/R_Low and accumulation are read from the crossbar.
    specs={"tech": "40nm", "supply": 1.1, "device": "RRAM", "bitcell": "1T1R",
           "sensing": "Current"},
)

C3CIM_XBAR = compose(
    "c3cim_xbar",
    Mapping(weight_bits=WEIGHT_BITS, weight_scaling=WEIGHT_SCALING,
            weight_encoding="twos_complement", conv="parallel",
            columns="contiguous"),  # drivers of empty column groups stay off
    [
        # Crossbar: 64x64 tiles of 1-bit cells, all rows driven at once (set
        # active_rows for row phases); its step "column_source" runs in every
        # activation. Built in:
        #   column_source  a constant current source per column, column_ua
        #                  each, on for the used columns (during `when`);
        #   column_driver  one per driver_group columns, driver_ua each, on
        #                  during the same window if its group holds weights.
        crossbars.c3cim_xbar(cell_bits=1, r_on=2e3, r_off=20e3, rows=64, cols=64,
                             time_ns=177.0, supply_v=1.1,
                             when=("column_source.start", "lif.end"),  # sources and drivers on until the LIF has fired
                             column_ua=0.1, column_area_um2=0.0,
                             driver_ua=11.87, driver_group=32, driver_area_um2=0.0),
        Component("vi", count="physical_columns",                   # V-I converter per column,
                  powered="used_columns",                           # on for the used columns
                  time_ns=10.0,                                     # step: after the sources, per activation
                  when=("vi.start", "lif.end"),                     # from its first step until the LIF has fired
                  supply_v=1.1, static_ua=24.3),
        Component("lif", count="outputs",                           # one per output neuron
                  time_ns=2.0,                                      # step: once per time bin, after the reads
                  supply_v=VDD, static_ua=6.0),                     # powered during its step
    ],
    specs={"tech": "40nm", "supply": 1.1, "device": "RRAM", "bitcell": "2T1R",
           "sensing": "Voltage"},
)

C3CIM_OP_XBAR = compose(
    "c3cim_op_xbar",
    Mapping(weight_bits=WEIGHT_BITS, weight_scaling=WEIGHT_SCALING,
            weight_encoding="twos_complement", conv="parallel",
            columns="contiguous"),  # drivers of empty column groups stay off
    [
        # Crossbar: 64x64 tiles of 1-bit cells, all rows driven at once (set
        # active_rows for row phases); its step "column_source" runs in every
        # activation. Built in:
        #   column_source  a constant current source per column, column_ua
        #                  each, on for the used columns (during `when`);
        #   column_driver  one per driver_group columns, driver_ua each, on
        #                  during the same window if its group holds weights.
        crossbars.c3cim_xbar(cell_bits=1, r_on=2e3, r_off=20e3, rows=64, cols=64,
                             time_ns=23.0, supply_v=1.1,
                             when=("column_source.start", "lif.end"),  # sources and drivers on until the LIF has fired
                             column_ua=0.1, column_area_um2=0.0,
                             driver_ua=11.87, driver_group=32, driver_area_um2=0.0),
        Component("vi", count="physical_columns",                   # V-I converter per column,
                  powered="used_columns",                           # on for the used columns
                  time_ns=10.0,                                     # step: after the sources, per activation
                  when=("vi.start", "lif.end"),                     # from its first step until the LIF has fired
                  supply_v=1.1, static_ua=24.3),
        Component("lif", count="outputs",                           # one per output neuron
                  time_ns=2.0,                                      # step: once per time bin, after the reads
                  supply_v=VDD, static_ua=6.0),                     # powered during its step
    ],
    specs={"tech": "40nm", "supply": 1.1, "device": "RRAM", "bitcell": "2T1R",
           "sensing": "Voltage"},
)

# Designs evaluated and compared side by side (same weight quantization:
# they share one recorded forward pass).
ARCHITECTURES = [RRAM_1BIT_XBAR, C3CIM_XBAR, C3CIM_OP_XBAR]
# Our work: listed last in the paper table (tools/latex_table.py), where its
# values that beat every other row are bold.
OURWORK = [C3CIM_XBAR, C3CIM_OP_XBAR]

# ============================================================================
# 4. METRICS: switch each reported metric on or off
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

LOG_DIR = os.path.join(ROOT, "logs")


def main():
    """4. Record the forward pass (or reuse the recording), 5. estimate the metrics."""
    parser = argparse.ArgumentParser(description="SNN inference + CIM hardware metrics")
    parser.add_argument("--model", default=MODEL, choices=["nmnist", "gesture"])
    parser.add_argument("--data", default=DATASET_DIR, help="dataset folder (overrides DATASET_DIR)")
    parser.add_argument("--samples", type=models.samples_argument, default=MAX_SAMPLES,
                        help="first N test samples to evaluate; -1 = all (default: MAX_SAMPLES)")
    parser.add_argument("--every", type=models.positive_argument, default=1000,
                        help="print progress every N samples (default 1000)")
    parser.add_argument("--parallel", type=models.positive_argument, default=PARALLEL,
                        help="samples processed at once: when recording (default: N-MNIST 50, "
                             "gesture 2) and per hardware evaluation step (default: one recorded file)")
    parser.add_argument("--recordings", default=RECORDING_DIR, help="folder of recorded forward passes")
    parser.add_argument("--rerecord", action="store_true", help="record the forward pass again")
    args = parser.parse_args()

    results = evaluate(args.model, ARCHITECTURES, data_dir=args.data,
                       recording_dir=args.recordings,
                       max_samples=args.samples,
                       parallel=args.parallel, rerecord=args.rerecord, log_every=args.every)
    print(format_results(results, METRICS))
    for row in export(results, ARCHITECTURES, args.model, METRICS, LOG_DIR):
        print(f"saved {row['result_json']}")


if __name__ == "__main__":
    main()
