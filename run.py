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

    python run.py                       # our work, on every dataset in MODELS
    python run.py --literature          # also the published macros (literature_macros.COMPARED)
    python run.py --model cifar10_thermo
    python run.py --table               # then write the paper's table, commit and push it

Command-line flags override the settings for one run. Results: printed, and
saved under logs/ (a JSON per architecture and logs/comparison_summary.csv).
"""
import argparse
import os
import subprocess
import sys

import crossbars
import literature_macros
import models
from evaluation.report import export, format_results, totals
from evaluation.runner import evaluate
from hardware import Component, Mapping, compose

ROOT = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# 1. MODELS, PRETRAINED WEIGHTS AND DATASET
# ============================================================================
# Models (datasets) evaluated, in order; pretrained weights in pretrained/.
# "nmnist17": N-MNIST downscaled to 17x17, first 100 ms (as ANP-I).
# "gesture16": DVS Gesture downscaled to 16x16, 10 classes (as DS-CIM, ReckOn).
# Also available: "nmnist", "gesture" (full-size sensor), "cifar10" (rate-coded CIFAR-10).
MODELS = ["nmnist17", "gesture16", "cifar10_thermo"]
# Folder holding the dataset folders (names starting with N-MNIST / Gesture),
# or the dataset folder itself.
DATASET_DIR = "/shares/bulk/yashbiyani/c3cim_sys_cluster/datasets/"   # None = none set
MAX_SAMPLES = -1       # first test samples to evaluate; -1 = the full test set
# Samples processed at once, per model (speed and memory only, not results):
# the batch when recording and the samples per hardware evaluation step.
# Models not listed use their defaults (recording: the model's batch_size;
# hardware evaluation: one recorded file per step). --parallel N sets all.
#PARALLEL = {"nmnist": 250, "gesture": 20, "gesture16": 240, "cifar10_thermo": 100}
PARALLEL = {"nmnist": 1000, "nmnist17": 1000, "gesture": 264, "gesture16": 240, "cifar10_thermo": 500}
# Recorded forward passes (layer inputs and outputs), reused across runs.
RECORDING_DIR = os.path.join(ROOT, "recordings")   # in the repository, committed with git

# ============================================================================
# 2. WEIGHT QUANTIZATION
# ============================================================================
WEIGHT_BITS = 6        # None = float weights (analog encoding only)
WEIGHT_SCALING = "std3"  # "std<k>", "max" or "mse" (see README, Mapping)

# ============================================================================
# 3. HARDWARE: the crossbar first, then its components (names must be unique)
# ============================================================================
VDD = 1.1
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
                  supply_v=1.0, static_ua=24.3),
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
                  supply_v=1.0, static_ua=24.3),
        Component("lif", count="outputs",                           # one per output neuron
                  time_ns=2.0,                                      # step: once per time bin, after the reads
                  supply_v=VDD, static_ua=6.0),                     # powered during its step
    ],
    specs={"tech": "40nm", "supply": 1.1, "device": "RRAM", "bitcell": "2T1R",
           "sensing": "Voltage"},
)

# Our work: always evaluated; listed last in the paper table
# (tools/latex_table.py), where its values that beat every other row are bold.
OURWORK = [C3CIM_XBAR, C3CIM_OP_XBAR]

# Published macros, evaluated only with --literature: the list COMPARED in
# literature_macros.py (how each number follows from its paper is written
# there). They store the same 6-bit weights, each in its paper's cells (at
# most 3 bits), so they share our recordings.
LITERATURE = literature_macros.COMPARED

# Every design defined here (architectures with the same weight quantization
# share one recorded forward pass).
ARCHITECTURES = OURWORK + LITERATURE

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

# ============================================================================
# 6. THE PAPER TABLE (--table)
# ============================================================================
# A clone of the paper repository (default: next to this repository) and the
# table's path in it. --table writes the comparison with other neuromorphic
# designs in DS-CIM's style there (tools/neuromorphic_table.py: our columns
# from this run's results, the other works' published numbers from its WORKS,
# all scaled to 40 nm with DeepScaleTool), commits and pushes it.
PAPER_REPO = os.path.join(os.path.dirname(ROOT), "C3CIM_journal_TCAS_v1")
PAPER_TABLE = os.path.join("Chapters", "neuromorphic_table.tex")


def summary(all_results):
    """One line per model and architecture: the network totals."""
    lines = ["", "=" * 96, "SUMMARY (per inference)", "=" * 96,
             f"{'model':<16}{'architecture':<26}{'accuracy %':>11}{'energy nJ':>12}"
             f"{'latency us':>12}{'power mW':>10}{'TOPS/W':>9}"]
    for model, results in all_results.items():
        for name, (accuracy, costs) in results.items():
            t = totals(costs)
            lines.append(f"{model:<16}{name:<26}{accuracy:>11.2f}{t['energy']:>12.4g}"
                         f"{t['latency']:>12.4g}{t['power']:>10.4g}{t['tops_per_w']:>9.4g}")
    return "\n".join(lines)


def _git(repo, *args, check=True):
    return subprocess.run(["git", "-C", repo, *args], check=check, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def publish_table(repo=PAPER_REPO, path=PAPER_TABLE, logs_dir=LOG_DIR, push=True):
    """Write the neuromorphic comparison table (our work from the results in
    logs_dir, the other works from their papers, scaled to 40 nm) into the
    paper repository `repo` (updated first with git pull), then commit and
    push it if it changed. Returns the table's path."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import neuromorphic_table
    if not os.path.isdir(os.path.join(repo, ".git")):
        raise SystemExit(f"--table: no git clone of the paper repository at {repo}; clone it "
                         "there (git clone https://github.com/DRAD-GIT/C3CIM_journal_TCAS_v1.git) "
                         "or pass --paper <path>")
    pulled = _git(repo, "pull", "--ff-only", check=False)
    if pulled.returncode:
        raise SystemExit(f"--table: git pull in {repo} failed:\n{pulled.stdout}")
    table = neuromorphic_table.make_table([a.name for a in OURWORK], logs_dir)
    out = os.path.join(repo, path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as file:
        file.write(table)
    print(f"saved {out}")
    _git(repo, "add", path)
    if _git(repo, "diff", "--cached", "--quiet", "--", path, check=False).returncode == 0:
        print("the table is unchanged: nothing to commit")
        return out
    _git(repo, "commit", "-m", "Neuromorphic comparison table from sys_level_snn", "--", path)
    if push:
        pushed = _git(repo, "push", check=False)
        if pushed.returncode:
            raise SystemExit(f"--table: committed, but git push failed:\n{pushed.stdout}")
        print(f"pushed the table to the paper repository ({repo})")
    return out


def main():
    """4. Reuse the recorded forward pass (record it if missing), 5. estimate the metrics."""
    parser = argparse.ArgumentParser(description="SNN inference + CIM hardware metrics")
    parser.add_argument("--model", nargs="+", default=MODELS, choices=sorted(models.MODEL_MODULES),
                        help="models (datasets) to evaluate (default: MODELS)")
    parser.add_argument("--literature", action="store_true",
                        help="also evaluate the published macros (literature_macros.COMPARED)")
    parser.add_argument("--data", default=DATASET_DIR, help="dataset folder (overrides DATASET_DIR)")
    parser.add_argument("--samples", type=models.samples_argument, default=MAX_SAMPLES,
                        help="first N test samples to evaluate; -1 = all (default: MAX_SAMPLES)")
    parser.add_argument("--every", type=models.positive_argument, default=1000,
                        help="print progress every N samples (default 1000)")
    parser.add_argument("--parallel", type=models.positive_argument,
                        help="samples processed at once, for every model: when recording and per "
                             "hardware evaluation step (default: PARALLEL)")
    parser.add_argument("--recordings", default=RECORDING_DIR, help="folder of recorded forward passes")
    parser.add_argument("--rerecord", action="store_true", help="record the forward pass again")
    parser.add_argument("--table", action="store_true",
                        help="then write the neuromorphic comparison table (our results, the "
                             "other works' published numbers, at 40 nm) into PAPER_REPO, commit "
                             "and push it")
    parser.add_argument("--paper", default=PAPER_REPO, help="paper repository clone (default: PAPER_REPO)")
    args = parser.parse_args()

    # --table also needs the published macros (their columns in the table).
    architectures = OURWORK + (LITERATURE if args.literature or args.table else [])
    all_results = {}
    for model in args.model:
        parallel = args.parallel or PARALLEL.get(model)
        results = evaluate(model, architectures, data_dir=args.data,
                           recording_dir=args.recordings,
                           max_samples=args.samples,
                           parallel=parallel, rerecord=args.rerecord, log_every=args.every)
        print(format_results(results, METRICS))
        for row in export(results, architectures, model, METRICS, LOG_DIR):
            print(f"saved {row['result_json']}")
        all_results[model] = results
    print(summary(all_results))
    if args.table:
        publish_table(repo=args.paper)


if __name__ == "__main__":
    main()
