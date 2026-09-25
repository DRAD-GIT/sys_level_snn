"""Main script: define the hardware below, then run

    python run.py

Edit the RUN SETTINGS and HARDWARE sections. Every value is optional: a field
you delete falls back to the default in hardware/configs.py. Command-line flags
(python run.py --help) override the run settings for one run without editing.

Units: resistance ohm, voltage V, current uA, latency ns, area um^2 per
instance. Results go to the console, logs/<run>.txt, a JSON per architecture,
and the comparison table logs/comparison_summary.csv.
"""
import argparse

from hardware import C3HardwareConfig, ConvHardwareConfig, load_config
from evaluation.runner import evaluate

# ============================================================================
# RUN SETTINGS
# ============================================================================
MODEL = "nmnist"       # "nmnist" or "gesture"
BATCH_SIZE = 1         # gesture is capped at 2
MAX_BATCHES = 1        # None = full test set (can take a long time)
WEIGHT_BITS = 0        # 0 = trained weights; e.g. 4 = quantize to 4-bit before inference
QUANT_STD_STEP = 2     # N-MNIST quantization range = mean +/- k*std  (-k)
QUANT_DECIMALS = 1     # decimal rounding of quantization levels       (-d)

# ============================================================================
# HARDWARE: conventional current-mode CIM
# ============================================================================
CONVENTIONAL = ConvHardwareConfig(
    # Crossbar tile and weight mapping
    xbar_row=64,             # physical rows per tile
    xbar_col=64,             # physical columns per tile
    active_rows=8,           # rows enabled per read phase (None = all rows at once)
    min_res=2e3,             # lowest programmable resistance (ohm)
    max_res=2e5,             # highest programmable resistance (ohm)
    vread=0.1,               # read voltage across the array (V)
    vdd=1.1,                 # supply voltage (V)
    xbar_lat=4.5,            # read time per phase (ns)
    xbar_area=136.67,        # area per tile (um^2)
    # Output periphery
    DA_curr=6.1,             # DA circuit current per active column (uA)
    DA_area=30.22,           # DA area per column (um^2)
    reference_array=True,    # physical reference array for signed weights
    ref_sub_curr=0.0,        # reference subtraction current per output (uA); 0 = ideal
    ref_sub_area=0.0,        # reference subtraction area per output (um^2)
    ref_sub_lat=0.0,         # reference subtraction latency (ns)
    # LIF neuron
    lif_curr=6.0,            # LIF current while integrating/emitting (uA)
    lif_lat=2.0,             # LIF emission latency per time bin (ns)
    lif_area=86.79,          # LIF area per output position (um^2)
    # Advanced: add/patch components or replace the stage schedule (README section 9)
    components=[],
    schedule=None,
)

# ============================================================================
# HARDWARE: C3CIM (constant-current column drive, VI output)
# ============================================================================
C3CIM = C3HardwareConfig(
    # Crossbar tile and weight mapping
    xbar_row=64,             # physical rows per tile
    xbar_col=64,             # physical columns per tile
    active_rows=8,           # rows enabled per read phase (None = all rows at once)
    min_res=2e3,             # lowest programmable resistance (ohm)
    max_res=2e4,             # highest programmable resistance (ohm)
    vdd=1.1,                 # supply voltage (V)
    # Constant-current columns
    col_curr=0.1,            # source current per active column (uA) = 100 nA
    col_lat=50.0,            # column read time per phase (ns)
    col_area=4.27,           # area per column (um^2)
    driver_part=32,          # columns sharing one current driver (per tile)
    driver_curr=11.87,       # overhead current per shared driver (uA)
    driver_area=86.36,       # area per driver (um^2)
    # VI converter
    VI_supp=1.0,             # VI supply voltage (V)
    VI_curr=24.3,            # VI current per active column (uA)
    VI_lat=10.0,             # VI conversion time per phase (ns)
    VI_area=29.79,           # VI area per column (um^2)
    # LIF neuron
    lif_curr=6.0,            # LIF current while integrating/emitting (uA)
    lif_lat=2.0,             # LIF emission latency per time bin (ns)
    lif_area=86.79,          # LIF area per output position (um^2)
    # Advanced: add/patch components or replace the stage schedule (README section 9).
    # Example: an extra bias block per tile, powered during the read stage:
    # components=[{
    #     "name": "extra_bias", "model": "fixed_current", "group": "input_periphery",
    #     "count": {"rule": "tiles"}, "activity": {"rule": "tiles"}, "timing": ["read"],
    #     "params": {"supply_v": 1.1, "current_ua": 2.0, "area_um2": 5.0},
    # }],
    components=[],
    schedule=None,
)

# Architectures to evaluate: remove an entry to skip it. The name is used in
# the logs and the comparison CSV.
ARCHITECTURES = {"conventional": CONVENTIONAL, "c3cim": C3CIM}


def main():
    parser = argparse.ArgumentParser(description="SNN inference + CIM hardware metrics")
    parser.add_argument("--model", default=MODEL, choices=["nmnist", "gesture"])
    parser.add_argument("-b", "--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--batches", type=int, default=MAX_BATCHES,
                        help="max batches to run; -1 = full test set")
    parser.add_argument("-n", "--bits", type=int, default=WEIGHT_BITS,
                        help="weight bit precision; 0 = no quantization")
    parser.add_argument("-k", type=float, default=QUANT_STD_STEP, help="quantization std step")
    parser.add_argument("-d", type=int, default=QUANT_DECIMALS, help="quantization level decimals")
    parser.add_argument("--conv-config", help="JSON file replacing the CONVENTIONAL config above")
    parser.add_argument("--c3-config", help="JSON file replacing the C3CIM config above")
    args = parser.parse_args()

    architectures = dict(ARCHITECTURES)
    if args.conv_config:
        architectures["conventional"] = load_config(ConvHardwareConfig, args.conv_config)
    if args.c3_config:
        architectures["c3cim"] = load_config(C3HardwareConfig, args.c3_config)

    evaluate(
        args.model, architectures,
        batch_size=args.batch_size,
        max_batches=None if args.batches in (None, -1) else args.batches,
        weight_bits=args.bits,
        quant_std_step=args.k,
        quant_decimals=args.d,
    )


if __name__ == "__main__":
    main()
