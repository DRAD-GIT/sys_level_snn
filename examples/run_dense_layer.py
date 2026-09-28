"""run.py for a dummy layer: one random 128 -> 64 spiking dense layer.

    python examples/run_dense_layer.py

Same structure as run.py (settings, hardware, metrics) with random data in
place of a dataset. The engine's result is checked against a hand calculation.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import crossbars  # noqa: E402
from evaluation.report import format_results  # noqa: E402
from hardware import Component, Mapping, compose, evaluate_layer  # noqa: E402

# ============================================================================
# LAYER AND DATA
# ============================================================================
INPUTS, OUTPUTS = 128, 64
TIME_BINS = 10              # SNN time bins; inputs are binary spikes in every bin
SPIKE_PROBABILITY = 0.2     # chance that an input spikes in a time bin
WEIGHT_BITS = 4             # weights are the only multi-bit values
SEED = 0

# ============================================================================
# HARDWARE
# ============================================================================
VDD = 1.1

ARCH = compose(
    "rram_1bit_conv_xbar",
    Mapping(weight_bits=WEIGHT_BITS, weight_encoding="twos_complement"),
    [
        # 1-bit RRAM (20k / 200k ohm), 64x64 tiles, 0.2 V read made by the OTA
        # from VDD, 5 ns to settle; the array conducts until the LIF ends.
        crossbars.conv_xbar(cell_bits=1, r_on=20e3, r_off=200e3, rows=64, cols=64,
                            v_read=0.2, cell_supply_v=VDD, time_ns=5.0,
                            when=("cells", "lif")),
        # An OTA per column (source line), 10 uA static, on from the read until
        # the LIF ends, only when its tile receives a spike.
        Component("sl_ota", count="physical_columns",
                  powered={"rule": "used_columns", "gated": True},
                  when=("cells", "lif"), supply_v=VDD, static_ua=10.0),
        # A LIF per output: its 2 ns step once per time bin, after the read;
        # its comparator draws 10 uA during it.
        Component("lif", count="outputs", time_ns=2.0, supply_v=VDD, static_ua=10.0),
    ],
)

# ============================================================================
# METRICS
# ============================================================================
METRICS = {"energy": True, "latency": True, "power": True, "area": False,
           "tops_per_w": True, "pj_per_sop": True, "layers": True, "components": True}


def random_layer():
    g = torch.Generator().manual_seed(SEED)
    top = 2 ** (WEIGHT_BITS - 1) - 1
    weights = torch.randint(-top, top + 1, (OUTPUTS, INPUTS, 1, 1), generator=g)
    spikes = (torch.rand(1, INPUTS, 1, 1, TIME_BINS, generator=g) < SPIKE_PROBABILITY).float()
    return weights, spikes


def hand_calculation(weights, spikes):
    """Independent per-component energy (nJ) and latency (ns)."""
    w = weights[:, :, 0, 0].numpy() % 2 ** WEIGHT_BITS        # two's complement bits
    s = spikes[0, :, 0, 0, :].numpy()                          # (inputs, bins)
    g = lambda bit: np.where(bit, 1 / 20e3, 1 / 200e3)         # cell conductance (S)
    on_ns = 5.0 + 2.0                                          # read + fire, per bin
    # Array current per bin (A): 0.2 V across every cell of every spiking row.
    current = [sum(0.2 * float(g((w[o] >> b) & 1) @ s[:, t])
                   for o in range(OUTPUTS) for b in range(WEIGHT_BITS)) for t in range(TIME_BINS)]
    # 128 rows -> 2 row tiles; an OTA runs in a bin only if its tile gets a spike.
    busy = sum(bool(s[r * 64:(r + 1) * 64, t].any()) for r in range(2) for t in range(TIME_BINS))
    used_columns = OUTPUTS * WEIGHT_BITS                       # per row tile
    return {"cells": VDD * sum(current) * on_ns,               # V * A * ns = nJ
            "sl_ota": VDD * 10e-6 * used_columns * busy * on_ns,
            "lif": VDD * 10e-6 * OUTPUTS * TIME_BINS * 2.0}, TIME_BINS * on_ns


if __name__ == "__main__":
    weights, spikes = random_layer()
    cost = evaluate_layer(ARCH, spikes, weights)
    print(f"{int(spikes.sum())} input spikes over {TIME_BINS} time bins")
    print(format_results({ARCH.name: (None, {"dense": cost})}, METRICS))

    expected, latency = hand_calculation(weights, spikes)
    print("\nhand check:")
    for name, energy in expected.items():
        got = cost.components[name].energy_nj
        print(f"  {name:<8} engine {got:.6g} nJ, hand {energy:.6g} nJ")
        assert abs(got - energy) <= 1e-9 * energy, name
    print(f"  latency  engine {cost.latency_ns:g} ns, hand {latency:g} ns")
    assert cost.latency_ns == latency
    print("engine matches the hand calculation")
