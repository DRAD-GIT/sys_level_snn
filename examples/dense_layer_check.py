"""Worked example: one 128 -> 64 spiking dense layer on a 1-bit RRAM crossbar.

    python examples/dense_layer_check.py

Hardware: binary spikes on the word lines; 1-bit RRAM cells (20 kOhm /
200 kOhm); an OTA per source line holds 0.2 V (bit lines grounded into the
neurons) and draws 10 uA while operating; 64x64 tiles settling in 5 ns per
read; supply 1.1 V. Each weight bit has its own column (4-bit weights on
1-bit cells: 4 columns per weight) feeding the output's LIF neuron, whose
comparator draws 10 uA during its 2 ns fire step.

Data: random 4-bit weights and random binary input spikes. In an SNN the
input of every time bin is a spike (0 or 1); only the weights are multi-bit.

The script prints the engine's result next to an independent hand calculation.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from architectures import crossbars, memories, neurons, periphery  # noqa: E402
from hardware import Precision, compose, evaluate_layer  # noqa: E402

ARCH = compose(
    "rram_1bit_conv_xbar",
    Precision(weight_bits=4, weight_encoding="twos_complement"),
    blocks=[
        crossbars.conv_xbar(memories.RRAM_1BIT, rows=64, cols=64, v_read=0.2, read_ns=5.0,
                            supply_v=1.1),
        periphery.source_line_ota(static_ua=10.0, supply_v=1.1),   # on only when spikes arrive
        neurons.lif_neuron(static_ua=10.0, fire_ns=2.0, supply_v=1.1),
    ],
)

IN, OUT, BINS = 128, 64, 10     # 10 SNN time bins
SPIKE_PROBABILITY = 0.2         # chance that an input spikes in a time bin


def random_layer(seed=0):
    g = torch.Generator().manual_seed(seed)
    weights = torch.randint(-7, 8, (OUT, IN, 1, 1), generator=g)      # signed 4-bit codes
    spikes = (torch.rand(1, IN, 1, 1, BINS, generator=g) < SPIKE_PROBABILITY).float()  # binary
    return weights, spikes


def hand_calculation(weights, spikes):
    w = weights[:, :, 0, 0].numpy() % 16                               # two's complement bits
    s = spikes[0, :, 0, 0, :].numpy()                                  # (inputs, bins)
    g = lambda bit: np.where(bit, 1 / 20e3, 1 / 200e3)                 # cell conductance
    # Current of weight-bit b's columns over all outputs and bins (A, summed over reads).
    bit_current = [sum(0.2 * float(g((w[o] >> b) & 1) @ s[:, t])
                       for o in range(OUT) for t in range(BINS)) for b in range(4)]
    # 128 rows -> 2 row tiles; an OTA works for a read only if its tile gets a spike.
    busy_tile_reads = sum(bool(s[r * 64:(r + 1) * 64, t].any()) for r in range(2) for t in range(BINS))
    return {
        "cells": 1.1 * sum(bit_current) * 5.0,                          # V * A * ns = nJ
        "sl_ota": 1.1 * 10e-6 * 256 * busy_tile_reads * 5.0,           # 256 used columns per row tile
        "lif": 1.1 * 10e-6 * OUT * BINS * 2.0,                         # 64 comparators x 2 ns per bin
    }, BINS * (5.0 + 2.0)


if __name__ == "__main__":
    weights, spikes = random_layer()
    result = evaluate_layer(ARCH, spikes, weights)
    g = result.geometry
    print(f"mapping: {g.rows_needed} rows x {g.used_columns} columns ({g.out_channels} outputs x "
          f"{g.columns_per_weight} weight bits) -> {g.row_tiles} x {g.column_tiles} tiles of 64x64")
    print(f"inputs: {int(spikes.sum())} binary spikes over {BINS} time bins; "
          f"reads per bin: {g.reads_per_timestep}")
    expected, latency = hand_calculation(weights, spikes)
    print(f"\n{'component':<16}{'installed':>10}{'engine nJ':>14}{'hand nJ':>14}")
    for name, cost in result.components.items():
        print(f"{name:<16}{cost.installed:>10}{cost.energy_nj:>14.6g}{expected[name]:>14.6g}")
    print(f"{'total':<16}{'':>10}{result.energy_nj:>14.6g}{sum(expected.values()):>14.6g}")
    print(f"\nlatency {result.latency_ns:g} ns (hand {latency:g} ns); "
          f"power {result.energy_nj / result.latency_ns * 1e3:.4g} mW")
    print(f"synaptic ops {result.synaptic_ops:g} ({int(spikes.sum())} spikes x 64 outputs) -> "
          f"{result.energy_nj * 1e3 / result.synaptic_ops:.4g} pJ/SOP")
    assert all(abs(result.components[k].energy_nj - v) <= 1e-9 * v for k, v in expected.items())
    assert result.latency_ns == latency
    print("engine matches the hand calculation")
