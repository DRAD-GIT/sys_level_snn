"""Example: 128 -> 64 dense layer on 64x64 1-bit RRAM tiles.

4-bit weights (two's complement, bit-sliced over 4 one-bit columns) and
4-bit inputs applied bit-serially on binary word lines. An OTA holds each
source line at 0.2 V (bit lines grounded into the neurons); the array
settles in 5 ns per read. Each LIF neuron's comparator draws 10 uA while
the neuron operates.

    python cim/examples/dense_1bit_rram.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from cim import Architecture, Component, Crossbar, Precision, Stage, evaluate_layer  # noqa: E402

ARCH = Architecture(
    name="1bit_rram_ota",
    crossbar=Crossbar(rows=64, cols=64, cell_bits=1, r_on=20e3, r_off=200e3, v_read=0.2),
    precision=Precision(weight_bits=4, weight_encoding="twos_complement",
                        input_bits=4, input_bits_per_read=1),
    stages=[
        Stage("read", 5.0),                           # settle + sense, every read
        Stage("fire", 2.0, level="timestep"),         # LIF compare/fire after the reads (placeholder)
    ],
    components=[
        Component("cells", model="crossbar_read", count="tiles", during=["read"],
                  supply_v=1.1, area_um2=0.0, group="crossbar"),
        Component("sl_ota", count="physical_columns", on="used_columns", during=["read"],
                  supply_v=1.1, static_ua=10.0, area_um2=0.0, group="input_periphery"),
        Component("lif_comparator", count="outputs", on="outputs", during=["timestep"],
                  supply_v=1.1, static_ua=10.0, area_um2=0.0, group="lif"),
    ],
)


def example_data(seed=0):
    g = torch.Generator().manual_seed(seed)
    weights = torch.randint(-8, 8, (64, 128, 1, 1), generator=g)    # signed 4-bit codes
    inputs = torch.randint(0, 16, (1, 128, 1, 1, 1), generator=g)   # 4-bit codes, 1 timestep
    return inputs, weights


if __name__ == "__main__":
    x, w = example_data()
    print(evaluate_layer(ARCH, x, w).summary())
