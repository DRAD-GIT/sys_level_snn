"""Modular compute-in-memory cost model for spiking layers.

architecture.py  memory, crossbar, mapping, components (with their stages and
                 power intervals); Block and compose()
mapping.py       layer -> windows, tiles, weight slices; spike activity
timeline.py      stage placement: serial, parallel, overlapping, pipelined
engine.py        energy / latency / area of a layer
"""
from hardware.architecture import (Architecture, Block, Component, Crossbar, Mapping, Memory,
                                   compose)
from hardware.engine import LayerCost, evaluate_layer
from hardware.mapping import quantize_weights

__all__ = ["Architecture", "Block", "Component", "Crossbar", "Mapping", "Memory", "compose",
           "LayerCost", "evaluate_layer", "quantize_weights"]
