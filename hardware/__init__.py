"""Modular compute-in-memory cost model for spiking layers.

architecture.py  memory, crossbar, precision, stages, components; blocks and compose()
mapping.py       layer -> windows, tiles, weight slices; spike activity
timeline.py      stage placement: serial, parallel, overlapping, pipelined
engine.py        energy / latency / area of a layer
"""
from hardware.architecture import (Architecture, Block, Component, Crossbar, Memory, Precision,
                                   Stage, compose)
from hardware.engine import LayerCost, evaluate_layer
from hardware.mapping import quantize_weights

__all__ = ["Architecture", "Block", "Component", "Crossbar", "Memory", "Precision", "Stage", "compose",
           "LayerCost", "evaluate_layer", "quantize_weights"]
