"""Modular compute-in-memory cost model for spiking layers.

architecture.py  crossbar, precision, conv mapping, stages and components
mapping.py       layer -> windows, tiles, weight slices; spike activity
timeline.py      stage placement: serial, parallel, overlapping, pipelined
engine.py        energy / latency / area of a layer
"""
from hardware.architecture import Architecture, Component, Crossbar, Precision, Stage
from hardware.engine import LayerCost, evaluate_layer
from hardware.mapping import quantize_weights

__all__ = ["Architecture", "Component", "Crossbar", "Precision", "Stage",
           "LayerCost", "evaluate_layer", "quantize_weights"]
