"""Modular compute-in-memory cost model (prototype).

architecture.py  crossbar, precision, stages (timeline) and components
timeline.py      stage placement: serial, parallel, overlapping, pipelined
engine.py        maps a layer onto tiles and evaluates energy/latency/area
"""
from cim.architecture import Architecture, Component, Crossbar, Precision, Stage
from cim.engine import evaluate_layer, quantize_symmetric

__all__ = ["Architecture", "Component", "Crossbar", "Precision", "Stage",
           "evaluate_layer", "quantize_symmetric"]
