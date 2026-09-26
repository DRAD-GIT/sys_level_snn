"""Hardware definitions.

memories.py     memory technologies (bits per cell, conductance levels)
ota_cim.py      OTA-regulated current-mode CIM with slice mirrors and comparator LIFs
conventional.py current-mode CIM with analog cells and a reference array
c3cim.py        C3CIM macro (fixed currents)

Each design module has build(name, memory, **parameters) -> hardware.Architecture,
so any memory can be combined with any design; run.py builds the ones to evaluate.
"""
