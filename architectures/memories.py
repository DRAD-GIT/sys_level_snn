"""Memory technologies: bits per cell and conductance levels.

Levels are linear in conductance from 1/r_off to 1/r_on unless `levels_s`
lists them (siemens, ascending) for nonuniform devices, e.g.
    Memory("rram_2bit", cell_bits=2, levels_s=(5e-6, 20e-6, 35e-6, 50e-6))
"""
from hardware import Memory

RRAM_1BIT = Memory("rram_1bit", cell_bits=1, r_on=20e3, r_off=200e3)
RRAM_ANALOG = Memory("rram_analog", cell_bits=1, r_on=2e3, r_off=200e3)   # conventional CIM
RRAM_C3 = Memory("rram_c3", cell_bits=1, r_on=2e3, r_off=20e3)            # C3CIM
