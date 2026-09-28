"""Crossbar types. Each returns the Crossbar (with its memory cells given
directly) together with the array's own components and its per-activation
step. Compose one with your own Components:

    compose(name, Mapping(...), [crossbars.conv_xbar(...), Component(...), ...])

Memory cells: cell_bits, and either r_on / r_off (levels linear in
conductance from 1/r_off at level 0 to 1/r_on at the top level) or levels_s,
every level's conductance in siemens, ascending (nonuniform cells). Every
value is a parameter; areas default to 0.

conv_xbar.py   current-mode crossbar (cell current G x v_read into each column)
c3cim_xbar.py  C3CIM crossbar (constant-current columns, shared drivers)
"""
from crossbars.c3cim_xbar import c3cim_xbar
from crossbars.conv_xbar import conv_xbar

__all__ = ["conv_xbar", "c3cim_xbar"]
