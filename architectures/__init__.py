"""Hardware parts. An architecture = one crossbar + any Stages and Components:

    compose(name, Precision(...), [crossbar, Stage(...), Component(...), ...])

crossbars.py  crossbar types: conv_xbar (current-mode), c3cim_xbar (constant-current
              columns); memory cells are crossbar parameters
designs.py    reference designs (conventional, c3cim) composed from these parts
"""
