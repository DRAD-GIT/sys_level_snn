"""Hardware building blocks. An architecture = one crossbar + periphery + neurons:

    compose(name, Precision(...), blocks=[crossbar, *periphery, neuron])

memories.py   memory technologies (bits per cell, conductance levels)
crossbars.py  crossbar types: conv_xbar (current-mode), c3cim_xbar (constant-current columns)
periphery.py  source-line OTAs, slice mirrors, DA, reference subtractor, VI converter
neurons.py    LIF neurons
designs.py    reference designs composed from these blocks
"""
