"""Current-mode CIM with multi-level cells and an offset-reference array.

Weights map linearly to conductance (one analog cell each); a reference
column per output reads G(0) in the same phase for signed weights. Read
energy is supply-side: vdd * cell current * read time. Per-column DA stage,
ideal (zero-cost) reference subtraction, LIF bank powered through the bin.
"""
from hardware import Architecture, Component, Crossbar, Precision, Stage

VDD = 1.1

ARCH = Architecture(
    name="conventional",
    crossbar=Crossbar(rows=64, cols=64, r_on=2e3, r_off=2e5, v_read=0.1, active_rows=8,
                      reference_columns=True),
    precision=Precision(weight_bits=None, weight_encoding="analog"),
    conv_mapping="sequential",
    stages=[
        Stage("read", 4.5),
        Stage("subtract", 0.0),                      # reference subtraction (ideal)
        Stage("lif", 2.0, level="timestep"),
    ],
    components=[
        Component("crossbar", model="crossbar_read", count="tiles", during=["read"],
                  supply_v=VDD, area_um2=136.67, group="crossbar"),
        Component("reference_array", model="reference_read", count="tiles", during=["read"],
                  supply_v=VDD, area_um2=136.67, group="crossbar"),
        Component("DA", count="physical_columns", on="used_columns", during=["read"],
                  supply_v=VDD, static_ua=6.1, area_um2=30.22, group="output_periphery"),
        Component("reference_subtractor", count="output_bank", on="outputs", during=["subtract"],
                  supply_v=VDD, static_ua=0.0, area_um2=0.0, group="output_periphery"),
        Component("LIF", count="output_bank", on="outputs", during=["timestep"],
                  supply_v=VDD, static_ua=6.0, area_um2=86.79, group="lif"),
    ],
)
