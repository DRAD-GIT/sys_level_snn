"""C3CIM macro: constant-current-driven columns, voltage-domain MAC, VI output.

Macro cost model with fixed currents (no voltage-transfer simulation): a
100 nA source per active column, one shared driver per 32 columns of a tile,
a VI converter per column and a LIF bank powered through the time bin.
"""
from hardware import Architecture, Component, Crossbar, Precision, Stage

VDD = 1.1
DRIVER_GROUP = {"rule": "column_groups", "size": 32}

ARCH = Architecture(
    name="c3cim",
    crossbar=Crossbar(rows=64, cols=64, r_on=2e3, r_off=2e4, active_rows=8),
    precision=Precision(weight_bits=None, weight_encoding="analog"),
    conv_mapping="sequential",
    stages=[Stage("read", 50.0), Stage("VI", 10.0), Stage("lif", 2.0, level="timestep")],
    components=[
        Component("column", count="physical_columns", on="used_columns", during=["read"],
                  supply_v=VDD, static_ua=0.1, area_um2=4.27, group="crossbar"),
        Component("column_driver", count=DRIVER_GROUP, during=["read"], supply_v=VDD,
                  static_ua=11.87, area_um2=86.36, group="input_periphery"),
        Component("VI", count="physical_columns", on="used_columns", during=["VI"],
                  supply_v=1.0, static_ua=24.3, area_um2=29.79, group="output_periphery"),
        Component("LIF", count="output_bank", on="outputs", during=["timestep"],
                  supply_v=VDD, static_ua=6.0, area_um2=86.79, group="lif"),
    ],
)
