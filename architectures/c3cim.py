"""C3CIM macro: constant-current-driven columns, voltage-domain MAC, VI output.

Macro cost model with fixed currents (no voltage-transfer simulation): a
100 nA source per active column, one shared driver per 32 columns of a tile,
a VI converter per column and a LIF bank powered through the time bin.
"""
from hardware import Architecture, Component, Crossbar, Precision, Stage


def build(name, memory, *, supply_v=1.1, conv_mapping="sequential"):
    driver_group = {"rule": "column_groups", "size": 32}
    return Architecture(
        name=name,
        crossbar=Crossbar(memory=memory, rows=64, cols=64, active_rows=8),
        precision=Precision(weight_bits=None, weight_encoding="analog"),
        conv_mapping=conv_mapping,
        stages=[Stage("read", 50.0), Stage("VI", 10.0), Stage("lif", 2.0, level="timestep")],
        components=[
            Component("column", count="physical_columns", on="used_columns", during=["read"],
                      supply_v=supply_v, static_ua=0.1, area_um2=4.27, group="crossbar"),
            Component("column_driver", count=driver_group, during=["read"], supply_v=supply_v,
                      static_ua=11.87, area_um2=86.36, group="input_periphery"),
            Component("VI", count="physical_columns", on="used_columns", during=["VI"],
                      supply_v=1.0, static_ua=24.3, area_um2=29.79, group="output_periphery"),
            Component("LIF", count="output_bank", on="outputs", during=["timestep"],
                      supply_v=supply_v, static_ua=6.0, area_um2=86.79, group="lif"),
        ],
    )
