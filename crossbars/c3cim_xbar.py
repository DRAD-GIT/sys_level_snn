"""C3CIM crossbar: columns driven by constant current sources (a fixed current
per active column) with drivers shared by groups of columns; the MAC happens
in the voltage domain (not simulated). The column sources are the
per-activation step "column_source", lasting `time_ns`; the drivers are
powered during it.
"""
from hardware import Block, Component, Crossbar, Memory


def c3cim_xbar(*, cell_bits=1, r_on=None, r_off=None, levels_s=None, rows=64, cols=64,
               active_rows=8, time_ns=50.0, supply_v=1.1, column_ua=0.1,
               column_area_um2=0.0, driver_ua=11.87, driver_group=32, driver_area_um2=0.0):
    return Block(
        components=[
            Component("column_source", count="physical_columns", powered="used_columns",
                      time_ns=time_ns, supply_v=supply_v, static_ua=column_ua,
                      area_um2=column_area_um2, group="crossbar"),
            Component("column_driver", count={"rule": "column_groups", "size": driver_group},
                      when="column_source", supply_v=supply_v, static_ua=driver_ua,
                      area_um2=driver_area_um2, group="input_periphery"),
        ],
        crossbar=Crossbar(Memory(cell_bits, r_on, r_off, levels_s), rows=rows,
                          cols=cols, active_rows=active_rows))
