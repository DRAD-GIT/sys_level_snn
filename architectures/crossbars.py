"""Crossbar types. Each returns a Block: the Crossbar, its read stage and the
costs of the array itself. Every value is a parameter; areas default to 0.

conv_xbar   current-mode crossbar: binary spikes on the word lines, each cell
            conducts G x v_read into its column; the cell current (computed
            from the spikes and conductances) is drawn from supply_v during
            the read. Optional G(0) reference columns for analog weights.
c3cim_xbar  C3CIM crossbar: columns driven by constant current sources (a
            fixed current per active column) with drivers shared by groups of
            columns; the MAC happens in the voltage domain (not simulated).
"""
from hardware import Block, Component, Crossbar, Stage


def conv_xbar(memory, *, rows=64, cols=64, v_read=0.2, active_rows=None, read_ns=5.0,
              supply_v=1.1, reference_columns=False, tile_area_um2=0.0):
    components = [Component("cells", model="crossbar_read", count="tiles", during=["read"],
                            supply_v=supply_v, area_um2=tile_area_um2, group="crossbar")]
    if reference_columns:
        components.append(Component("reference_cells", model="reference_read", count="tiles",
                                    during=["read"], supply_v=supply_v, area_um2=tile_area_um2,
                                    group="crossbar"))
    return Block(stages=[Stage("read", read_ns)], components=components,
                 crossbar=Crossbar(memory, rows=rows, cols=cols, v_read=v_read,
                                   active_rows=active_rows, reference_columns=reference_columns))


def c3cim_xbar(memory, *, rows=64, cols=64, active_rows=8, read_ns=50.0, supply_v=1.1,
               column_ua=0.1, column_area_um2=0.0, driver_ua=11.87, driver_group=32,
               driver_area_um2=0.0):
    group = {"rule": "column_groups", "size": driver_group}
    return Block(
        stages=[Stage("read", read_ns)],
        components=[
            Component("column_source", count="physical_columns", on="used_columns",
                      during=["read"], supply_v=supply_v, static_ua=column_ua,
                      area_um2=column_area_um2, group="crossbar"),
            Component("column_driver", count=group, during=["read"], supply_v=supply_v,
                      static_ua=driver_ua, area_um2=driver_area_um2, group="input_periphery"),
        ],
        crossbar=Crossbar(memory, rows=rows, cols=cols, active_rows=active_rows))
