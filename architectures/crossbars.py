"""Crossbar types. Each returns a Block: the Crossbar, its read stage and the
costs of the array itself. Every value is a parameter; areas default to 0.

Memory cells are given directly: cell_bits, and either r_on / r_off (levels
linear in conductance from 1/r_off at level 0 to 1/r_on at the top level) or
levels_s, every level's conductance in siemens, ascending (nonuniform cells).

conv_xbar   current-mode crossbar: binary spikes on the word lines, each cell
            conducts G x v_read into its column. The cell current (computed
            from the spikes and conductances) is charged as
            cell_supply_v x current x powered time, where cell_supply_v is the
            rail the current is drawn from:
              an OTA/regulator derives v_read from VDD -> cell_supply_v = VDD
                (supply-side energy; the cells dissipate v_read x I and the
                regulator the rest);
              the source line is driven directly by a v_read supply ->
                cell_supply_v = v_read (energy of the cells alone).
            during/window: when the array conducts (default: the read; e.g.
            ["read", "fire"] keeps it on until the neurons have fired).
            Optional G(0) reference columns for analog weights.
c3cim_xbar  C3CIM crossbar: columns driven by constant current sources (a
            fixed current per active column) with drivers shared by groups of
            columns; the MAC happens in the voltage domain (not simulated).
"""
from hardware import Block, Component, Crossbar, Memory, Stage


def conv_xbar(*, cell_bits=1, r_on=None, r_off=None, levels_s=None, rows=64, cols=64,
              v_read=0.2, active_rows=None, read_ns=5.0, cell_supply_v=1.1,
              reference_columns=False, tile_area_um2=0.0, during="read", window=None):
    powered = dict(during=[] if window else during, window=window)
    components = [Component("cells", model="crossbar_read", count="tiles", **powered,
                            supply_v=cell_supply_v, area_um2=tile_area_um2, group="crossbar")]
    if reference_columns:
        components.append(Component("reference_cells", model="reference_read", count="tiles",
                                    **powered, supply_v=cell_supply_v,
                                    area_um2=tile_area_um2, group="crossbar"))
    return Block(stages=[Stage("read", read_ns)], components=components,
                 crossbar=Crossbar(Memory(cell_bits, r_on, r_off, levels_s), rows=rows,
                                   cols=cols, v_read=v_read, active_rows=active_rows,
                                   reference_columns=reference_columns))


def c3cim_xbar(*, cell_bits=1, r_on=None, r_off=None, levels_s=None, rows=64, cols=64,
               active_rows=8, read_ns=50.0, supply_v=1.1, column_ua=0.1, column_area_um2=0.0,
               driver_ua=11.87, driver_group=32, driver_area_um2=0.0):
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
        crossbar=Crossbar(Memory(cell_bits, r_on, r_off, levels_s), rows=rows,
                          cols=cols, active_rows=active_rows))
