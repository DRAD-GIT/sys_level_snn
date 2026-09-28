"""Current-mode crossbar: binary spikes on the word lines, each cell conducts
G x v_read into its column.

The cell current (computed from the spikes and conductances) is charged as
cell_supply_v x current x powered time, where cell_supply_v is the rail the
current is drawn from:
  an OTA/regulator derives v_read from VDD -> cell_supply_v = VDD (supply-side
    energy; the cells dissipate v_read x I and the regulator the rest);
  the source line is driven directly by a v_read supply -> cell_supply_v =
    v_read (energy of the cells alone).
The array is the per-activation step "cells", lasting `time_ns`, and conducts
during it unless `when` says otherwise (e.g. when=("cells", "lif") keeps it
on until the neurons' step "lif" ends). Optional G(0) reference columns for
analog weights.
"""
from hardware import Block, Component, Crossbar, Memory


def conv_xbar(*, cell_bits=1, r_on=None, r_off=None, levels_s=None, rows=64, cols=64,
              v_read=0.2, active_rows=None, time_ns=5.0, when=None, cell_supply_v=1.1,
              reference_columns=False, tile_area_um2=0.0):
    components = [Component("cells", model="crossbar_read", count="tiles", time_ns=time_ns,
                            when=when, supply_v=cell_supply_v, area_um2=tile_area_um2,
                            group="crossbar")]
    if reference_columns:
        components.append(Component("reference_cells", model="reference_read", count="tiles",
                                    when=when or "cells", supply_v=cell_supply_v,
                                    area_um2=tile_area_um2, group="crossbar"))
    return Block(components=components,
                 crossbar=Crossbar(Memory(cell_bits, r_on, r_off, levels_s), rows=rows,
                                   cols=cols, v_read=v_read, active_rows=active_rows,
                                   reference_columns=reference_columns))
