"""Reference designs composed from a crossbar plus plain Stages and Components."""
from architectures import crossbars, memories
from hardware import Component, Precision, Stage, compose

VDD = 1.1


def conventional(name="conventional", memory=memories.RRAM_ANALOG, conv_mapping="sequential"):
    """Current-mode crossbar with analog cells and G(0) reference columns, a DA
    per column, ideal reference subtraction and LIFs integrating through the bin."""
    return compose(name, Precision(weight_bits=None, weight_encoding="analog"), [
        crossbars.conv_xbar(memory, v_read=0.1, active_rows=8, read_ns=4.5, cell_supply_v=VDD,
                            reference_columns=True, tile_area_um2=136.67),
        Stage("subtract", 0.0),
        Stage("fire", 2.0, level="timestep"),
        Component("da", count="physical_columns", on="used_columns", during="read",
                  supply_v=VDD, static_ua=6.1, area_um2=30.22),
        Component("reference_subtractor", count="output_bank", on="outputs", during="subtract",
                  supply_v=VDD),
        Component("lif", count="output_bank", on="outputs", during="timestep",
                  supply_v=VDD, static_ua=6.0, area_um2=86.79),
    ], conv_mapping=conv_mapping)


def c3cim(name="c3cim", memory=memories.RRAM_C3, conv_mapping="sequential"):
    """C3CIM crossbar with a VI converter per column and LIFs integrating
    through the bin (fixed macro currents)."""
    return compose(name, Precision(weight_bits=None, weight_encoding="analog"), [
        crossbars.c3cim_xbar(memory, supply_v=VDD, column_area_um2=4.27, driver_area_um2=86.36),
        Stage("vi", 10.0),
        Stage("fire", 2.0, level="timestep"),
        Component("vi", count="physical_columns", on="used_columns", during="vi",
                  supply_v=1.0, static_ua=24.3, area_um2=29.79),
        Component("lif", count="output_bank", on="outputs", during="timestep",
                  supply_v=VDD, static_ua=6.0, area_um2=86.79),
    ], conv_mapping=conv_mapping)
