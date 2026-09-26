"""Reference designs composed from the building blocks (used as baselines)."""
from architectures import crossbars, memories, neurons, periphery
from hardware import Precision, compose


def conventional(name="conventional", memory=memories.RRAM_ANALOG, conv_mapping="sequential"):
    """Current-mode crossbar with analog cells and G(0) reference columns, a DA
    per column, ideal reference subtraction and LIFs integrating through the bin."""
    return compose(name, Precision(weight_bits=None, weight_encoding="analog"), [
        crossbars.conv_xbar(memory, v_read=0.1, active_rows=8, read_ns=4.5,
                            reference_columns=True, tile_area_um2=136.67),
        periphery.da_converter(6.1, area_um2=30.22),
        periphery.reference_subtractor(0.0, 0.0),
        neurons.lif_neuron(6.0, 2.0, powered="time_bin", count="output_bank", area_um2=86.79),
    ], conv_mapping=conv_mapping)


def c3cim(name="c3cim", memory=memories.RRAM_C3, conv_mapping="sequential"):
    """C3CIM crossbar with a VI converter per column and LIFs integrating
    through the bin (fixed macro currents)."""
    return compose(name, Precision(weight_bits=None, weight_encoding="analog"), [
        crossbars.c3cim_xbar(memory, column_area_um2=4.27, driver_area_um2=86.36),
        periphery.vi_converter(24.3, 10.0, area_um2=29.79),
        neurons.lif_neuron(6.0, 2.0, powered="time_bin", count="output_bank", area_um2=86.79),
    ], conv_mapping=conv_mapping)
