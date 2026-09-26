"""1-bit RRAM crossbar with OTA-regulated source lines and comparator LIFs.

Binary spikes drive the word lines; an OTA holds each source line at 0.2 V
(bit lines grounded into the neurons) and is powered only in reads where its
tile receives a spike. 4-bit weights are bit-sliced over four 1-bit columns.
Each LIF neuron's comparator draws a static current for the whole time bin.
Values left at 0 (areas) or marked "placeholder" are not yet characterised.
"""
from hardware import Architecture, Component, Crossbar, Precision, Stage

ARCH = Architecture(
    name="rram_1bit",
    crossbar=Crossbar(rows=64, cols=64, cell_bits=1, r_on=20e3, r_off=200e3, v_read=0.2),
    precision=Precision(weight_bits=4, weight_encoding="twos_complement"),
    conv_mapping="sequential",
    stages=[
        Stage("read", 5.0),                          # array settles and integrates
        Stage("fire", 2.0, level="timestep"),        # LIF compare/fire per time bin (placeholder)
    ],
    components=[
        Component("cells", model="crossbar_read", count="tiles", during=["read"],
                  supply_v=1.1, area_um2=0.0, group="crossbar"),
        Component("sl_ota", count="physical_columns", on={"rule": "used_columns", "gated": True},
                  during=["read"], supply_v=1.1, static_ua=10.0, area_um2=0.0,
                  group="input_periphery"),
        Component("lif_comparator", count="outputs", during=["timestep"],
                  supply_v=1.1, static_ua=10.0, area_um2=0.0, group="lif"),
    ],
)
