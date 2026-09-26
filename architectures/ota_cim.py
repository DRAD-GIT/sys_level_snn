"""OTA-regulated current-mode CIM with binary-weighted slice mirrors.

Binary spikes drive the word lines. An OTA holds each source line at v_read
(bit lines grounded into the neurons); it is powered only in reads where its
tile receives a spike. Weights are bit-sliced over one column per slice; each
slice column's current is mirrored with a binary gain (most significant slice
1, the next 1/2, ...) and summed into one LIF neuron, whose comparator draws
a static current during the fire step of every time bin.
Areas default to 0 (not characterised); switch the area metric off or fill them in.
"""
from hardware import Architecture, Component, Crossbar, Precision, Stage


def build(name, memory, *, rows=64, cols=64, v_read=0.2, weight_bits=4,
          weight_encoding="twos_complement", conv_mapping="sequential", supply_v=1.1,
          settle_ns=5.0, fire_ns=2.0, ota_ua=10.0, comparator_ua=10.0,
          tile_area_um2=0.0, ota_area_um2=0.0, mirror_area_um2=0.0, lif_area_um2=0.0):
    return Architecture(
        name=name,
        crossbar=Crossbar(memory=memory, rows=rows, cols=cols, v_read=v_read),
        precision=Precision(weight_bits=weight_bits, weight_encoding=weight_encoding),
        conv_mapping=conv_mapping,
        stages=[
            Stage("read", settle_ns),                  # array settles, mirrors sum into the LIFs
            Stage("fire", fire_ns, level="timestep"),  # LIF compare/fire, once per time bin
        ],
        components=[
            Component("cells", model="crossbar_read", count="tiles", during=["read"],
                      supply_v=supply_v, area_um2=tile_area_um2, group="crossbar"),
            Component("sl_ota", count="physical_columns",
                      on={"rule": "used_columns", "gated": True}, during=["read"],
                      supply_v=supply_v, static_ua=ota_ua, area_um2=ota_area_um2,
                      group="input_periphery"),
            Component("slice_mirrors", model="slice_mirror", count="used_columns",
                      during=["read"], supply_v=supply_v, area_um2=mirror_area_um2,
                      group="output_periphery"),
            Component("lif_comparator", count="outputs", during=["fire"], supply_v=supply_v,
                      static_ua=comparator_ua, area_um2=lif_area_um2, group="lif"),
        ],
    )
