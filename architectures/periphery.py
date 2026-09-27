"""Peripheral circuit blocks; add any of them to any crossbar. Each returns a
Block (its components, plus a stage if it adds time to every read). `during`
names the stage(s) the block is powered in (e.g. "read", or ["read", "fire"];
"timestep" = the whole time bin); alternatively `window` powers it from one
stage edge to another (see hardware.Component); areas default to 0.
"""
from hardware import Block, Component, Stage


def _stages(during):
    return [during] if isinstance(during, str) else list(during)


def source_line_ota(static_ua, *, supply_v=1.1, gated=True, area_um2=0.0, during="read",
                    window=None):
    """An OTA per column holding the source line at v_read. gated: powered only
    when its tile receives a spike (in that read, or in that time bin if it is
    also powered during time-bin stages such as "fire")."""
    return Block(components=[Component(
        "sl_ota", count="physical_columns", on={"rule": "used_columns", "gated": gated},
        during=[] if window else _stages(during), window=window, supply_v=supply_v,
        static_ua=static_ua, area_um2=area_um2, group="input_periphery")])


def slice_mirrors(*, supply_v=1.1, gains=None, area_um2=0.0, during="read"):
    """A current mirror per weight-slice column copying its current into the
    neuron with a gain (default binary: most significant slice x1, then x1/2...)."""
    return Block(components=[Component(
        "slice_mirrors", model="slice_mirror", count="used_columns", during=_stages(during),
        supply_v=supply_v, slice_gains=gains, area_um2=area_um2, group="output_periphery")])


def da_converter(static_ua, *, supply_v=1.1, area_um2=0.0, during="read", window=None):
    """A DA circuit per column, powered for used columns during the read."""
    return Block(components=[Component(
        "da", count="physical_columns", on="used_columns",
        during=[] if window else _stages(during), window=window, supply_v=supply_v,
        static_ua=static_ua, area_um2=area_um2, group="output_periphery")])


def reference_subtractor(static_ua=0.0, latency_ns=0.0, *, supply_v=1.1, area_um2=0.0,
                         stage="subtract"):
    """Subtracts the reference column per output, in its own stage after the read."""
    return Block(stages=[Stage(stage, latency_ns)], components=[Component(
        "reference_subtractor", count="output_bank", on="outputs", during=[stage],
        supply_v=supply_v, static_ua=static_ua, area_um2=area_um2, group="output_periphery")])


def vi_converter(static_ua, latency_ns, *, supply_v=1.0, area_um2=0.0, stage="vi"):
    """A voltage-to-current converter per column, in its own stage after the read."""
    return Block(stages=[Stage(stage, latency_ns)], components=[Component(
        "vi", count="physical_columns", on="used_columns", during=[stage], supply_v=supply_v,
        static_ua=static_ua, area_um2=area_um2, group="output_periphery")])
