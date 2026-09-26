"""Neuron blocks. Each adds its fire step (once per time bin, after the reads)."""
from hardware import Block, Component, Stage


def lif_neuron(static_ua, fire_ns, *, supply_v=1.1, powered="fire", count="outputs",
               area_um2=0.0):
    """A LIF neuron per output drawing a static current (no dynamic energy).

    powered: "fire" = only during its fire step (e.g. a comparator switched on
    to evaluate); "time_bin" = through the whole time bin (integrating).
    count: installed neurons, "outputs" (one per output neuron) or
    "output_bank" (one per column position of a weight copy).
    """
    if powered not in ("fire", "time_bin"):
        raise ValueError("powered must be 'fire' or 'time_bin'")
    return Block(stages=[Stage("fire", fire_ns, level="timestep")], components=[Component(
        "lif", count=count, on="outputs", during=["fire" if powered == "fire" else "timestep"],
        supply_v=supply_v, static_ua=static_ua, area_um2=area_um2, group="lif")])
