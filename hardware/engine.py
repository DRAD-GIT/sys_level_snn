"""Energy, latency and area of one weighted layer on a CIM architecture.

evaluate_layer(arch, spikes, weights) maps the layer (mapping.py), extracts
the spike activity, places the steps on the timeline (timeline.py) and
charges every component:
  static   supply_v * static_ua * (powered time per activation or time bin)
           * powered instances, summed over all activations or time bins
  events   event_pj * (powered instances per activation or time bin, or output spikes)
  data     supply_v * (current computed from spikes and conductances)
           * powered time,
           for the array ("crossbar_read", "reference_read") and the
           weight-slice current mirrors ("slice_mirror")
Powered instances follow the component's `powered` rule: all valid instances, or
("gated") only those whose tile/window/layer receives a spike.
"""
import math
from dataclasses import dataclass

from hardware.architecture import DATA_RULES, TILE_RULES, rule_of
from hardware.mapping import (Geometry, conductance_slices, default_slice_gains, layer_geometry,
                              spike_activity)
from hardware.timeline import Timeline, build_timeline


@dataclass
class ComponentCost:
    group: str
    installed: int
    area_um2: float
    energy_nj: float = 0.0   # summed over all evaluated inferences
    used: int = 0            # installed instances its `powered` rule can power


@dataclass
class LayerCost:
    """Costs of a layer over `inferences` evaluated samples; add() merges batches."""
    geometry: Geometry
    timeline: Timeline
    components: dict
    inferences: int
    macs: int = 0              # dense MACs per inference (all inputs, all time bins)
    synaptic_ops: float = 0.0  # total over inferences: input spikes x outputs reached
    output_spikes: float = 0.0  # total over inferences (if provided)

    @property
    def latency_ns(self):  # per inference; fixed by the schedule
        return self.timeline.latency_ns

    @property
    def energy_nj(self):  # per inference
        return sum(c.energy_nj for c in self.components.values()) / self.inferences

    @property
    def area_um2(self):
        return sum(c.area_um2 for c in self.components.values())

    def add(self, other):
        for name, cost in other.components.items():
            self.components[name].energy_nj += cost.energy_nj
        self.inferences += other.inferences
        self.synaptic_ops += other.synaptic_ops
        self.output_spikes += other.output_spikes


def _per_unit(rule, g):
    """Instances per unit: per row tile of a tile set (tile rules; used
    columns per weight copy), per window ("outputs"), per tile set
    ("output_bank") or per layer."""
    return {"tiles": g.column_tiles,
            "physical_rows": g.column_tiles * g.tile_rows,
            "physical_columns": g.column_tiles * g.tile_cols,
            "used_columns": g.used_columns,
            "column_groups": g.column_tiles * math.ceil(g.tile_cols / rule.get("size", 1)),
            "outputs": g.out_channels,
            "output_bank": g.column_tiles * g.tile_cols,
            "one": 1,
            "fixed": rule.get("value", 0)}[rule["rule"]]


def installed(rule, g):
    r = rule["rule"]
    if r == "used_columns":
        return g.copies * g.row_tiles * _per_unit(rule, g)
    if r in TILE_RULES:
        return g.tile_sets * g.row_tiles * _per_unit(rule, g)
    if r == "outputs":
        return g.windows * g.out_channels
    if r == "output_bank":
        return g.tile_sets * _per_unit(rule, g)
    return _per_unit(rule, g)


def powered_per_activation(rule, g, act, sequential):
    """Sum over all activations of the powered instances."""
    r, gated, frames = rule["rule"], rule["gated"], act.frames
    if r == "spiking_rows":  # every word-line segment (one per column tile) with a spike
        return act.spikes_on_rows * g.column_tiles
    if r == "used_columns":  # the used columns of every tile activated
        units = act.tile_read_copies if gated else \
            frames * sum(n * c * sum(p) for n, c, p in g.slot_kinds)
    elif r in TILE_RULES:
        units = act.tile_reads if gated else frames * sum(n * sum(p) for n, _, p in g.slot_kinds)
    elif r == "output_bank":
        units = act.slot_reads if gated else frames * g.slots * g.phases
    elif r == "outputs" or sequential:  # a sequential activation = one window
        units = act.window_reads if gated else frames * g.windows * g.phases
    else:
        units = act.layer_reads if gated else frames * g.phases
    return units * _per_unit(rule, g)


def powered_per_bin(rule, g, act, sequential):
    """Sum over all time bins of the powered instances (whole-bin components)."""
    r, gated, frames = rule["rule"], rule["gated"], act.frames
    if r in TILE_RULES and sequential:
        units = act.shared_tile_bins if gated else frames * g.row_tiles
    elif r == "used_columns":
        units = act.tile_bin_copies if gated else frames * g.windows * g.row_tiles
    elif r in TILE_RULES:
        units = act.tile_bins if gated else frames * g.slots * g.row_tiles
    elif r == "outputs":
        units = act.window_bins if gated else frames * g.windows
    elif r == "output_bank" and not sequential:
        units = act.slot_bins if gated else frames * g.slots
    else:
        units = act.layer_bins if gated else frames
    return units * _per_unit(rule, g)


def evaluate_layer(arch, spikes, weights, *, stride=1, padding=0, output_spikes=None,
                   activity_cache=None):
    """Cost of one layer for a batch.

    spikes: input [batch, channels, height, width, time bins], nonzero = spike.
    weights: [out, in, kh, kw] (a trailing time axis of 1 is dropped); integer
    codes for bit-sliced encodings, floats or codes for "analog".
    output_spikes: total LIF output spikes of the batch (for events="output_spike").
    activity_cache: a dict shared by calls with the same spikes and layer, to
    reuse the spike activity across architectures with the same tile height
    and active rows.
    """
    if weights.dim() == 5:
        weights = weights[..., 0]
    batch = spikes.shape[0]
    g = layer_geometry(arch, spikes.shape[1:4], weights.shape, stride, padding)
    key = (g.tile_rows, g.phase_rows, g.copies_per_tile)
    if activity_cache is not None and key in activity_cache:
        act = activity_cache[key]
    else:
        act = spike_activity(spikes, g)
        if activity_cache is not None:
            activity_cache[key] = act
    tl = build_timeline(arch, g, spikes.shape[4])
    sequential = arch.mapping.conv == "sequential"

    # Current (A) of each column slice summed over all reads: each spike on
    # row k drives the conductances of row k in every column, and the level-0
    # cells of the other weight copies packed in its tile (leak).
    slices, g_reference = conductance_slices(arch, weights)
    v_read, g_empty = arch.crossbar.v_read, arch.crossbar.memory.conductances()[0]
    leak_a = v_read * g_empty * act.leak_drive * g.out_channels   # per slice (out columns)
    slice_a = [(v_read * float(g_slice.sum(0) @ act.row_drive) + leak_a, s) for g_slice, s in slices]
    reference_a = v_read * g_reference * g.out_channels * act.spikes_on_rows + leak_a \
        if arch.crossbar.reference_columns else 0.0

    components = {}
    for c in arch.components:
        count = rule_of(c.count)
        on = rule_of(c.powered, activity=True)
        on = dict(count, gated=on["gated"]) if on["rule"] == "all" else on
        n = installed(count, g)
        # Instances the powered rule can ever power (e.g. used columns of the
        # installed columns); data rules (spiking rows) can reach all.
        used = n if on["rule"] in DATA_RULES else min(n, installed(on, g))
        # Powered time per activation or per time bin, and the powered
        # instances summed over all activations or time bins.
        power = arch.power[c.name]
        per_activation = power is not None and power.level == "activation"
        if per_activation:
            on_ns = tl.on_time(power)
            if on_ns > tl.activation_interval and tl.activations > 1:
                raise ValueError(f"{c.name} is powered longer than the activation interval "
                                 "(it would serve two activations at once)")
            powered = powered_per_activation(on, g, act, sequential)
        elif power is not None:
            on_ns = tl.on_time(power)
            powered = powered_per_bin(on, g, act, sequential)
        else:
            on_ns = powered = 0.0
        energy = c.supply_v * c.static_ua * 1e-6 * on_ns * powered
        if c.model != "static":  # data-driven current (summed over activations) while powered
            if not per_activation and tl.activations > 1:
                raise ValueError(f"{c.name}: {c.model} powered beyond its activation is only "
                                 f"defined with one activation per time bin (this layer has "
                                 f"{tl.activations})")
            if c.model == "crossbar_read":
                current = sum(a for a, _ in slice_a)
            elif c.model == "reference_read":
                current = reference_a
            else:
                gains = c.slice_gains or default_slice_gains(arch)
                current = sum(gains[s] * a for a, s in slice_a)
            energy += c.supply_v * current * on_ns  # A * V * ns = nJ
        if c.event_pj:
            if c.events == "activation":
                events = powered_per_activation(on, g, act, sequential)
            elif c.events == "time_bin":
                events = powered_per_bin(on, g, act, sequential)
            elif output_spikes is None:
                raise ValueError(f"{c.name} counts output spikes: pass output_spikes")
            else:
                events = float(output_spikes)
            energy += c.event_pj * 1e-3 * events
        components[c.name] = ComponentCost(c.group, n, n * c.area_um2, energy, used)

    return LayerCost(g, tl, components, batch,
                     macs=g.rows_needed * g.outputs * spikes.shape[4],
                     synaptic_ops=act.spikes_on_rows * g.out_channels,
                     output_spikes=float(output_spikes or 0.0))
