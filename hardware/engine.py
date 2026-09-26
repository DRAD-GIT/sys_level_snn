"""Energy, latency and area of one weighted layer on a CIM architecture.

evaluate_layer(arch, spikes, weights) maps the layer (mapping.py), extracts
the spike activity, places the stages on the timeline (timeline.py) and
charges every component:
  static   supply_v * static_ua * (powered time per read or time bin)
           * powered instances, summed over all reads or time bins
  events   event_pj * (powered instances per read or time bin, or output spikes)
  data     supply_v * (current computed from spikes and conductances) * read time,
           for the array ("crossbar_read", "reference_read") and the
           weight-slice current mirrors ("slice_mirror")
Powered instances follow the component's `on` rule: all valid instances, or
("gated") only those whose tile/window/layer receives a spike.
"""
import math
from dataclasses import dataclass

from hardware.architecture import TILE_RULES, rule_of
from hardware.mapping import (Geometry, conductance_slices, default_slice_gains, layer_geometry,
                              spike_activity)
from hardware.timeline import Timeline, build_timeline


@dataclass
class ComponentCost:
    group: str
    installed: int
    area_um2: float
    energy_nj: float = 0.0   # summed over all evaluated inferences
    active_ns: float = 0.0   # time its stages run per inference (before gating)


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
    """Instances per unit: per row tile (tile rules), per window (window
    rules) or per layer."""
    r = rule["rule"]
    return {"tiles": g.column_tiles,
            "physical_rows": g.column_tiles * g.tile_rows,
            "physical_columns": g.column_tiles * g.tile_cols,
            "used_columns": g.used_columns,
            "column_groups": g.column_tiles * math.ceil(g.tile_cols / rule.get("size", 1)),
            "outputs": g.out_channels,
            "output_bank": g.column_tiles * g.tile_cols,
            "one": 1,
            "fixed": rule.get("value", 0)}[r]


def installed(rule, g):
    r = rule["rule"]
    if r in TILE_RULES:
        return g.copies * g.row_tiles * _per_unit(rule, g)
    if r == "outputs":
        return g.windows * g.out_channels
    if r == "output_bank":
        return g.copies * _per_unit(rule, g)
    return _per_unit(rule, g)


def powered_per_read(rule, g, act, sequential):
    """Sum over all reads of the powered instances."""
    r = rule["rule"]
    if r == "spiking_rows":  # every word-line segment (one per column tile) with a spike
        return act.spikes_on_rows * g.column_tiles
    gated = rule["gated"]
    if r in TILE_RULES:
        units = act.tile_reads if gated else act.frames * g.windows * sum(g.tile_phases)
    elif r in ("outputs", "output_bank") or sequential:  # a sequential read = one window
        units = act.window_reads if gated else act.frames * g.windows * g.phases
    else:
        units = act.layer_reads if gated else act.frames * g.phases
    return units * _per_unit(rule, g)


def powered_per_bin(rule, g, act, sequential):
    """Sum over all time bins of the powered instances (whole-bin components)."""
    r = rule["rule"]
    gated = rule["gated"]
    if r in TILE_RULES:
        if sequential:
            units = act.shared_tile_bins if gated else act.frames * g.row_tiles
        else:
            units = act.tile_bins if gated else act.frames * g.windows * g.row_tiles
    elif r == "outputs" or (r == "output_bank" and not sequential):
        units = act.window_bins if gated else act.frames * g.windows
    else:
        units = act.layer_bins if gated else act.frames
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
    key = (arch.crossbar.rows, arch.crossbar.active_rows)
    if activity_cache is not None and key in activity_cache:
        act = activity_cache[key]
    else:
        act = spike_activity(spikes, g, arch.crossbar.active_rows)
        if activity_cache is not None:
            activity_cache[key] = act
    tl = build_timeline(arch, g.reads_per_timestep, spikes.shape[4])
    sequential = arch.conv_mapping == "sequential"

    # Current (A) of each column slice summed over all reads: each spike on
    # row k drives the conductances of row k in every column.
    slices, g_reference = conductance_slices(arch, weights)
    v_read = arch.crossbar.v_read
    slice_a = [(v_read * float(g_slice.sum(0) @ act.row_drive), s) for g_slice, s in slices]
    reference_a = v_read * g_reference * g.out_channels * act.spikes_on_rows \
        if arch.crossbar.reference_columns else 0.0

    components = {}
    for c in arch.components:
        count = rule_of(c.count)
        on = rule_of(c.on, activity=True)
        on = dict(count, gated=on["gated"]) if on["rule"] == "all" else on
        n = installed(count, g)
        read_level = bool(c.during) and set(c.during) <= arch.read_stages
        energy = active = 0.0
        if read_level:
            per_read = tl.read_on_time(c.during)
            if per_read > tl.read_interval and tl.reads > 1:
                raise ValueError(f"{c.name} is powered longer than the read interval "
                                 "(it would serve two reads at once)")
            energy += c.supply_v * c.static_ua * 1e-6 * per_read * powered_per_read(on, g, act, sequential)
            active = per_read * tl.reads * tl.timesteps
        elif c.during:
            per_bin = tl.timestep_on_time(c.during)
            energy += c.supply_v * c.static_ua * 1e-6 * per_bin * powered_per_bin(on, g, act, sequential)
            active = per_bin * tl.timesteps
        if c.model != "static":  # data-driven current drawn from supply_v for the stage
            start, end = tl.read[c.during[0]]
            if c.model == "crossbar_read":
                current = sum(a for a, _ in slice_a)
            elif c.model == "reference_read":
                current = reference_a
            else:
                gains = c.slice_gains or default_slice_gains(arch)
                current = sum(gains[s] * a for a, s in slice_a)
            energy += c.supply_v * current * (end - start)  # A * V * ns = nJ
        if c.event_pj:
            if c.events == "read":
                events = powered_per_read(on, g, act, sequential)
            elif c.events == "timestep":
                events = powered_per_bin(on, g, act, sequential)
            elif output_spikes is None:
                raise ValueError(f"{c.name} counts output spikes: pass output_spikes")
            else:
                events = float(output_spikes)
            energy += c.event_pj * 1e-3 * events
        components[c.name] = ComponentCost(c.group, n, n * c.area_um2, energy, active)

    return LayerCost(g, tl, components, batch,
                     macs=g.rows_needed * g.outputs * spikes.shape[4],
                     synaptic_ops=act.spikes_on_rows * g.out_channels,
                     output_spikes=float(output_spikes or 0.0))
