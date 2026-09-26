"""Evaluate one weighted layer (conv or dense) on a modular CIM architecture.

Mapping: the layer is unrolled into a K x O matrix, K = in_channels * kh * kw
rows and O = out_channels * output pixels logical outputs. Each weight takes
`columns_per_weight` physical columns (weight slices; x2 for differential).
Row tiles run in parallel; row phases and input slices are sequential reads.
The array current is computed from the actual input codes and conductances;
each cell is read once per input slice, whatever the row phasing.
"""
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from cim.architecture import rule_of
from cim.timeline import build_timeline


@dataclass
class Geometry:
    rows_needed: int        # K
    outputs: int            # O (neurons / logical outputs)
    columns_per_weight: int
    used_columns: int       # L = O * columns_per_weight
    row_tiles: int          # Nr
    column_tiles: int       # Nc
    tile_phases: list       # row phases of each row tile
    input_slices: int
    timesteps: int
    batch: int

    @property
    def phases(self):
        return max(self.tile_phases)


@dataclass
class ComponentResult:
    group: str
    installed: int
    area_um2: float
    energy_nj: float
    powered_ns: float       # powered time per instance per inference


@dataclass
class LayerResult:
    geometry: Geometry
    components: dict
    latency_ns: float       # per inference
    macs: int               # per inference, at the layer's precision
    timeline: object = None
    array_current_ua: float = 0.0   # mean total cell current during a read

    @property
    def energy_nj(self):
        return sum(c.energy_nj for c in self.components.values())

    @property
    def area_um2(self):
        return sum(c.area_um2 for c in self.components.values())

    @property
    def tops_per_w(self):
        return 2 * self.macs / self.energy_nj * 1e-3 if self.energy_nj else 0.0

    def summary(self):
        g = self.geometry
        lines = [f"tiles {g.row_tiles} x {g.column_tiles} (rows {g.rows_needed}, columns "
                 f"{g.used_columns} = {g.outputs} outputs x {g.columns_per_weight}); "
                 f"reads/timestep {g.input_slices} input slices x {g.phases} phases; "
                 f"timesteps {g.timesteps}",
                 f"{'component':<18}{'group':<18}{'installed':>10}{'area um^2':>12}"
                 f"{'energy nJ':>13}{'on ns':>9}"]
        for name, c in self.components.items():
            lines.append(f"{name:<18}{c.group:<18}{c.installed:>10}{c.area_um2:>12.6g}"
                         f"{c.energy_nj:>13.6g}{c.powered_ns:>9.4g}")
        lines.append(f"total: energy {self.energy_nj:.6g} nJ, latency {self.latency_ns:.6g} ns, "
                     f"area {self.area_um2:.6g} um^2, {self.macs} MACs, "
                     f"{self.tops_per_w:.4g} TOPS/W")
        return "\n".join(lines)


def quantize_symmetric(weights, bits):
    """Float weights -> signed integer codes in [-(2^(b-1)-1), 2^(b-1)-1]."""
    top = 2 ** (bits - 1) - 1
    scale = weights.abs().max() / top if weights.abs().max() > 0 else 1.0
    return torch.round(weights / scale).clamp(-top, top).to(torch.int64), scale


def _geometry(arch, x, w, padding):
    xb, pr = arch.crossbar, arch.precision
    batch, _, height, width, timesteps = x.shape
    out_ch, in_ch, kh, kw = w.shape
    out_pixels = (height - kh + 2 * padding + 1) * (width - kw + 2 * padding + 1)
    k_rows = in_ch * kh * kw
    outputs = out_ch * out_pixels
    if pr.weight_encoding == "analog":
        cpw = 1
    elif pr.weight_encoding == "differential":
        cpw = 2 * math.ceil((pr.weight_bits - 1) / xb.cell_bits)
    else:
        cpw = math.ceil(pr.weight_bits / xb.cell_bits)
    used = outputs * cpw
    active = xb.active_rows or xb.rows
    phases = [math.ceil(min(xb.rows, k_rows - start) / active) for start in range(0, k_rows, xb.rows)]
    return Geometry(k_rows, outputs, cpw, used, math.ceil(k_rows / xb.rows),
                    math.ceil(used / xb.cols), phases,
                    math.ceil(pr.input_bits / pr.input_bits_per_read), timesteps, batch)


def _instances(rule, g, xb, phased):
    """Installed instances, or (phased) the sum over row phases of powered ones."""
    rows = sum(g.tile_phases) if phased else g.row_tiles
    everywhere = g.phases if phased else 1
    r = rule["rule"]
    if r == "one":
        return everywhere
    if r == "fixed":
        return rule["value"] * everywhere
    if r == "tiles":
        return rows * g.column_tiles
    if r == "physical_rows":
        return rows * g.column_tiles * xb.rows
    if r == "physical_columns":
        return rows * g.column_tiles * xb.cols
    if r == "used_columns":
        return rows * g.used_columns
    if r == "column_groups":
        return rows * g.column_tiles * math.ceil(xb.cols / rule["size"])
    if r == "output_bank":
        return everywhere * g.column_tiles * xb.cols
    if r == "outputs":
        return everywhere * g.outputs
    raise ValueError(r)


def _conductance_slices(arch, w, levels):
    """Conductance kernels (siemens), one per physical column slice."""
    xb, pr = arch.crossbar, arch.precision
    g_on, g_off = 1 / xb.r_on, 1 / xb.r_off
    if pr.weight_encoding == "analog":
        # Linear map including zero, so G(0) is representable for the reference.
        lo = min(float(levels.min()), 0.0)
        hi = max(float(levels.max()), 0.0)
        if hi == lo:
            return [torch.full_like(w, g_off, dtype=torch.float64)], g_off
        slope = (g_on - g_off) / (hi - lo)
        g = (slope * w.double() + g_off - slope * lo).clamp(g_off, g_on)
        return [g.double()], g_off - slope * lo
    if w.is_floating_point():
        raise ValueError("bit-sliced encodings need integer weight codes (see quantize_symmetric)")
    bits, cell = pr.weight_bits, xb.cell_bits
    top = 2 ** cell - 1
    level_to_g = lambda level: g_off + level.double() / top * (g_on - g_off)
    if pr.weight_encoding == "twos_complement":
        if w.min() < -2 ** (bits - 1) or w.max() > 2 ** (bits - 1) - 1:
            raise ValueError(f"weight codes outside the signed {bits}-bit range")
        planes = [w % 2 ** bits]
    elif pr.weight_encoding == "offset":
        if w.min() < -2 ** (bits - 1) or w.max() > 2 ** (bits - 1) - 1:
            raise ValueError(f"weight codes outside the signed {bits}-bit range")
        planes = [w + 2 ** (bits - 1)]
    else:  # differential: magnitudes of the positive and negative parts
        if w.abs().max() > 2 ** (bits - 1) - 1:
            raise ValueError(f"differential weight codes must be within +/-{2 ** (bits - 1) - 1}")
        bits = bits - 1
        planes = [w.clamp(min=0), (-w).clamp(min=0)]
    slices = []
    for value in planes:
        for s in range(math.ceil(bits / cell)):
            slices.append(level_to_g((value >> (s * cell)) & top))
    return slices, None


def _window_sums(inputs, kh, kw, padding):
    """For each kernel offset (c, i, j): the input summed over every position the
    kernel visits, i.e. the total drive that one cell row receives."""
    padded = F.pad(inputs, (padding,) * 4)
    oh, ow = padded.shape[1] - kh + 1, padded.shape[2] - kw + 1
    sums = torch.empty(inputs.shape[0], kh, kw, dtype=torch.float64)
    for i in range(kh):
        for j in range(kw):
            sums[:, i, j] = padded[:, i:i + oh, j:j + ow].sum((1, 2))
    return sums


def _array_current(arch, x, w, padding, levels, g):
    """Total cell current (A) summed over all reads, split data / reference.

    sum over outputs of conv(x, G) = sum_(c,i,j) [sum_o G(o,c,i,j)] * [window
    sum of x at (c,i,j)], so no convolution is needed. float64 throughout.
    """
    xb, pr = arch.crossbar, arch.precision
    if x.min() < 0 or x.max() > 2 ** pr.input_bits - 1:
        raise ValueError(f"input codes must be within [0, {2 ** pr.input_bits - 1}]")
    codes = x.to(torch.int64)
    slices, g_zero = _conductance_slices(arch, w, levels)
    kh, kw = w.shape[2], w.shape[3]
    top = 2 ** pr.input_bits_per_read - 1
    data = reference = 0.0
    for s in range(g.input_slices):
        # Word-line drive of this input slice, summed over batch and timesteps.
        drive = (((codes >> (s * pr.input_bits_per_read)) & top).double() / top).sum((0, 4))
        windows = _window_sums(drive, kh, kw, padding)
        for kernel in slices:
            data += float((kernel.sum(0) * windows).sum())
        if xb.reference_columns:
            reference += float(windows.sum()) * w.shape[0] * g_zero
    return data * xb.v_read, reference * xb.v_read


def evaluate_layer(arch, x, w, *, padding=0, levels=None, output_spikes=None):
    """Hardware cost of one layer for a batch of inputs.

    x: input codes [batch, channels, height, width, timesteps] (spikes: 0/1).
    w: weights [out, in, kh, kw]; integer codes for bit-sliced encodings,
       floats (with `levels`, the quantization levels) for "analog".
    output_spikes: total output spikes, for components with events="output_spike".
    Returns per-inference energy/latency (batch-averaged) and installed area.
    """
    xb = arch.crossbar
    if w.dim() == 5:
        w = w[..., 0]
    if levels is None:
        levels = torch.stack([w.min(), w.max()]).float()
    g = _geometry(arch, x, w, padding)
    tl = build_timeline(arch, g.input_slices * g.phases, g.timesteps)
    data_a, reference_a = _array_current(arch, x, w, padding, levels, g)
    runs = g.batch * g.timesteps  # timesteps executed over the batch

    components = {}
    for c in arch.components:
        count_rule = rule_of(c.count)
        on_rule = count_rule if rule_of(c.on)["rule"] == "all" else rule_of(c.on)
        installed = _instances(count_rule, g, xb, phased=False)
        read_level = bool(c.during) and all(s in tl.read for s in c.during)
        if read_level:
            per_read = tl.read_on_time(c.during)
            if per_read > tl.read_interval + 1e-12 and g.input_slices * g.phases > 1:
                raise ValueError(f"{c.name} is powered longer than the read interval "
                                 "(it would be on for two reads at once)")
            instance_ns = per_read * _instances(on_rule, g, xb, phased=True) * g.input_slices
            powered_ns = per_read * g.input_slices * g.phases * g.timesteps
        elif c.during:
            per_step = tl.timestep_on_time(c.during)
            instance_ns = per_step * _instances(on_rule, g, xb, phased=False)
            powered_ns = per_step * g.timesteps
        else:
            instance_ns = powered_ns = 0.0
        energy = c.supply_v * c.static_ua * 1e-6 * instance_ns * runs  # nJ
        if c.model != "static":  # cell current drawn from supply_v for the read stage
            start, end = tl.read[c.during[0]]
            current = data_a if c.model == "crossbar_read" else reference_a
            energy += c.supply_v * current * (end - start)  # A * V * ns = nJ
        if c.event_pj:
            if c.events == "read":
                events = _instances(on_rule, g, xb, phased=True) * g.input_slices * runs
            elif c.events == "timestep":
                events = _instances(on_rule, g, xb, phased=False) * runs
            else:
                if output_spikes is None:
                    raise ValueError(f"{c.name} counts output spikes: pass output_spikes")
                events = float(output_spikes)
            energy += c.event_pj * 1e-3 * events
        components[c.name] = ComponentResult(c.group, installed, installed * c.area_um2,
                                             energy / g.batch, powered_ns)

    reads = g.batch * g.timesteps * g.input_slices * g.phases
    return LayerResult(g, components, tl.latency_ns, g.rows_needed * g.outputs * g.timesteps,
                       tl, (data_a + reference_a) * 1e6 / reads if reads else 0.0)
