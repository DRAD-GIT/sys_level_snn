"""Layer -> crossbar mapping: windows, tiles, weight slices, spike activity.

Each kernel (one output channel) occupies K = in_channels * kh * kw rows of
one column per weight slice; a window's input patch drives those rows. One
weight copy holds all kernels: K rows x (out_channels * columns_per_weight)
columns, split into row_tiles x column_tiles tiles (the fewest that fit).
"sequential" uses one copy and applies the windows one after another;
"parallel" uses one copy per window so all windows run at once. Row tiles of
a copy work in parallel, each reading its rows in phases of `active_rows`.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from hardware.architecture import slices_per_group


@dataclass(frozen=True)
class Geometry:
    in_channels: int
    out_channels: int
    kernel: tuple            # (kh, kw)
    stride: int
    padding: int
    windows: int             # output pixels
    columns_per_weight: int
    copies: int              # weight copies: windows ("parallel") or 1 ("sequential")
    row_tiles: int           # per copy
    column_tiles: int        # per copy
    tile_phases: tuple       # row phases of each row tile
    tile_rows: int
    tile_cols: int
    reads_per_timestep: int

    @property
    def rows_needed(self):
        return self.in_channels * self.kernel[0] * self.kernel[1]

    @property
    def used_columns(self):  # per weight copy
        return self.out_channels * self.columns_per_weight

    @property
    def outputs(self):
        return self.out_channels * self.windows

    @property
    def phases(self):
        return max(self.tile_phases)


def layer_geometry(arch, input_shape, weight_shape, stride=1, padding=0):
    """input_shape: (channels, height, width); weight_shape: (out, in, kh, kw)."""
    xb = arch.crossbar
    channels, height, width = input_shape
    out_ch, in_ch, kh, kw = weight_shape
    if channels != in_ch:
        raise ValueError(f"input has {channels} channels, weights expect {in_ch}")
    windows = ((height + 2 * padding - kh) // stride + 1) * ((width + 2 * padding - kw) // stride + 1)
    groups = 2 if arch.precision.weight_encoding == "differential" else 1
    per_weight = groups * slices_per_group(arch)
    k_rows = in_ch * kh * kw
    active = xb.active_rows or xb.rows
    phases = tuple(math.ceil(min(xb.rows, k_rows - start) / active)
                   for start in range(0, k_rows, xb.rows))
    copies = windows if arch.conv_mapping == "parallel" else 1
    return Geometry(in_ch, out_ch, (kh, kw), stride, padding, windows, per_weight, copies,
                    len(phases), math.ceil(out_ch * per_weight / xb.cols), phases,
                    xb.rows, xb.cols, (windows // copies) * max(phases))


def quantize_weights(weights, bits):
    """Symmetric uniform quantization: integer codes in [-(2^(b-1)-1), 2^(b-1)-1]
    and the scale (weights ~= codes * scale). bits=None returns the weights."""
    if bits is None:
        return weights, 1.0
    weights = weights.detach()
    top = 2 ** (bits - 1) - 1
    peak = float(weights.abs().max())
    scale = peak / top if peak > 0 else 1.0
    return torch.round(weights / scale).clamp(-top, top).to(torch.int64), scale


def conductance_slices(arch, weights):
    """Every physical column slice as (conductance [out, K] in S, slice index
    within its sign group), plus the reference conductance G(0) for analog."""
    memory, pr = arch.crossbar.memory, arch.precision
    w = weights.reshape(weights.shape[0], -1)
    levels = torch.tensor(memory.conductances(), dtype=torch.float64)
    if pr.weight_encoding == "analog":
        # G linear in the weight over [-max|w|, max|w|] across the memory's
        # conductance range; G(0) is the midpoint.
        g_min, g_max = float(levels[0]), float(levels[-1])
        g_mid = (g_min + g_max) / 2
        peak = float(w.abs().max()) or 1.0
        return [(g_mid + w.double() / peak * (g_max - g_min) / 2, 0)], g_mid
    if w.is_floating_point():
        raise ValueError("bit-sliced encodings need integer weight codes (see quantize_weights)")
    bits, cell = pr.weight_bits, memory.cell_bits
    top = 2 ** (bits - 1) - 1
    if w.abs().max() > top:
        raise ValueError(f"weight codes must be within +/-{top} for {bits}-bit weights")
    if pr.weight_encoding == "twos_complement":
        planes = [w % 2 ** bits]
    elif pr.weight_encoding == "offset":
        planes = [w + 2 ** (bits - 1)]
    else:  # differential: magnitudes of the positive and negative parts
        planes = [w.clamp(min=0), (-w).clamp(min=0)]
    mask = 2 ** cell - 1
    return [(levels[(plane >> (s * cell)) & mask], s)
            for plane in planes for s in range(slices_per_group(arch))], None


def default_slice_gains(arch):
    """Binary mirror gains per slice (least significant first): the most
    significant slice 1, each lower slice 2^-cell_bits of the next."""
    n, cell = slices_per_group(arch), arch.crossbar.memory.cell_bits
    return tuple(2.0 ** (cell * (s - n + 1)) for s in range(n))


@dataclass
class Activity:
    """Spike activity of a batch, summed over frames (samples x time bins).

    *_reads count (read, unit) pairs with at least one input spike; *_bins
    count (time bin, unit) pairs. Units: tile = a row tile of a window's
    weight copy; window = a window; layer = the whole layer.
    """
    frames: int
    row_drive: torch.Tensor  # [K] spikes per row, over frames and windows
    tile_reads: int          # (frame, window, row tile, phase)
    window_reads: int        # (frame, window, phase)
    layer_reads: int         # (frame, phase), all windows at once ("parallel" reads)
    tile_bins: int           # (frame, window, row tile) ("parallel" copies)
    shared_tile_bins: int    # (frame, row tile), any window ("sequential" copy)
    window_bins: int         # (frame, window)
    layer_bins: int          # frame

    @property
    def spikes_on_rows(self):
        return float(self.row_drive.sum())


def spike_activity(spikes, geometry, active_rows=None, max_elements=2 ** 24):
    """spikes: [batch, channels, height, width, time bins]; nonzero = spike."""
    g = geometry
    batch, channels, height, width, bins = spikes.shape
    frames = (spikes != 0).permute(0, 4, 1, 2, 3).reshape(-1, channels, height, width)
    k_rows, rows = g.rows_needed, g.tile_rows
    per_phase = active_rows or rows
    phases_per_tile = math.ceil(rows / per_phase)
    row_drive = torch.zeros(k_rows, dtype=torch.float64)
    totals = dict.fromkeys(("tile_reads", "window_reads", "layer_reads", "tile_bins",
                            "shared_tile_bins", "window_bins", "layer_bins"), 0)
    chunk = max(1, max_elements // (k_rows * g.windows))
    for frame_chunk in frames.split(chunk):
        # [f, K, windows]: each window's input patch, rows ordered like the weights.
        patches = F.unfold(frame_chunk.float(), g.kernel, padding=g.padding, stride=g.stride)
        row_drive += patches.sum((0, 2)).double()
        # Group rows by (row tile, phase): pad K to whole tiles and whole phases.
        padded = F.pad(patches, (0, 0, 0, g.row_tiles * rows - k_rows))
        padded = padded.view(len(patches), g.row_tiles, rows, g.windows)
        padded = F.pad(padded, (0, 0, 0, phases_per_tile * per_phase - rows))
        active = padded.view(len(patches), g.row_tiles, phases_per_tile, per_phase, g.windows) \
            .amax(3) > 0                                         # [f, tile, phase, window]
        totals["tile_reads"] += int(active.sum())
        by_window = active.any(1)                                # [f, phase, window]
        totals["window_reads"] += int(by_window.sum())
        totals["layer_reads"] += int(by_window.any(2).sum())
        tile_bin = active.any(2)                                 # [f, tile, window]
        totals["tile_bins"] += int(tile_bin.sum())
        totals["shared_tile_bins"] += int(tile_bin.any(2).sum())
        totals["window_bins"] += int(tile_bin.any(1).sum())
        totals["layer_bins"] += int(tile_bin.flatten(1).any(1).sum())
    return Activity(len(frames), row_drive, **totals)
