"""Layer -> crossbar mapping: windows, tiles, weight slices, spike activity.

Each kernel (one output channel) occupies K = in_channels * kh * kw rows of
one column per weight slice; a window's input patch drives those rows. One
weight copy holds all kernels: K rows x (out_channels * columns_per_weight)
columns, split into row_tiles x column_tiles tiles (the fewest that fit).
"sequential" uses one copy and applies the windows one after another;
"parallel" uses one copy per window so all windows run at once. Parallel
copies that fit in a tile share it, packed block-diagonally: each copy has
its own rows (its window's inputs) and columns (its outputs); the other
copies' cells on its rows hold level 0 and leak into their columns. Columns
without weights stay off. Row tiles work in parallel, each reading its rows
in phases of `active_rows`.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from hardware.architecture import scaling_std, slices_per_group


def _row_phases(rows_used, tile_rows, phase_rows):
    """Row phases of each row tile holding `rows_used` rows."""
    return tuple(math.ceil(min(tile_rows, rows_used - start) / phase_rows)
                 for start in range(0, rows_used, tile_rows))


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
    copies_per_tile: int     # parallel copies packed into one tile (1 if a copy doesn't fit)
    row_tiles: int           # per tile set
    column_tiles: int        # per tile set
    tile_phases: tuple       # row phases of each row tile of a full tile set
    tile_rows: int
    tile_cols: int
    phase_rows: int          # rows enabled per read (active_rows)
    reads_per_timestep: int
    # Slots: the windows read together, copies_per_tile at a time (the last
    # slot may hold fewer); (number of slots, copies, row phases per row tile).
    slot_kinds: tuple

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

    @property
    def slots(self):
        return sum(n for n, _, _ in self.slot_kinds)

    @property
    def tile_sets(self):  # physical tile sets: one per slot, shared by all when sequential
        return self.slots if self.copies > 1 else 1


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
    k_rows, used = in_ch * kh * kw, out_ch * per_weight
    active = xb.active_rows or xb.rows
    parallel = arch.conv_mapping == "parallel"
    per_tile = min(xb.rows // k_rows, xb.cols // used, windows) \
        if parallel and k_rows <= xb.rows and used <= xb.cols else 1
    full, rest = divmod(windows, per_tile)
    slot_kinds = tuple((n, c, _row_phases(c * k_rows, xb.rows, active))
                       for n, c in ((full, per_tile), (1 if rest else 0, rest)) if n)
    phases = slot_kinds[0][2]
    return Geometry(in_ch, out_ch, (kh, kw), stride, padding, windows, per_weight,
                    windows if parallel else 1, per_tile, len(phases),
                    math.ceil(per_tile * used / xb.cols), phases, xb.rows, xb.cols, active,
                    (1 if parallel else windows) * max(phases), slot_kinds)


def quantize_weights(weights, bits, scaling="max"):
    """Symmetric uniform quantization: integer codes in [-(2^(b-1)-1), 2^(b-1)-1]
    and the scale (weights ~= codes * scale); weights beyond the clip range
    saturate. bits=None returns the weights. The range (clip = top code *
    scale) follows `scaling`:
      "max"     the largest |weight| (nothing is clipped);
      "mse"     the clip, among 200 steps up to the largest |weight|, with the
                least squared error of the quantized weights;
      "std<k>"  k standard deviations of the weights (e.g. "std3").
    """
    if bits is None:
        return weights, 1.0
    weights = weights.detach()
    top = 2 ** (bits - 1) - 1
    peak = float(weights.abs().max())
    if peak == 0:
        return torch.zeros_like(weights, dtype=torch.int64), 1.0
    k = scaling_std(scaling)
    if k is not None:
        clip = min(k * float(weights.std()), peak)
    elif scaling == "mse":
        clip = min((peak * step / 200 for step in range(1, 201)),
                   key=lambda c: float((_quantize(weights, top, c / top) - weights).pow(2).sum()))
    else:
        clip = peak
    scale = clip / top
    return _codes(weights, top, scale), scale


def _codes(weights, top, scale):
    return torch.round(weights / scale).clamp(-top, top).to(torch.int64)


def _quantize(weights, top, scale):
    return _codes(weights, top, scale) * scale


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
    count (time bin, unit) pairs. Units: tile = a row tile of a slot (the
    windows read together); window; layer = the whole layer. *_copies weight
    each tile by the weight copies it holds.
    """
    frames: int
    row_drive: torch.Tensor  # [K] spikes per row, over frames and windows
    leak_drive: float        # spikes x other copies sharing the spiking row's tile
    tile_reads: int          # (frame, slot, row tile, phase)
    tile_read_copies: int
    slot_reads: int          # (frame, slot, phase)
    window_reads: int        # (frame, window, phase)
    layer_reads: int         # (frame, phase), all windows at once ("parallel" reads)
    tile_bins: int           # (frame, slot, row tile) ("parallel" tile sets)
    tile_bin_copies: int
    shared_tile_bins: int    # (frame, row tile), any slot ("sequential" tile set)
    slot_bins: int           # (frame, slot)
    window_bins: int         # (frame, window)
    layer_bins: int          # frame

    @property
    def spikes_on_rows(self):
        return float(self.row_drive.sum())


def spike_activity(spikes, geometry, max_elements=2 ** 24):
    """spikes: [batch, channels, height, width, time bins]; nonzero = spike."""
    g = geometry
    frames = (spikes != 0).permute(0, 4, 1, 2, 3).reshape(-1, *spikes.shape[1:4])
    k_rows, rows, per_phase, per_tile = g.rows_needed, g.tile_rows, g.phase_rows, g.copies_per_tile
    slots, phases = g.slots, g.phases
    slot_rows = per_tile * k_rows
    copies = torch.full((slots,), per_tile, dtype=torch.int64)                      # copies per slot
    copies[-1] = g.windows - (slots - 1) * per_tile
    # Row q of a slot belongs to window slot q // K and is read in phase
    # (q % tile_rows) // phase_rows (of its row tile).
    q = torch.arange(slot_rows)
    window_phase = F.one_hot(q // k_rows * phases + q % rows // per_phase,
                             per_tile * phases).float()           # [slot rows, windows x phases]
    row_drive = torch.zeros(k_rows, dtype=torch.float64)
    leak_drive = 0.0
    totals = dict.fromkeys(("tile_reads", "tile_read_copies", "slot_reads", "window_reads",
                            "layer_reads", "tile_bins", "tile_bin_copies", "shared_tile_bins",
                            "slot_bins", "window_bins", "layer_bins"), 0)
    chunk = max(1, max_elements // (k_rows * g.windows))
    for frame_chunk in frames.split(chunk):
        f = len(frame_chunk)
        # [f, K, windows]: each window's input patch, rows ordered like the weights.
        patches = F.unfold(frame_chunk.float(), g.kernel, padding=g.padding, stride=g.stride)
        row_drive += patches.sum((0, 2)).double()
        per_window = patches.sum((0, 1)).double()                          # [windows]
        leak_drive += float(per_window @ (copies - 1).repeat_interleave(per_tile)[:g.windows].double())
        # [f, slot, slot rows]: the windows of a slot stacked along the rows.
        x = F.pad(patches.transpose(1, 2), (0, 0, 0, slots * per_tile - g.windows))
        x = x.reshape(f, slots, slot_rows)
        # Group rows by (row tile, phase): pad to whole tiles and whole phases.
        tiles = F.pad(x, (0, g.row_tiles * rows - slot_rows)).view(f, slots, g.row_tiles, rows)
        tiles = F.pad(tiles, (0, phases * per_phase - rows))
        active = tiles.view(f, slots, g.row_tiles, phases, per_phase).amax(4) > 0
        windows = (x @ window_phase > 0).view(f, slots * per_tile, phases)[:, :g.windows]
        tile_bin = active.any(3)                                 # [f, slot, tile]
        totals["tile_reads"] += int(active.sum())
        totals["tile_read_copies"] += int(active.sum((0, 2, 3)) @ copies)
        totals["slot_reads"] += int(active.any(2).sum())
        totals["window_reads"] += int(windows.sum())
        totals["layer_reads"] += int(windows.any(1).sum())
        totals["tile_bins"] += int(tile_bin.sum())
        totals["tile_bin_copies"] += int(tile_bin.sum((0, 2)) @ copies)
        totals["shared_tile_bins"] += int(tile_bin.any(1).sum())
        totals["slot_bins"] += int(tile_bin.any(2).sum())
        totals["window_bins"] += int(windows.any(2).sum())
        totals["layer_bins"] += int(tile_bin.flatten(1).any(1).sum())
    return Activity(len(frames), row_drive, leak_drive, **totals)
