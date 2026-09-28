"""Hardware engine checks.

ReferenceModel re-derives every cost by literally enumerating samples, time
bins, windows, row tiles, phases and cells; the vectorized engine must match
it. Plus hand calculations, earlier worked examples, timeline placement and
validation.
"""
import math
import unittest

import numpy as np
import torch

from architectures import crossbars
from hardware import (Architecture, Component, Crossbar, Memory, Precision, Stage, compose,
                      evaluate_layer, quantize_weights)
from hardware.architecture import LAYER_RULES, TILE_RULES, rule_of
from hardware.mapping import layer_geometry

LINEAR_1BIT = Memory(cell_bits=1, r_on=1e3, r_off=1e6)
NONUNIFORM_2BIT = Memory(cell_bits=2, levels_s=(1e-6, 3e-4, 5e-4, 1e-3))


def rram_ota_design(conv_mapping="sequential"):
    """The 1-bit RRAM current-mode design of run.py, with slice mirrors."""
    return compose("rram_ota", Precision(4, "twos_complement"), [
        crossbars.conv_xbar(cell_bits=1, r_on=20e3, r_off=200e3, rows=64, cols=64, v_read=0.2,
                            read_ns=5.0),
        Stage("fire", 2.0, level="timestep"),
        Component("sl_ota", count="physical_columns", on={"rule": "used_columns", "gated": True},
                  during="read", static_ua=10.0),
        Component("slice_mirrors", model="slice_mirror", count="used_columns", during="read"),
        Component("lif", count="outputs", during="fire", static_ua=10.0),
    ], conv_mapping=conv_mapping)


def conventional_example():
    """Worked example: current-mode crossbar with analog cells and G(0)
    reference columns, a DA per column and LIFs on for the whole time bin."""
    return compose("conventional", Precision(None, "analog"), [
        crossbars.conv_xbar(r_on=2e3, r_off=200e3, v_read=0.1, active_rows=8, read_ns=4.5,
                            cell_supply_v=1.1, reference_columns=True, tile_area_um2=136.67),
        Stage("subtract", 0.0),
        Stage("fire", 2.0, level="timestep"),
        Component("da", count="physical_columns", on="used_columns", during="read",
                  static_ua=6.1, area_um2=30.22),
        Component("reference_subtractor", count="output_bank", on="outputs", during="subtract"),
        Component("lif", count="output_bank", on="outputs", during="timestep",
                  static_ua=6.0, area_um2=86.79)])


def c3cim_example():
    """Worked example: C3CIM crossbar with a VI converter per column."""
    return compose("c3cim", Precision(None, "analog"), [
        crossbars.c3cim_xbar(r_on=2e3, r_off=20e3, column_area_um2=4.27, driver_area_um2=86.36),
        Stage("vi", 10.0),
        Stage("fire", 2.0, level="timestep"),
        Component("vi", count="physical_columns", on="used_columns", during="vi",
                  supply_v=1.0, static_ua=24.3, area_um2=29.79),
        Component("lif", count="output_bank", on="outputs", during="timestep",
                  static_ua=6.0, area_um2=86.79)])


def reference_cost(arch, spikes, weights, stride, padding, output_spikes=0.0):
    """Per-inference energy per component and latency, by brute force.

    Stages must be serial (default placement) so powered times are sums of
    durations; timeline placement itself is tested separately.
    """
    xb, pr = arch.crossbar, arch.precision
    batch, channels, height, width, bins = spikes.shape
    out, _, kh, kw = weights.shape
    k_rows = channels * kh * kw
    rows, active_rows = xb.rows, xb.active_rows or xb.rows
    sequential = arch.conv_mapping == "sequential"

    # Windows and their input patches, rows ordered (channel, i, j).
    xp = torch.zeros(batch, channels, height + 2 * padding, width + 2 * padding, bins)
    xp[:, :, padding:padding + height, padding:padding + width] = (spikes != 0).float()
    positions = [(y, x) for y in range((height + 2 * padding - kh) // stride + 1)
                 for x in range((width + 2 * padding - kw) // stride + 1)]

    def patch(b, t, y, x):
        return [int(xp[b, c, y * stride + i, x * stride + j, t])
                for c in range(channels) for i in range(kh) for j in range(kw)]

    # Physical columns: (conductance of each cell, slice index), built from the
    # codes bit by bit and the memory's level conductances.
    levels = xb.memory.conductances()
    g_min, g_max = levels[0], levels[-1]
    g_mid = (g_min + g_max) / 2
    flat = weights.reshape(out, -1)
    columns = []
    if pr.weight_encoding == "analog":
        peak = float(flat.abs().max())
        n_slices = 1
        for o in range(out):
            columns.append(([g_mid + float(v) / peak * (g_max - g_min) / 2 for v in flat[o]], 0))
    else:
        bits, cell = pr.weight_bits, xb.memory.cell_bits
        top = 2 ** cell - 1
        for o in range(out):
            codes = [int(v) for v in flat[o]]
            if pr.weight_encoding == "twos_complement":
                planes, n = [[c % 2 ** bits for c in codes]], bits
            elif pr.weight_encoding == "offset":
                planes, n = [[c + 2 ** (bits - 1) for c in codes]], bits
            else:
                planes, n = [[max(c, 0) for c in codes], [max(-c, 0) for c in codes]], bits - 1
            n_slices = math.ceil(n / cell)
            for plane in planes:
                for s in range(n_slices):
                    columns.append(([levels[(v >> (s * cell)) & top] for v in plane], s))
    used = len(columns)

    # Physical layout. A slot = the windows read together: one window per read
    # ("sequential"), or the parallel copies packed block-diagonally into one
    # tile set (as many as fit in a tile, else one copy per tile set). Copy j
    # of a slot owns slot rows j*K.. and columns j*used..; the rest of its rows
    # hold level-0 cells in the other copies' columns. Unused columns are off.
    per_tile = 1
    if not sequential and k_rows <= rows and used <= xb.cols:
        while (per_tile + 1) * k_rows <= rows and (per_tile + 1) * used <= xb.cols \
                and per_tile < len(positions):
            per_tile += 1
    slots = [positions[i:i + per_tile] for i in range(0, len(positions), per_tile)]
    column_tiles = math.ceil(per_tile * used / xb.cols)

    def layout(slot):
        """Row tiles of the slot -> phases -> rows (slot row q = j * K + k)."""
        q_rows = len(slot) * k_rows
        tiles = [range(r, min(r + rows, q_rows)) for r in range(0, q_rows, rows)]
        return [[t[p:p + active_rows] for p in range(0, len(t), active_rows)] for t in tiles]

    def tile_used_columns(slot, rt):   # used columns of copies with rows in row tile rt
        return used * sum(1 for j in range(len(slot)) if j * k_rows < (rt + 1) * rows
                          and (j + 1) * k_rows > rt * rows)

    max_phases = max(len(p) for p in layout(slots[0]))
    n_slices = max(index for _, index in columns) + 1
    slice_current = [0.0] * n_slices

    def per_unit(rule):
        return {"tiles": column_tiles, "physical_rows": column_tiles * rows,
                "physical_columns": column_tiles * xb.cols,
                "column_groups": column_tiles * math.ceil(xb.cols / rule.get("size", 1)),
                "outputs": out, "output_bank": column_tiles * xb.cols, "one": 1,
                "fixed": rule.get("value", 0)}[rule["rule"]]

    def tile_units(rule, slot, rt):
        return tile_used_columns(slot, rt) if rule["rule"] == "used_columns" else per_unit(rule)

    def rules(c):
        count = rule_of(c.count)
        on = rule_of(c.on, activity=True)
        return dict(count, gated=on["gated"]) if on["rule"] == "all" else on

    read_powered = {c.name: 0.0 for c in arch.components}
    bin_powered = {c.name: 0.0 for c in arch.components}
    reference_current = 0.0
    reads_per_bin = max_phases * (len(positions) if sequential else 1)
    for b in range(batch):
        for t in range(bins):
            patches = {w: patch(b, t, *w) for w in positions}
            # Cell currents: every row of a slot drives all of the slot's used
            # columns (and reference columns): its own copy's weights, level 0
            # in the other copies' columns.
            for slot in slots:
                for j, w in enumerate(slot):
                    for k, s in enumerate(patches[w]):
                        for owner in range(len(slot)):
                            for cells, index in columns:
                                g = cells[k] if owner == j else g_min
                                slice_current[index] += s * g * xb.v_read
                            reference_current += s * out * (g_mid if owner == j else g_min) * xb.v_read
            slot_spikes = {tuple(slot): [s for w in slot for s in patches[w]] for slot in slots}
            for slot in slots:
                phases = layout(slot)
                x = slot_spikes[tuple(slot)]
                for p in range(max_phases):
                    spike = {r: any(x[q] for q in phases[r][p])
                             for r in range(len(phases)) if p < len(phases[r])}
                    window_spike = {w: any(x[q] for r in spike for q in phases[r][p]
                                           if q // k_rows == j) for j, w in enumerate(slot)}
                    for c in arch.components:
                        rule = rules(c)
                        if rule["rule"] == "spiking_rows":
                            n = sum(x[q] for r in spike for q in phases[r][p])
                            read_powered[c.name] += n * column_tiles
                        elif rule["rule"] in TILE_RULES:
                            read_powered[c.name] += sum(tile_units(rule, slot, r) for r, s in spike.items()
                                                        if s or not rule["gated"])
                        elif rule["rule"] == "outputs":
                            read_powered[c.name] += per_unit(rule) * sum(
                                1 for w in slot if window_spike[w] or not rule["gated"])
                        elif rule["rule"] == "output_bank":
                            read_powered[c.name] += per_unit(rule) * (
                                not rule["gated"] or any(spike.values()))
            # Layer units: per read, i.e. per sequential window read or per
            # parallel phase (all slots at once).
            reads = [([slot], p) for slot in slots for p in range(max_phases)] if sequential else \
                [(slots, p) for p in range(max_phases)]
            for together, p in reads:
                hit = any(slot_spikes[tuple(s)][q] for s in together
                          for tile in layout(s) if p < len(tile) for q in tile[p])
                for c in arch.components:
                    rule = rules(c)
                    if rule["rule"] in LAYER_RULES:
                        read_powered[c.name] += per_unit(rule) * (hit or not rule["gated"])
            # Whole-bin units: a row tile / window / tile set / layer that
            # receives any spike during this time bin.
            any_tile = {(tuple(slot), r): any(slot_spikes[tuple(slot)][q] for q in range(
                r * rows, min((r + 1) * rows, len(slot) * k_rows)))
                for slot in slots for r in range(len(layout(slot)))}
            for c in arch.components:
                rule = rules(c)
                if rule["rule"] == "spiking_rows":   # counted per read only
                    continue
                if rule["rule"] in TILE_RULES:
                    if sequential:   # one shared tile set; a slot = a window
                        units = [per_unit(rule) if rule["rule"] != "used_columns" else used
                                 for r in range(len(layout(slots[0])))
                                 if not rule["gated"] or any(any_tile[(tuple(s), r)] for s in slots)]
                    else:
                        units = [tile_units(rule, list(s), r) for (s, r), hit in any_tile.items()
                                 if hit or not rule["gated"]]
                    bin_powered[c.name] += sum(units)
                elif rule["rule"] == "outputs":
                    bin_powered[c.name] += per_unit(rule) * sum(
                        1 for w in positions if not rule["gated"] or any(patches[w]))
                elif rule["rule"] == "output_bank" and not sequential:
                    bin_powered[c.name] += per_unit(rule) * sum(
                        1 for s in slots if not rule["gated"] or any(slot_spikes[tuple(s)]))
                else:
                    bin_powered[c.name] += per_unit(rule) * (
                        not rule["gated"] or any(any(v) for v in patches.values()))

    durations = {s.name: s.duration_ns for s in arch.stages}
    read_span = sum(s.duration_ns for s in arch.stages if s.level == "read")
    # Serial stage edges within a time bin: read stages repeat reads_per_bin times,
    # then the time-bin stages follow.
    edges, t = {}, 0.0
    for stage in (s for s in arch.stages if s.level == "read"):
        edges[stage.name] = (t, (reads_per_bin - 1) * read_span + t + stage.duration_ns)
        t += stage.duration_ns
    t = reads_per_bin * read_span
    for stage in (s for s in arch.stages if s.level == "timestep"):
        edges[stage.name] = (t, t + stage.duration_ns)
        t += stage.duration_ns
    bin_span = reads_per_bin * read_span + sum(s.duration_ns for s in arch.stages if s.level == "timestep")
    energy = {}
    for c in arch.components:
        e = 0.0
        if c.during and set(c.during) <= arch.read_stages:
            e += c.supply_v * c.static_ua * 1e-6 * sum(durations[s] for s in c.during) * read_powered[c.name]
            events = read_powered[c.name] if c.events == "read" else bin_powered[c.name]
        elif c.window:
            (s0, e0, o0), (s1, e1, o1) = c.window
            span = edges[s1][e1 == "end"] + o1 - edges[s0][e0 == "end"] - o0
            e += c.supply_v * c.static_ua * 1e-6 * span * bin_powered[c.name]
            events = bin_powered[c.name] if c.events == "timestep" else read_powered[c.name]
        else:
            e += c.supply_v * c.static_ua * 1e-6 * bin_span * bin_powered[c.name] if c.during else 0.0
            events = bin_powered[c.name] if c.events == "timestep" else read_powered[c.name]
        if c.events == "output_spike":
            events = output_spikes
        e += c.event_pj * 1e-3 * events
        if c.model == "crossbar_read":
            e += c.supply_v * sum(slice_current) * durations[c.during[0]]
        elif c.model == "reference_read":
            e += c.supply_v * reference_current * durations[c.during[0]]
        elif c.model == "slice_mirror":  # binary: MSB x1, each lower slice / 2^cell_bits
            cell = xb.memory.cell_bits
            gains = c.slice_gains or [2.0 ** (cell * (s - n_slices + 1)) for s in range(n_slices)]
            e += c.supply_v * sum(gains[s] * slice_current[s] for s in range(n_slices)) \
                * durations[c.during[0]]
        energy[c.name] = e / batch
    return energy, bins * bin_span


def test_arch(mapping, precision, crossbar, custom_gains=None):
    gated = lambda rule, **kw: dict(rule=rule, gated=True, **kw)
    return Architecture(
        "test", crossbar, precision,
        [Stage("drive", 2.0), Stage("sense", 3.0), Stage("fire", 1.0, level="timestep")],
        [Component("cells", model="crossbar_read", during=["sense"], supply_v=1.0),
         Component("mirrors", model="slice_mirror", during=["sense"], supply_v=1.2),
         Component("mirrors_custom", model="slice_mirror", during=["drive"], supply_v=0.9,
                   slice_gains=custom_gains),
         Component("ota", count="physical_columns", on=gated("used_columns"), during=["sense"],
                   static_ua=10.0, event_pj=0.5),
         Component("bias", count="physical_columns", during=["drive", "sense"], static_ua=1.0),
         Component("driver", count={"rule": "column_groups", "size": 3},
                   on=gated("column_groups", size=3), during=["drive"], static_ua=4.0),
         Component("wl_driver", count="physical_rows", on="spiking_rows", during=["drive"],
                   static_ua=2.0, event_pj=0.1),
         Component("tile_ctrl", count="tiles", on=gated("tiles"), during=["timestep"], static_ua=3.0),
         Component("neuron", count="outputs", on=gated("outputs"), during=["timestep"],
                   static_ua=5.0, event_pj=0.2, events="output_spike"),
         Component("bank", count="output_bank", during=["timestep"], static_ua=0.5,
                   event_pj=0.3, events="timestep"),
         Component("sense_amp", count="output_bank", on=gated("output_bank"), during=["sense"],
                   static_ua=0.7),
         Component("controller", count="one", on=gated("one"), during=["sense"], event_pj=1.0),
         Component("clock", count={"rule": "fixed", "value": 5}, during=["timestep"], static_ua=0.2),
         Component("windowed", count="physical_columns", on=gated("used_columns"),
                   window=(("sense", "start", 0.5), ("fire", "end", -0.25)), static_ua=6.0)],
        conv_mapping=mapping)


class ReferenceModelTests(unittest.TestCase):
    def check(self, arch, spikes, weights, stride=1, padding=0):
        result = evaluate_layer(arch, spikes, weights, stride=stride, padding=padding, output_spikes=17)
        expected, latency = reference_cost(arch, spikes, weights, stride, padding, output_spikes=17)
        for name, energy in expected.items():
            got = result.components[name].energy_nj / result.inferences
            self.assertAlmostEqual(got, energy, delta=1e-12 + 1e-9 * abs(energy), msg=name)
        self.assertAlmostEqual(result.latency_ns, latency, places=9)

    def test_conv_and_dense_both_mappings(self):
        g = torch.Generator().manual_seed(0)
        conv_spikes = (torch.rand(2, 3, 6, 6, 3, generator=g) > 0.8).float()
        conv_weights = torch.randint(-7, 8, (5, 3, 3, 3), generator=g)
        dense_spikes = (torch.rand(2, 40, 1, 1, 2, generator=g) > 0.7).float()
        dense_weights = torch.randint(-7, 8, (7, 40, 1, 1), generator=g)
        encodings = ("twos_complement", "offset", "differential", "analog")
        for mapping in ("sequential", "parallel"):
            for memory in (NONUNIFORM_2BIT, LINEAR_1BIT):
                # 16-row tiles with 6 active rows: partial tiles, 3 phases.
                crossbar = Crossbar(memory, rows=16, cols=8, active_rows=6)
                for encoding in encodings:
                    analog = encoding == "analog"
                    magnitude_bits = 3 if encoding == "differential" else 4
                    slices = 1 if analog else math.ceil(magnitude_bits / memory.cell_bits)
                    gains = tuple(0.3 + 0.5 * s for s in range(slices))   # not binary
                    arch = test_arch(mapping, Precision(None if analog else 4, encoding),
                                     crossbar, gains)
                    w_conv = conv_weights.float() if analog else conv_weights
                    w_dense = dense_weights.float() if analog else dense_weights
                    with self.subTest(mapping=mapping, cell_bits=memory.cell_bits, encoding=encoding):
                        self.check(arch, conv_spikes, w_conv, stride=2, padding=1)
                        self.check(arch, conv_spikes, w_conv, stride=1, padding=0)
                        self.check(arch, dense_spikes, w_dense)

    def test_parallel_copies_packed_in_shared_tiles(self):
        """Small kernels: several parallel copies per tile, row phases that
        cross copy boundaries, and a last tile set holding fewer copies."""
        g = torch.Generator().manual_seed(2)
        spikes = (torch.rand(2, 2, 4, 4, 3, generator=g) > 0.6).float()   # 25 windows
        weights = torch.randint(-3, 4, (2, 2, 2, 2), generator=g)          # K = 8 rows
        for memory in (NONUNIFORM_2BIT, LINEAR_1BIT):
            for encoding in ("twos_complement", "differential", "analog"):
                analog = encoding == "analog"
                arch = test_arch("parallel", Precision(None if analog else 3, encoding),
                                 Crossbar(memory, rows=30, cols=20, active_rows=7, reference_columns=analog))
                if analog:
                    arch.components.append(Component("reference", model="reference_read",
                                                     during=["sense"]))
                w = weights.float() if analog else weights
                with self.subTest(cell_bits=memory.cell_bits, encoding=encoding):
                    geometry = layer_geometry(arch, spikes.shape[1:4], w.shape, padding=1)
                    self.assertGreater(geometry.copies_per_tile, 1)
                    self.assertNotEqual(geometry.windows % geometry.copies_per_tile, 0)
                    self.check(arch, spikes, w, stride=1, padding=1)
                    self.check(arch, spikes, w, stride=2, padding=0)

    def test_silent_input_powers_only_ungated_parts(self):
        arch = test_arch("parallel", Precision(4), Crossbar(LINEAR_1BIT, rows=8, cols=8))
        silent = torch.zeros(1, 2, 4, 4, 2)
        result = evaluate_layer(arch, silent, torch.ones(3, 2, 3, 3, dtype=torch.int64),
                                output_spikes=0)
        for name in ("cells", "mirrors", "mirrors_custom", "ota", "driver", "wl_driver",
                     "tile_ctrl", "neuron", "sense_amp", "controller", "windowed"):
            self.assertEqual(result.components[name].energy_nj, 0.0, name)
        for name in ("bias", "bank", "clock"):
            self.assertGreater(result.components[name].energy_nj, 0.0, name)


class HandCalculationTests(unittest.TestCase):
    def test_rram_ota_dense_layer(self):
        """128 -> 64 dense layer, 4-bit weights on 1-bit RRAM, 4 time bins."""
        arch = rram_ota_design()
        g = torch.Generator().manual_seed(0)
        weights = torch.randint(-7, 8, (64, 128, 1, 1), generator=g)
        spikes = torch.randint(0, 2, (1, 128, 1, 1, 4), generator=g)
        r = evaluate_layer(arch, spikes, weights)
        codes, s = weights[:, :, 0, 0].numpy() % 16, spikes[0, :, 0, 0, :].numpy()
        # Current of weight-bit b's columns, over all outputs and time bins.
        bit_current = [sum(0.2 * np.sum(s[:, t] * np.where((codes[o] >> b) & 1, 1 / 20e3, 1 / 200e3))
                           for o in range(64) for t in range(4)) for b in range(4)]
        # OTAs: 256 used columns in each of the 2 row tiles, per time bin in
        # which that row tile receives a spike.
        busy = sum(bool(s[r * 64:(r + 1) * 64, t].any()) for r in range(2) for t in range(4))
        expected = {"cells": 1.1 * sum(bit_current) * 5.0,
                    "sl_ota": 1.1 * 10e-6 * 256 * busy * 5.0,
                    # mirrors: MSB x1, then 1/2, 1/4, 1/8
                    "slice_mirrors": 1.1 * sum(2.0 ** (b - 3) * bit_current[b] for b in range(4)) * 5.0,
                    # comparators: 64 neurons on for the 2 ns fire step of each of 4 bins
                    "lif": 1.1 * 10e-6 * 64 * 4 * 2.0}
        for name, energy in expected.items():
            self.assertAlmostEqual(r.components[name].energy_nj, energy, places=9)
        self.assertEqual(r.latency_ns, 4 * (5.0 + 2.0))
        self.assertEqual(r.components["sl_ota"].installed, 512)
        self.assertEqual((r.macs, r.synaptic_ops), (128 * 64 * 4, s.sum() * 64))

    def test_array_conducting_until_neurons_fire(self):
        g = torch.Generator().manual_seed(1)
        weights = torch.randint(-7, 8, (8, 20, 1, 1), generator=g)
        spikes = (torch.rand(1, 20, 1, 1, 3, generator=g) > 0.5).float()

        def cells(during, **crossbar):
            arch = compose("x", Precision(4), [
                crossbars.conv_xbar(r_on=1e3, r_off=1e6, read_ns=5.0, during=during, **crossbar),
                Stage("fire", 2.0, level="timestep"),
                Component("lif", count="outputs", during="fire", static_ua=1.0)])
            return evaluate_layer(arch, spikes, weights).components["cells"].energy_nj

        # One read per time bin: the bin's current flows for 5 + 2 ns instead of 5.
        self.assertAlmostEqual(cells(["read", "fire"]), cells("read") * 7.0 / 5.0, places=12)
        with self.assertRaisesRegex(ValueError, "one read per time bin"):
            cells(["read", "fire"], rows=8, active_rows=4)   # 3 row tiles x 2 phases

    def test_worked_examples(self):
        x, w = torch.ones(1, 96, 1, 1, 1), torch.ones(2, 96, 1, 1)
        r = evaluate_layer(c3cim_example(), x, w)
        for name, energy in dict(column_source=.000132, column_driver=.0156684, vi=.005832,
                                 lif=.0063624).items():
            self.assertAlmostEqual(r.components[name].energy_nj, energy, places=12)
        self.assertAlmostEqual(r.latency_ns, 482.0)
        self.assertAlmostEqual(r.area_um2, 10259.68, places=6)
        r = evaluate_layer(conventional_example(), x, w)
        for name, energy in dict(cells=.04752, reference_cells=.0239976, da=.00072468,
                                 reference_subtractor=0.0, lif=.0005016).items():
            self.assertAlmostEqual(r.components[name].energy_nj, energy, places=12)
        self.assertAlmostEqual(r.latency_ns, 38.0)
        self.assertAlmostEqual(r.area_um2, 9969.4, places=6)

    def test_conv_mappings_trade_area_for_latency(self):
        spikes = torch.ones(1, 2, 5, 5, 1)
        weights = torch.ones(4, 2, 3, 3, dtype=torch.int64)
        seq = evaluate_layer(rram_ota_design("sequential"), spikes, weights, padding=1)
        par = evaluate_layer(rram_ota_design("parallel"), spikes, weights, padding=1)
        self.assertEqual(seq.geometry.windows, 25)
        # A copy is 18 rows x 16 columns: 3 fit block-diagonally in a 64x64
        # tile, so 25 copies take 9 tiles (the last holds one copy).
        self.assertEqual(par.geometry.copies_per_tile, 3)
        self.assertEqual((seq.components["cells"].installed, par.components["cells"].installed), (1, 9))
        self.assertEqual(par.components["sl_ota"].installed, 9 * 64)
        # Used: only the 16 weight columns of each of the 25 copies can be powered.
        self.assertEqual(par.components["sl_ota"].used, 25 * 16)
        self.assertEqual((seq.components["sl_ota"].installed, seq.components["sl_ota"].used), (64, 16))
        self.assertEqual(par.components["lif"].used, par.components["lif"].installed)
        self.assertEqual((seq.latency_ns, par.latency_ns), (25 * 5.0 + 2.0, 5.0 + 2.0))
        # Packed copies leak: a spike on a copy's row drives the level-0 cells
        # (200 kohm) in the 2 x 16 columns of the other two copies of its tile.
        per_window = torch.nn.functional.unfold(spikes[..., 0], (3, 3), padding=1).sum((0, 1))
        leak = 0.2 / 200e3 * float(per_window[:24].sum()) * 2 * 16 * 1.1 * 5.0
        self.assertAlmostEqual(par.components["cells"].energy_nj,
                               seq.components["cells"].energy_nj + leak, places=12)
        # Slice mirrors of every used column: 25 copies x 16, but no mirror of
        # the empty columns; OTAs likewise only for the used columns.
        self.assertEqual(par.components["slice_mirrors"].installed, 25 * 16)
        self.assertAlmostEqual(par.components["sl_ota"].energy_nj, 1.1 * 10e-6 * 25 * 16 * 5.0, places=12)


class TimelineTests(unittest.TestCase):
    def run_arch(self, stages, components, x=None, **kw):
        crossbar = kw.pop("crossbar", Crossbar(LINEAR_1BIT))
        arch = Architecture("t", crossbar, Precision(None, "analog"), stages, components, **kw)
        return evaluate_layer(arch, torch.ones(1, 64, 1, 1, 3) if x is None else x,
                              torch.ones(1, 64, 1, 1))

    def test_serial_parallel_and_overlap(self):
        stages = [Stage("a", 4.0), Stage("b", 6.0),               # b after a
                  Stage("c", 3.0, after=[]),                      # c parallel with a
                  Stage("d", 5.0, after=["b"], offset_ns=-2.0)]   # d overlaps b by 2 ns
        comps = [Component("ac", during=["a", "c"], static_ua=1.0),
                 Component("bd", during=["b", "d"], static_ua=1.0)]
        r = self.run_arch(stages, comps)
        self.assertEqual(r.timeline.read, {"a": (0, 4), "b": (4, 10), "c": (0, 3), "d": (8, 13)})
        self.assertEqual(r.latency_ns, 3 * 13.0)
        self.assertEqual(r.timeline.read_on_time(["a", "c"]), 4.0)  # union of a and c
        self.assertEqual(r.timeline.read_on_time(["b", "d"]), 9.0)  # 4..13

    def test_pipelined_reads_and_overlapping_bins(self):
        stages = [Stage("drive", 2.0), Stage("sense", 3.0), Stage("fire", 1.0, level="timestep")]
        comps = [Component("amp", during=["sense"], static_ua=1.0),
                 Component("neuron", count="outputs", during=["timestep"], static_ua=1.0)]
        r = self.run_arch(stages, comps, x=torch.ones(1, 64, 1, 1, 2),
                          crossbar=Crossbar(LINEAR_1BIT, active_rows=16),       # 4 reads per bin
                          read_interval_ns=3.0, timestep_interval_ns=10.0)
        self.assertEqual(r.timeline.timestep["fire"], (14.0, 15.0))  # 3*3 + 5, then fire
        self.assertEqual(r.latency_ns, 25.0)                         # next bin starts at 10
        self.assertEqual(r.timeline.read_on_time(["sense"]), 3.0)
        self.assertEqual(r.timeline.timestep_on_time(["timestep"]), 15.0)
        # 1 uA from 1.1 V: 4 reads x 3 ns and 15 ns per bin, over 2 bins.
        self.assertAlmostEqual(r.components["amp"].energy_nj, 1.1e-6 * 2 * 4 * 3.0, places=15)
        self.assertAlmostEqual(r.components["neuron"].energy_nj, 1.1e-6 * 2 * 15.0, places=15)

    def test_component_cannot_serve_overlapping_reads(self):
        with self.assertRaisesRegex(ValueError, "two reads at once"):
            self.run_arch([Stage("drive", 2.0), Stage("sense", 3.0)],
                          [Component("both", during=["drive", "sense"], static_ua=1.0)],
                          crossbar=Crossbar(LINEAR_1BIT, active_rows=32), read_interval_ns=3.0)

    def test_power_window_between_stage_edges(self):
        # 4 reads per bin of drive 2 + sense 3 ns, then fire 1 ns: sense starts
        # at 2 in the first read; fire ends at 4 * 5 + 1 = 21.
        window = (("sense", "start", 1.0), ("fire", "end", -1.0))
        r = self.run_arch([Stage("drive", 2.0), Stage("sense", 3.0), Stage("fire", 1.0, level="timestep")],
                          [Component("ota", window=window, static_ua=1.0)],
                          crossbar=Crossbar(LINEAR_1BIT, active_rows=16))
        self.assertEqual(r.timeline.anchor("sense", "start"), 2.0)
        self.assertEqual(r.timeline.anchor("sense", "end"), 20.0)
        self.assertEqual(r.timeline.window_on_time(window), 21.0 - 1.0 - 3.0)
        self.assertAlmostEqual(r.components["ota"].energy_nj, 1.1e-6 * 3 * 17.0, places=15)
        with self.assertRaisesRegex(ValueError, "ends before it starts"):
            self.run_arch([Stage("read", 5.0)],
                          [Component("x", window=(("read", "end", 0.0), ("read", "start", 0.0)),
                                     static_ua=1.0)])

    def test_timestep_stage_before_reads(self):
        r = self.run_arch([Stage("read", 5.0), Stage("precharge", 1.0, level="timestep", after=[]),
                           Stage("fire", 2.0, level="timestep", after=["reads"])], [])
        self.assertEqual(r.timeline.timestep["precharge"], (0.0, 1.0))
        self.assertEqual(r.latency_ns, 3 * 7.0)


class CompositionTests(unittest.TestCase):
    def test_parts_compose_in_order(self):
        arch = rram_ota_design()
        self.assertEqual([(s.name, s.level) for s in arch.stages],
                         [("read", "read"), ("fire", "timestep")])
        self.assertEqual([c.name for c in arch.components],
                         ["cells", "sl_ota", "slice_mirrors", "lif"])
        self.assertEqual(arch.components[1].during, ["read"])   # a single stage name is a list

    def test_compose_needs_one_crossbar(self):
        with self.assertRaisesRegex(ValueError, "exactly one crossbar"):
            compose("x", Precision(4), [Stage("read", 1.0)])
        with self.assertRaisesRegex(ValueError, "exactly one crossbar"):
            compose("x", Precision(4), [crossbars.conv_xbar(r_on=1e3, r_off=1e6),
                                         crossbars.c3cim_xbar(r_on=1e3, r_off=1e6)])
        with self.assertRaisesRegex(ValueError, "Blocks, Stages or Components"):
            compose("x", Precision(4), [crossbars.conv_xbar(r_on=1e3, r_off=1e6), "lif"])


class ValidationTests(unittest.TestCase):
    def test_quantize_weights(self):
        codes, scale = quantize_weights(torch.tensor([-1.0, 0.6, 1.0]), 4, "max")
        self.assertEqual(codes.tolist(), [-7, 4, 7])
        self.assertAlmostEqual(scale, 1 / 7)
        weights = torch.tensor([0.3])
        self.assertIs(quantize_weights(weights, None)[0], weights)

    def test_quantization_scalings(self):
        g = torch.Generator().manual_seed(3)
        weights = torch.randn(4000, generator=g)
        weights[:5] *= 15                                            # a few outliers
        std = float(weights.std())

        def error(bits, scaling):
            codes, scale = quantize_weights(weights, bits, scaling)
            return float((codes * scale - weights).pow(2).sum())
        for bits in (2, 3, 4, 6, 8):
            # "mse" searches clips up to the largest |weight|, which is "max":
            # never worse, and much better with outliers at few bits.
            self.assertLessEqual(error(bits, "mse"), error(bits, "max") * (1 + 1e-12))
        self.assertLess(error(3, "mse"), 0.5 * error(3, "max"))
        codes, scale = quantize_weights(weights, 4, "std3")
        self.assertAlmostEqual(scale * 7, 3 * std, places=5)         # clip at 3 sigma
        self.assertEqual(int(codes.abs().max()), 7)                  # outliers saturate
        _, scale = quantize_weights(torch.tensor([-1.0, 0.5]), 4, "std10")
        self.assertAlmostEqual(scale * 7, 1.0)                       # never beyond the max
        self.assertEqual(quantize_weights(torch.zeros(3), 4, "mse")[0].tolist(), [0, 0, 0])
        for bad in ("std", "std0", "min", "std-1"):
            with self.assertRaisesRegex(ValueError, "weight_scaling"):
                quantize_weights(weights, 4, bad)
            with self.assertRaisesRegex(ValueError, "weight_scaling"):
                test_arch("sequential", Precision(4, weight_scaling=bad), Crossbar(LINEAR_1BIT))

    def test_weight_code_checks(self):
        arch = test_arch("sequential", Precision(4), Crossbar(LINEAR_1BIT))
        with self.assertRaisesRegex(ValueError, "within"):
            evaluate_layer(arch, torch.ones(1, 1, 1, 1, 1), torch.tensor([[[[8]]]]))
        with self.assertRaisesRegex(ValueError, "integer weight codes"):
            evaluate_layer(arch, torch.ones(1, 1, 1, 1, 1), torch.ones(1, 1, 1, 1))

    def test_invalid_architectures(self):
        read = [Stage("read", 1.0)]
        bad = [dict(stages=[Stage("t", 1.0, level="timestep")]),
               dict(components=[Component("x", during=["missing"])]),
               dict(components=[Component("x", static_ua=1.0)]),
               dict(components=[Component("x", model="crossbar_read")]),
               dict(components=[Component("x", count={"rule": "column_groups"})]),
               dict(components=[Component("x", count={"rule": "tiles", "gated": True})]),
               dict(components=[Component("x", on="spiking_rows", during=["timestep"], static_ua=1.0)]),
               dict(components=[Component("x", count="spiking_rows")]),
               dict(precision=Precision(None, "twos_complement")),
               dict(conv_mapping="diagonal"),
               dict(components=[Component("x", during=["read"],
                                          window=(("read", "start", 0), ("read", "end", 0)))]),
               dict(components=[Component("x", window=(("missing", "start", 0), ("read", "end", 0)))]),
               dict(components=[Component("x", window=(("read", "middle", 0), ("read", "end", 0)))])]
        for kwargs in bad:
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}), self.assertRaises(ValueError):
                Architecture("x", Crossbar(LINEAR_1BIT), kwargs.pop("precision", Precision()),
                             kwargs.pop("stages", read), kwargs.pop("components", []), **kwargs)


if __name__ == "__main__":
    unittest.main()
