"""Hardware engine checks.

ReferenceModel re-derives every cost by literally enumerating samples, time
bins, windows, row tiles, phases and cells; the vectorized engine must match
it. Plus hand calculations, earlier worked examples, timeline placement and
validation.
"""
import dataclasses
import math
import unittest

import numpy as np
import torch

import crossbars
from hardware import (Architecture, Component, Crossbar, Mapping, Memory, compose, evaluate_layer,
                      quantize_weights)
from hardware.architecture import LAYER_RULES, TILE_RULES, rule_of
from hardware.mapping import layer_geometry

LINEAR_1BIT = Memory(cell_bits=1, r_on=1e3, r_off=1e6)
NONUNIFORM_2BIT = Memory(cell_bits=2, levels_s=(1e-6, 3e-4, 5e-4, 1e-3))
WHOLE_BIN = dict(when="bin")


def rram_ota_design(conv="sequential"):
    """The 1-bit RRAM current-mode design of run.py, with slice mirrors."""
    return compose("rram_ota", Mapping(4, "max", "twos_complement", conv), [
        crossbars.conv_xbar(cell_bits=1, r_on=20e3, r_off=200e3, rows=64, cols=64, v_read=0.2,
                            time_ns=5.0),
        Component("sl_ota", count="physical_columns",
                  powered={"rule": "used_columns", "gated": True}, when="cells", static_ua=10.0),
        Component("slice_mirrors", model="slice_mirror", count="used_columns", when="cells"),
        Component("lif", count="outputs", time_ns=2.0, static_ua=10.0),
    ])


def conventional_example():
    """Worked example: current-mode crossbar with analog cells and G(0)
    reference columns, a DA per column and LIFs on for the whole time bin."""
    return compose("conventional", Mapping(None, "max", "analog", "sequential"), [
        crossbars.conv_xbar(r_on=2e3, r_off=200e3, v_read=0.1, active_rows=8, time_ns=4.5,
                            cell_supply_v=1.1, reference_columns=True, tile_area_um2=136.67),
        Component("da", count="physical_columns", powered="used_columns", when="cells",
                  static_ua=6.1, area_um2=30.22),
        Component("reference_subtractor", count="output_bank", powered="outputs",
                  time_ns=0.0, per="activation"),
        Component("lif", count="output_bank", powered="outputs", time_ns=2.0,
                  **WHOLE_BIN, static_ua=6.0, area_um2=86.79)])


def c3cim_example():
    """Worked example: C3CIM crossbar with a VI converter per column."""
    return compose("c3cim", Mapping(None, "max", "analog", "sequential"), [
        crossbars.c3cim_xbar(r_on=2e3, r_off=20e3, column_area_um2=4.27, driver_area_um2=86.36),
        Component("vi", count="physical_columns", powered="used_columns", time_ns=10.0,
                  supply_v=1.0, static_ua=24.3, area_um2=29.79),
        Component("lif", count="output_bank", powered="outputs", time_ns=2.0,
                  **WHOLE_BIN, static_ua=6.0, area_um2=86.79)])


def reference_cost(arch, spikes, weights, stride, padding, output_spikes=0.0):
    """Per-inference energy per component and latency, by brute force.

    Steps must be serial (default placement) so powered times are sums of
    durations; timeline placement itself is tested separately. Serial steps
    last as long as the fullest column group, with the weight columns of each
    tile dealt to its groups in turn.
    """
    xb, pr = arch.crossbar, arch.mapping
    batch, channels, height, width, bins = spikes.shape
    out, _, kh, kw = weights.shape
    k_rows = channels * kh * kw
    rows, active_rows = xb.rows, xb.active_rows or xb.rows
    sequential = pr.conv == "sequential"

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
        on = rule_of(c.powered, activity=True)
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

    def duration(stage):
        if stage.serial_size is None:
            return stage.time_ns
        groups = math.ceil(xb.cols / stage.serial_size)
        fullest = 0
        for slot in slots:           # physical tiles: the slot's columns, cols at a time
            columns = len(slot) * used
            for first in range(0, columns, xb.cols):
                load = [0] * groups
                for column in range(min(xb.cols, columns - first)):
                    load[column % groups] += 1
                fullest = max(fullest, max(load))
        return fullest * stage.time_ns

    # Serial step positions: within one activation, and within a time bin,
    # where the per-activation steps repeat reads_per_bin times and the
    # time-bin steps follow.
    in_activation, t = {}, 0.0
    for stage in (s for s in arch.stages if s.level == "activation"):
        in_activation[stage.name] = (t, t + duration(stage))
        t += duration(stage)
    span = t
    edges = {name: (a, (reads_per_bin - 1) * span + b) for name, (a, b) in in_activation.items()}
    t = reads_per_bin * span
    for stage in (s for s in arch.stages if s.level == "time_bin"):
        edges[stage.name] = (t, t + duration(stage))
        t += duration(stage)
    bin_span = t
    edges["bin"] = (0.0, bin_span)
    energy = {}
    for c in arch.components:
        p = arch.power[c.name]
        places = in_activation if p is not None and p.level == "activation" else edges
        on = 0.0 if p is None else \
            places[p.end[0]][p.end[1] == "end"] + p.end[2] - places[p.start[0]][p.start[1] == "end"] - p.start[2]
        powered = read_powered[c.name] if places is in_activation else bin_powered[c.name]
        e = c.supply_v * c.static_ua * 1e-6 * on * powered
        events = {"activation": read_powered[c.name], "time_bin": bin_powered[c.name],
                  "output_spike": output_spikes}[c.events]
        e += c.event_pj * 1e-3 * events
        if c.model == "crossbar_read":
            e += c.supply_v * sum(slice_current) * on
        elif c.model == "reference_read":
            e += c.supply_v * reference_current * on
        elif c.model == "slice_mirror":  # binary: MSB x1, each lower slice / 2^cell_bits
            cell = xb.memory.cell_bits
            gains = c.slice_gains or [2.0 ** (cell * (s - n_slices + 1)) for s in range(n_slices)]
            e += c.supply_v * sum(gains[s] * slice_current[s] for s in range(n_slices)) * on
        energy[c.name] = e / batch
    return energy, bins * bin_span


def test_arch(conv, mapping, crossbar, custom_gains=None):
    """Per activation: wl_driver 2 ns (word-line drivers), cells 3 ns (the
    read), adc 0.5 ns per column of its group; per time bin: neuron 1 ns.
    Every rule, gated and not."""
    gated = lambda rule, **kw: dict(rule=rule, gated=True, **kw)
    return Architecture(
        "test", crossbar, dataclasses.replace(mapping, conv=conv),
        [Component("wl_driver", count="physical_rows", powered="spiking_rows", time_ns=2.0,
                   static_ua=2.0, event_pj=0.1),
         Component("cells", model="crossbar_read", time_ns=3.0, supply_v=1.0),
         Component("mirrors", model="slice_mirror", when="cells", supply_v=1.2),
         Component("mirrors_custom", model="slice_mirror", when="wl_driver",
                   supply_v=0.9, slice_gains=custom_gains),
         Component("ota", count="physical_columns", powered=gated("used_columns"), when="cells",
                   static_ua=10.0, event_pj=0.5),
         Component("bias", count="physical_columns", when=("wl_driver", "cells"), static_ua=1.0),
         Component("driver", count={"rule": "column_groups", "size": 3},
                   powered=gated("column_groups", size=3), when="wl_driver", static_ua=4.0),
         Component("adc", count={"rule": "column_groups", "size": 3}, time_ns=0.5, serial=True,
                   powered=gated("column_groups", size=3), static_ua=3.0, event_pj=0.05),
         Component("tile_ctrl", count="tiles", powered=gated("tiles"), **WHOLE_BIN, static_ua=3.0),
         Component("neuron", count="outputs", powered=gated("outputs"), time_ns=1.0,
                   **WHOLE_BIN, static_ua=5.0, event_pj=0.2, events="output_spike"),
         Component("bank", count="output_bank", **WHOLE_BIN, static_ua=0.5,
                   event_pj=0.3, events="time_bin"),
         Component("sense_amp", count="output_bank", powered=gated("output_bank"), when="cells",
                   static_ua=0.7),
         Component("controller", count="one", powered=gated("one"), when="cells", event_pj=1.0),
         Component("clock", count={"rule": "fixed", "value": 5}, **WHOLE_BIN, static_ua=0.2),
         Component("windowed", count="physical_columns", powered=gated("used_columns"),
                   when=("cells.start+0.5", "neuron.end-0.25"), static_ua=6.0)])


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
        for conv in ("sequential", "parallel"):
            for memory in (NONUNIFORM_2BIT, LINEAR_1BIT):
                # 16-row tiles with 6 active rows: partial tiles, 3 phases.
                crossbar = Crossbar(memory, rows=16, cols=8, active_rows=6)
                for encoding in encodings:
                    analog = encoding == "analog"
                    magnitude_bits = 3 if encoding == "differential" else 4
                    slices = 1 if analog else math.ceil(magnitude_bits / memory.cell_bits)
                    gains = tuple(0.3 + 0.5 * s for s in range(slices))   # not binary
                    arch = test_arch(conv, Mapping(None if analog else 4, "max", encoding),
                                     crossbar, gains)
                    w_conv = conv_weights.float() if analog else conv_weights
                    w_dense = dense_weights.float() if analog else dense_weights
                    with self.subTest(conv=conv, cell_bits=memory.cell_bits, encoding=encoding):
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
                crossbar = Crossbar(memory, rows=30, cols=20, active_rows=7, reference_columns=analog)
                arch = test_arch("parallel", Mapping(None if analog else 3, "max", encoding), crossbar)
                if analog:
                    arch = Architecture("test", crossbar, arch.mapping, arch.components + [
                        Component("reference", model="reference_read", when="cells")])
                w = weights.float() if analog else weights
                with self.subTest(cell_bits=memory.cell_bits, encoding=encoding):
                    geometry = layer_geometry(arch, spikes.shape[1:4], w.shape, padding=1)
                    self.assertGreater(geometry.copies_per_tile, 1)
                    self.assertNotEqual(geometry.windows % geometry.copies_per_tile, 0)
                    self.check(arch, spikes, w, stride=1, padding=1)
                    self.check(arch, spikes, w, stride=2, padding=0)

    def test_silent_input_powers_only_ungated_parts(self):
        arch = test_arch("parallel", Mapping(4, "max"), Crossbar(LINEAR_1BIT, rows=8, cols=8))
        silent = torch.zeros(1, 2, 4, 4, 2)
        result = evaluate_layer(arch, silent, torch.ones(3, 2, 3, 3, dtype=torch.int64),
                                output_spikes=0)
        for name in ("cells", "mirrors", "mirrors_custom", "ota", "driver", "adc", "wl_driver",
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

        def cells(when=None, **crossbar):
            arch = compose("x", Mapping(4, "max"), [
                crossbars.conv_xbar(r_on=1e3, r_off=1e6, time_ns=5.0, when=when, **crossbar),
                Component("lif", count="outputs", time_ns=2.0, static_ua=1.0)])
            return evaluate_layer(arch, spikes, weights).components["cells"].energy_nj

        # One activation per time bin: the bin's current flows for 5 + 2 ns instead of 5.
        self.assertAlmostEqual(cells(when=("cells", "lif")), cells() * 7.0 / 5.0, places=12)
        with self.assertRaisesRegex(ValueError, "one activation per time bin"):
            cells(when=("cells", "lif"), rows=8, active_rows=4)   # 3 row tiles x 2 phases

    def test_shared_adc_converting_its_columns(self):
        """64x64 tile, 8 rows at a time, columns multiplexed to ADCs (6 ns per
        conversion, 10 uA), LIF 2 ns per time bin."""
        def design(group, bits=6):
            return compose("adc", Mapping(bits, "max", conv="parallel"), [
                crossbars.conv_xbar(r_on=20e3, r_off=200e3, active_rows=8, time_ns=5.0),
                Component("adc", count={"rule": "column_groups", "size": group},
                          time_ns=6.0, serial=True, supply_v=1.1, static_ua=10.0),
                Component("lif", count="outputs", time_ns=2.0)])
        g = torch.Generator().manual_seed(0)
        weights = torch.randint(-31, 32, (10, 64, 1, 1), generator=g)      # 60 of 64 columns
        spikes = (torch.rand(1, 64, 1, 1, 1, generator=g) < 0.2).float()  # one time bin
        r = evaluate_layer(design(8), spikes, weights)
        self.assertEqual(r.timeline.stages, (("cells", 5.0, "activation"), ("adc", 48.0, "activation"),
                                             ("lif", 2.0, "time_bin")))
        self.assertEqual(r.geometry.activations_per_bin, 8)
        self.assertEqual(r.latency_ns, 8 * (5.0 + 48.0) + 2.0)                # 426 ns
        self.assertEqual(r.components["adc"].installed, 8)
        self.assertAlmostEqual(r.components["adc"].energy_nj, 1.1 * 10e-6 * 8 * 48.0 * 8, places=12)
        self.assertIn("adc 48 ns (per activation, x8)", r.timeline.describe())
        # Half the group: twice the ADCs; 60 columns interleaved over 16 ADCs
        # take 4 conversions.
        r4 = evaluate_layer(design(4), spikes, weights)
        self.assertEqual((r4.components["adc"].installed, r4.latency_ns), (16, 8 * (5.0 + 24.0) + 2.0))
        # 32 of 64 columns (8 outputs x 4 slices), interleaved: every ADC
        # converts 4 columns, not the 8 of its group.
        codes = torch.randint(-7, 8, (8, 64, 1, 1), generator=g)
        r32 = evaluate_layer(design(8, bits=4), spikes, codes)
        self.assertEqual(r32.latency_ns, 8 * (5.0 + 4 * 6.0) + 2.0)
        self.assertAlmostEqual(r32.components["adc"].energy_nj, 1.1 * 10e-6 * 8 * 24.0 * 8, places=12)

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


def stage(name, ns, **kw):
    """A component that only defines a step (per activation unless its count
    makes it per time bin)."""
    return Component(name, time_ns=ns, **kw)


class TimelineTests(unittest.TestCase):
    def run_arch(self, components, x=None, **kw):
        crossbar = kw.pop("crossbar", Crossbar(LINEAR_1BIT))
        arch = Architecture("t", crossbar, Mapping(None, "max", "analog"), components, **kw)
        return evaluate_layer(arch, torch.ones(1, 64, 1, 1, 3) if x is None else x,
                              torch.ones(1, 64, 1, 1))

    def test_serial_parallel_and_overlap(self):
        r = self.run_arch([stage("a", 4.0),
                           stage("c", 3.0, at="a.start"),                   # c parallel with a
                           stage("b", 6.0),                                 # after a and c
                           stage("d", 5.0, at="b.end-2"),                   # d overlaps b by 2 ns
                           stage("e", 1.0),                                 # after all of them
                           Component("bd", when=("b", "d"), static_ua=1.0)])
        self.assertEqual(r.timeline.activation,
                         {"a": (0, 4), "c": (0, 3), "b": (4, 10), "d": (8, 13), "e": (13, 14)})
        self.assertEqual(r.latency_ns, 3 * 14.0)
        self.assertAlmostEqual(r.components["bd"].energy_nj, 1.1e-6 * 3 * 9.0, places=15)  # 4..13

    def test_pipelined_activations_and_overlapping_bins(self):
        comps = [stage("drive", 2.0), stage("sense", 3.0),
                 Component("neuron", count="outputs", time_ns=1.0, **WHOLE_BIN, static_ua=1.0),
                 Component("amp", when="sense", static_ua=1.0)]
        r = self.run_arch(comps, x=torch.ones(1, 64, 1, 1, 2),
                          crossbar=Crossbar(LINEAR_1BIT, active_rows=16),   # 4 activations per bin
                          activation_interval_ns=3.0, time_bin_interval_ns=10.0)
        self.assertEqual(r.timeline.time_bin["neuron"], (14.0, 15.0))  # 3*3 + 5, then the neuron
        self.assertEqual(r.latency_ns, 25.0)                           # next bin starts at 10
        # 1 uA from 1.1 V: 4 activations x 3 ns and 15 ns per bin, over 2 bins.
        self.assertAlmostEqual(r.components["amp"].energy_nj, 1.1e-6 * 2 * 4 * 3.0, places=15)
        self.assertAlmostEqual(r.components["neuron"].energy_nj, 1.1e-6 * 2 * 15.0, places=15)

    def test_component_cannot_serve_overlapping_activations(self):
        with self.assertRaisesRegex(ValueError, "two activations at once"):
            self.run_arch([stage("drive", 2.0), stage("sense", 3.0),
                           Component("both", when=("drive", "sense"), static_ua=1.0)],
                          crossbar=Crossbar(LINEAR_1BIT, active_rows=32), activation_interval_ns=3.0)

    def test_power_interval_between_step_edges(self):
        # 4 activations per bin of drive 2 + sense 3 ns, then fire 1 ns: sense
        # starts at 2 in the first activation; fire ends at 4 * 5 + 1 = 21.
        r = self.run_arch([stage("drive", 2.0), stage("sense", 3.0),
                           Component("ota", when=("sense.start+1", "fire.end-1"), static_ua=1.0),
                           stage("fire", 1.0, count="outputs")],        # defined after its use
                          crossbar=Crossbar(LINEAR_1BIT, active_rows=16))
        self.assertEqual(r.timeline.anchor("sense", "start"), 2.0)
        self.assertEqual(r.timeline.anchor("sense", "end"), 20.0)
        self.assertAlmostEqual(r.components["ota"].energy_nj, 1.1e-6 * 3 * (21.0 - 1.0 - 3.0), places=15)
        with self.assertRaisesRegex(ValueError, "ends before it starts"):
            self.run_arch([stage("read", 5.0),
                           Component("x", when=("read.end", "read.start"), static_ua=1.0)])

    def test_time_bin_steps_at_anchors(self):
        r = self.run_arch([stage("read", 5.0),
                           stage("precharge", 1.0, count="outputs", at="bin.start"),
                           stage("fire", 2.0, count="outputs")])       # after the activations
        self.assertEqual(r.timeline.time_bin["precharge"], (0.0, 1.0))
        self.assertEqual(r.timeline.time_bin["fire"], (5.0, 7.0))
        self.assertEqual(r.latency_ns, 3 * 7.0)
        # A time-bin step at a per-activation step: its last activation's end
        # (2 activations of 5 ns), and a step placed at one defined later.
        r = self.run_arch([stage("read", 5.0),
                           stage("early", 1.0, count="outputs", at="fire.start-1"),
                           stage("fire", 2.0, count="outputs", at="read.end-1")],
                          crossbar=Crossbar(LINEAR_1BIT, active_rows=32))
        self.assertEqual((r.timeline.time_bin["fire"], r.timeline.time_bin["early"]),
                         ((9.0, 11.0), (8.0, 9.0)))
        with self.assertRaisesRegex(ValueError, "cycle"):
            self.run_arch([stage("read", 5.0), stage("x", 1.0, at="y.end"),
                           stage("y", 1.0, at="x.end")])
        with self.assertRaisesRegex(ValueError, "before its level starts"):
            self.run_arch([stage("read", 5.0), stage("x", 1.0, at="read.start-1")])


class CompositionTests(unittest.TestCase):
    def test_parts_compose_in_order(self):
        arch = rram_ota_design()
        self.assertEqual([(s.name, s.level) for s in arch.stages],
                         [("cells", "activation"), ("lif", "time_bin")])
        self.assertEqual([c.name for c in arch.components],
                         ["cells", "sl_ota", "slice_mirrors", "lif"])
        self.assertEqual(arch.power["sl_ota"].level, "activation")
        self.assertIsNone(Architecture("x", Crossbar(LINEAR_1BIT), Mapping(), [
            stage("read", 1.0), Component("area_only", area_um2=5.0)]).power["area_only"])

    def test_step_frequency_inferred_or_set(self):
        arch = Architecture("x", Crossbar(LINEAR_1BIT), Mapping(), [
            stage("read", 1.0),                                              # tiles: per activation
            Component("adc", count={"rule": "column_groups", "size": 4}, time_ns=2.0, serial=True),
            Component("acc", count="outputs", time_ns=1.0, per="activation"),
            Component("lif", count="outputs", time_ns=1.0),                  # neurons: per time bin
            Component("ctrl", count="one", time_ns=1.0)])
        self.assertEqual({s.name: (s.time_ns, s.serial_size, s.level) for s in arch.stages},
                         {"read": (1.0, None, "activation"), "adc": (2.0, 4, "activation"),
                          "acc": (1.0, None, "activation"), "lif": (1.0, None, "time_bin"),
                          "ctrl": (1.0, None, "time_bin")})

    def test_anchors_and_windows(self):
        from hardware.architecture import parse_anchor
        self.assertEqual(parse_anchor("fire", "start"), ("fire", "start", 0.0))
        self.assertEqual(parse_anchor("fire", "end"), ("fire", "end", 0.0))
        self.assertEqual(parse_anchor("read.start+1", "end"), ("read", "start", 1.0))
        self.assertEqual(parse_anchor("fire.end - 0.5", "start"), ("fire", "end", -0.5))
        self.assertEqual(parse_anchor("bin.start", "start"), ("bin", "start", 0.0))
        for bad in ("read.middle", "read+", "+1", "read.start*2", 3):
            with self.assertRaisesRegex(ValueError, "bad anchor"):
                parse_anchor(bad, "start")
        arch = Architecture("x", Crossbar(LINEAR_1BIT), Mapping(), [
            stage("read", 1.0), stage("lif", 1.0, count="outputs"),
            Component("own", time_ns=1.0), Component("during", when="read"),
            Component("span", when=("read", "lif")), Component("whole", when="bin")])
        self.assertEqual({name: (p.level, p.start, p.end) for name, p in arch.power.items() if p},
                         {"read": ("activation", ("read", "start", 0.0), ("read", "end", 0.0)),
                          "lif": ("time_bin", ("lif", "start", 0.0), ("lif", "end", 0.0)),
                          "own": ("activation", ("own", "start", 0.0), ("own", "end", 0.0)),
                          "during": ("activation", ("read", "start", 0.0), ("read", "end", 0.0)),
                          "span": ("time_bin", ("read", "start", 0.0), ("lif", "end", 0.0)),
                          "whole": ("time_bin", ("bin", "start", 0.0), ("bin", "end", 0.0))})

    def test_compose_needs_one_crossbar(self):
        with self.assertRaisesRegex(ValueError, "exactly one crossbar"):
            compose("x", Mapping(), [stage("read", 1.0)])
        with self.assertRaisesRegex(ValueError, "exactly one crossbar"):
            compose("x", Mapping(), [crossbars.conv_xbar(r_on=1e3, r_off=1e6),
                                     crossbars.c3cim_xbar(r_on=1e3, r_off=1e6)])
        with self.assertRaisesRegex(ValueError, "a crossbar and Components"):
            compose("x", Mapping(), [crossbars.conv_xbar(r_on=1e3, r_off=1e6), "lif"])


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
                test_arch("sequential", Mapping(4, bad), Crossbar(LINEAR_1BIT))

    def test_weight_code_checks(self):
        arch = test_arch("sequential", Mapping(4, "max"), Crossbar(LINEAR_1BIT))
        with self.assertRaisesRegex(ValueError, "within"):
            evaluate_layer(arch, torch.ones(1, 1, 1, 1, 1), torch.tensor([[[[8]]]]))
        with self.assertRaisesRegex(ValueError, "integer weight codes"):
            evaluate_layer(arch, torch.ones(1, 1, 1, 1, 1), torch.ones(1, 1, 1, 1))

    def test_invalid_architectures(self):
        read = stage("read", 1.0)
        fire = stage("fire", 1.0, count="outputs")
        bad = {
            "no per-activation step": [stage("t", 1.0, count="outputs")],
            "unknown step in when": [read, Component("x", when=("missing", "read"))],
            "static current, no power": [read, Component("x", static_ua=1.0)],
            "data model, no power": [read, Component("x", model="crossbar_read")],
            "column_groups without size": [read, Component("x", count={"rule": "column_groups"})],
            "gated count": [read, Component("x", count={"rule": "tiles", "gated": True})],
            "spiking rows over a time bin": [read, Component("x", powered="spiking_rows",
                                                             **WHOLE_BIN, static_ua=1.0)],
            "spiking rows as count": [read, Component("x", count="spiking_rows")],
            "at without time_ns": [read, Component("x", at="read")],
            "serial without time_ns": [read, Component("x", count={"rule": "column_groups",
                                                                   "size": 2}, serial=True)],
            "serial without column groups": [read, stage("x", 1.0, serial=True)],
            "when is a point": [read, Component("x", when="read.start")],
            "when of three": [read, Component("x", when=("read", "read", "read"))],
            "bad anchor": [read, Component("x", when=("read.middle", "read"))],
            "at unknown step": [read, stage("x", 1.0, at="missing")],
            "activation step at a time-bin step": [read, fire, stage("x", 1.0, at="fire.end")],
            "activation step at the bin": [read, stage("x", 1.0, at="bin.start")],
            "step at the bin's end": [read, stage("x", 1.0, count="outputs", at="bin.end")],
            "duplicate name": [read, stage("read", 2.0, count="outputs")],
            "bad per": [read, stage("x", 1.0, per="cycle")],
            "reserved step name": [read, stage("bin", 1.0)],
            "step name not an identifier": [read, stage("my adc", 1.0)],
        }
        for label, components in bad.items():
            with self.subTest(label), self.assertRaises(ValueError):
                Architecture("x", Crossbar(LINEAR_1BIT), Mapping(), components)
        for label, mapping in {"unquantized bit-sliced": Mapping(None, "max", "twos_complement"),
                               "unknown conv": Mapping(conv="diagonal")}.items():
            with self.subTest(label), self.assertRaises(ValueError):
                Architecture("x", Crossbar(LINEAR_1BIT), mapping, [read])
        with self.assertRaisesRegex(ValueError, "the steps are read"):   # the error lists them
            Architecture("x", Crossbar(LINEAR_1BIT), Mapping(),
                         [read, Component("r2", when=("fier.start", "read"))])


if __name__ == "__main__":
    unittest.main()
