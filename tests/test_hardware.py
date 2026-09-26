"""Hardware engine checks.

ReferenceModel re-derives every cost by literally enumerating samples, time
bins, windows, row tiles, phases and cells; the vectorized engine must match
it. Plus hand calculations, the README examples, timeline placement and
validation.
"""
import math
import unittest

import numpy as np
import torch

from architectures import crossbars, designs, memories, neurons, periphery
from hardware import (Architecture, Component, Crossbar, Memory, Precision, Stage, compose,
                      evaluate_layer, quantize_weights)

LINEAR_1BIT = Memory("linear_1bit", cell_bits=1, r_on=1e3, r_off=1e6)


def rram_ota_design(conv_mapping="sequential"):
    """The 1-bit RRAM current-mode design of run.py, composed from blocks."""
    return compose("rram_ota", Precision(4, "twos_complement"), [
        crossbars.conv_xbar(memories.RRAM_1BIT, rows=64, cols=64, v_read=0.2, read_ns=5.0),
        periphery.source_line_ota(static_ua=10.0),
        periphery.slice_mirrors(),
        neurons.lif_neuron(static_ua=10.0, fire_ns=2.0),
    ], conv_mapping=conv_mapping)
NONUNIFORM_2BIT = Memory("nonuniform_2bit", cell_bits=2, levels_s=(1e-6, 3e-4, 5e-4, 1e-3))
from hardware.architecture import TILE_RULES, rule_of


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
    slice_current = [0.0] * n_slices
    column_tiles = math.ceil(used / xb.cols)
    tiles = [range(r, min(r + rows, k_rows)) for r in range(0, k_rows, rows)]
    tile_phases = [[t[p:p + active_rows] for p in range(0, len(t), active_rows)] for t in tiles]
    max_phases = max(len(p) for p in tile_phases)

    def per_unit(rule):
        return {"tiles": column_tiles, "physical_rows": column_tiles * rows,
                "physical_columns": column_tiles * xb.cols, "used_columns": used,
                "column_groups": column_tiles * math.ceil(xb.cols / rule.get("size", 1)),
                "outputs": out, "output_bank": column_tiles * xb.cols, "one": 1,
                "fixed": rule.get("value", 0)}[rule["rule"]]

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
            for w, p_in in patches.items():
                for k, s in enumerate(p_in):
                    for cells, index in columns:
                        slice_current[index] += s * cells[k] * xb.v_read
                    reference_current += s * out * g_mid * xb.v_read
            groups = [[w] for w in positions] if sequential else [positions]
            for group in groups:
                for p in range(max_phases):
                    spike = {(w, r): any(patches[w][k] for k in tile_phases[r][p])
                             for w in group for r in range(len(tiles)) if p < len(tile_phases[r])}
                    for c in arch.components:
                        rule = rules(c)
                        if rule["rule"] == "spiking_rows":
                            n = sum(patches[w][k] for (w, r) in spike for k in tile_phases[r][p])
                            read_powered[c.name] += n * column_tiles
                            continue
                        if rule["rule"] in TILE_RULES:
                            units = [key for key, s in spike.items() if s or not rule["gated"]]
                        elif rule["rule"] in ("outputs", "output_bank"):
                            units = [w for w in group if not rule["gated"]
                                     or any(s for (u, _), s in spike.items() if u == w)]
                        else:
                            units = [0] if not rule["gated"] or any(spike.values()) else []
                        read_powered[c.name] += len(units) * per_unit(rule)
            # Whole-bin units: a physical row tile / window / bank / layer that
            # receives any spike during this time bin.
            any_tile = {(w, r): any(patches[w][k] for k in tiles[r])
                        for w in positions for r in range(len(tiles))}
            for c in arch.components:
                rule = rules(c)
                if rule["rule"] == "spiking_rows":   # counted per read only
                    continue
                if rule["rule"] in TILE_RULES:
                    if sequential:
                        keys = [r for r in range(len(tiles))
                                if not rule["gated"] or any(any_tile[(w, r)] for w in positions)]
                    else:
                        keys = [k for k, s in any_tile.items() if s or not rule["gated"]]
                elif rule["rule"] == "outputs" or (rule["rule"] == "output_bank" and not sequential):
                    keys = [w for w in positions if not rule["gated"] or any(patches[w])]
                else:
                    keys = [0] if not rule["gated"] or any(any(v) for v in patches.values()) else []
                bin_powered[c.name] += len(keys) * per_unit(rule)

    durations = {s.name: s.duration_ns for s in arch.stages}
    read_span = sum(s.duration_ns for s in arch.stages if s.level == "read")
    bin_span = reads_per_bin * read_span + sum(s.duration_ns for s in arch.stages if s.level == "timestep")
    energy = {}
    for c in arch.components:
        e = 0.0
        if c.during and set(c.during) <= arch.read_stages:
            e += c.supply_v * c.static_ua * 1e-6 * sum(durations[s] for s in c.during) * read_powered[c.name]
            events = read_powered[c.name] if c.events == "read" else bin_powered[c.name]
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
         Component("clock", count={"rule": "fixed", "value": 5}, during=["timestep"], static_ua=0.2)],
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
                    with self.subTest(mapping=mapping, memory=memory.name, encoding=encoding):
                        self.check(arch, conv_spikes, w_conv, stride=2, padding=1)
                        self.check(arch, conv_spikes, w_conv, stride=1, padding=0)
                        self.check(arch, dense_spikes, w_dense)

    def test_silent_input_powers_only_ungated_parts(self):
        arch = test_arch("parallel", Precision(4), Crossbar(LINEAR_1BIT, rows=8, cols=8))
        silent = torch.zeros(1, 2, 4, 4, 2)
        result = evaluate_layer(arch, silent, torch.ones(3, 2, 3, 3, dtype=torch.int64),
                                output_spikes=0)
        for name in ("cells", "mirrors", "mirrors_custom", "ota", "driver", "wl_driver",
                     "tile_ctrl", "neuron", "sense_amp", "controller"):
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

    def test_readme_examples(self):
        x, w = torch.ones(1, 96, 1, 1, 1), torch.ones(2, 96, 1, 1)
        r = evaluate_layer(designs.c3cim(), x, w)
        for name, energy in dict(column_source=.000132, column_driver=.0156684, vi=.005832,
                                 lif=.0063624).items():
            self.assertAlmostEqual(r.components[name].energy_nj, energy, places=12)
        self.assertAlmostEqual(r.latency_ns, 482.0)
        self.assertAlmostEqual(r.area_um2, 10259.68, places=6)
        r = evaluate_layer(designs.conventional(), x, w)
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
        self.assertEqual((seq.components["cells"].installed, par.components["cells"].installed), (1, 25))
        self.assertEqual((seq.latency_ns, par.latency_ns), (25 * 5.0 + 2.0, 5.0 + 2.0))
        self.assertAlmostEqual(seq.components["cells"].energy_nj, par.components["cells"].energy_nj)


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
        self.assertEqual(r.components["ac"].active_ns, 3 * 4.0)  # union of a and c
        self.assertEqual(r.components["bd"].active_ns, 3 * 9.0)  # 4..13

    def test_pipelined_reads_and_overlapping_bins(self):
        stages = [Stage("drive", 2.0), Stage("sense", 3.0), Stage("fire", 1.0, level="timestep")]
        comps = [Component("amp", during=["sense"], static_ua=1.0),
                 Component("neuron", count="outputs", during=["timestep"], static_ua=1.0)]
        r = self.run_arch(stages, comps, x=torch.ones(1, 64, 1, 1, 2),
                          crossbar=Crossbar(LINEAR_1BIT, active_rows=16),       # 4 reads per bin
                          read_interval_ns=3.0, timestep_interval_ns=10.0)
        self.assertEqual(r.timeline.timestep["fire"], (14.0, 15.0))  # 3*3 + 5, then fire
        self.assertEqual(r.latency_ns, 25.0)                         # next bin starts at 10
        self.assertEqual(r.components["amp"].active_ns, 2 * 4 * 3.0)
        self.assertEqual(r.components["neuron"].active_ns, 2 * 15.0)

    def test_component_cannot_serve_overlapping_reads(self):
        with self.assertRaisesRegex(ValueError, "two reads at once"):
            self.run_arch([Stage("drive", 2.0), Stage("sense", 3.0)],
                          [Component("both", during=["drive", "sense"], static_ua=1.0)],
                          crossbar=Crossbar(LINEAR_1BIT, active_rows=32), read_interval_ns=3.0)

    def test_timestep_stage_before_reads(self):
        r = self.run_arch([Stage("read", 5.0), Stage("precharge", 1.0, level="timestep", after=[]),
                           Stage("fire", 2.0, level="timestep", after=["reads"])], [])
        self.assertEqual(r.timeline.timestep["precharge"], (0.0, 1.0))
        self.assertEqual(r.latency_ns, 3 * 7.0)


class CompositionTests(unittest.TestCase):
    def test_blocks_compose_in_order(self):
        arch = rram_ota_design()
        self.assertEqual([s.name for s in arch.stages], ["read", "fire"])
        self.assertEqual([c.name for c in arch.components],
                         ["cells", "sl_ota", "slice_mirrors", "lif"])
        extra = compose("x", Precision(4), [crossbars.conv_xbar(LINEAR_1BIT),
                                             periphery.vi_converter(1.0, 3.0),
                                             neurons.lif_neuron(1.0, 1.0)])
        self.assertEqual([(s.name, s.level) for s in extra.stages],
                         [("read", "read"), ("vi", "read"), ("fire", "timestep")])

    def test_compose_needs_one_crossbar(self):
        with self.assertRaisesRegex(ValueError, "exactly one crossbar"):
            compose("x", Precision(4), [periphery.slice_mirrors()])
        with self.assertRaisesRegex(ValueError, "exactly one crossbar"):
            compose("x", Precision(4), [crossbars.conv_xbar(LINEAR_1BIT),
                                         crossbars.c3cim_xbar(LINEAR_1BIT)])
        with self.assertRaises(ValueError):
            neurons.lif_neuron(1.0, 1.0, powered="always")


class ValidationTests(unittest.TestCase):
    def test_quantize_weights(self):
        codes, scale = quantize_weights(torch.tensor([-1.0, 0.6, 1.0]), 4)
        self.assertEqual(codes.tolist(), [-7, 4, 7])
        self.assertAlmostEqual(scale, 1 / 7)
        weights = torch.tensor([0.3])
        self.assertIs(quantize_weights(weights, None)[0], weights)

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
               dict(components=[Component("x", model="crossbar_read", during=["timestep"])]),
               dict(components=[Component("x", count={"rule": "column_groups"})]),
               dict(components=[Component("x", count={"rule": "tiles", "gated": True})]),
               dict(components=[Component("x", on="spiking_rows", during=["timestep"], static_ua=1.0)]),
               dict(components=[Component("x", count="spiking_rows")]),
               dict(precision=Precision(None, "twos_complement")),
               dict(conv_mapping="diagonal")]
        for kwargs in bad:
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}), self.assertRaises(ValueError):
                Architecture("x", Crossbar(LINEAR_1BIT), kwargs.pop("precision", Precision()),
                             kwargs.pop("stages", read), kwargs.pop("components", []), **kwargs)


if __name__ == "__main__":
    unittest.main()
