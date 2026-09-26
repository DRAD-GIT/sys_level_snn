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

from architectures import c3cim, conventional, rram_1bit
from hardware import (Architecture, Component, Crossbar, Precision, Stage, evaluate_layer,
                      quantize_weights)
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

    # Physical columns: conductance of each cell, built from the codes bit by bit.
    g_on, g_off = 1 / xb.r_on, 1 / xb.r_off
    flat = weights.reshape(out, -1)
    columns = []
    if pr.weight_encoding == "analog":
        peak = float(flat.abs().max())
        for o in range(out):
            columns.append([(g_on + g_off) / 2 + float(v) / peak * (g_on - g_off) / 2 for v in flat[o]])
    else:
        bits, cell = pr.weight_bits, xb.cell_bits
        top = 2 ** cell - 1
        for o in range(out):
            codes = [int(v) for v in flat[o]]
            if pr.weight_encoding == "twos_complement":
                planes, n = [[c % 2 ** bits for c in codes]], bits
            elif pr.weight_encoding == "offset":
                planes, n = [[c + 2 ** (bits - 1) for c in codes]], bits
            else:
                planes, n = [[max(c, 0) for c in codes], [max(-c, 0) for c in codes]], bits - 1
            for plane in planes:
                for s in range(math.ceil(n / cell)):
                    columns.append([g_off + ((v >> (s * cell)) & top) / top * (g_on - g_off)
                                    for v in plane])
    used = len(columns)
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
    data_current = reference_current = 0.0
    reads_per_bin = max_phases * (len(positions) if sequential else 1)
    for b in range(batch):
        for t in range(bins):
            patches = {w: patch(b, t, *w) for w in positions}
            for w, p_in in patches.items():
                for k, s in enumerate(p_in):
                    data_current += s * sum(col[k] for col in columns) * xb.v_read
                    reference_current += s * out * (g_on + g_off) / 2 * xb.v_read
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
            e += c.supply_v * data_current * durations[c.during[0]]
        elif c.model == "reference_read":
            e += c.supply_v * reference_current * durations[c.during[0]]
        energy[c.name] = e / batch
    return energy, bins * bin_span


def test_arch(mapping, precision, crossbar):
    gated = lambda rule, **kw: dict(rule=rule, gated=True, **kw)
    return Architecture(
        "test", crossbar, precision,
        [Stage("drive", 2.0), Stage("sense", 3.0), Stage("fire", 1.0, level="timestep")],
        [Component("cells", model="crossbar_read", during=["sense"], supply_v=1.0),
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
        crossbar = Crossbar(rows=16, cols=8, cell_bits=2, active_rows=6)   # partial tiles, 3 phases
        precisions = [Precision(4, "twos_complement"), Precision(4, "offset"),
                      Precision(4, "differential"), Precision(None, "analog")]
        for mapping in ("sequential", "parallel"):
            for precision in precisions:
                arch = test_arch(mapping, precision, crossbar)
                w_conv = conv_weights.float() if precision.weight_bits is None else conv_weights
                w_dense = dense_weights.float() if precision.weight_bits is None else dense_weights
                with self.subTest(mapping=mapping, encoding=precision.weight_encoding):
                    self.check(arch, conv_spikes, w_conv, stride=2, padding=1)
                    self.check(arch, conv_spikes, w_conv, stride=1, padding=0)
                    self.check(arch, dense_spikes, w_dense)

    def test_silent_input_powers_only_ungated_parts(self):
        arch = test_arch("parallel", Precision(4), Crossbar(rows=8, cols=8))
        silent = torch.zeros(1, 2, 4, 4, 2)
        result = evaluate_layer(arch, silent, torch.ones(3, 2, 3, 3, dtype=torch.int64),
                                output_spikes=0)
        for name in ("cells", "ota", "driver", "wl_driver", "tile_ctrl", "neuron", "sense_amp",
                     "controller"):
            self.assertEqual(result.components[name].energy_nj, 0.0, name)
        for name in ("bias", "bank", "clock"):
            self.assertGreater(result.components[name].energy_nj, 0.0, name)


class HandCalculationTests(unittest.TestCase):
    def test_rram_1bit_dense_layer(self):
        g = torch.Generator().manual_seed(0)
        weights = torch.randint(-7, 8, (64, 128, 1, 1), generator=g)
        spikes = torch.randint(0, 2, (1, 128, 1, 1, 4), generator=g)
        r = evaluate_layer(rram_1bit.ARCH, spikes, weights)
        codes, s = weights[:, :, 0, 0].numpy() % 16, spikes[0, :, 0, 0, :].numpy()
        current = sum(0.2 * np.sum(s[:, t] * np.where((codes[o] >> b) & 1, 1 / 20e3, 1 / 200e3))
                      for o in range(64) for b in range(4) for t in range(4))
        # OTAs: 256 used columns in each of the 2 row tiles, per time bin in
        # which that row tile receives a spike.
        busy = sum(bool(s[r * 64:(r + 1) * 64, t].any()) for r in range(2) for t in range(4))
        expected = {"cells": 1.1 * current * 5.0,
                    "sl_ota": 1.1 * 10e-6 * 256 * busy * 5.0,
                    "lif_comparator": 1.1 * 10e-6 * 64 * 4 * 7.0}
        for name, energy in expected.items():
            self.assertAlmostEqual(r.components[name].energy_nj, energy, places=9)
        self.assertEqual(r.latency_ns, 28.0)
        self.assertEqual(r.components["sl_ota"].installed, 512)
        self.assertEqual((r.macs, r.synaptic_ops), (128 * 64 * 4, s.sum() * 64))

    def test_readme_examples(self):
        x, w = torch.ones(1, 96, 1, 1, 1), torch.ones(2, 96, 1, 1)
        r = evaluate_layer(c3cim.ARCH, x, w)
        for name, energy in dict(column=.000132, column_driver=.0156684, VI=.005832,
                                 LIF=.0063624).items():
            self.assertAlmostEqual(r.components[name].energy_nj, energy, places=12)
        self.assertAlmostEqual(r.latency_ns, 482.0)
        self.assertAlmostEqual(r.area_um2, 10259.68, places=6)
        r = evaluate_layer(conventional.ARCH, x, w)
        for name, energy in dict(crossbar=.04752, reference_array=.0239976, DA=.00072468,
                                 reference_subtractor=0.0, LIF=.0005016).items():
            self.assertAlmostEqual(r.components[name].energy_nj, energy, places=12)
        self.assertAlmostEqual(r.latency_ns, 38.0)
        self.assertAlmostEqual(r.area_um2, 9969.4, places=6)

    def test_conv_mappings_trade_area_for_latency(self):
        spikes = torch.ones(1, 2, 5, 5, 1)
        weights = torch.ones(4, 2, 3, 3, dtype=torch.int64)
        seq = evaluate_layer(rram_1bit.ARCH, spikes, weights, padding=1)
        par_arch = Architecture(**{**vars(rram_1bit.ARCH), "conv_mapping": "parallel"})
        par = evaluate_layer(par_arch, spikes, weights, padding=1)
        self.assertEqual(seq.geometry.windows, 25)
        self.assertEqual((seq.components["cells"].installed, par.components["cells"].installed), (1, 25))
        self.assertEqual((seq.latency_ns, par.latency_ns), (25 * 5.0 + 2.0, 5.0 + 2.0))
        self.assertAlmostEqual(seq.components["cells"].energy_nj, par.components["cells"].energy_nj)


class TimelineTests(unittest.TestCase):
    def run_arch(self, stages, components, x=None, **kw):
        crossbar = kw.pop("crossbar", Crossbar())
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
                          crossbar=Crossbar(active_rows=16),       # 4 reads per bin
                          read_interval_ns=3.0, timestep_interval_ns=10.0)
        self.assertEqual(r.timeline.timestep["fire"], (14.0, 15.0))  # 3*3 + 5, then fire
        self.assertEqual(r.latency_ns, 25.0)                         # next bin starts at 10
        self.assertEqual(r.components["amp"].active_ns, 2 * 4 * 3.0)
        self.assertEqual(r.components["neuron"].active_ns, 2 * 15.0)

    def test_component_cannot_serve_overlapping_reads(self):
        with self.assertRaisesRegex(ValueError, "two reads at once"):
            self.run_arch([Stage("drive", 2.0), Stage("sense", 3.0)],
                          [Component("both", during=["drive", "sense"], static_ua=1.0)],
                          crossbar=Crossbar(active_rows=32), read_interval_ns=3.0)

    def test_timestep_stage_before_reads(self):
        r = self.run_arch([Stage("read", 5.0), Stage("precharge", 1.0, level="timestep", after=[]),
                           Stage("fire", 2.0, level="timestep", after=["reads"])], [])
        self.assertEqual(r.timeline.timestep["precharge"], (0.0, 1.0))
        self.assertEqual(r.latency_ns, 3 * 7.0)


class ValidationTests(unittest.TestCase):
    def test_quantize_weights(self):
        codes, scale = quantize_weights(torch.tensor([-1.0, 0.6, 1.0]), 4)
        self.assertEqual(codes.tolist(), [-7, 4, 7])
        self.assertAlmostEqual(scale, 1 / 7)
        weights = torch.tensor([0.3])
        self.assertIs(quantize_weights(weights, None)[0], weights)

    def test_weight_code_checks(self):
        arch = test_arch("sequential", Precision(4), Crossbar())
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
                Architecture("x", Crossbar(), kwargs.pop("precision", Precision()),
                             kwargs.pop("stages", read), kwargs.pop("components", []), **kwargs)


if __name__ == "__main__":
    unittest.main()
