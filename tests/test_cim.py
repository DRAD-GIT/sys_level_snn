"""Modular CIM engine: hand calculations, legacy equivalence, timeline, slicing."""
import math
import unittest

import numpy as np
import torch

from cim import Architecture, Component, Crossbar, Precision, Stage, evaluate_layer, quantize_symmetric
from cim.examples.dense_1bit_rram import ARCH as DENSE_ARCH, example_data
from cim.presets import c3cim, conventional_cim
from hardware import (C3HardwareConfig, ConvHardwareConfig, calculate_c3_metrics,
                      calculate_conv_metrics)


def static(name, during, ua=1.0, count="tiles", on="all", **kw):
    return Component(name, count=count, on=on, during=during, supply_v=1.0, static_ua=ua, **kw)


def arch(stages, components, **kw):
    precision = kw.pop("precision", Precision(weight_encoding="analog"))
    crossbar = kw.pop("crossbar", Crossbar())
    return Architecture("test", crossbar, precision, stages, components, **kw)


class HandCalculationTests(unittest.TestCase):
    def test_dense_1bit_rram_example(self):
        x, w = example_data()
        result = evaluate_layer(DENSE_ARCH, x, w)
        # Independent calculation, cell by cell.
        codes = w[:, :, 0, 0].numpy() % 16                  # two's complement bits
        spikes = x[0, :, 0, 0, :].numpy()                   # (128 inputs, 4 time bins)
        current = 0.0
        for o in range(64):
            for s in range(4):                              # weight bit / column
                g = np.where((codes[o] >> s) & 1, 1 / 20e3, 1 / 200e3)
                for t in range(4):                          # time bin / read
                    current += 0.2 * np.sum(spikes[:, t] * g)
        expected = {"cells": 1.1 * current * 5.0,
                    "sl_ota": 1.1 * 10e-6 * 512 * 4 * 5.0,  # 512 used columns x 4 reads x 5 ns
                    "lif_comparator": 1.1 * 10e-6 * 64 * 4 * (5.0 + 2.0)}
        for name, energy in expected.items():
            self.assertAlmostEqual(result.components[name].energy_nj, energy, places=9)
        self.assertEqual(result.latency_ns, 4 * 7.0)        # read + fire per time bin
        self.assertEqual((result.geometry.row_tiles, result.geometry.column_tiles), (2, 4))
        self.assertEqual(result.components["sl_ota"].installed, 512)
        self.assertEqual(result.macs, 128 * 64 * 4)                   # every input, every bin
        self.assertEqual(result.synaptic_ops, spikes.sum() * 64)      # each spike reaches 64 outputs

    def test_readme_c3_example(self):
        x, w = torch.ones(1, 96, 1, 1, 1), torch.ones(2, 96, 1, 1)
        r = evaluate_layer(c3cim(C3HardwareConfig(active_rows=8)), x, w)
        for name, energy in dict(column=.000132, column_driver=.0156684, VI=.005832,
                                 LIF=.0063624).items():
            self.assertAlmostEqual(r.components[name].energy_nj, energy, places=12)
        self.assertAlmostEqual(r.latency_ns, 482.0)
        self.assertAlmostEqual(r.area_um2, 10259.68, places=6)

    def test_readme_conventional_example(self):
        x, w = torch.ones(1, 96, 1, 1, 1), torch.ones(2, 96, 1, 1)
        r = evaluate_layer(conventional_cim(ConvHardwareConfig(active_rows=8)), x, w,
                           levels=torch.tensor([-1.0, 1.0]))
        for name, energy in dict(crossbar=.04752, reference_array=.0239976, DA=.00072468,
                                 reference_subtractor=0.0, LIF=.0005016).items():
            self.assertAlmostEqual(r.components[name].energy_nj, energy, places=9)
        self.assertAlmostEqual(r.latency_ns, 38.0)
        self.assertAlmostEqual(r.area_um2, 9969.4, places=6)


class LegacyEquivalenceTests(unittest.TestCase):
    def test_matches_legacy_engine_on_random_layers(self):
        g = torch.Generator().manual_seed(3)
        cases = [((2, 6, 12, 12, 7), (16, 6, 5, 5), 2), ((2, 120, 1, 1, 4), (84, 120, 1, 1), 0)]
        configs = [(ConvHardwareConfig(active_rows=8), calculate_conv_metrics, conventional_cim),
                   (ConvHardwareConfig(active_rows=16, ref_sub_curr=3.0, ref_sub_lat=7.0),
                    calculate_conv_metrics, conventional_cim),
                   (C3HardwareConfig(active_rows=8), calculate_c3_metrics, c3cim),
                   (C3HardwareConfig(driver_part=16, xbar_row=32, xbar_col=128),
                    calculate_c3_metrics, c3cim)]
        for xs, ws, pad in cases:
            x = (torch.rand(xs, generator=g) > 0.7).float()
            w = torch.randn(ws, generator=g)
            levels = torch.stack([w.min(), w.max()])
            for config, legacy, preset in configs:
                old = legacy(config, x, w, levels, pad).normalize()
                new = evaluate_layer(preset(config), x, w, padding=pad, levels=levels)
                with self.subTest(case=xs, config=type(config).__name__):
                    # float32 sums in the legacy engine: agree to ~1e-7.
                    self.assertAlmostEqual(new.energy_nj / old.energy_nj, 1.0, places=6)
                    self.assertAlmostEqual(new.latency_ns, old.latency_us * 1e3, places=6)
                    self.assertAlmostEqual(new.area_um2, old.area_mm2 * 1e6, places=6)
                    self.assertEqual(2 * new.macs, old.ops)
                    for name, component in old.components.items():
                        self.assertEqual(new.components[name].installed, component.count)


class TimelineTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.ones(1, 64, 1, 1, 3)       # 3 timesteps, one read each
        self.w = torch.ones(1, 64, 1, 1)

    def run_arch(self, stages, components, **kw):
        return evaluate_layer(arch(stages, components, **kw), self.x, self.w)

    def test_serial_parallel_and_overlap(self):
        stages = [Stage("a", 4.0), Stage("b", 6.0),            # b after a (serial)
                  Stage("c", 3.0, after=[]),                   # c parallel with a
                  Stage("d", 5.0, after=["b"], offset_ns=-2.0)]  # d overlaps end of b by 2 ns
        comps = [static("on_a_c", ["a", "c"]), static("on_b_d", ["b", "d"])]
        r = self.run_arch(stages, comps)
        read = r.timeline.read
        self.assertEqual(read, {"a": (0, 4), "b": (4, 10), "c": (0, 3), "d": (8, 13)})
        self.assertEqual(r.latency_ns, 3 * 13.0)
        self.assertEqual(r.components["on_a_c"].powered_ns, 3 * 4.0)   # union of a and c
        self.assertEqual(r.components["on_b_d"].powered_ns, 3 * 9.0)   # 4..13

    def test_pipelined_reads_and_overlapping_timesteps(self):
        x = torch.ones(1, 64, 1, 1, 2)
        a = arch([Stage("drive", 2.0), Stage("sense", 3.0), Stage("fire", 1.0, level="timestep")],
                 [static("sense_amp", ["sense"]), static("neuron", ["timestep"])],
                 crossbar=Crossbar(active_rows=16),          # 4 row phases = 4 reads per bin
                 read_interval_ns=3.0, timestep_interval_ns=10.0)
        r = evaluate_layer(a, x, self.w)
        # One read: drive 0-2, sense 2-5. Reads start every 3 ns -> block 3*3+5 = 14,
        # fire 14-15; next timestep starts 10 ns later -> latency 10 + 15.
        self.assertEqual(r.timeline.timestep["fire"], (14.0, 15.0))
        self.assertEqual(r.latency_ns, 25.0)
        self.assertEqual(r.components["sense_amp"].powered_ns, 2 * 4 * 3.0)
        self.assertEqual(r.components["neuron"].powered_ns, 2 * 15.0)

    def test_component_cannot_serve_overlapping_reads(self):
        a = arch([Stage("drive", 2.0), Stage("sense", 3.0)], [static("both", ["drive", "sense"])],
                 crossbar=Crossbar(active_rows=32), read_interval_ns=3.0)
        with self.assertRaisesRegex(ValueError, "two reads at once"):
            evaluate_layer(a, torch.ones(1, 64, 1, 1, 1), self.w)

    def test_timestep_stage_before_reads(self):
        stages = [Stage("read", 5.0), Stage("precharge", 1.0, level="timestep", after=[]),
                  Stage("fire", 2.0, level="timestep", after=["reads"])]
        r = self.run_arch(stages, [])
        self.assertEqual(r.timeline.timestep["precharge"], (0.0, 1.0))
        self.assertEqual(r.latency_ns, 3 * 7.0)


class SlicingTests(unittest.TestCase):
    def cells_energy(self, precision, crossbar, x, w):
        a = Architecture("t", crossbar, precision, [Stage("read", 1.0)],
                         [Component("cells", model="crossbar_read", during=["read"], supply_v=1.0)])
        r = evaluate_layer(a, x, w)
        return r.components["cells"].energy_nj, r.geometry

    def test_two_bit_cells_over_time_bins(self):
        xb = Crossbar(rows=4, cols=4, cell_bits=2, r_on=1e3, r_off=4e3, v_read=1.0)
        pr = Precision(weight_bits=4, weight_encoding="offset")
        w = torch.tensor([[[[5]]]])                      # offset code 13 = cells (1, 3)
        x = torch.tensor([[[[[1, 0, 1]]]]])              # spikes in 2 of 3 time bins
        energy, geometry = self.cells_energy(pr, xb, x, w)
        g = lambda level: 1 / 4e3 + level / 3 * (1 / 1e3 - 1 / 4e3)
        self.assertAlmostEqual(energy, 2 * (g(1) + g(3)), places=12)
        self.assertEqual((geometry.columns_per_weight, geometry.timesteps), (2, 3))

    def test_differential_and_twos_complement(self):
        xb = Crossbar(rows=8, cols=8, r_on=1e3, r_off=1e6, v_read=1.0)
        x = torch.ones(1, 1, 1, 1, 1, dtype=torch.int64)
        w = torch.tensor([[[[-3]]]])
        g1, g0 = 1e-3, 1e-6
        diff, geometry = self.cells_energy(Precision(weight_bits=3, weight_encoding="differential"), xb, x, w)
        self.assertEqual(geometry.columns_per_weight, 4)      # 2 magnitude bits x (pos, neg)
        self.assertAlmostEqual(diff, 2 * g0 + 2 * g1, places=12)   # pos: 00, neg: 11
        twos, _ = self.cells_energy(Precision(weight_bits=3, weight_encoding="twos_complement"), xb, x, w)
        self.assertAlmostEqual(twos, 2 * g1 + g0, places=12)      # -3 = 101

    def test_quantize_and_range_checks(self):
        codes, scale = quantize_symmetric(torch.tensor([-1.0, 0.6, 1.0]), 4)
        self.assertEqual(codes.tolist(), [-7, 4, 7])
        self.assertAlmostEqual(scale, 1 / 7)
        with self.assertRaisesRegex(ValueError, "range"):
            self.cells_energy(Precision(weight_bits=2), Crossbar(), torch.ones(1, 1, 1, 1, 1),
                              torch.tensor([[[[5]]]]))
        with self.assertRaisesRegex(ValueError, "integer weight codes"):
            self.cells_energy(Precision(weight_bits=4), Crossbar(), torch.ones(1, 1, 1, 1, 1),
                              torch.ones(1, 1, 1, 1))

    def test_validation(self):
        bad = [dict(stages=[Stage("t", 1.0, level="timestep")]),               # no read stage
               dict(components=[static("x", ["missing"])]),
               dict(components=[static("x", [])]),
               dict(components=[Component("x", model="crossbar_read", during=["read", "read2"])]),
               dict(components=[static("x", ["read"], count={"rule": "column_groups"})])]
        for kwargs in bad:
            stages = kwargs.pop("stages", [Stage("read", 1.0), Stage("read2", 1.0)])
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}), self.assertRaises(ValueError):
                arch(stages, kwargs.get("components", []))


if __name__ == "__main__":
    unittest.main()
