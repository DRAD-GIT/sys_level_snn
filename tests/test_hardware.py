import csv
import json
import logging
import os
import tempfile
import unittest

import torch

from evaluation.report import export_results
from hardware import (
    C3HardwareConfig,
    ConvHardwareConfig,
    HardwareMetrics,
    calculate_c3_metrics,
    calculate_conv_metrics,
    load_config,
)
from hardware.mapping import map_weights
from hardware.metrics import component, metrics_from_components


class HardwareTests(unittest.TestCase):
    def setUp(self):
        self.weights = torch.tensor([[[[-1.0]]], [[[1.0]]]])
        self.levels = torch.tensor([-1.0, 1.0])
        self.spikes = torch.tensor([[[[[1.0, 0.0]]]]])

    def test_offset_mapping(self):
        mapped = map_weights(self.weights, self.levels, 2000, 200000)
        zero = map_weights(torch.zeros_like(self.weights), self.levels, 2000, 200000)
        self.assertTrue(torch.all(mapped > 0))
        self.assertTrue(torch.all(zero > 0))
        self.assertAlmostEqual((mapped[0] + mapped[1]).item(), 2 * zero[0].item(), places=8)

    def test_batch_shares_area(self):
        for fn, config in ((calculate_conv_metrics, ConvHardwareConfig(xbar_row=1, xbar_col=1)),
                           (calculate_c3_metrics, C3HardwareConfig(xbar_row=1, xbar_col=1))):
            one = fn(config, self.spikes, self.weights, self.levels, 0)
            two = fn(config, self.spikes.repeat(2, 1, 1, 1, 1), self.weights, self.levels, 0)
            self.assertAlmostEqual(one.area_mm2, two.area_mm2)
            self.assertAlmostEqual(one.energy_nj, two.energy_nj / 2)
            self.assertAlmostEqual(one.latency_us, two.latency_us / 2)
            self.assertEqual(one.ops, two.ops / 2)
            combined = HardwareMetrics()
            combined.add(one)
            combined.add(two)
            self.assertAlmostEqual(combined.normalize().area_mm2, one.area_mm2)
            total = HardwareMetrics()
            total.add(one, distinct_layer=True)
            total.add(two, distinct_layer=True)
            self.assertAlmostEqual(total.area_mm2, 2 * one.area_mm2)

    def test_spike_activity_changes_conventional_energy(self):
        config = ConvHardwareConfig(xbar_row=1, xbar_col=1)
        active = calculate_conv_metrics(config, self.spikes, self.weights, self.levels, 0)
        idle = calculate_conv_metrics(config, torch.zeros_like(self.spikes), self.weights, self.levels, 0)
        self.assertGreater(active.energy_nj, idle.energy_nj)
        self.assertEqual(active.area_mm2, idle.area_mm2)

    def test_signed_weights_require_reference(self):
        with self.assertRaisesRegex(ValueError, "Signed weights"):
            calculate_conv_metrics(
                ConvHardwareConfig(reference_array=False),
                self.spikes, self.weights, self.levels, 0,
            )

    def test_eight_rows_phased_in_parallel_across_tiles(self):
        # 96 inputs occupy two 64-row tiles. A full tile determines eight
        # phases, NOT ceil(96 / 8) = 12 phases.
        x = torch.ones((1, 96, 1, 1, 2))
        w = torch.ones((2, 96, 1, 1))
        for fn, cls, read_component in (
            (calculate_conv_metrics, ConvHardwareConfig, "crossbar"),
            (calculate_c3_metrics, C3HardwareConfig, "column"),
        ):
            single = fn(cls(xbar_row=64), x, w, self.levels, 0)
            phased = fn(cls(xbar_row=64, active_rows=8), x, w, self.levels, 0)
            self.assertEqual(single.row_phases, 1)
            self.assertEqual(phased.row_phases, 8)
            self.assertEqual(single.area_mm2, phased.area_mm2)
            self.assertEqual(single.ops, phased.ops)
            self.assertEqual(single.components[read_component].count,
                             phased.components[read_component].count)
            self.assertAlmostEqual(single.components[read_component].latency_us * 8,
                                   phased.components[read_component].latency_us)
            self.assertAlmostEqual(sum(c.energy_nj for c in phased.components.values()),
                                   phased.energy_nj)
            self.assertAlmostEqual(sum(c.area_mm2 for c in phased.components.values()),
                                   phased.area_mm2)
            self.assertAlmostEqual(sum(c.latency_us for c in phased.components.values()),
                                   phased.latency_us)
        # The second tile has only 32 valid rows: it stops after four phases.
        c3 = calculate_c3_metrics(C3HardwareConfig(active_rows=8), x, w, self.levels, 0)
        self.assertAlmostEqual(c3.components["column"].energy_nj,
                               (2 * (8 + 4)) * C3HardwareConfig().col_pow
                               * 1e-6 * C3HardwareConfig().col_lat * 2)
        conv_one = calculate_conv_metrics(ConvHardwareConfig(), x, w, self.levels, 0)
        conv_eight = calculate_conv_metrics(ConvHardwareConfig(active_rows=8), x, w, self.levels, 0)
        self.assertAlmostEqual(conv_one.components["crossbar"].energy_nj,
                               conv_eight.components["crossbar"].energy_nj)
        self.assertAlmostEqual(conv_one.components["DA"].energy_nj * 6,
                               conv_eight.components["DA"].energy_nj)

    def test_new_component_contributes_automatically(self):
        blocks = {"custom_buffer": component(3, 12.0, 0.5, 4.0)}
        result = metrics_from_components(blocks, 1, 0, 0, 1)
        self.assertAlmostEqual(result.energy_nj, 0.5)
        self.assertAlmostEqual(result.area_mm2, 12e-6)
        self.assertAlmostEqual(result.latency_us, 0.004)
        self.assertIn("custom_buffer", result.format_summary())

    def test_driver_partition_changes_only_driver_contributions(self):
        x = torch.ones((1, 8, 1, 1, 1))
        w = torch.ones((1, 8, 1, 1))
        a = calculate_c3_metrics(C3HardwareConfig(driver_part=32), x, w, self.levels, 0)
        b = calculate_c3_metrics(C3HardwareConfig(driver_part=8), x, w, self.levels, 0)
        self.assertEqual(b.components["column_driver"].count,
                         4 * a.components["column_driver"].count)
        for name in ("column", "VI", "LIF"):
            self.assertEqual(a.components[name], b.components[name])

    def test_component_json_and_table_upsert(self):
        x = self.spikes
        w = self.weights
        per_layer = {"SC1": calculate_conv_metrics(ConvHardwareConfig(), x, w, self.levels, 0)}
        total = per_layer["SC1"].normalize()
        with tempfile.TemporaryDirectory() as log_dir:
            for _ in range(2):
                export_results(
                    log_dir, "nmnist", "NMNIST", "conventional", ConvHardwareConfig(),
                    0, 2, 1, per_layer, total, logging.getLogger("test_export"),
                )
            with open(os.path.join(log_dir, "comparison_summary.csv"), newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["energy_nj_per_inference"], str(total.energy_nj))
            with open(rows[0]["result_json"]) as file:
                report = json.load(file)
            self.assertIn("crossbar", report["layers"][0]["metrics"]["components"])

    def test_config_rejects_unknown_keys(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as file:
            json.dump({"typo": 12}, file)
            file.flush()
            with self.assertRaisesRegex(ValueError, "unknown parameters"):
                load_config(ConvHardwareConfig, file.name)


if __name__ == "__main__":
    unittest.main()
