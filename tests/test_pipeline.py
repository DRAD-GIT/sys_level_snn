"""End-to-end pipeline: probes, runner, report switches and export."""
import dataclasses
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import torch

import models
import models.nmnist
from architectures import crossbars
from evaluation.probes import LayerProbe
from evaluation.report import export, format_results
from evaluation.runner import accuracy_sweep, evaluate, quantized_network
from hardware import Component, Precision, Stage, compose
from run import RRAM_1BIT_XBAR

# run.py's design plus one with analog weights, for a second precision group.
ANALOG = compose("analog_c3cim", Precision(None, "analog"), [
    crossbars.c3cim_xbar(r_on=2e3, r_off=20e3),
    Stage("fire", 2.0, level="timestep"),
    Component("lif", count="outputs", during="fire", static_ua=6.0)])
ARCHITECTURES = [RRAM_1BIT_XBAR, ANALOG]


class RandomSpikes(torch.utils.data.Dataset):
    def __init__(self, **_):
        self.gen = torch.Generator().manual_seed(0)

    def __len__(self):
        return 2

    def __getitem__(self, index):
        target = torch.zeros(10, 1, 1, 1)
        target[index] = 1
        return index, (torch.rand(2, 34, 34, 300, generator=self.gen) > 0.97).float(), target, index


class ProbeTests(unittest.TestCase):
    def test_inputs_and_output_spikes_per_layer(self):
        spec = models.get_spec("nmnist")
        net = models.load_pretrained(spec).eval()
        x = (torch.rand(1, 2, 34, 34, 50, generator=torch.Generator().manual_seed(1)) > 0.9).float()
        with torch.no_grad():  # expected counts from an unprobed forward pass
            sc1_spikes = net.slayer.spike(net.slayer.psp(net.SC1(x)))
            output = net(x)
        probe = LayerProbe(net, spec.layers)
        with torch.no_grad():
            net(x)
        self.assertTrue(torch.equal(probe.inputs["SC1"], x))
        self.assertEqual(probe.output_spikes["SC1"], int(torch.count_nonzero(sc1_spikes)))
        self.assertEqual(probe.output_spikes["SF2"], int(torch.count_nonzero(output)))
        probe.remove()
        self.assertNotIn("spike", vars(net.slayer))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        spec = dataclasses.replace(models.get_spec("nmnist"), dataset_class=RandomSpikes)
        self.spec_patch = patch.object(models.nmnist, "SPEC", spec)
        self.spec_patch.start()
        with tempfile.TemporaryDirectory() as data:
            os.mkdir(os.path.join(data, "N-MNIST"))
            self.results = evaluate("nmnist", ARCHITECTURES, data_dir=data, batch_size=2,
                                    max_batches=1, num_workers=0, log=lambda *_: None)

    def tearDown(self):
        self.spec_patch.stop()

    def test_results_per_architecture_and_layer(self):
        self.assertEqual(set(self.results), {"rram_1bit_conv_xbar", "analog_c3cim"})
        for accuracy, costs in self.results.values():
            self.assertEqual(list(costs), list(models.get_spec("nmnist").layers))
            self.assertTrue(0 <= accuracy <= 100)
            for cost in costs.values():
                self.assertEqual(cost.inferences, 2)
                self.assertGreater(cost.energy_nj, 0)
        sc1 = self.results["rram_1bit_conv_xbar"][1]["SC1"]
        self.assertEqual(sc1.geometry.windows, 28 * 28)          # 34x34 input, 7x7 kernel
        self.assertGreater(sc1.output_spikes, 0)

    def test_accuracy_sweep_matches_pipeline(self):
        with tempfile.TemporaryDirectory() as data:
            os.mkdir(os.path.join(data, "N-MNIST"))
            sweep = accuracy_sweep("nmnist", [4, None], data_dir=data, batch_size=2,
                                   max_batches=1, num_workers=0, log=lambda *_: None)
        # Same quantization as the pipeline: 4-bit = run.py's design, float = analog.
        self.assertEqual(sweep[4], self.results["rram_1bit_conv_xbar"][0])
        self.assertEqual(sweep[None], self.results["analog_c3cim"][0])

    def test_quantized_network_is_a_copy(self):
        spec = models.get_spec("nmnist")
        net = models.load_pretrained(spec)
        original = net.SC1.weight.clone()
        quantized, codes = quantized_network(net, spec.layers, 3)
        self.assertTrue(torch.equal(net.SC1.weight, original))
        for name in spec.layers:
            self.assertLessEqual(len(torch.unique(getattr(quantized, name).weight)), 7)  # +/-3
            self.assertLessEqual(int(codes[name].abs().max()), 3)

    def test_metric_switches(self):
        metrics = {"energy": True, "latency": True, "area": False, "layers": True, "components": False}
        text = format_results(self.results, metrics)
        self.assertIn("energy/inference", text)
        self.assertNotIn("area", text)
        self.assertNotIn("component", text)
        self.assertNotIn("TOPS/W", text)
        with tempfile.TemporaryDirectory() as folder:
            rows = export(self.results, ARCHITECTURES, "nmnist", metrics, folder)
            rows = export(self.results, ARCHITECTURES, "nmnist", metrics, folder)
            with open(rows[0]["result_json"]) as file:
                report = json.load(file)
            self.assertEqual(set(report["network"]), {"energy", "latency"})
            self.assertNotIn("components", report["layers"]["SC1"])
            with open(os.path.join(folder, "comparison_summary.csv")) as file:
                self.assertEqual(len(file.read().strip().splitlines()), 3)   # header + 2 (upserted)


if __name__ == "__main__":
    unittest.main()
