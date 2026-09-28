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
# Same design with weights quantized by the least-squared-error clip.
RRAM_MSE = dataclasses.replace(
    RRAM_1BIT_XBAR, name="rram_mse",
    precision=dataclasses.replace(RRAM_1BIT_XBAR.precision, weight_scaling="mse"))
ARCHITECTURES = [RRAM_1BIT_XBAR, ANALOG, RRAM_MSE]


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
        spec = dataclasses.replace(models.get_spec("nmnist"), dataset_class=RandomSpikes,
                                   batch_size=2)
        self.spec_patch = patch.object(models.nmnist, "SPEC", spec)
        self.spec_patch.start()
        with tempfile.TemporaryDirectory() as data:
            os.mkdir(os.path.join(data, "N-MNIST"))
            self.results = evaluate("nmnist", ARCHITECTURES, data_dir=data, max_samples=2,
                                    num_workers=0, log=lambda *_: None)

    def tearDown(self):
        self.spec_patch.stop()

    def test_results_per_architecture_and_layer(self):
        self.assertEqual(set(self.results), {"rram_1bit_conv_xbar", "analog_c3cim", "rram_mse"})
        for accuracy, costs in self.results.values():
            self.assertEqual(list(costs), list(models.get_spec("nmnist").layers))
            self.assertTrue(0 <= accuracy <= 100)
            for cost in costs.values():
                self.assertEqual(cost.inferences, 2)
                self.assertGreater(cost.energy_nj, 0)
        # The mse design stores other weight codes: other cell conductances.
        for layer in models.get_spec("nmnist").layers:
            self.assertNotEqual(self.results["rram_mse"][1][layer].components["cells"].energy_nj,
                                self.results["rram_1bit_conv_xbar"][1][layer].components["cells"].energy_nj)
        sc1 = self.results["rram_1bit_conv_xbar"][1]["SC1"]
        self.assertEqual(sc1.geometry.windows, 28 * 28)          # 34x34 input, 7x7 kernel
        self.assertGreater(sc1.output_spikes, 0)

    def test_accuracy_sweep_matches_pipeline(self):
        with tempfile.TemporaryDirectory() as data:
            os.mkdir(os.path.join(data, "N-MNIST"))
            sweep = accuracy_sweep("nmnist", {"6 std3": (6, "std3"), "6 mse": (6, "mse"),
                                              "float": (None, "max")},
                                   data_dir=data, max_samples=2, num_workers=0,
                                   log=lambda *_: None)
        # Same quantization as the pipeline's architectures.
        self.assertEqual(sweep["6 std3"], self.results["rram_1bit_conv_xbar"][0])
        self.assertEqual(sweep["6 mse"], self.results["rram_mse"][0])
        self.assertEqual(sweep["float"], self.results["analog_c3cim"][0])

    def test_sample_limit_and_batches(self):
        spec = models.get_spec("nmnist")                        # patched: 2 samples, batches of 2
        params = models.load_params(spec.path(spec.params_yaml))
        with tempfile.TemporaryDirectory() as data:
            os.mkdir(os.path.join(data, "N-MNIST"))
            one = models.test_loader(spec, params, data, max_samples=1, num_workers=0)
            everything = models.test_loader(spec, params, data, max_samples=50, num_workers=0)
        self.assertEqual((len(one.dataset), one.batch_size), (1, 2))
        with tempfile.TemporaryDirectory() as data:
            os.mkdir(os.path.join(data, "N-MNIST"))
            self.assertEqual(models.test_loader(spec, params, data, parallel=7).batch_size, 7)
        with self.assertRaisesRegex(ValueError, "parallel"):
            models.test_loader(spec, params, None, parallel=0)
        self.assertEqual(len(everything.dataset), 2)            # capped at the test set
        with self.assertRaisesRegex(ValueError, "max_samples"):
            models.test_loader(spec, params, data, max_samples=0)

    def test_quantized_network_is_a_copy(self):
        spec = models.get_spec("nmnist")
        net = models.load_pretrained(spec)
        original = net.SC1.weight.clone()
        quantized, codes = quantized_network(net, spec.layers, 3)
        self.assertTrue(torch.equal(net.SC1.weight, original))
        for name in spec.layers:
            self.assertLessEqual(len(torch.unique(getattr(quantized, name).weight)), 7)  # +/-3
            self.assertLessEqual(int(codes[name].abs().max()), 3)
        # Per-layer bit widths: listed layers quantized, the others float.
        mixed, codes = quantized_network(net, spec.layers, {"SF2": 3, "SC1": None})
        self.assertLessEqual(len(torch.unique(mixed.SF2.weight)), 7)
        self.assertTrue(torch.equal(mixed.SC1.weight, net.SC1.weight))
        self.assertTrue(torch.equal(mixed.SF1.weight, net.SF1.weight))
        with self.assertRaisesRegex(ValueError, "unknown layers"):
            quantized_network(net, spec.layers, {"SC9": 4})

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
                self.assertEqual(len(file.read().strip().splitlines()), 4)   # header + 3 (upserted)


if __name__ == "__main__":
    unittest.main()
