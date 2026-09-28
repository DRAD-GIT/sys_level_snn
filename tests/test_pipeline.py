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
import crossbars
from evaluation.probes import LayerProbe
from evaluation.report import export, format_results
from evaluation.recording import open_recording, record
from evaluation.runner import accuracy_sweep, evaluate, quantized_network
from evaluation.software import predict_class
from hardware import Component, Mapping, compose
from run import RRAM_1BIT_XBAR

# run.py's design plus one with analog weights, for a second precision group.
ANALOG = compose("analog_c3cim", Mapping(None, "max", "analog", "sequential"), [
    crossbars.c3cim_xbar(r_on=2e3, r_off=20e3),
    Component("lif", count="outputs", time_ns=2.0, static_ua=6.0)])
# Same design with weights quantized by the least-squared-error clip.
RRAM_MSE = dataclasses.replace(
    RRAM_1BIT_XBAR, name="rram_mse",
    mapping=dataclasses.replace(RRAM_1BIT_XBAR.mapping, weight_scaling="mse"))
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


class Unreadable(RandomSpikes):
    def __getitem__(self, index):
        raise AssertionError("the dataset was read although a recording exists")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        spec = dataclasses.replace(models.get_spec("nmnist"), dataset_class=RandomSpikes,
                                   batch_size=2)
        self.spec_patch = patch.object(models.nmnist, "SPEC", spec)
        self.spec_patch.start()
        self.recordings = tempfile.TemporaryDirectory()
        self.data = tempfile.TemporaryDirectory()
        os.mkdir(os.path.join(self.data.name, "N-MNIST"))
        self.results = evaluate("nmnist", ARCHITECTURES, data_dir=self.data.name,
                                recording_dir=self.recordings.name, max_samples=2,
                                num_workers=0, log=lambda *_: None)

    def tearDown(self):
        self.spec_patch.stop()
        self.recordings.cleanup()
        self.data.cleanup()

    def test_recording_is_reused_with_identical_results(self):
        # One recording per weight quantization (6-bit std3, 6-bit mse, float).
        self.assertEqual(sorted(os.listdir(self.recordings.name)),
                         ["nmnist_6b_mse", "nmnist_6b_std3", "nmnist_float"])
        spec = dataclasses.replace(models.get_spec("nmnist"), dataset_class=Unreadable)
        with patch.object(models.nmnist, "SPEC", spec):
            again = evaluate("nmnist", ARCHITECTURES, data_dir=self.data.name,
                             recording_dir=self.recordings.name, max_samples=2,
                             num_workers=0, log=lambda *_: None)
        for name, (accuracy, costs) in self.results.items():
            self.assertEqual(again[name][0], accuracy)
            for layer, cost in costs.items():
                for component, c in cost.components.items():
                    self.assertEqual(again[name][1][layer].components[component].energy_nj,
                                     c.energy_nj)

    def test_recording_round_trip_is_exact(self):
        spec = models.get_spec("nmnist")
        recording = open_recording(self.recordings.name, "nmnist", 6, "std3", 2)
        chunk = next(recording.chunks())
        net = quantized_network(models.load_pretrained(spec).eval(), spec.layers, 6, "std3")[0]
        probe = LayerProbe(net, spec.layers)
        spikes = next(iter(models.test_loader(spec, models.load_params(spec.path(spec.params_yaml)),
                                              self.data.name, num_workers=0)))[1]
        with torch.no_grad():
            predicted = predict_class(net(spikes))
        for name in spec.layers:
            self.assertTrue(torch.equal(chunk.inputs[name], probe.inputs[name]), name)
            self.assertEqual(chunk.output_counts[name].tolist(), probe.output_counts[name].tolist())
        self.assertEqual(chunk.predictions.tolist(), predicted.tolist())
        self.assertEqual(len(next(recording.chunks(max_samples=1))), 1)

    def test_stale_or_short_recordings_are_not_used(self):
        folder = self.recordings.name
        self.assertIsNotNone(open_recording(folder, "nmnist", 6, "std3", 2))
        self.assertIsNotNone(open_recording(folder, "nmnist", 6, "std3", None))  # the whole (2-sample) set
        self.assertIsNone(open_recording(folder, "nmnist", 5, "std3", 2))        # never recorded
        self.assertIsNone(open_recording(folder, "nmnist", 6, "std3", 2, full_outputs=True))
        meta_path = os.path.join(folder, "nmnist_6b_std3", "meta.json")
        with open(meta_path) as file:
            meta = json.load(file)
        meta["fingerprint"] = "changed checkpoint"
        with open(meta_path, "w") as file:
            json.dump(meta, file)
        self.assertIsNone(open_recording(folder, "nmnist", 6, "std3", 2))

    def test_full_outputs(self):
        spec = models.get_spec("nmnist")
        with tempfile.TemporaryDirectory() as folder:
            recording = record("nmnist", [(6, "std3")], data_dir=self.data.name, recording_dir=folder,
                               full_outputs=True, num_workers=0, log=lambda *_: None)[(6, "std3")]
            chunk = next(recording.chunks())
            for name in spec.layers:
                self.assertEqual(chunk.outputs[name].shape[0], 2)
                self.assertEqual([int((o != 0).sum()) for o in chunk.outputs[name]],
                                 chunk.output_counts[name].tolist())

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
        shapes = {layer: (cost.geometry.in_channels, cost.geometry.input_size,
                          cost.geometry.out_channels, cost.geometry.output_size)
                  for layer, cost in self.results["rram_1bit_conv_xbar"][1].items()}
        self.assertEqual(shapes["SC1"], (2, (34, 34), 6, (28, 28)))
        self.assertEqual(shapes["SF2"][2:], (10, (1, 1)))
        text = format_results(self.results, {"layers": True})
        self.assertIn("input 2x34x34 -> output 6x28x28 x 300 time bins", text)
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
        with tempfile.TemporaryDirectory() as data:
            os.mkdir(os.path.join(data, "N-MNIST"))
            self.assertEqual(len(models.test_loader(spec, params, data, max_samples=-1,
                                                    num_workers=0).dataset), 2)   # -1 = all
        self.assertIsNotNone(open_recording(self.recordings.name, "nmnist", 6, "std3", -1))
        with self.assertRaisesRegex(ValueError, "samples"):
            models.sample_limit(-2)
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
