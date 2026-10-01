"""models/lif.py and models/cifar10.py: LIF neurons, BatchNorm folding, the
CIFAR-10 reader and rate coding, and the VGG-11 network on the engine."""
import os
import pickle
import tempfile
import unittest

import numpy as np
import torch

import crossbars
import models
import models.cifar10
import models.gesture
import models.nmnist
from evaluation.probes import LayerProbe
from hardware import Component, Mapping, compose, evaluate_layer, quantize_weights
from models.cifar10 import SPEC, CIFAR10Dataset, VGG11Network, read_batches
from models.lif import LIFLayer, fold_batchnorm, rate_code

PARAMS = {"simulation": {"Ts": 1.0, "tSample": 4},
          "neuron": {"type": "LIF", "theta": 1.0, "beta": 0.5, "alpha": 2.0}}


class LIFTests(unittest.TestCase):
    def test_spikes_follow_leaky_integration_with_soft_reset(self):
        layer = LIFLayer(PARAMS["neuron"], PARAMS["simulation"])
        current = torch.tensor([0.6, 0.6, 0.0, 2.5])
        # v: 0.6 | 0.9 | 0.45 | 2.725 -> fires, v = 1.725
        self.assertEqual(layer.spike(current).tolist(), [0, 0, 0, 1])
        current = torch.tensor([1.0, 0.0, 3.0, 0.0])
        # v: 1 fires (0) | 0 | 3 fires (2) | 1 fires (0)
        self.assertEqual(layer.spike(current).tolist(), [1, 0, 1, 1])

    def test_surrogate_gradient_flows(self):
        layer = LIFLayer(PARAMS["neuron"], PARAMS["simulation"])
        current = torch.tensor([0.9, 0.2, 0.3, 0.1], requires_grad=True)
        layer.spike(current).sum().backward()
        self.assertTrue(torch.all(current.grad > 0))

    def test_rate_code_is_binary_with_the_pixel_probability(self):
        g = torch.Generator().manual_seed(0)
        images = torch.tensor([0.0, 0.25, 1.0]).reshape(1, 3, 1, 1)
        spikes = rate_code(images, 4000, g)
        self.assertEqual(spikes.shape, (1, 3, 1, 1, 4000))
        self.assertTrue(set(spikes.unique().tolist()) <= {0.0, 1.0})
        rates = spikes.mean(-1).flatten().tolist()
        self.assertEqual(rates[0], 0.0)
        self.assertAlmostEqual(rates[1], 0.25, delta=0.03)
        self.assertEqual(rates[2], 1.0)


class VGG11Tests(unittest.TestCase):
    def test_batchnorm_folding_gives_identical_spikes(self):
        torch.manual_seed(0)
        trained = VGG11Network(PARAMS, batchnorm=True)
        for name, module in trained.named_modules():          # non-trivial statistics
            if isinstance(module, torch.nn.BatchNorm3d):
                module.running_mean.uniform_(-0.2, 0.2)
                module.running_var.uniform_(0.5, 1.5)
                module.weight.data.uniform_(0.5, 1.5)
                module.bias.data.uniform_(-0.1, 0.1)
        trained.eval()
        folded = VGG11Network(PARAMS)
        folded.load_state_dict(trained.folded_state_dict())
        folded.eval()
        spikes = (torch.rand(2, 3, 32, 32, 4) < 0.3).float()
        with torch.no_grad():
            reference = trained(spikes)
            result = folded(spikes)
        self.assertEqual(result.shape, (2, 10, 1, 1, 4))
        self.assertLess(float((reference != result).float().mean()), 0.01)

    def test_fold_batchnorm_matches_conv_then_bn(self):
        conv = torch.nn.Conv3d(3, 4, (3, 3, 1), padding=(1, 1, 0))
        bn = torch.nn.BatchNorm3d(4).eval()
        bn.running_mean.uniform_(-1, 1)
        bn.running_var.uniform_(0.5, 2)
        weight, bias = fold_batchnorm(conv, bn)
        x = torch.randn(1, 3, 5, 5, 2)
        expected = bn(conv(x))
        result = torch.nn.functional.conv3d(x, weight, bias, padding=(1, 1, 0))
        self.assertTrue(torch.allclose(expected, result, atol=1e-5))

    def test_every_weighted_layer_gets_binary_spikes_and_runs_on_the_engine(self):
        torch.manual_seed(0)
        net = VGG11Network(PARAMS).eval()
        for name in SPEC.layers:                                 # enough activity to fire
            getattr(net, name).bias.data.fill_(0.5)
        probe = LayerProbe(net, SPEC.layers)
        with torch.no_grad():
            net((torch.rand(1, 3, 32, 32, 4) < 0.3).float())
        arch = compose("c3cim", Mapping(weight_bits=6, weight_encoding="twos_complement"), [
            crossbars.c3cim_xbar(r_on=2e3, r_off=20e3, rows=64, cols=64, time_ns=23.0),
            Component("lif", count="outputs", time_ns=2.0, static_ua=6.0)])
        for name in SPEC.layers:
            with self.subTest(layer=name):
                spikes = probe.inputs[name]
                self.assertTrue(set(spikes.unique().tolist()) <= {0.0, 1.0})
                layer = getattr(net, name)
                weights, _ = quantize_weights(layer.weight.detach().squeeze(-1), 6, "std3")
                cost = evaluate_layer(arch, spikes, weights, padding=layer.padding[0])
                self.assertGreater(cost.latency_ns, 0)
        probe.remove()


class CIFAR10DataTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp()
        rng = np.random.default_rng(0)
        self.labels = rng.integers(0, 10, 5).tolist()
        self.data = rng.integers(0, 256, (5, 3072)).astype(np.uint8)
        with open(os.path.join(self.folder, "test_batch"), "wb") as file:
            pickle.dump({b"data": self.data, b"labels": self.labels}, file)

    def test_reader_and_reproducible_rate_coding(self):
        images, labels = read_batches([os.path.join(self.folder, "test_batch")])
        self.assertEqual(images.shape, (5, 3, 32, 32))
        self.assertEqual(labels.tolist(), self.labels)
        self.assertEqual(images[1, 2, 0, 0], self.data[1, 2048])
        dataset = CIFAR10Dataset(None, os.path.join(self.folder, "test_batch"), 1.0, 8)
        index, spikes, desired, label = dataset[3]
        self.assertEqual((index, label), (3, self.labels[3]))
        self.assertEqual(spikes.shape, (3, 32, 32, 8))
        self.assertEqual(int(desired.argmax()), label)
        self.assertTrue(torch.equal(spikes, dataset[3][1]))      # same coding every time

    def test_existing_models_keep_their_recording_sources(self):
        # Recordings of N-MNIST and gesture are fingerprinted with these files.
        for spec in (models.nmnist.SPEC, models.gesture.SPEC):
            self.assertEqual(spec.sources, ("models/srm.py", "models/events.py"))
        self.assertEqual(SPEC.sources, ("models/lif.py", "models/cifar10.py"))
        self.assertIn("cifar10", models.MODEL_MODULES)


if __name__ == "__main__":
    unittest.main()
