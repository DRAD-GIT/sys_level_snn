"""SRMLayer against literal transcriptions of slayerSNN's computations.

The reference functions below follow slayerSNN's CUDA kernels
(convKernel, getSpikesKernel) and Python layers loop by loop, so the
vectorized implementation must reproduce them exactly.
"""
import unittest

import numpy as np
import torch
import torch.nn.functional as F

import models
from evaluation.software import num_spikes_loss, predict_class
from models.srm import Dense, Pool, SRMLayer, alpha_kernel

NEURON = {"type": "SRMALPHA", "theta": 10, "tauSr": 10.0, "tauRef": 1.0,
          "scaleRef": 2, "tauRho": 1, "scaleRho": 1}
SIMULATION = {"Ts": 1.0, "tSample": 300}


def reference_psp(x, kernel, ts):
    """slayerCuda convKernel: per neuron, result = sum_i x[t-i]*k[i]; * Ts."""
    flat = x.reshape(-1, x.shape[-1])
    out = torch.zeros_like(flat)
    for n in range(flat.shape[0]):
        for t in range(flat.shape[1]):
            result = torch.tensor(0.0)
            for i in range(len(kernel)):
                if t - i >= 0:
                    result = result + flat[n, t - i] * kernel[i]
            out[n, t] = result * ts
    return out.reshape(x.shape)


def reference_spikes(u, ref, theta, ts):
    """slayerCuda getSpikesKernel, one neuron at a time."""
    flat = u.reshape(-1, u.shape[-1]).clone()
    spikes = torch.zeros_like(flat)
    n_steps = flat.shape[1]
    for n in range(flat.shape[0]):
        for i in range(n_steps):
            if flat[n, i] >= theta:
                spikes[n, i] = 1 / ts
                for j in range(len(ref)):
                    if i + j < n_steps:
                        flat[n, i + j] += ref[j]
    return spikes.reshape(u.shape)


class KernelTests(unittest.TestCase):
    def test_kernels_regenerate_trained_kernels_exactly(self):
        for name in ("nmnist", "gesture"):
            spec = models.get_spec(name)
            state = torch.load(spec.path(spec.checkpoint), weights_only=True)["state_dict"]
            params = models.load_params(spec.path(spec.params_yaml))
            layer = SRMLayer(params["neuron"], params["simulation"])
            self.assertTrue(torch.equal(layer.srmKernel, state["slayer.srmKernel"]))
            self.assertTrue(torch.equal(layer.refKernel, state["slayer.refKernel"]))

    def test_alpha_kernel_formula(self):
        k = alpha_kernel(tau=2.0, t_sample=100, ts=1.0)
        self.assertEqual(k[0].item(), 0.0)
        self.assertAlmostEqual(k[2].item(), 1.0)  # peak value 1 at t = tau
        self.assertAlmostEqual(k[1].item(), 0.5 * np.exp(0.5), places=6)
        self.assertGreaterEqual(abs(k[-1].item()), 0.01)  # truncated below epsilon


class NeuronTests(unittest.TestCase):
    def setUp(self):
        self.layer = SRMLayer(NEURON, SIMULATION)
        self.gen = torch.Generator().manual_seed(0)

    def test_psp_matches_cuda_loop_exactly(self):
        for ts in (1.0, 6.0):
            layer = SRMLayer(NEURON, {"Ts": ts, "tSample": 300})
            x = torch.randn(2, 3, 1, 1, 40, generator=self.gen)
            self.assertTrue(torch.equal(layer.psp(x), reference_psp(x, layer.srmKernel, ts)))

    def test_psp_kernel_longer_than_signal(self):
        x = torch.randn(1, 2, 1, 1, 5, generator=self.gen)  # 77-tap kernel, 5 steps
        self.assertTrue(torch.equal(self.layer.psp(x),
                                    reference_psp(x, self.layer.srmKernel, 1.0)))

    def test_spike_matches_cuda_loop_exactly(self):
        for ts in (1.0, 4.0):
            layer = SRMLayer(NEURON, {"Ts": ts, "tSample": 300})
            u = 14 * torch.rand(3, 4, 2, 2, 60, generator=self.gen)
            spikes = layer.spike(u)
            self.assertTrue(torch.equal(spikes, reference_spikes(u, layer.refKernel, 10, ts)))
            self.assertTrue(spikes.sum() > 0)

    def test_spike_refractory_and_threshold_equality(self):
        # Constant drive exactly at threshold: fires at t=0, then refKernel
        # (-20 at t=1, decaying) suppresses firing until it recovers.
        u = torch.full((1, 1, 1, 1, 30), 10.0)
        spikes = self.layer.spike(u)[0, 0, 0, 0]
        self.assertEqual(spikes[0].item(), 1.0)
        self.assertTrue(torch.equal(spikes, reference_spikes(u, self.layer.refKernel, 10, 1.0)[0, 0, 0, 0]))
        self.assertEqual(u[0, 0, 0, 0, 1].item(), 10.0)  # input left untouched

    def test_spike_near_end_of_window(self):
        u = torch.zeros(1, 1, 1, 1, 8)
        u[..., -1] = 11
        self.assertEqual(self.layer.spike(u)[..., -1].item(), 1.0)


class LayerTests(unittest.TestCase):
    def setUp(self):
        self.gen = torch.Generator().manual_seed(1)

    def test_pool_is_scaled_window_sum(self):
        pool = Pool(10, 2)
        x = torch.rand(2, 3, 8, 6, 4, generator=self.gen)
        expected = 11.0 * F.avg_pool3d(x, (2, 2, 1)) * 4
        self.assertTrue(torch.allclose(pool(x), expected, atol=1e-5))

    def test_pool_odd_size_padding_as_slayer(self):
        # 17 rows: SLAYER appends 17 % 2 = 1 zero row -> 18 -> 9 outputs.
        x = torch.rand(1, 2, 17, 9, 3, generator=self.gen)
        out = Pool(10, 2)(x)
        self.assertEqual(tuple(out.shape), (1, 2, 9, 5, 3))
        padded = F.pad(x, (0, 0, 0, 1, 0, 1))
        self.assertTrue(torch.allclose(out, 11.0 * 4 * F.avg_pool3d(padded, (2, 2, 1)), atol=1e-5))

    def test_dense_tuple_input_spans_whole_image(self):
        dense = Dense((8, 8, 32), 512)
        self.assertEqual(tuple(dense.weight.shape), (512, 32, 8, 8, 1))
        out = dense(torch.rand(1, 32, 8, 8, 5, generator=self.gen))
        self.assertEqual(tuple(out.shape), (1, 512, 1, 1, 5))

    def test_pretrained_networks_forward(self):
        for name, shape, classes in (("nmnist", (1, 2, 34, 34, 20), 10),
                                     ("gesture", (1, 2, 128, 128, 20), 11)):
            net = models.load_pretrained(models.get_spec(name)).eval()
            x = (torch.rand(shape, generator=self.gen) > 0.9).float()
            with torch.no_grad():
                out = net(x)
            self.assertEqual(tuple(out.shape), (1, classes, 1, 1, 20))
            self.assertTrue(set(out.unique().tolist()) <= {0.0, 1.0})


class SoftwareMetricTests(unittest.TestCase):
    def test_predict_class_counts_spikes(self):
        out = torch.zeros(2, 3, 1, 1, 5)
        out[0, 2, ..., :3] = 1
        out[1, 0, ..., :1] = 1
        self.assertEqual(predict_class(out).tolist(), [2, 0])

    def test_num_spikes_loss_zero_at_target_counts(self):
        params = {"simulation": {"Ts": 1.0},
                  "training": {"error": {"tgtSpikeRegion": {"start": 0, "stop": 10},
                                         "tgtSpikeCount": {True: 4, False: 1}}}}
        target = torch.zeros(1, 2, 1, 1, 1)
        target[0, 0] = 1
        out = torch.zeros(1, 2, 1, 1, 10)
        out[0, 0, ..., :4] = 1
        out[0, 1, ..., :1] = 1
        psp = SRMLayer(NEURON, SIMULATION).psp
        self.assertEqual(num_spikes_loss(out, target, params, psp).item(), 0.0)
        out[0, 1, ..., 1] = 1  # one surplus spike: error 1/10 over 10 steps
        self.assertAlmostEqual(num_spikes_loss(out, target, params, psp).item(), 0.5 * 10 * 0.01)


if __name__ == "__main__":
    unittest.main()
