"""models/gesture16.py: 16x16 downscaling of the events, time windows, the
10-class test set and the network (BatchNorm folding)."""
import os
import tempfile
import unittest

import numpy as np
import torch

import models
from models.gesture16 import (CLASSES, FINE_MS, SPEC, Gesture16Dataset, Gesture16Network,
                              fine_spikes, window)


class Gesture16Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        trial = os.path.join(self.tmp.name, "user01_led")
        os.makedirs(trial)
        # Events (x, y, p, t in ms): pixel (17, 9) -> block (2, 1); one late event.
        events = np.array([[17, 9, 1, 5.0], [16, 8, 1, 7.0], [127, 127, 0, 60.0],
                           [0, 0, 1, 1999.0]])
        for c in range(11):
            np.save(os.path.join(trial, f"{c}.npy"), events)
        self.list = os.path.join(self.tmp.name, "trials.txt")
        with open(self.list, "w") as file:
            file.write("user01_led.aedat\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_blocks_bins_and_windows(self):
        fine = fine_spikes(os.path.join(self.tmp.name, "user01_led", "3.npy"), 1000)
        self.assertEqual(fine.shape, (2, 16, 16, int(1000 / FINE_MS)))
        self.assertEqual(int(fine.sum()), 2)                 # two blocks; the late event dropped
        self.assertEqual(int(fine[1, 1, 2, 0]), 1)           # (x 17, y 9) -> row 1, column 2
        self.assertEqual(int(fine[0, 15, 15, 5]), 1)         # 60 ms -> fine bin 5 (55 ms after the first)
        steps = window(fine, 0, 3, 50.0)
        self.assertEqual(steps.shape, (2, 16, 16, 3))
        self.assertEqual(int(steps[0, 15, 15, 1]), 1)        # 55 ms falls in the second 50 ms step
        self.assertTrue(set(steps.unique().tolist()) <= {0.0, 1.0})

    def test_ten_classes_and_folding(self):
        params = models.load_params(SPEC.path(SPEC.params_yaml))
        data = Gesture16Dataset(self.tmp.name + os.sep, self.list, 50.0, 1500)
        self.assertEqual(len(data), CLASSES)
        _, spikes, desired, label = data[CLASSES - 1]
        self.assertEqual(label, CLASSES - 1)
        self.assertEqual(spikes.shape, (2, 16, 16, 30))
        self.assertEqual(desired.shape, (CLASSES, 1, 1, 1))
        torch.manual_seed(0)
        net = Gesture16Network(params, batchnorm=True)
        net.BN1.running_mean.uniform_(-0.2, 0.2)
        net.BN1.running_var.uniform_(0.5, 1.5)
        net.eval()
        folded = Gesture16Network(params)
        folded.load_state_dict(net.folded_state_dict())
        x = (torch.rand(2, 2, 16, 16, 30) < 0.2).float()
        with torch.no_grad():
            self.assertTrue(torch.equal(folded.eval()(x), net(x)))
            self.assertEqual(net(x).shape, (2, CLASSES, 1, 1, 30))


if __name__ == "__main__":
    unittest.main()


class NMNIST17Tests(unittest.TestCase):
    def test_blocks_window_and_folding(self):
        from models.nmnist17 import NMNIST17Dataset, NMNIST17Network, fine_spikes
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "Test"))
            # Two events: (x 5, y 3, p 1, 2 ms) -> block (2, 1), bin 0; (33, 33, p 0, 27 ms) -> (16, 16), bin 5.
            raw = np.zeros((2, 5), np.uint8)
            for row, (x, y, p, t_us) in enumerate(((5, 3, 1, 2000), (33, 33, 0, 27000))):
                raw[row] = (x, y, (p << 7) | ((t_us >> 16) & 0x7F), (t_us >> 8) & 0xFF, t_us & 0xFF)
            raw.tofile(os.path.join(root, "Test", "00007.bin"))
            with open(os.path.join(root, "Test.txt"), "w") as file:
                file.write("#sample label\n7 4\n8 1\n")   # header skipped; sample 8 has no file
            fine = fine_spikes(os.path.join(root, "Test", "00007.bin"), 100)
            self.assertEqual(fine.shape, (2, 17, 17, 20))
            self.assertEqual((int(fine[1, 1, 2, 0]), int(fine[0, 16, 16, 5])), (1, 1))
            data = NMNIST17Dataset(os.path.join(root, "Test") + os.sep,
                                   os.path.join(root, "Test.txt"), 10.0, 100)
            self.assertEqual(len(data), 1)
            _, spikes, _, label = data[0]
            self.assertEqual((tuple(spikes.shape), label), ((2, 17, 17, 10), 4))
            self.assertEqual(int(spikes[0, 16, 16, 2]), 1)  # 27 ms -> third 10 ms step
        params = models.load_params(models.get_spec("nmnist17").path("models/nmnist17.yaml"))
        net = NMNIST17Network(params, batchnorm=True).eval()
        folded = NMNIST17Network(params)
        folded.load_state_dict(net.folded_state_dict())
        x = (torch.rand(2, 2, 17, 17, 10) < 0.2).float()
        with torch.no_grad():
            self.assertTrue(torch.equal(folded.eval()(x), net(x)))
