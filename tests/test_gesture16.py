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
