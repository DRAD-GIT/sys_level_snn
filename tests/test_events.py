"""Event readers and binning, against hand-built files."""
import os
import tempfile
import unittest

import numpy as np
import torch

from models.events import Events, read_nmnist_bin, read_npy_events


def encode_nmnist(events):
    """Encode (x, y, p, t_us) tuples in the 5-byte N-MNIST format."""
    data = bytearray()
    for x, y, p, t in events:
        data += bytes([x, y, (p << 7) | ((t >> 16) & 0x7F), (t >> 8) & 0xFF, t & 0xFF])
    return bytes(data)


class EventTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.dir.cleanup()

    def test_nmnist_bin_decoding_and_binning(self):
        path = os.path.join(self.dir.name, "00000.bin")
        with open(path, "wb") as f:
            f.write(encode_nmnist([(3, 5, 1, 1500), (33, 0, 0, 2500), (34, 1, 1, 0),
                                   (0, 0, 0, 8_388_607)]))
        events = read_nmnist_bin(path)
        self.assertEqual(events.x.tolist(), [3, 33, 34, 0])
        self.assertEqual(events.y.tolist(), [5, 0, 1, 0])
        self.assertEqual(events.p.tolist(), [1, 0, 1, 0])
        self.assertEqual(events.t_ms.tolist(), [1.5, 2.5, 0.0, 8388.607])
        spikes = events.to_spike_tensor((2, 34, 34, 300), 1.0)
        # Round half to even: 1.5 ms -> bin 2, 2.5 ms -> bin 2. x=34 and
        # t=8388 ms are outside the tensor and dropped.
        self.assertEqual(spikes.sum().item(), 2)
        self.assertEqual(spikes[1, 5, 3, 2].item(), 1)
        self.assertEqual(spikes[0, 0, 33, 2].item(), 1)

    def test_polarity_shifted_to_zero_and_or_binning(self):
        # Only ON events: SLAYER shifts polarity by its minimum, so they land
        # in channel 0. Two events in one bin give a single 1/Ts entry.
        events = Events(np.array([1, 1]), np.array([2, 2]), np.array([1, 1]), np.array([0.4, 0.2]))
        spikes = events.to_spike_tensor((2, 4, 4, 3), 2.0)
        self.assertEqual(spikes[0, 2, 1, 0].item(), 0.5)
        self.assertEqual(spikes.sum().item(), 0.5)

    def test_npy_reader_columns_and_units(self):
        path = os.path.join(self.dir.name, "0.npy")
        np.save(path, np.array([[10.0, 20.0, 1.0, 7.4], [127.0, 3.0, 0.0, 12.0]]))
        events = read_npy_events(path)
        self.assertEqual(events.x.tolist(), [10, 127])
        self.assertEqual(events.y.tolist(), [20, 3])
        self.assertEqual(events.p.tolist(), [1, 0])
        self.assertTrue(np.allclose(events.t_ms, [7.4, 12.0]))
        spikes = events.to_spike_tensor((2, 128, 128, 20), 1.0)
        self.assertEqual(spikes[1, 20, 10, 7].item(), 1)
        self.assertEqual(spikes[0, 3, 127, 12].item(), 1)
        self.assertEqual(spikes.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
