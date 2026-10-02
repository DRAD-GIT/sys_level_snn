"""IBM DVS Gesture downscaled to 16x16 with 10 classes (the "other gestures"
class left out): the version DS-CIM (TCAS-I'24) and ReckOn (ISSCC'22)
evaluate. A small fully connected SNN with LIF neurons: 512 inputs (2
polarities x 16 x 16) - 512 - 10.

Input: each 8x8 block of the 128x128 sensor is one input pixel; time steps of
Ts ms (models/gesture16.yaml); an input is 1 in a step if its block had an
event of that polarity in it (binary spikes into every layer). The test set
uses each gesture's first tSample ms; training (tools/train_gesture16.py)
also uses later windows of the same length.

Trained with BatchNorm after the hidden layer; the checkpoint has it folded
into SF1's weights and bias (a constant current into each neuron).
"""
import os

import numpy as np
import torch

from .base import ModelSpec, NNetwork
from .events import read_npy_events
from .lif import LIFLayer, fold_batchnorm

SENSOR = 128
SIZE = 16                      # downscaled width and height
BLOCK = SENSOR // SIZE         # 8x8 sensor pixels per input
CLASSES = 10                   # the 11th class ("other gestures") is left out
HIDDEN = 512                   # default hidden width (the YAML's network: hidden)
FINE_MS = 10.0                 # resolution kept for drawing training windows


def fine_spikes(path, length_ms):
    """A gesture's events (models.events.read_npy_events) as a uint8 tensor
    [2, 16, 16, length_ms / FINE_MS]: 1 where an 8x8 block had an event of
    that polarity in a FINE_MS bin, from the gesture's first event on."""
    events = read_npy_events(path)
    bins = int(round(length_ms / FINE_MS))
    tensor = torch.zeros((2, SIZE, SIZE, bins), dtype=torch.uint8)
    if not len(events.t_ms):
        return tensor
    t = np.floor((events.t_ms - events.t_ms.min()) / FINE_MS).astype(np.int64)
    x, y, p = events.x // BLOCK, events.y // BLOCK, events.p
    valid = (p >= 0) & (p < 2) & (x >= 0) & (x < SIZE) & (y >= 0) & (y < SIZE) & (t < bins)
    index = tuple(torch.from_numpy(a[valid].astype(np.int64)) for a in (p, y, x, t))
    tensor[index] = 1
    return tensor


def window(fine, start_bin, steps, step_ms):
    """Binary spikes [..., steps] of `steps` time steps of step_ms, starting
    at fine bin `start_bin` of a fine_spikes tensor (any event in a step = 1)."""
    per_step = int(round(step_ms / FINE_MS))
    part = fine[..., start_bin:start_bin + steps * per_step].float()
    if part.shape[-1] < steps * per_step:
        part = torch.nn.functional.pad(part, (0, steps * per_step - part.shape[-1]))
    return part.reshape(*part.shape[:-1], steps, per_step).amax(-1)


def gesture_files(data_path, samples_file):
    """(file, class) of every gesture of the trials in samples_file
    (data_path/<trial>/<class>.npy, classes 0-9) that exists: DVS Gesture's
    errata lists trials that lack some gestures."""
    trials = [line.split()[0].split(".")[0] for line in open(samples_file) if line.strip()]
    files = [(os.path.join(data_path, t, f"{c}.npy"), c) for t in trials for c in range(CLASSES)]
    return [(f, c) for f, c in files if os.path.exists(f)]


class Gesture16Dataset(torch.utils.data.Dataset):
    """The test set: every gesture (classes 0-9) of the trials listed in
    samples_file that exists (data_path/<trial>/<class>.npy), its first
    sample_length ms in time steps of sampling_time ms."""

    def __init__(self, data_path, samples_file, sampling_time, sample_length):
        self.files = gesture_files(data_path, samples_file)
        self.step_ms = float(sampling_time)
        self.steps = int(round(sample_length / sampling_time))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path, label = self.files[index]
        fine = fine_spikes(path, self.steps * self.step_ms)
        desired = torch.zeros((CLASSES, 1, 1, 1))
        desired[label, ...] = 1
        return index + 1, window(fine, 0, self.steps, self.step_ms), desired, label


class Gesture16Network(NNetwork):
    """512 inputs (2 x 16 x 16) - SF1 hidden LIF (network: hidden in the YAML,
    default 512) - SF2 10 LIF. batchnorm=True
    adds a BatchNorm after SF1 (training); the evaluated checkpoint has it
    folded (batchnorm=False)."""

    def __init__(self, net_params: dict, do_enable=False, backend=None, batchnorm=False):
        super().__init__(net_params, backend or LIFLayer)
        self.batchnorm = batchnorm
        hidden = int(net_params.get("network", {}).get("hidden", HIDDEN))
        self.SF1 = self.slayer.dense((SIZE, SIZE, 2), hidden, bias=not batchnorm)
        if batchnorm:
            self.BN1 = torch.nn.BatchNorm3d(hidden)
        self.SF2 = self.slayer.dense(hidden, CLASSES)
        self.SD = self.slayer.dropout(0.2 if do_enable else 0.0)

    def forward(self, s_in):
        current = self.SF1(s_in)                                       # 512 x 1 x 1
        if self.batchnorm:
            current = self.BN1(current)
        s = self.SD(self.slayer.spike(self.slayer.psp(current)))
        return self.slayer.spike(self.slayer.psp(self.SF2(s)))         # 10 x 1 x 1

    def folded_state_dict(self):
        """State dict of the same network with BN1 folded into SF1 (for
        Gesture16Network(batchnorm=False))."""
        if not self.batchnorm:
            return self.state_dict()
        state = {k: v for k, v in self.state_dict().items()
                 if not k.startswith("BN1") and not k.startswith("SF1")}
        state["SF1.weight"], state["SF1.bias"] = fold_batchnorm(self.SF1, self.BN1)
        return state


SPEC = ModelSpec(
    name="gesture16",
    display_name="IBM-Gesture 16x16",
    network_class=Gesture16Network,
    dataset_class=Gesture16Dataset,
    checkpoint="pretrained/gesture16.pth",
    params_yaml="models/gesture16.yaml",
    layers=("SF1", "SF2"),
    batch_size=50,
    dataset_folders=("Gesture", "DVS-Gesture"),
    sources=("models/lif.py", "models/events.py", "models/gesture16.py"),
)
