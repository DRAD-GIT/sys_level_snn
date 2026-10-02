"""N-MNIST downscaled to 17x17, the input ANP-I (JSSC'24) uses: a small fully
connected SNN with LIF neurons, 578 inputs (2 polarities x 17 x 17) - 512 -
10.

Input: each 2x2 block of the 34x34 sensor is one input pixel; time steps of
Ts ms over the sample's first tSample ms (models/nmnist17.yaml); an input is
1 in a step if its block had an event of that polarity in it (binary spikes
into every layer). Trained (tools/train_event_snn.py --model nmnist17) with
BatchNorm after the hidden layer, folded into SF1's weights and bias in the
checkpoint.
"""
import os

import numpy as np
import torch

from .base import ModelSpec, NNetwork
from .events import read_nmnist_bin
from .lif import LIFLayer, fold_batchnorm

SENSOR = 34
SIZE = 17                      # downscaled width and height
BLOCK = SENSOR // SIZE         # 2x2 sensor pixels per input
CLASSES = 10
HIDDEN = 512                   # default hidden width (the YAML's network: hidden)
FINE_MS = 5.0                  # resolution kept for drawing training windows
KEEP_MS = 300.0                # an N-MNIST sample lasts 300 ms


def fine_spikes(path, length_ms):
    """A sample's events as a uint8 tensor [2, 17, 17, length_ms / FINE_MS]:
    1 where a 2x2 block had an event of that polarity in a FINE_MS bin."""
    events = read_nmnist_bin(path)
    bins = int(round(length_ms / FINE_MS))
    tensor = torch.zeros((2, SIZE, SIZE, bins), dtype=torch.uint8)
    t = np.floor(events.t_ms / FINE_MS).astype(np.int64)
    x, y, p = events.x // BLOCK, events.y // BLOCK, events.p
    valid = (p >= 0) & (p < 2) & (x >= 0) & (x < SIZE) & (y >= 0) & (y < SIZE) & (t >= 0) & (t < bins)
    index = tuple(torch.from_numpy(a[valid].astype(np.int64)) for a in (p, y, x, t))
    tensor[index] = 1
    return tensor


def window(fine, start_bin, steps, step_ms):
    """Binary spikes [..., steps] of `steps` time steps of step_ms (a multiple
    of FINE_MS), from fine bin `start_bin` of a fine_spikes tensor."""
    per_step = int(round(step_ms / FINE_MS))
    if per_step < 1 or abs(per_step * FINE_MS - step_ms) > 1e-6:
        raise ValueError(f"time step {step_ms:g} ms: use a multiple of {FINE_MS:g} ms")
    part = fine[..., start_bin:start_bin + steps * per_step].float()
    if part.shape[-1] < steps * per_step:
        part = torch.nn.functional.pad(part, (0, steps * per_step - part.shape[-1]))
    return part.reshape(*part.shape[:-1], steps, per_step).amax(-1)


def sample_files(data_path, samples_file):
    """(file, label) of every sample listed in samples_file ("<index> <label>"
    per line, "#" comments skipped; data_path/<index:05>.bin) that exists."""
    files = []
    for line in open(samples_file):
        fields = line.split("#", 1)[0].split()          # "#" starts a comment (np.loadtxt's rule)
        try:
            index, label = int(float(fields[0])), int(float(fields[1]))
        except (IndexError, ValueError):
            continue
        path = os.path.join(data_path, f"{index:05}.bin")
        if os.path.exists(path):
            files.append((path, label))
    return files


class NMNIST17Dataset(torch.utils.data.Dataset):
    """The test set: every sample in samples_file, its first sample_length ms
    in time steps of sampling_time ms."""

    def __init__(self, data_path, samples_file, sampling_time, sample_length):
        self.files = sample_files(data_path, samples_file)
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


class NMNIST17Network(NNetwork):
    """578 inputs (2 x 17 x 17) - SF1 hidden LIF (network: hidden, default
    512) - SF2 10 LIF. batchnorm=True adds a BatchNorm after SF1 (training);
    the evaluated checkpoint has it folded (batchnorm=False)."""

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
        current = self.SF1(s_in)
        if self.batchnorm:
            current = self.BN1(current)
        s = self.SD(self.slayer.spike(self.slayer.psp(current)))
        return self.slayer.spike(self.slayer.psp(self.SF2(s)))

    def folded_state_dict(self):
        if not self.batchnorm:
            return self.state_dict()
        state = {k: v for k, v in self.state_dict().items()
                 if not k.startswith("BN1") and not k.startswith("SF1")}
        state["SF1.weight"], state["SF1.bias"] = fold_batchnorm(self.SF1, self.BN1)
        return state


SPEC = ModelSpec(
    name="nmnist17",
    display_name="NMNIST 17x17",
    network_class=NMNIST17Network,
    dataset_class=NMNIST17Dataset,
    checkpoint="pretrained/nmnist17.pth",
    params_yaml="models/nmnist17.yaml",
    layers=("SF1", "SF2"),
    batch_size=500,
    dataset_folders=("N-MNIST", "NMNIST"),
    sources=("models/lif.py", "models/events.py", "models/nmnist17.py"),
)
