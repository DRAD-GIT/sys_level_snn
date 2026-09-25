import torch
from .base import ModelSpec, NDataset, NNetwork
from .events import read_nmnist_bin

class NMNISTDataset(NDataset):
    sensor_shape = (2, 34, 34)
    read_events = staticmethod(read_nmnist_bin)

    def event_file(self, index):
        return f"{self.path}{int(self.samples[index, 0]):05}.bin"

    def __getitem__(self, index):
        input_index = int(self.samples[index, 0])
        class_label = int(self.samples[index, 1])

        spikes_in = self.read_events(self.event_file(index)) \
            .to_spike_tensor((*self.sensor_shape, self.n_time_bins), self.sampling_time)
        desired_class = torch.zeros((10, 1, 1, 1))
        desired_class[class_label, ...] = 1

        return input_index, spikes_in, desired_class, class_label


class NMNISTNetwork(NNetwork):
    def __init__(self, net_params: dict, do_enable=False, backend=None):
        super(NMNISTNetwork, self).__init__(net_params, backend)

        self.SC1 = self.slayer.conv(2, 16, 5, padding=1) #in_channel, out_channel, kernel, stride=1
        self.SC2 = self.slayer.conv(16, 32, 3, padding=1)
        self.SC3 = self.slayer.conv(32, 64, 3, padding=1)

        self.SP1 = self.slayer.pool(2)
        self.SP2 = self.slayer.pool(2)

        self.SF1 = self.slayer.dense((8, 8, 64), 10)

        self.SDC = self.slayer.dropout(0.10 if do_enable else 0.0)
        self.SDF = self.slayer.dropout(0.25 if do_enable else 0.0)

    def forward(self, s_in):
        s_out = self.slayer.spike(self.slayer.psp(self.SC1(s_in)))   # 16, 32, 32
        s_out = self.slayer.spike(self.slayer.psp(self.SP1(s_out)))  # 16, 16, 16

        s_out = self.SDC(s_out)
        s_out = self.slayer.spike(self.slayer.psp(self.SC2(s_out)))  # 32, 16, 16
        s_out = self.slayer.spike(self.slayer.psp(self.SP2(s_out)))  # 32, 8,  8

        s_out = self.SDC(s_out)
        s_out = self.slayer.spike(self.slayer.psp(self.SC3(s_out)))  # 64, 8,  8

        s_out = self.SDF(s_out)
        s_out = self.slayer.spike(self.slayer.psp(self.SF1(s_out)))  # 10

        return s_out


class LeNetNetwork(NNetwork):
    def __init__(self, net_params: dict, do_enable=False, backend=None):
        super(LeNetNetwork, self).__init__(net_params, backend)

        self.SC1 = self.slayer.conv(2, 6, 7)
        self.SC2 = self.slayer.conv(6, 16, 5)
        self.SC3 = self.slayer.conv(16, 120, 5)

        self.SP1 = self.slayer.pool(2)

        self.SP2 = self.slayer.pool(2)

        self.SF1 = self.slayer.dense(120, 84)
        self.SF2 = self.slayer.dense(84, 10)

        self.SDC = self.slayer.dropout(0.10 if do_enable else 0.0)
        self.SDF = self.slayer.dropout(0.25 if do_enable else 0.0)

    def forward(self, s_in):
        s_out = self.slayer.spike(self.slayer.psp(self.SC1(s_in)))   # 6, 28, 28
        s_out = self.slayer.spike(self.slayer.psp(self.SP1(s_out)))  # 6, 14, 14

        s_out = self.SDC(s_out)
        s_out = self.slayer.spike(self.slayer.psp(self.SC2(s_out)))  # 16, 10, 10
        s_out = self.slayer.spike(self.slayer.psp(self.SP2(s_out)))  # 16,  5,  5

        s_out = self.SDC(s_out)
        s_out = self.slayer.spike(self.slayer.psp(self.SC3(s_out)))  # 120, 1, 1

        s_out = self.SDF(s_out)
        s_out = self.slayer.spike(self.slayer.psp(self.SF1(s_out)))  # 84

        s_out = self.SDF(s_out)
        s_out = self.slayer.spike(self.slayer.psp(self.SF2(s_out)))  # 10

        return s_out


SPEC = ModelSpec(
    name="nmnist",
    display_name="NMNIST",
    network_class=LeNetNetwork,
    dataset_class=NMNISTDataset,
    checkpoint="pretrained/nmnist_lenet.pth",
    params_yaml="models/nmnist.yaml",
    layers=(("SC1", 0), ("SC2", 0), ("SC3", 0), ("SF1", 0), ("SF2", 0)),
    std_quantization=True,
)
