import numpy as np
import torch
import slayerSNN as snn # type: ignore 

class NDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, samples_file: str, sampling_time: int, sample_length: int):
        self.path = data_path
        self.samples = np.loadtxt(samples_file, dtype='str')
        self.sampling_time = sampling_time
        self.n_time_bins = int(sample_length / sampling_time)

    def __len__(self):
        return self.samples.shape[0]


class NNetwork(torch.nn.Module):
    def __init__(self, net_params: snn.params):
        super(NNetwork, self).__init__()

        self.slayer = snn.layer(net_params['neuron'], net_params['simulation'])
