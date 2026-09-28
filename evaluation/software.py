"""Software classification (replaces slayerSNN's predict)."""
import torch


def predict_class(output):
    """Class with the most output spikes over the whole simulation."""
    counts = torch.sum(output, 4, keepdim=True).cpu()
    return torch.max(counts.reshape(counts.shape[0], -1), 1)[1]
