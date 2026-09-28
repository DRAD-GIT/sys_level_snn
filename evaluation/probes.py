"""Forward probes on a network: the input spikes of each profiled layer and
the spikes its LIF neurons emit."""
import torch


class LayerProbe:
    """Records, for the named weighted layers, their input tensor and the
    spikes of the neuron stage that follows them: the batch total
    (`output_spikes`), per sample (`output_counts`), and with keep_outputs the
    spike tensor itself (`outputs`).

    A layer's neurons are the first ``net.slayer.spike`` call after the
    layer runs; any other conv-like module (e.g. pooling) in between resets
    the attribution.
    """

    def __init__(self, net, layer_names, keep_outputs=False):
        self.inputs, self.output_spikes, self.output_counts, self.outputs = {}, {}, {}, {}
        self._current = None
        self._handles = []
        self._net = net
        names = set(layer_names)
        for name, module in net.named_children():
            if isinstance(module, torch.nn.Conv3d):
                self._handles.append(module.register_forward_hook(self._hook(name, name in names)))
        spike = net.slayer.spike

        def counting_spike(membrane):
            spikes = spike(membrane)
            if self._current is not None:
                per_sample = (spikes != 0).reshape(len(spikes), -1).sum(1)
                self.output_counts[self._current] = per_sample.cpu()
                self.output_spikes[self._current] = int(per_sample.sum())
                if keep_outputs:
                    self.outputs[self._current] = spikes.detach()
                self._current = None
            return spikes
        net.slayer.spike = counting_spike

    def _hook(self, name, profiled):
        def record(_module, inputs, _output):
            self._current = name if profiled else None
            if profiled:
                self.inputs[name] = inputs[0].detach()
        return record

    def clear(self):
        for store in (self.inputs, self.output_spikes, self.output_counts, self.outputs):
            store.clear()

    def remove(self):
        for handle in self._handles:
            handle.remove()
        del self._net.slayer.spike  # restore the class method
