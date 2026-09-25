"""Software SNN inference with per-layer CIM hardware estimation.

Runs the pretrained network on the test set, hooks the input of every
weighted (conv/dense) layer, and feeds those spikes to the hardware estimator
of each requested architecture. Hardware estimation does not replace the
software forward pass or simulate analog accuracy.
"""
import os

import torch
from torch.utils.data import DataLoader

import models
from hardware import (C3HardwareConfig, ConvHardwareConfig, HardwareMetrics,
                      calculate_c3_metrics, calculate_conv_metrics)
from hardware.mapping import quantize_weights
from evaluation.report import export_results, format_final_table, setup_logger

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LOG_DIR = os.path.join(REPO_ROOT, "logs")

# Which estimator handles which hardware config type.
ESTIMATORS = {ConvHardwareConfig: calculate_conv_metrics,
              C3HardwareConfig: calculate_c3_metrics}


class ActivationHook:
    """Stores the input tensor of each hooked layer during a forward pass."""

    def __init__(self):
        self.inputs = {}
        self.handles = []

    def register(self, key, module):
        def capture(_module, inputs, _output):
            self.inputs[key] = inputs[0].detach()
        self.handles.append(module.register_forward_hook(capture))

    def clear(self):
        self.inputs.clear()

    def remove_all(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def evaluate(model, architectures, *, batch_size=1, max_batches=None,
             weight_bits=0, quant_std_step=2, quant_decimals=1,
             num_workers=4, log_dir=DEFAULT_LOG_DIR):
    """Run inference and hardware estimation.

    model: "nmnist" or "gesture" (see models/__init__.py).
    architectures: {name: ConvHardwareConfig | C3HardwareConfig}.
    max_batches: None for the full test set.
    weight_bits: 0 keeps trained weights; >0 quantizes them before inference.

    Returns (accuracy_percent, {name: network HardwareMetrics per inference}).
    """
    import slayerSNN as snn  # type: ignore  # imported here so hardware-only use needs no slayer

    if batch_size <= 0 or (max_batches is not None and max_batches <= 0):
        raise ValueError("batch_size and max_batches must be positive")
    if not architectures:
        raise ValueError("Define at least one hardware architecture")
    for name, config in architectures.items():
        if type(config) not in ESTIMATORS:
            raise TypeError(f"{name}: unsupported hardware config {type(config).__name__}")
        if config.temporal_map:
            raise ValueError("Only spatial-parallel mapping is supported")

    spec = models.get_spec(model)
    if spec.max_batch_size is not None:
        batch_size = min(batch_size, spec.max_batch_size)
    logger = setup_logger(log_dir, spec.name, batch_size,
                          "all" if max_batches is None else max_batches, weight_bits)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = models.load_pretrained(spec, device).to(device)
    net_params = snn.params(spec.path(spec.params_yaml))
    net.eval()

    layer_names = [name for name, _ in spec.layers]
    padding = dict(spec.layers)
    act_hook = ActivationHook()
    for name in layer_names:
        act_hook.register(name, getattr(net, name))

    # Optional weight quantization, then the 4-D weights seen by the hardware.
    levels, hw_weights = {}, {}
    with torch.no_grad():
        for name in layer_names:
            module = getattr(net, name)
            if weight_bits > 0:
                quantized, levels[name] = quantize_weights(
                    module.weight, spec.std_quantization, weight_bits,
                    quant_std_step, quant_decimals)
                module.weight = torch.nn.Parameter(quantized)
                levels[name] = levels[name].to(device)
            w_hw = module.weight.clone().detach()
            if w_hw.dim() == 5:  # slayer weights carry a trailing time axis
                w_hw = w_hw[:, :, :, :, 0]
            hw_weights[name] = w_hw.to(device)
            if weight_bits == 0:
                levels[name] = torch.tensor([torch.min(w_hw), torch.max(w_hw)]).to(device)

    paths = net_params["training"]["path"]
    test_set = spec.dataset_class(
        data_path=os.path.join(REPO_ROOT, paths["dir_test"]),
        samples_file=os.path.join(REPO_ROOT, paths["list_test"]),
        sampling_time=net_params["simulation"]["Ts"],
        sample_length=net_params["simulation"]["tSample"],
    )
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers)

    error = snn.loss(net_params).to(device)
    stats = snn.learningStats.learningStats()
    per_layer = {arch: {name: HardwareMetrics() for name in layer_names}
                 for arch in architectures}

    logger.info(f"{' & '.join(architectures)} hardware @ {spec.display_name}")
    logger.info("Starting combined inference and hardware metrics evaluation...\n")

    for b_idx, (_, x_in, target, label) in enumerate(test_loader):
        if max_batches is not None and b_idx == max_batches:
            break
        x_in, target = x_in.to(device), target.to(device)

        act_hook.clear()
        with torch.no_grad():
            output = net(x_in)

        # Software metrics
        stats.testing.correctSamples += torch.sum(snn.predict.getClass(output) == label).item()
        stats.testing.numSamples += len(label)
        stats.testing.lossSum += error.numSpikes(output, target).cpu().item()
        test_acc = round(100 * stats.testing.correctSamples / stats.testing.numSamples, 2)
        if (b_idx + 1) % 10 == 0 or b_idx == 0:
            logger.info(f"Batch: {b_idx + 1}; Loss: {round(stats.testing.loss(), 2)}; "
                        f"Accuracy: {test_acc}%")

        # Hardware metrics: any nonzero layer input counts as a binary spike.
        with torch.no_grad():
            for name in layer_names:
                x = act_hook.inputs[name].bool().float()
                for arch, config in architectures.items():
                    batch_metrics = ESTIMATORS[type(config)](
                        config, x, hw_weights[name], levels[name], padding[name])
                    per_layer[arch][name].add(batch_metrics)
    act_hook.remove_all()

    logger.info(f"\nInference done. Final Accuracy: {test_acc}%\n")

    results = {}
    for arch, config in architectures.items():
        logger.info("=" * 40)
        logger.info(f"{arch.upper()} HARDWARE METRICS SUMMARY")
        logger.info("=" * 40)
        total = HardwareMetrics()
        for i, name in enumerate(layer_names, 1):
            logger.info(f"{i}.) Layer {name}")
            norm_metrics = per_layer[arch][name].normalize()  # per image
            logger.info(norm_metrics.format_summary())
            total.add(norm_metrics, distinct_layer=True)
        logger.info(f"\nOVERALL NETWORK METRICS ({arch}):")
        logger.info(total.format_summary())
        export_results(log_dir, spec.name, spec.display_name, arch, config, weight_bits,
                       quant_std_step, quant_decimals, per_layer[arch], total, logger)
        results[arch] = total

    logger.info("\n" + "=" * 40 + "\nFINAL METRICS (per inference)\n" + "=" * 40)
    logger.info(format_final_table(results, test_acc))
    return test_acc, results
