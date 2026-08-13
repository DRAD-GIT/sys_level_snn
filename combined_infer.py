import argparse
import logging
import os
import sys
import warnings

import slayerSNN as snn  # type: ignore
import torch

# Import our new clean hardware evaluation logic
from clean_hardware import (
    C3HardwareConfig,
    ConvHardwareConfig,
    HardwareMetrics,
    calculate_c3_metrics,
    calculate_conv_metrics,
    quantize_weights,
)
from torch.utils.data import DataLoader

# Add original script path to sys.path so we can import SlayerSNN, demo
script_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Final SNN script")
)
sys.path.append(script_dir)

warnings.filterwarnings("ignore", category=UserWarning)


class ActivationHook:
    """Class-based hook to store intermediate layer inputs, replacing global variables."""

    def __init__(self):
        self.inputs = []
        self.handles = []

    def hook(self, module, input, output):
        self.inputs.append(input[0])  # store the actual tensor

    def register(self, module):
        self.handles.append(module.register_forward_hook(self.hook))

    def clear(self):
        self.inputs.clear()

    def remove_all(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combined Inference & Hardware Metrics"
    )
    parser.add_argument(
        "--model", default="", type=str, help="SNN Model name (nmnist or gesture)"
    )
    parser.add_argument("-b", default=12, type=int, help="Batch size")
    parser.add_argument(
        "-n", default=0, type=int, help="Bit-Precision (0 for no quantization)"
    )
    parser.add_argument("-k", default=2, type=int, help="Standard Deviation step")
    parser.add_argument(
        "-d", default=1, type=int, help="Quantization levels decimal precision"
    )
    parser.add_argument(
        "--batches", default=-1, type=int, help="Max no. of batches to run (-1 for all)"
    )
    return parser.parse_args()


def get_module(net, layer_name):
    """Helper to get module attribute instead of using exec"""
    if layer_name.startswith("net."):
        layer_name = layer_name[4:]
    parts = layer_name.split(".")
    mod = net
    for p in parts:
        mod = getattr(mod, p)
    return mod


def set_module_weight(net, layer_name, new_weight):
    """Helper to set module weight parameter instead of using exec"""
    if layer_name.startswith("net."):
        layer_name = layer_name[4:]
    parts = layer_name.split(".")
    mod = net
    for p in parts[:-1]:
        mod = getattr(mod, p)
    setattr(mod, parts[-1], torch.nn.Parameter(new_weight))


def setup_logger(
    model_name: str, batch_size: int, batches: str, nbit: int
) -> logging.Logger:
    """Setup standard Python logging to output to both console and a log file."""
    logger = logging.getLogger("combined_infer")
    logger.setLevel(logging.INFO)

    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(
        log_dir, f"{model_name}_B{batch_size}_N{batches}_b{nbit}.txt"
    )

    file_handler = logging.FileHandler(log_file_path, mode="w")
    console_handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def main():
    args = parse_args()
    model_name = args.model or input("Enter model name: ")
    batches = args.batches if args.batches != -1 else "all"

    logger = setup_logger(model_name, args.b, batches, args.n)

    # We define standard configs using dataclasses for hardware logic
    conv_config = ConvHardwareConfig()
    c3_config = C3HardwareConfig()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if model_name == "nmnist":
        from demo.nets.nmnist import NMNISTDataset as Dataset

        disp_model_name = "NMNIST"
        net = torch.load(
            os.path.join(script_dir, "NMNIST_SNN/nmnist-lenet_net2.pt")
        ).to(device)
        net_params = snn.params(os.path.join(script_dir, "NMNIST_SNN/nmnist.yaml"))
        batch_size = args.b
        weight_list = [1, 4, 7, 9, 11]
        padding = [0, 0, 0, 0, 0]
        layer_dict = {
            1: "net.SC1.weight",
            4: "net.SC2.weight",
            7: "net.SC3.weight",
            9: "net.SF1.weight",
            11: "net.SF2.weight",
        }
    elif model_name == "gesture":
        from demo.nets.gesture import GestureDataset as Dataset

        disp_model_name = "IBM-Gesture"
        net = torch.load(
            os.path.join(script_dir, "Gesture_SNN/gesture-do_net4.pt"),
            weights_only=False,
        ).to(device)
        net_params = snn.params(os.path.join(script_dir, "Gesture_SNN/gesture.yaml"))
        batch_size = min(2, args.b)
        weight_list = [2, 5, 8, 10]
        padding = [2, 1, 0, 0]
        layer_dict = {
            2: "net.SC1.weight",
            5: "net.SC2.weight",
            8: "net.SF1.weight",
            10: "net.SF2.weight",
        }
    else:
        logger.error(f"Unknown model: {model_name}")
        sys.exit(1)

    act_hook = ActivationHook()
    for layer in net.children():
        act_hook.register(layer)

    net.eval()

    levels_dict = {}
    if args.n > 0:
        with torch.no_grad():
            for i in weight_list:
                orig_weight = get_module(net, layer_dict[i])
                x, wl = quantize_weights(
                    orig_weight, model_name, args.n, args.k, args.d
                )
                set_module_weight(net, layer_dict[i], x)
                levels_dict[i] = wl.to(device)

    # Pre-compute hw_weights for inference to avoid repeating it per batch
    hw_weights = {}
    for w_layer_idx in weight_list:
        layer_weight = get_module(net, layer_dict[w_layer_idx])
        w_shape = layer_weight.shape
        w_hw = layer_weight.clone().detach()
        if len(w_shape) == 5:
            w_hw = w_hw[:, :, :, :, 0]
        hw_weights[w_layer_idx] = w_hw.to(device)

        # Ensure levels fallback exists if no quantization is used
        if args.n == 0:
            min_w = torch.min(w_hw)
            max_w = torch.max(w_hw)
            levels_dict[w_layer_idx] = torch.tensor([min_w, max_w]).to(device)

    # Dataset loading
    net_params["training"]["path"]["dir_test"] = os.path.join(
        script_dir, net_params["training"]["path"]["dir_test"]
    )
    net_params["training"]["path"]["list_test"] = os.path.join(
        script_dir, net_params["training"]["path"]["list_test"]
    )

    test_set = Dataset(
        data_path=net_params["training"]["path"]["dir_test"],
        samples_file=net_params["training"]["path"]["list_test"],
        sampling_time=net_params["simulation"]["Ts"],
        sample_length=net_params["simulation"]["tSample"],
    )
    test_loader = DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=4
    )

    error = snn.loss(net_params).to(device)
    stats = snn.learningStats.learningStats()

    # Hardware Objects Metrics initialization
    metrics_conv = {idx: HardwareMetrics() for idx in weight_list}
    metrics_c3 = {idx: HardwareMetrics() for idx in weight_list}

    logger.info(f"Conventional & C3CIM Hardware @ {disp_model_name}")
    logger.info("Starting combined inference and hardware metrics evaluation...\n")

    for b_idx, (_, x_in, target, label) in enumerate(test_loader):
        if b_idx == args.batches:
            break

        x_in, target = x_in.to(device), target.to(device)

        act_hook.clear()  # Reset before forward
        output = net(x_in)

        # Software metrics
        stats.testing.correctSamples += torch.sum(
            snn.predict.getClass(output) == label
        ).item()
        stats.testing.numSamples += len(label)
        loss = error.numSpikes(output, target)
        stats.testing.lossSum += loss.cpu().item()

        test_loss = round(stats.testing.loss(), 2)
        test_acc = round(stats.testing.accuracy() * 100, 2)

        if (b_idx + 1) % 10 == 0 or b_idx == 0:
            logger.info(f"Batch: {b_idx + 1}; Loss: {test_loss}; Accuracy: {test_acc}%")

        # Hardware metrics for this batch
        with torch.no_grad():
            for idx, w_layer_idx in enumerate(weight_list):
                actual_input_idx = w_layer_idx - 1
                x = act_hook.inputs[actual_input_idx].bool().float()

                w_hw = hw_weights[w_layer_idx]
                levels = levels_dict[w_layer_idx]
                p = padding[idx]

                # Accumulate conv hardware metrics
                batch_conv = calculate_conv_metrics(conv_config, x, w_hw, levels, p)
                metrics_conv[w_layer_idx].add(batch_conv)

                # Accumulate C3 hardware metrics
                batch_c3 = calculate_c3_metrics(c3_config, x, w_hw, levels, p)
                metrics_c3[w_layer_idx].add(batch_c3)

    logger.info(f"\nInference done. Final Accuracy: {test_acc}%\n")

    logger.info("=" * 40)
    logger.info("CONVENTIONAL HARDWARE METRICS SUMMARY")
    logger.info("=" * 40)

    total_conv = HardwareMetrics()
    for idx, w_layer_idx in enumerate(weight_list):
        logger.info(f"{idx + 1}.) Layer {w_layer_idx} : {layer_dict[w_layer_idx]}")
        # Normalize metrics per image
        norm_metrics = metrics_conv[w_layer_idx].normalize()
        logger.info(norm_metrics.format_summary())
        total_conv.add(norm_metrics)

    logger.info("\nOVERALL NETWORK METRICS (CONV):")
    logger.info(total_conv.format_summary())

    logger.info("=" * 40)
    logger.info("C3CIM HARDWARE METRICS SUMMARY")
    logger.info("=" * 40)

    total_c3 = HardwareMetrics()
    for idx, w_layer_idx in enumerate(weight_list):
        logger.info(f"{idx + 1}.) Layer {w_layer_idx} : {layer_dict[w_layer_idx]}")
        # Normalize metrics per image
        norm_metrics = metrics_c3[w_layer_idx].normalize()
        logger.info(norm_metrics.format_summary())
        total_c3.add(norm_metrics)

    logger.info("\nOVERALL NETWORK METRICS (C3CIM):")
    logger.info(total_c3.format_summary())


if __name__ == "__main__":
    main()
