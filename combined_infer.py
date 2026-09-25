import argparse
import csv
import fcntl
import hashlib
import json
import logging
import os
import sys
import tempfile
import warnings
from dataclasses import asdict, fields

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

# All checkpoint/config assets are local. Dataset paths in the YAMLs are
# resolved relative to this directory, not the caller's working directory.
project_dir = os.path.dirname(os.path.abspath(__file__))
asset_dir = os.path.join(project_dir, "models")

warnings.filterwarnings("ignore", category=UserWarning)


class ActivationHook:
    """Class-based hook to store intermediate layer inputs, replacing global variables."""

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
    parser.add_argument(
        "--temporal-map",
        action="store_true",
        help="Enable temporal convolution mapping instead of fully spatially unrolled.",
    )
    parser.add_argument("--conv-config", type=str, help="JSON file overriding conventional hardware parameters")
    parser.add_argument("--c3-config", type=str, help="JSON file overriding C3 hardware parameters")
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


def load_config(cls, path):
    if path is None:
        return cls()
    with open(path, encoding="utf-8") as file:
        overrides = json.load(file)
    if not isinstance(overrides, dict):
        raise ValueError(f"{path}: expected a JSON object")
    valid = {field.name for field in fields(cls)}
    unknown = set(overrides) - valid
    if unknown:
        raise ValueError(f"{path}: unknown parameters: {sorted(unknown)}")
    if overrides.get("temporal_map", False):
        raise ValueError("Only spatial-parallel mapping is supported")
    # Shared validation covers scalar, registry and schedule configuration.
    return cls(**overrides)


def export_results(model_name, dataset_name, architecture, config, nbit,
                   quant_k, quant_d, layer_dict, per_layer, overall, logger):
    """Export exact per-component measurements and upsert a comparison row."""
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    config_data = asdict(config)
    fingerprint = hashlib.sha256(json.dumps(config_data, sort_keys=True).encode()).hexdigest()[:12]
    quant_id = f"k{quant_k}_d{quant_d}" if nbit else "raw"
    images = next(iter(per_layer.values())).images_processed
    json_path = os.path.join(
        log_dir, f"{model_name}_{architecture}_b{nbit}_{quant_id}_{fingerprint}_{images}images.json"
    )
    def serialize_metrics(metrics):
        data = asdict(metrics)
        data["groups"] = {name: asdict(group) for name, group in metrics.groups.items()}
        return data

    network_metrics = serialize_metrics(overall)
    network_metrics["images_processed"] = images
    from hardware_components import resolve_components
    resolved_components, schedule = resolve_components(config)
    result = {
        "schema_version": 2,
        "model": model_name,
        "dataset": dataset_name,
        "architecture": architecture,
        "weight_bits": nbit,
        "quantization": {"std_step": quant_k, "decimal_precision": quant_d} if nbit else None,
        "images_processed": images,
        "configuration": config_data,
        "resolved_components": resolved_components,
        "schedule": schedule,
        "notes": "Spatial tiles parallel; named stages sequential; LIF integrates through emission. C3 uses FIXED macro currents, not a voltage-transfer model.",
        "layers": [
            {"name": layer_dict[idx], "metrics": serialize_metrics(metrics.normalize())}
            for idx, metrics in per_layer.items()
        ],
        "network": network_metrics,
    }
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    logger.info(f"Component-level JSON: {json_path}")

    summary_path = os.path.join(log_dir, "comparison_summary.csv")
    row = {
        "model": model_name,
        "dataset": dataset_name,
        "architecture": architecture,
        "weight_bits": nbit,
        "quantization_id": quant_id,
        "images_processed": images,
        "configuration_id": fingerprint,
        "active_rows": config.active_rows or config.xbar_row,
        "average_power_mw": overall.power_mw,
        "energy_nj_per_inference": overall.energy_nj,
        "latency_us_per_inference": overall.latency_us,
        "energy_efficiency_tops_per_w": overall.topsw,
        "inferences_per_joule": 1e9 / overall.energy_nj if overall.energy_nj else 0,
        "area_mm2": overall.area_mm2,
        "result_json": json_path,
    }
    key_fields = ("model", "architecture", "weight_bits", "quantization_id",
                  "configuration_id", "images_processed")
    key = tuple(str(row[field]) for field in key_fields)
    # Two datasets may be evaluated in separate processes at the same time.
    with open(summary_path + ".lock", "w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = []
        if os.path.exists(summary_path):
            with open(summary_path, newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
        rows = [item for item in rows if
                tuple(item.get(field) for field in key_fields) != key]
        rows.append(row)
        with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8",
                                         dir=log_dir, delete=False) as file:
            temporary_path = file.name
            writer = csv.DictWriter(file, fieldnames=row.keys())
            writer.writeheader()
            writer.writerows({column: item.get(column, "") for column in row}
                             for item in rows)
        os.replace(temporary_path, summary_path)
    logger.info(f"Comparison table: {summary_path}")


def main():
    args = parse_args()
    model_name = args.model or input("Enter model name: ")
    batches = args.batches if args.batches != -1 else "all"

    if args.b <= 0 or args.batches == 0 or args.batches < -1:
        raise ValueError("Batch size must be positive and batches must be -1 or positive")
    logger = setup_logger(model_name, args.b, batches, args.n)

    # We define standard configs using dataclasses for hardware logic
    if args.temporal_map:
        raise ValueError("Only spatial-parallel mapping is supported")
    conv_config = load_config(ConvHardwareConfig, args.conv_config)
    c3_config = load_config(C3HardwareConfig, args.c3_config)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if model_name == "nmnist":
        from demo.nets.nmnist import NMNISTDataset as Dataset

        disp_model_name = "NMNIST"
        net = torch.load(
            os.path.join(asset_dir, "NMNIST_SNN/nmnist-lenet_net2.pt"),
            weights_only=False,
        ).to(device)
        net_params = snn.params(os.path.join(asset_dir, "NMNIST_SNN/nmnist.yaml"))
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
            os.path.join(asset_dir, "Gesture_SNN/gesture-do_net4.pt"),
            weights_only=False,
        ).to(device)
        net_params = snn.params(os.path.join(asset_dir, "Gesture_SNN/gesture.yaml"))
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
    for key in weight_list:
        act_hook.register(key, get_module(net, layer_dict[key].removesuffix(".weight")))

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
        project_dir, net_params["training"]["path"]["dir_test"]
    )
    net_params["training"]["path"]["list_test"] = os.path.join(
        project_dir, net_params["training"]["path"]["list_test"]
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
        with torch.no_grad():
            output = net(x_in)

        # Software metrics
        stats.testing.correctSamples += torch.sum(
            snn.predict.getClass(output) == label
        ).item()
        stats.testing.numSamples += len(label)
        loss = error.numSpikes(output, target)
        stats.testing.lossSum += loss.cpu().item()

        test_loss = round(stats.testing.loss(), 2)
        test_acc = round(100 * stats.testing.correctSamples / stats.testing.numSamples, 2)

        if (b_idx + 1) % 10 == 0 or b_idx == 0:
            logger.info(f"Batch: {b_idx + 1}; Loss: {test_loss}; Accuracy: {test_acc}%")

        # Hardware metrics for this batch
        with torch.no_grad():
            for idx, w_layer_idx in enumerate(weight_list):
                x = act_hook.inputs[w_layer_idx].bool().float()

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
        total_conv.add(norm_metrics, distinct_layer=True)

    logger.info("\nOVERALL NETWORK METRICS (CONV):")
    logger.info(total_conv.format_summary())
    export_results(model_name, disp_model_name, "conventional", conv_config, args.n,
                   args.k, args.d, layer_dict, metrics_conv, total_conv, logger)

    logger.info("=" * 40)
    logger.info("C3CIM HARDWARE METRICS SUMMARY")
    logger.info("=" * 40)

    total_c3 = HardwareMetrics()
    for idx, w_layer_idx in enumerate(weight_list):
        logger.info(f"{idx + 1}.) Layer {w_layer_idx} : {layer_dict[w_layer_idx]}")
        # Normalize metrics per image
        norm_metrics = metrics_c3[w_layer_idx].normalize()
        logger.info(norm_metrics.format_summary())
        total_c3.add(norm_metrics, distinct_layer=True)

    logger.info("\nOVERALL NETWORK METRICS (C3CIM):")
    logger.info(total_c3.format_summary())
    export_results(model_name, disp_model_name, "c3cim", c3_config, args.n,
                   args.k, args.d, layer_dict, metrics_c3, total_c3, logger)
    act_hook.remove_all()


if __name__ == "__main__":
    main()
