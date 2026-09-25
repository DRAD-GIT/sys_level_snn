"""Logging and result export (text log, per-run JSON, comparison CSV)."""
import csv
import hashlib
import json
import logging
import os
import sys
import tempfile
from dataclasses import asdict

try:
    import fcntl
except ImportError:  # Windows: no cross-process CSV lock.
    fcntl = None

from hardware.components import resolve_components


def setup_logger(log_dir: str, model_name: str, batch_size: int, batches, nbit: int) -> logging.Logger:
    """Log to both console and logs/<model>_B<batch>_N<batches>_b<bits>.txt."""
    logger = logging.getLogger("snn_eval")
    logger.setLevel(logging.INFO)
    for handler in list(logger.handlers):  # avoid duplicates on repeated runs
        logger.removeHandler(handler)
        handler.close()

    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{model_name}_B{batch_size}_N{batches}_b{nbit}.txt")

    formatter = logging.Formatter("%(message)s")
    for handler in (logging.FileHandler(log_file_path, mode="w"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def export_results(log_dir, model_name, dataset_name, architecture, config, nbit,
                   quant_k, quant_d, per_layer, overall, logger):
    """Export exact per-component measurements and upsert a comparison row.

    per_layer maps layer name -> accumulated (not yet normalized) HardwareMetrics;
    overall is the normalized network total.
    """
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
            {"name": name, "metrics": serialize_metrics(metrics.normalize())}
            for name, metrics in per_layer.items()
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
        if fcntl is not None:
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


def format_final_table(results, accuracy):
    """One-line-per-architecture summary of the network totals."""
    header = (f"{'architecture':<14}{'energy/inf (nJ)':>17}{'latency/inf (us)':>18}"
              f"{'power (mW)':>12}{'TOPS/W':>10}{'area (mm^2)':>13}{'GOPS/mm^2':>12}")
    lines = [f"Software accuracy: {accuracy}%", header, "-" * len(header)]
    for name, m in results.items():
        lines.append(f"{name:<14}{m.energy_nj:>17.4g}{m.latency_us:>18.4g}{m.power_mw:>12.4g}"
                     f"{m.topsw:>10.4g}{m.area_mm2:>13.4g}{m.topsmm2 * 1e3:>12.4g}")
    return "\n".join(lines)
