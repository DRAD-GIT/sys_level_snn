"""Reports: per-layer and network metrics, printed and exported (JSON + CSV).

Every metric has a switch (see METRIC_NAMES); disabled metrics are neither
printed nor exported. Network totals: energy, latency, area and operations
add up over the layers, which run one after another.
"""
import csv
import hashlib
import json
import os
from dataclasses import asdict

try:
    import fcntl
except ImportError:  # Windows: no cross-process lock for the CSV
    fcntl = None

# name: (label, unit, format)
METRIC_NAMES = {
    "accuracy": ("accuracy", "%", ".2f"),
    "energy": ("energy/inference", "nJ", ".6g"),
    "latency": ("latency/inference", "us", ".6g"),
    "power": ("average power", "mW", ".6g"),
    "area": ("area", "mm^2", ".6g"),
    "tops_per_w": ("TOPS/W (dense MACs)", "", ".4g"),
    "pj_per_sop": ("energy per synaptic op", "pJ", ".4g"),
    "components": ("per-component breakdown", "", ""),
    "layers": ("per-layer results", "", ""),
}


def totals(costs):
    """Network metrics per inference from {layer: LayerCost}."""
    layers = list(costs.values())
    energy = sum(c.energy_nj for c in layers)
    latency_ns = sum(c.latency_ns for c in layers)
    sops = sum(c.synaptic_ops / c.inferences for c in layers)
    macs = sum(c.macs for c in layers)
    return {"energy": energy,
            "latency": latency_ns * 1e-3,
            "power": energy / latency_ns * 1e3 if latency_ns else 0.0,
            "area": sum(c.area_um2 for c in layers) * 1e-6,
            "tops_per_w": 2 * macs / energy * 1e-3 if energy else 0.0,
            "pj_per_sop": energy * 1e3 / sops if sops else 0.0}


def _lines(values, metrics, indent=""):
    for key, value in values.items():
        if metrics.get(key):
            label, unit, fmt = METRIC_NAMES[key]
            yield f"{indent}{label:<24}{value:{fmt}} {unit}".rstrip()


def format_results(results, metrics):
    """Text report for {arch name: (accuracy, {layer: LayerCost})}."""
    lines = []
    for name, (accuracy, costs) in results.items():
        lines += ["=" * 60, f"{name}", "=" * 60]
        lines += _lines({"accuracy": accuracy, **totals(costs)}, metrics)
        if metrics.get("layers"):
            for layer, cost in costs.items():
                g = cost.geometry
                lines.append(f"\n  layer {layer}: {g.windows} window(s) x {g.copies} weight "
                             f"cop{'y' if g.copies == 1 else 'ies'}, {g.row_tiles}x{g.column_tiles} "
                             f"tiles each, {g.reads_per_timestep} reads/time bin")
                lines += _lines(totals({layer: cost}), metrics, "    ")
                if metrics.get("components"):
                    lines += _component_lines(cost, metrics)
    return "\n".join(lines)


def _component_lines(cost, metrics):
    header = f"    {'component':<18}{'group':<18}{'installed':>10}{'energy nJ':>13}"
    header += f"{'area um^2':>12}" if metrics.get("area") else ""
    rows = [header]
    for name, c in cost.components.items():
        row = f"    {name:<18}{c.group:<18}{c.installed:>10}{c.energy_nj / cost.inferences:>13.6g}"
        rows.append(row + (f"{c.area_um2:>12.6g}" if metrics.get("area") else ""))
    return rows


def _fingerprint(arch):
    return hashlib.sha256(json.dumps(asdict(arch), sort_keys=True, default=str)
                          .encode()).hexdigest()[:12]


def _enabled(values, metrics):
    return {k: v for k, v in values.items() if metrics.get(k)}


def _layer_report(cost, metrics):
    entry = _enabled(totals({"layer": cost}), metrics)
    if metrics.get("components"):
        entry["components"] = {
            name: {"group": c.group, "installed": c.installed,
                   "energy_nj": c.energy_nj / cost.inferences,
                   **({"area_um2": c.area_um2} if metrics.get("area") else {})}
            for name, c in cost.components.items()}
    return entry


def export(results, architectures, model, metrics, log_dir):
    """Write one JSON per architecture and upsert rows of comparison_summary.csv
    (keyed by model, architecture, configuration and sample count)."""
    os.makedirs(log_dir, exist_ok=True)
    by_name = {a.name: a for a in architectures}
    rows = []
    for name, (accuracy, costs) in results.items():
        arch, config = by_name[name], _fingerprint(by_name[name])
        samples = next(iter(costs.values())).inferences
        network = _enabled({"accuracy": accuracy, **totals(costs)}, metrics)
        report = {"model": model, "samples": samples, "architecture": asdict(arch),
                  "network": network}
        if metrics.get("layers"):
            report["layers"] = {layer: _layer_report(c, metrics) for layer, c in costs.items()}
        path = os.path.join(log_dir, f"{model}_{name}_{config}_{samples}samples.json")
        with open(path, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, default=str)
        rows.append({"model": model, "architecture": name, "configuration": config,
                     "samples": samples, **network, "result_json": path})
    _upsert_csv(os.path.join(log_dir, "comparison_summary.csv"), rows,
                key=("model", "architecture", "configuration", "samples"))
    return rows


def _upsert_csv(path, new_rows, key):
    with open(path + ".lock", "w", encoding="utf-8") as lock:
        if fcntl is not None:
            fcntl.flock(lock, fcntl.LOCK_EX)
        rows = []
        if os.path.exists(path):
            with open(path, newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
        replaced = {tuple(str(r[k]) for k in key) for r in new_rows}
        rows = [r for r in rows if tuple(r.get(k) for k in key) not in replaced] + new_rows
        fields = list(dict.fromkeys(f for r in rows for f in r))
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields, restval="")
            writer.writeheader()
            writer.writerows(rows)
