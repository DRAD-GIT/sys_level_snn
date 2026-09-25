import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from hardware_components import EvaluationContext, evaluate_components, validate_config, resolve_components


@dataclass
class ConvHardwareConfig:
    min_res: float = 2e3
    max_res: float = 2e5
    vdd: float = 1.1
    xbar_row: int = 64
    xbar_col: int = 64
    active_rows: int | None = None  # None: read all physical rows simultaneously.
    vread: float = 0.1
    xbar_lat: float = 4.5
    xbar_area: float = 136.67
    DA_curr: float = 6.1
    DA_area: float = 30.22
    lif_curr: float = 6.0
    lif_lat: float = 2.0
    lif_area: float = 86.79
    temporal_map: bool = False
    reference_array: bool = True  # Physical array for offset subtraction.
    ref_sub_curr: float = 0.0  # Subtraction circuit per output column (uA).
    ref_sub_area: float = 0.0  # Subtraction circuit per output column (um^2).
    ref_sub_lat: float = 0.0  # Subtraction latency (ns).

    components: list[dict] = field(default_factory=list)
    schedule: list[dict] | None = None

    def __post_init__(self):
        validate_config(self)

    @property
    def DA_pow(self) -> float:
        return self.vdd * self.DA_curr

    @property
    def tot_lat(self) -> float:
        return self.xbar_lat + self.lif_lat + (self.ref_sub_lat if self.reference_array else 0.0)


@dataclass
class C3HardwareConfig:
    min_res: float = 2e3
    max_res: float = 2e4
    vdd: float = 1.1
    xbar_row: int = 64
    xbar_col: int = 64
    active_rows: int | None = None
    col_curr: float = 0.1
    col_lat: float = 50.0
    col_area: float = 4.27
    driver_part: int = 32
    driver_curr: float = 11.87
    driver_area: float = 86.36
    VI_supp: float = 1.0
    VI_curr: float = 24.3
    VI_lat: float = 10.0
    VI_area: float = 29.79
    lif_curr: float = 6.0
    lif_lat: float = 2.0
    lif_area: float = 86.79
    temporal_map: bool = False

    components: list[dict] = field(default_factory=list)
    schedule: list[dict] | None = None

    def __post_init__(self):
        validate_config(self)

    @property
    def xbar_area(self) -> float:
        return (
            self.xbar_col * self.col_area
            + math.ceil(self.xbar_col / self.driver_part) * self.driver_area
        )

    @property
    def col_pow(self) -> float:
        return self.vdd * self.col_curr

    @property
    def driver_pow(self) -> float:
        return self.vdd * self.driver_curr

    @property
    def VI_pow(self) -> float:
        return self.VI_supp * self.VI_curr

    @property
    def tot_lat(self) -> float:
        return self.col_lat + self.VI_lat + self.lif_lat


@dataclass
class ComponentMetrics:
    """Contribution of one physical component; latency is critical-path time.

    Components that operate alongside another component have zero incremental
    latency, but still contribute energy and area. Installed counts/area do not
    grow with the number of images processed.
    """
    energy_nj: float = 0.0
    latency_us: float = 0.0
    area_mm2: float = 0.0
    count: int = 0
    group: str = "crossbar"
    model: str = ""

    def add(self, other: "ComponentMetrics", *, distinct_layer: bool = False):
        self.group = other.group
        self.model = other.model
        self.energy_nj += other.energy_nj
        self.latency_us += other.latency_us
        if distinct_layer:
            self.area_mm2 += other.area_mm2
            self.count += other.count
        else:
            self.area_mm2 = max(self.area_mm2, other.area_mm2)
            self.count = max(self.count, other.count)

    def normalize(self, images: int) -> "ComponentMetrics":
        return ComponentMetrics(self.energy_nj / images, self.latency_us / images,
                                self.area_mm2, self.count, self.group, self.model)


@dataclass
class HardwareMetrics:
    latency_us: float = 0.0
    energy_nj: float = 0.0
    area_mm2: float = 0.0
    ops: float = 0.0
    read_current_ua: float = 0.0  # Total physical read current, averaged across row phases.
    images_processed: int = 0
    row_phases: int = 0
    components: dict[str, ComponentMetrics] = field(default_factory=dict)

    def add(self, other: "HardwareMetrics", *, distinct_layer: bool = False):
        self.latency_us += other.latency_us
        self.energy_nj += other.energy_nj
        # Images reuse installed tiles; distinct layers own separate tiles.
        self.area_mm2 = (self.area_mm2 + other.area_mm2) if distinct_layer else max(
            self.area_mm2, other.area_mm2
        )
        self.ops += other.ops
        self.read_current_ua += other.read_current_ua
        self.images_processed = (max(self.images_processed, other.images_processed)
                                 if distinct_layer else self.images_processed + other.images_processed)
        if not distinct_layer:
            self.row_phases = max(self.row_phases, other.row_phases)
        for name, contribution in other.components.items():
            self.components.setdefault(name, ComponentMetrics()).add(
                contribution, distinct_layer=distinct_layer
            )

    @property
    def groups(self) -> dict[str, ComponentMetrics]:
        from hardware_components import GROUPS
        result = {name: ComponentMetrics(group=name) for name in GROUPS}
        for c in self.components.values():
            result[c.group].add(c, distinct_layer=True)
            result[c.group].model = ""
        return result

    @property
    def power_mw(self) -> float:
        return self.energy_nj / self.latency_us if self.latency_us > 0 else 0.0

    @property
    def topsw(self) -> float:
        # OPS / (nJ * 1e-9) / 1e12 = OPS / nJ * 1e-3
        return (self.ops / self.energy_nj) * 1e-3 if self.energy_nj > 0 else 0.0

    @property
    def topsmm2(self) -> float:
        if self.area_mm2 > 0 and self.latency_us > 0:
            # OPS / (latency_us * 1e-6) = OPS/sec. TOPS = OPS/sec * 1e-12.
            # So TOPS = (ops / latency_us) * 1e-6
            tops = (self.ops / self.latency_us) * 1e-6
            return tops / self.area_mm2
        return 0.0

    def normalize(self) -> "HardwareMetrics":
        if self.images_processed == 0:
            return self
        return HardwareMetrics(
            latency_us=self.latency_us / self.images_processed,
            energy_nj=self.energy_nj / self.images_processed,
            area_mm2=self.area_mm2,
            ops=self.ops / self.images_processed,
            read_current_ua=self.read_current_ua / self.images_processed,
            images_processed=1,
            row_phases=self.row_phases,
            components={name: component.normalize(self.images_processed)
                        for name, component in self.components.items()},
        )

    def format_summary(self) -> str:
        lines = [
            f"Latency = {np.round(self.latency_us, 2)} us",
            f"Energy = {np.round(self.energy_nj, 1)} nJ",
            f"Area = {np.round(self.area_mm2, 2)} mm^2",
            f"Throughput = {np.round((self.ops * 1e-9), 2)} GOP",
            f"Mean read current (all columns) = {np.round(self.read_current_ua, 2)} uA",
            f"Power = {np.round(self.power_mw, 2)} mW",
            f"TOPS/W = {np.round(self.topsw, 2)}",
            f"GOPS/mm^2 = {np.round(self.topsmm2 * 1e3, 2)}\n",
        ]
        if self.row_phases:
            lines.insert(0, f"Row-read phases = {self.row_phases}")
        if self.components:
            lines.append("Component breakdown (count, area mm^2, energy nJ, critical-path us):")
            for name, c in self.components.items():
                lines.append(f"  {name} [{c.group}; {c.model}]: {c.count}, {c.area_mm2:.6g}, {c.energy_nj:.6g}, {c.latency_us:.6g}")
            lines.append("Group totals (count, area mm^2, energy nJ, critical-path us):")
            for name, c in self.groups.items():
                lines.append(f"  {name}: {c.count}, {c.area_mm2:.6g}, {c.energy_nj:.6g}, {c.latency_us:.6g}")
        return "\n".join(lines)


def row_tile_phases(vector_size: int, row_size: int,
                    active_rows: int | None) -> list[int]:
    """Read phases per physical row tile (last tile can be partly occupied)."""
    return [math.ceil(min(row_size, vector_size - start) / (active_rows or row_size))
            for start in range(0, vector_size, row_size)]


def row_read_phases(vector_size: int, row_size: int, active_rows: int | None) -> int:
    """Parallel tiles share phases; the fullest physical tile sets read latency."""
    return max(row_tile_phases(vector_size, row_size, active_rows))


def metrics_from_components(
    components: dict[str, ComponentMetrics], images: int, ops: float,
    read_current_ua: float, phases: int,
) -> HardwareMetrics:
    return HardwareMetrics(
        latency_us=sum(c.latency_us for c in components.values()),
        energy_nj=sum(c.energy_nj for c in components.values()),
        area_mm2=sum(c.area_mm2 for c in components.values()),
        ops=ops,
        read_current_ua=read_current_ua,
        images_processed=images,
        row_phases=phases,
        components=components,
    )


def component(count: int, area_um2: float, energy_nj: float,
              latency_ns: float = 0.0) -> ComponentMetrics:
    return ComponentMetrics(float(energy_nj), latency_ns * 1e-3,
                            area_um2 * 1e-6, count)


def quantize_weights(
    matrix: torch.Tensor, model_name: str, nbit: int, k: float, d_levels: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weights and return the quantized tensor and levels."""
    device = matrix.device
    mean = torch.mean(matrix)
    std_dev = torch.std(matrix)

    if model_name == "nmnist":
        min_val = mean - k * std_dev
        max_val = mean + k * std_dev
    elif model_name == "gesture":
        min_val = torch.min(matrix)
        max_val = torch.max(matrix)
    else:
        min_val = torch.min(matrix)
        max_val = torch.max(matrix)

    ndeci = 10**d_levels
    min_val = torch.round(min_val * ndeci) / ndeci
    max_val = torch.round(max_val * ndeci) / ndeci

    num_levels = 2**nbit
    matrix_flat = matrix.view(-1)

    levels = torch.linspace(min_val, max_val, num_levels).to(device)
    levels = torch.round(levels * ndeci) / ndeci

    diff_matrix = torch.abs(matrix_flat.unsqueeze(1) - levels.unsqueeze(0))
    idx_matrix = torch.argmin(diff_matrix, dim=1)
    quant_matrix_flat = levels[idx_matrix]
    quant_matrix = quant_matrix_flat.view(matrix.shape)

    return quant_matrix, levels


def map_weights(
    weights: torch.Tensor,
    levels: torch.Tensor,
    min_res: float,
    max_res: float,
    is_conv: bool = True,
) -> torch.Tensor:
    """Map weights to conductances or resistances for hardware representation."""
    min_level = torch.min(levels)
    max_level = torch.max(levels)

    if is_conv:
        g_min = 1 / max_res
        g_max = 1 / min_res
        # Include zero so G(0) is a physically representable reference.
        lo = min(min_level.item(), 0.0)
        hi = max(max_level.item(), 0.0)
        if hi == lo:
            return torch.full_like(weights, g_min)
        slope = (g_max - g_min) / (hi - lo)
        return (slope * weights + g_min - slope * lo).clamp(g_min, g_max)
    else:
        if max_level == min_level:
            return torch.full_like(weights, min_res)
        slope = (max_res - min_res) / (max_level - min_level)
        offset = min_res - slope * min_level

        res_matrix = slope * weights + offset
        ndeci = 10**0
        res_matrix = torch.round(res_matrix * ndeci) / ndeci
        return res_matrix


def xbar_partition(
    weight_shape: tuple, output_size: int, row_size: int, col_size: int, temporal_map: bool
) -> tuple:
    """Calculate hardware partitioning metrics based on shapes."""
    # Assuming weight_shape: [Out_channel, In_channel, Kernel, Kernel]
    is_fc = weight_shape[2] * weight_shape[3] == 1

    if is_fc:
        vector_size = weight_shape[1]
        out_total = weight_shape[0]
        time_steps = 1
    else:
        vector_size = weight_shape[1] * weight_shape[2] * weight_shape[3]
        if temporal_map:
            out_total = weight_shape[0]
            time_steps = output_size**2
        else:
            out_total = (output_size**2) * weight_shape[0]
            time_steps = 1

    nxbar_rowside = math.ceil(vector_size / row_size)
    nxbar_colside = math.ceil(out_total / col_size)

    return vector_size, out_total, time_steps, nxbar_rowside, nxbar_colside


def conv_out_size(in_size: int, kernel_size: int, padding: int, stride: int = 1) -> int:
    return (in_size - kernel_size + 2 * padding) // stride + 1


def _prepare_inputs(
    x: torch.Tensor,
    weights: torch.Tensor,
    levels: torch.Tensor,
    min_res: float,
    max_res: float,
    padding: int,
    is_conv: bool,
    xbar_row: int,
    xbar_col: int,
    temporal_map: bool,
):
    # Conventional read current uses the mapped weights and spike tensors.
    # C3 uses configured fixed macro currents, so avoid materializing both.
    batch_size = x.shape[0]
    num_steps = x.shape[4]
    if is_conv:
        qweights = map_weights(weights, levels, min_res, max_res, is_conv=True)
        x_mod = x.permute(0, 4, 1, 2, 3).reshape(-1, *x.shape[1:4]).float()
    else:
        qweights = None
        x_mod = None

    out_size = conv_out_size(x.shape[3], weights.shape[3], padding)

    vector_size, out_total, time_steps, nxbar_rowside, nxbar_colside = xbar_partition(
        weights.shape, out_size, xbar_row, xbar_col, temporal_map
    )

    xbar_total = nxbar_rowside * nxbar_colside
    col_total = nxbar_rowside * out_total

    return (
        qweights,
        x_mod,
        batch_size,
        num_steps,
        vector_size,
        out_total,
        time_steps,
        nxbar_rowside,
        nxbar_colside,
        xbar_total,
        col_total,
    )


def calculate_conv_metrics(
    config: ConvHardwareConfig,
    x: torch.Tensor,
    weights: torch.Tensor,
    levels: torch.Tensor,
    padding: int,
) -> HardwareMetrics:
    """Calculate Conv hardware metrics for a single batch."""
    validate_config(config)
    (
        qweights,
        x_mod,
        batch_size,
        num_steps,
        vector_size,
        out_total,
        time_steps,
        nxbar_rowside,
        nxbar_colside,
        xbar_total,
        col_total,
    ) = _prepare_inputs(
        x,
        weights,
        levels,
        config.min_res,
        config.max_res,
        padding,
        True,
        config.xbar_row,
        config.xbar_col,
        config.temporal_map,
    )

    if not config.reference_array and torch.any(weights < 0):
        raise ValueError("Signed weights require a reference array for offset subtraction")

    tile_phases = row_tile_phases(vector_size, config.xbar_row, config.active_rows)
    phases = max(tile_phases)
    # These spatially replicated tiles execute row phases concurrently.
    data_curr = torch.sum(
        F.conv2d(x_mod, qweights, padding=padding, stride=1)
    ) * config.vread
    # G(w) = G(0) + alpha*w. Reference array is read in the same phases,
    # with no extra read latency; its physical current is charged separately.
    ref_curr = torch.zeros_like(data_curr)
    if config.reference_array:
        g_zero = map_weights(torch.zeros_like(weights), levels, config.min_res,
                             config.max_res)[0, 0, 0, 0]
        ref_curr = torch.sum(
            F.conv2d(x_mod, torch.ones_like(qweights), padding=padding)
        ) * g_zero * config.vread

    # The summed physical current already includes every active row once;
    # specialized read evaluators do not multiply it by row phases again.
    components = evaluate_components(config, EvaluationContext(
        config, batch_size, time_steps * num_steps, tile_phases, nxbar_colside,
        out_total, [], data_curr.item(), ref_curr.item(),
    ))

    return metrics_from_components(
        components, batch_size,
        2 * vector_size * out_total * time_steps * num_steps * batch_size,
        ((data_curr + ref_curr) * 1e6 / (num_steps * phases)).item(), phases,
    )


def calculate_c3_metrics(
    config: C3HardwareConfig,
    x: torch.Tensor,
    weights: torch.Tensor,
    levels: torch.Tensor,
    padding: int,
) -> HardwareMetrics:
    """Calculate C3CIM hardware metrics for a single batch (FIXED macro model)."""
    validate_config(config)
    (
        qweights,
        x_mod,
        batch_size,
        num_steps,
        vector_size,
        out_total,
        time_steps,
        nxbar_rowside,
        nxbar_colside,
        xbar_total,
        col_total,
    ) = _prepare_inputs(
        x,
        weights,
        levels,
        config.min_res,
        config.max_res,
        padding,
        False,
        config.xbar_row,
        config.xbar_col,
        config.temporal_map,
    )

    tile_phases = row_tile_phases(vector_size, config.xbar_row, config.active_rows)
    phases = max(tile_phases)
    # C3 remains a FIXED macro estimate, independent of input/G. Per-column
    # current and constant-current-driver overhead are separate components.
    context = EvaluationContext(
        config, batch_size, time_steps * num_steps, tile_phases, nxbar_colside,
        out_total, [],
    )
    components = evaluate_components(config, context)
    # Read-current diagnostic includes configured C3 column models only, not
    # peripheral overhead. Report batch-summed, phase-averaged current.
    read_current = sum(
        context.instances(spec["activity"], phased=True) * spec["params"]["current_ua"]
        for spec in resolve_components(config)[0] if spec["model"] == "c3_column"
    ) * batch_size / phases
    return metrics_from_components(
        components, batch_size,
        2 * vector_size * out_total * time_steps * num_steps * batch_size,
        read_current, phases,
    )
