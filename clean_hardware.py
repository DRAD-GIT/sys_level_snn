import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class ConvHardwareConfig:
    min_res: float = 2e3
    max_res: float = 2e5
    vdd: float = 1.1
    xbar_row: int = 64
    xbar_col: int = 64
    vread: float = 0.1
    xbar_lat: float = 4.5
    xbar_area: float = 136.67
    DA_curr: float = 6.1
    DA_area: float = 30.22
    lif_curr: float = 6.0
    lif_lat: float = 2.0
    lif_area: float = 86.79

    @property
    def DA_pow(self) -> float:
        return self.vdd * self.DA_curr

    @property
    def tot_lat(self) -> float:
        return self.xbar_lat + self.lif_lat


@dataclass
class C3HardwareConfig:
    min_res: float = 2e3
    max_res: float = 2e4
    vdd: float = 1.1
    xbar_row: int = 64
    xbar_col: int = 64
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
class HardwareMetrics:
    latency_us: float = 0.0
    energy_nj: float = 0.0
    area_mm2: float = 0.0
    ops: float = 0.0
    images_processed: int = 0

    def add(self, other: "HardwareMetrics"):
        self.latency_us += other.latency_us
        self.energy_nj += other.energy_nj
        self.area_mm2 += other.area_mm2
        self.ops += other.ops
        self.images_processed += other.images_processed

    @property
    def power_mw(self) -> float:
        return self.energy_nj / self.latency_us if self.latency_us > 0 else 0.0

    @property
    def topsw(self) -> float:
        # OPS / (nJ * 1e-9) / 1e12 = OPS / nJ * 1e-3
        return (self.ops / self.energy_nj) * 1e-3 if self.energy_nj > 0 else 0.0

    @property
    def topsmm2(self) -> float:
        return (self.ops / self.area_mm2) * 1e-12 if self.area_mm2 > 0 else 0.0

    def normalize(self) -> "HardwareMetrics":
        if self.images_processed == 0:
            return self
        return HardwareMetrics(
            latency_us=self.latency_us / self.images_processed,
            energy_nj=self.energy_nj / self.images_processed,
            area_mm2=self.area_mm2 / self.images_processed,
            ops=self.ops / self.images_processed,
            images_processed=1,
        )

    def format_summary(self) -> str:
        lines = [
            f"Latency = {np.round(self.latency_us, 2)} us",
            f"Energy = {np.round(self.energy_nj, 1)} nJ",
            f"Area = {np.round(self.area_mm2, 2)} mm^2",
            f"Throughput = {np.round((self.ops * 1e-9), 2)} GOP",
            f"Power = {np.round(self.power_mw, 2)} mW",
            f"TOPS/W = {np.round(self.topsw, 2)}",
            f"GOPS/mm^2 = {np.round(self.topsmm2 * 1e3, 2)}\n",
        ]
        return "\n".join(lines)


def quantize_weights(
    matrix: torch.Tensor, model_name: str, nbit: int, k: float, d_levels: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weights and return the quantized tensor and levels."""
    device = matrix.device
    mean = torch.mean(matrix)
    variance = torch.var(matrix)

    if model_name == "nmnist":
        min_val = mean - k * variance
        max_val = mean + k * variance
    elif model_name == "gesture":
        min_val = torch.min(matrix)
        max_val = torch.max(matrix)
    else:
        min_val = torch.min(matrix)
        max_val = torch.max(matrix)

    ndeci = 10**0
    min_val = torch.round(min_val * ndeci) / ndeci
    max_val = torch.round(max_val * ndeci) / ndeci

    num_levels = 2**nbit
    matrix_flat = matrix.view(-1)

    levels = torch.linspace(min_val, max_val, num_levels).to(device)
    ndeci = 10**d_levels
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
        slope = (g_max - g_min) / (max_level - min_level)
        offset = g_min - slope * min_level

        cond_matrix = slope * weights + offset
        ndeci = 10**6
        cond_matrix = torch.round(cond_matrix * ndeci) / ndeci
        return cond_matrix
    else:
        slope = (max_res - min_res) / (max_level - min_level)
        offset = min_res - slope * min_level

        res_matrix = slope * weights + offset
        ndeci = 10**0
        res_matrix = torch.round(res_matrix * ndeci) / ndeci
        return res_matrix


def xbar_partition(
    weight_shape: tuple, output_size: int, row_size: int, col_size: int
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
        out_total = (output_size**2) * weight_shape[0]
        time_steps = 1  # The original logic sets it to 1 if fc=True for conv. Assuming standard conv here.
        # Wait, the original code had a weird fc flag in xbar_partition. We assume standard conv mode where time_steps=1 and out_total includes spatial dims.

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
):
    # Map weights
    qweights = map_weights(weights, levels, min_res, max_res, is_conv=is_conv)

    # Process Input
    # original x shape from SlayerSNN is usually [Batch, Channels, Height, Width, Time]
    # We reshape to [Batch*Time, Channels, Height, Width] for F.conv2d
    batch_size = x.shape[0]
    num_steps = x.shape[4]

    x_mod = x.permute(0, 4, 1, 2, 3).reshape(-1, *x.shape[1:4]).float()

    out_size = conv_out_size(x.shape[3], weights.shape[3], padding)

    vector_size, out_total, time_steps, nxbar_rowside, nxbar_colside = xbar_partition(
        weights.shape, out_size, xbar_row, xbar_col
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
    )

    # Run convolution to get spikes
    xbar_pow = (
        torch.sum(F.conv2d(x_mod, qweights, padding=padding, stride=1), dim=(1, 2, 3))
        * config.vread
    )

    da_pow_w = (config.DA_pow * col_total) * 1e-6  # in W

    # Calculate metrics over the batch
    mac_latency = ((config.tot_lat * time_steps) * num_steps) * batch_size
    mac_energy = (
        (torch.sum(xbar_pow) / batch_size + da_pow_w * num_steps)
        * (config.tot_lat * time_steps)
    ) * batch_size
    mac_area = (
        (config.xbar_area + config.DA_area * config.xbar_col) * xbar_total
        + config.lif_area * config.xbar_col * nxbar_colside
    ) * batch_size
    mac_ops = ((vector_size + 1) * (out_total * time_steps) * num_steps) * batch_size

    return HardwareMetrics(
        latency_us=mac_latency * 1e-3,
        energy_nj=mac_energy.item(),
        area_mm2=mac_area * 1e-6,
        ops=mac_ops,
        images_processed=batch_size,
    )


def calculate_c3_metrics(
    config: C3HardwareConfig,
    x: torch.Tensor,
    weights: torch.Tensor,
    levels: torch.Tensor,
    padding: int,
) -> HardwareMetrics:
    """Calculate C3CIM hardware metrics for a single batch."""
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
    )

    # Calculate power metrics per image
    xbar_pow_w = (
        col_total * config.col_pow
        + math.ceil(col_total / config.driver_part) * config.driver_pow
    ) * 1e-6
    vi_pow_w = (col_total * config.VI_pow) * 1e-6

    # Calculate metrics over the batch
    mac_latency = ((config.tot_lat * time_steps) * num_steps) * batch_size
    mac_energy = (
        (
            (xbar_pow_w * config.tot_lat + vi_pow_w * (config.VI_lat + config.lif_lat))
            * time_steps
        )
        * num_steps
    ) * batch_size
    mac_area = (
        (config.xbar_area + config.VI_area * config.xbar_col) * xbar_total
        + config.lif_area * config.xbar_col * nxbar_colside
    ) * batch_size
    mac_ops = ((vector_size + 1) * (out_total * time_steps) * num_steps) * batch_size

    return HardwareMetrics(
        latency_us=mac_latency * 1e-3,
        energy_nj=mac_energy,
        area_mm2=mac_area * 1e-6,
        ops=mac_ops,
        images_processed=batch_size,
    )
