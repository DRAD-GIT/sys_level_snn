"""Weight quantization, weight-to-device mapping and crossbar tiling geometry."""
import math

import torch


def row_tile_phases(vector_size: int, row_size: int,
                    active_rows: int | None) -> list[int]:
    """Read phases per physical row tile (last tile can be partly occupied)."""
    return [math.ceil(min(row_size, vector_size - start) / (active_rows or row_size))
            for start in range(0, vector_size, row_size)]


def row_read_phases(vector_size: int, row_size: int, active_rows: int | None) -> int:
    """Parallel tiles share phases; the fullest physical tile sets read latency."""
    return max(row_tile_phases(vector_size, row_size, active_rows))


def quantize_weights(
    matrix: torch.Tensor, std_range: bool, nbit: int, k: float, d_levels: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weights and return the quantized tensor and levels.

    std_range=True spans mean +/- k*std (clipping outliers); otherwise min/max.
    """
    device = matrix.device
    mean = torch.mean(matrix)
    std_dev = torch.std(matrix)

    if std_range:
        min_val = mean - k * std_dev
        max_val = mean + k * std_dev
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
