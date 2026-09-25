"""Per-batch hardware estimation for one weighted layer on each architecture.

calculate_conv_metrics / calculate_c3_metrics take the layer's binary input
spikes x [batch, channels, H, W, bins], its weights, the quantization levels
and the conv padding, and return HardwareMetrics summed over the batch.
"""
import torch
import torch.nn.functional as F

from hardware.components import EvaluationContext, evaluate_components, validate_config, resolve_components
from hardware.configs import C3HardwareConfig, ConvHardwareConfig
from hardware.mapping import conv_out_size, map_weights, row_tile_phases, xbar_partition
from hardware.metrics import HardwareMetrics, metrics_from_components


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
