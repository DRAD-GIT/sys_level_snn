"""CIM hardware cost models.

configs.py     - ConvHardwareConfig / C3HardwareConfig: the numbers you edit
components.py  - component registry, counting/activity rules, schedule
mapping.py     - quantization, weight->conductance mapping, tile geometry
estimators.py  - per-layer energy/latency/area for each architecture
metrics.py     - HardwareMetrics result containers and summaries
"""
from hardware.components import register_model
from hardware.configs import C3HardwareConfig, ConvHardwareConfig, load_config
from hardware.estimators import calculate_c3_metrics, calculate_conv_metrics
from hardware.metrics import HardwareMetrics

__all__ = ["C3HardwareConfig", "ConvHardwareConfig", "HardwareMetrics",
           "calculate_c3_metrics", "calculate_conv_metrics", "load_config",
           "register_model"]
