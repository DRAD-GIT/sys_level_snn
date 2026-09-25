"""Hardware parameter sets for the two CIM architectures.

Every field is a scalar default taken from the original hardware classes.
Units: resistance ohm, voltage V, current uA, latency ns, area um^2 (per
instance: one tile for xbar_area, one circuit otherwise). ``components`` and
``schedule`` optionally patch/extend the default component composition; see
hardware/components.py and the README.
"""
import json
import math
from dataclasses import dataclass, field, fields

from hardware.components import validate_config


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


def load_config(cls, path):
    """Build a config from a JSON file of overrides (omitted fields keep defaults)."""
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
