"""Result containers: per-component and per-layer/network metrics."""
from dataclasses import dataclass, field

import numpy as np

from hardware.components import GROUPS


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
