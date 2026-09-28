"""Timeline: where each step starts and ends, and how long components are powered.

Steps of a level start when all steps of that level defined before them have
ended (serial), or at their `at` anchor. Activations repeat every activation
interval (back to back unless activation_interval_ns pipelines them); the
whole block of a time bin's activations, "activations", starts with the time
bin, and time-bin steps follow it by default. Time bins repeat every time-bin
interval.
"""
from dataclasses import dataclass

from hardware.architecture import ACTIVATIONS, BIN


def conversions(stage, geometry):
    """Operations one after another in a step: 1, or for a serial step the
    weight columns of the fullest column group (see Mapping.columns)."""
    return 1 if stage.serial_size is None else geometry.fullest_group(stage.serial_size)


def _place(stages, durations, placed, outside=None):
    """Place `stages` (one level) after the already `placed` intervals.
    outside: name -> (start, end) of other-level steps they may be anchored to."""
    placed, outside = dict(placed), outside or {}
    base = list(placed)                      # "activations" for the time-bin level
    pending = list(stages)
    while pending:
        progress = False
        for stage in list(pending):
            earlier = [s.name for s in stages[:stages.index(stage)]]
            if stage.at is None:
                if any(name not in placed for name in earlier):
                    continue
                start = max((placed[name][1] for name in base + earlier), default=0.0)
            else:
                name, edge, offset = stage.at
                if name == BIN:
                    edges = (0.0, None)
                elif name in outside:
                    edges = outside[name]
                elif name in placed:
                    edges = placed[name]
                else:
                    continue
                start = edges[edge == "end"] + offset
                if start < 0:
                    raise ValueError(f"step {stage.name} would start before its level starts "
                                     f"(at {start:g} ns)")
            placed[stage.name] = (start, start + durations[stage.name])
            pending.remove(stage)
            progress = True
        if not progress:
            raise ValueError("steps placed at each other form a cycle: "
                             + ", ".join(s.name for s in pending))
    return placed


@dataclass
class Timeline:
    activation: dict             # step -> (start, end) within one activation
    time_bin: dict               # step -> (start, end) within one time bin, incl. "activations"
    activation_span: float       # duration of one activation
    activation_interval: float   # start-to-start time of consecutive activations
    activations: int             # activations per time bin
    time_bin_span: float
    time_bin_interval: float
    time_bins: int
    stages: tuple = ()           # (name, duration, level) in order, for reports

    @property
    def latency_ns(self):
        """Per inference: the last time bin runs its full span."""
        return (self.time_bins - 1) * self.time_bin_interval + self.time_bin_span

    def anchor(self, name, edge):
        """Time within a time bin of a step edge; a per-activation step starts
        with the first activation and ends with the last."""
        if name == BIN:
            return 0.0 if edge == "start" else self.time_bin_span
        if name in self.activation:
            start, end = self.activation[name]
            return start if edge == "start" else (self.activations - 1) * self.activation_interval + end
        start, end = self.time_bin[name]
        return start if edge == "start" else end

    def on_time(self, power):
        """Powered time per activation (per-activation intervals) or per time bin."""
        (s0, e0, o0), (s1, e1, o1) = power.start, power.end
        if power.level == "activation":
            start = self.activation[s0][e0 == "end"] + o0
            end = self.activation[s1][e1 == "end"] + o1
        else:
            start, end = self.anchor(s0, e0) + o0, self.anchor(s1, e1) + o1
        if end < start:
            raise ValueError(f"power interval {power.start} -> {power.end} ends before it "
                             f"starts ({start} -> {end} ns)")
        return end - start

    def describe(self):
        """One line: every step with its duration and how often it runs."""
        parts = [f"{name} {duration:g} ns ("
                 + (f"per activation, x{self.activations}" if level == "activation" else "per time bin")
                 + ")" for name, duration, level in self.stages]
        return " | ".join(parts) + f" = {self.time_bin_span:g} ns per time bin"


def build_timeline(arch, geometry, time_bins):
    """The timeline of a layer (serial steps depend on its columns)."""
    durations = {s.name: s.time_ns * conversions(s, geometry) for s in arch.stages}
    activations = geometry.activations_per_bin
    activation = _place([s for s in arch.stages if s.level == "activation"], durations, {})
    span = max(end for _, end in activation.values())
    interval = arch.activation_interval_ns or span
    block = (activations - 1) * interval + span
    in_bin = {name: (start, (activations - 1) * interval + end)
              for name, (start, end) in activation.items()}
    time_bin = _place([s for s in arch.stages if s.level == "time_bin"], durations,
                      {ACTIVATIONS: (0.0, block)}, in_bin)
    bin_span = max(end for _, end in time_bin.values())
    return Timeline(activation, time_bin, span, interval, activations, bin_span,
                    arch.time_bin_interval_ns or bin_span, time_bins,
                    tuple((s.name, durations[s.name], s.level) for s in arch.stages))
