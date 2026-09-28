"""Timeline: where each stage starts and ends, and how long components are powered.

Stages of a level are placed in the order they are defined: start = latest end
of the stages in `after` (default: the previous stage) + offset_ns.
Activations repeat every activation interval (back to back unless
activation_interval_ns pipelines them); the whole block of a time bin's
activations is the pseudo-stage "activations" of the time-bin level, which the
first time-bin stage follows by default. Time bins repeat every time-bin
interval.
"""
from dataclasses import dataclass

from hardware.architecture import ACTIVATIONS, BIN


def _place(stages, placed):
    placed = dict(placed)
    previous = list(placed)
    for stage in stages:
        deps = previous if stage.after is None else list(stage.after)
        unknown = [d for d in deps if d not in placed]
        if unknown:
            raise ValueError(f"stage {stage.name}: 'after' must name stages defined before it "
                             f"at the same level (or '{ACTIVATIONS}'), got {unknown}")
        start = max((placed[d][1] for d in deps), default=0.0) + stage.offset_ns
        if start < 0:
            raise ValueError(f"stage {stage.name} would start before its level starts")
        placed[stage.name] = (start, start + stage.duration_ns)
        previous = [stage.name]
    return placed


@dataclass
class Timeline:
    activation: dict             # stage -> (start, end) within one activation
    time_bin: dict               # stage -> (start, end) within one time bin, incl. "activations"
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

    def anchor(self, stage, edge):
        """Time within a time bin of a stage edge; a per-activation stage starts
        with the first activation and ends with the last."""
        if stage == BIN:
            return 0.0 if edge == "start" else self.time_bin_span
        if stage in self.activation:
            start, end = self.activation[stage]
            return start if edge == "start" else (self.activations - 1) * self.activation_interval + end
        start, end = self.time_bin[stage]
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
        """One line: every stage with its duration and how often it runs."""
        parts = [f"{name} {duration:g} ns ("
                 + (f"per activation, x{self.activations}" if level == "activation" else "per time bin")
                 + ")" for name, duration, level in self.stages]
        return " | ".join(parts) + f" = {self.time_bin_span:g} ns per time bin"


def build_timeline(arch, activations, time_bins):
    activation = _place([s for s in arch.stages if s.level == "activation"], {})
    span = max(end for _, end in activation.values())
    interval = arch.activation_interval_ns or span
    block = (activations - 1) * interval + span
    time_bin = _place([s for s in arch.stages if s.level == "time_bin"], {ACTIVATIONS: (0.0, block)})
    bin_span = max(end for _, end in time_bin.values())
    return Timeline(activation, time_bin, span, interval, activations, bin_span,
                    arch.time_bin_interval_ns or bin_span, time_bins,
                    tuple((s.name, s.duration_ns, s.level) for s in arch.stages))
