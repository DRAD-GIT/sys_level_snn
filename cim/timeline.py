"""Timeline: stage start/end times, block durations and powered time.

Stages of a level are placed in list order: start = max(end of `after`
stages) + offset_ns. Reads repeat every read interval (back to back unless
read_interval_ns pipelines them); the read block is the pseudo-stage "reads"
of the timestep level. Timesteps repeat every timestep interval.
"""
from dataclasses import dataclass


def _place(stages, preset=None):
    placed = dict(preset or {})
    previous = None
    for stage in stages:
        deps = stage.after if stage.after is not None else ([previous] if previous else [])
        if previous is None and stage.after is None and preset:
            deps = list(preset)  # first timestep stage follows the reads by default
        for dep in deps:
            if dep not in placed:
                raise ValueError(f"stage {stage.name}: 'after' must name an earlier stage "
                                 f"of the same level (or 'reads'), got {dep!r}")
        start = max((placed[d][1] for d in deps), default=0.0) + stage.offset_ns
        if start < 0:
            raise ValueError(f"stage {stage.name} would start before its level starts")
        placed[stage.name] = (start, start + stage.duration_ns)
        previous = stage.name
    return placed


def _union_length(intervals):
    total, current_start, current_end = 0.0, None, None
    for start, end in sorted(intervals):
        if current_end is None or start > current_end:
            if current_end is not None:
                total += current_end - current_start
            current_start, current_end = start, end
        else:
            current_end = max(current_end, end)
    if current_end is not None:
        total += current_end - current_start
    return total


@dataclass
class Timeline:
    read: dict              # stage -> (start, end) within one read
    timestep: dict          # stage -> (start, end) within one timestep, incl. "reads"
    read_span: float        # duration of one read
    read_interval: float    # start-to-start time of consecutive reads (> span = idle gap)
    reads: int              # reads per timestep
    timestep_span: float
    timestep_interval: float
    timesteps: int

    @property
    def latency_ns(self):
        """Inference latency; the last timestep ends after its full span."""
        return (self.timesteps - 1) * self.timestep_interval + self.timestep_span

    def read_on_time(self, stage_names):
        """Powered time per read for read-level stages (union of their intervals)."""
        return _union_length([self.read[s] for s in stage_names])

    def timestep_on_time(self, stage_names):
        """Powered time per timestep; read-level stages repeat for every read."""
        if "timestep" in stage_names:
            return self.timestep_span
        intervals = []
        for name in stage_names:
            if name in self.read:
                start, end = self.read[name]
                intervals += [(k * self.read_interval + start, k * self.read_interval + end)
                              for k in range(self.reads)]
            else:
                intervals.append(self.timestep[name])
        return _union_length(intervals)


def build_timeline(arch, reads, timesteps):
    read_stages = [s for s in arch.stages if s.level == "read"]
    step_stages = [s for s in arch.stages if s.level == "timestep"]
    read = _place(read_stages)
    read_span = max(end for _, end in read.values())
    interval = arch.read_interval_ns or read_span
    block = (reads - 1) * interval + read_span if reads else 0.0
    step = _place(step_stages, {"reads": (0.0, block)})
    step_span = max(end for _, end in step.values())
    step_interval = arch.timestep_interval_ns or step_span
    return Timeline(read, step, read_span, interval, reads, step_span, step_interval, timesteps)
