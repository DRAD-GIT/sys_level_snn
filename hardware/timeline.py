"""Timeline: where each stage starts and ends, and how long components are powered.

Stages of a level are placed in list order: start = latest end of the stages
in `after` (default: the previous stage) + offset_ns. Reads repeat every read
interval (back to back unless read_interval_ns pipelines them); the whole
block of reads is the pseudo-stage "reads" of the timestep level, which the
first timestep stage follows by default. Time bins repeat every timestep
interval.
"""
from dataclasses import dataclass


def _place(stages, placed):
    placed = dict(placed)
    previous = list(placed)
    for stage in stages:
        deps = previous if stage.after is None else stage.after
        unknown = [d for d in deps if d not in placed]
        if unknown:
            raise ValueError(f"stage {stage.name}: 'after' must name earlier stages of the "
                             f"same level (or 'reads'), got {unknown}")
        start = max((placed[d][1] for d in deps), default=0.0) + stage.offset_ns
        if start < 0:
            raise ValueError(f"stage {stage.name} would start before its level starts")
        placed[stage.name] = (start, start + stage.duration_ns)
        previous = [stage.name]
    return placed


def _union_length(intervals):
    total, reach = 0.0, float("-inf")
    for start, end in sorted(intervals):
        if end > reach:
            total += end - max(start, reach)
            reach = end
    return total


@dataclass
class Timeline:
    read: dict              # stage -> (start, end) within one read
    timestep: dict          # stage -> (start, end) within one time bin, incl. "reads"
    read_span: float        # duration of one read
    read_interval: float    # start-to-start time of consecutive reads
    reads: int              # reads per time bin
    timestep_span: float
    timestep_interval: float
    timesteps: int

    @property
    def latency_ns(self):
        """Per inference: the last time bin runs its full span."""
        return (self.timesteps - 1) * self.timestep_interval + self.timestep_span

    def read_on_time(self, stage_names):
        """Powered time per read of read-level stages (overlaps counted once)."""
        return _union_length([self.read[s] for s in stage_names])

    def timestep_on_time(self, stage_names):
        """Powered time per time bin; read-level stages repeat for every read."""
        if "timestep" in stage_names:
            return self.timestep_span
        intervals = [self.timestep[s] for s in stage_names if s not in self.read]
        for s in stage_names:
            if s in self.read:
                start, end = self.read[s]
                intervals += [(k * self.read_interval + start, k * self.read_interval + end)
                              for k in range(self.reads)]
        return _union_length(intervals)


def build_timeline(arch, reads, timesteps):
    read = _place([s for s in arch.stages if s.level == "read"], {})
    read_span = max(end for _, end in read.values())
    interval = arch.read_interval_ns or read_span
    block = (reads - 1) * interval + read_span
    step = _place([s for s in arch.stages if s.level == "timestep"], {"reads": (0.0, block)})
    step_span = max(end for _, end in step.values())
    return Timeline(read, step, read_span, interval, reads, step_span,
                    arch.timestep_interval_ns or step_span, timesteps)
