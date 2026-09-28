"""Description of a CIM macro: crossbar, mapping, components and their timing.

Units: resistance ohm, voltage V, current uA, time ns, event energy pJ, area
um^2 per installed instance.

A spiking layer runs one time bin after another. Within a time bin the
crossbar is activated (a group of rows driven, its columns read) once per row
phase (`active_rows` smaller than the tile height) and, with the "sequential"
conv mapping, once per convolution window: an **activation**. Input spikes of
every time bin carry the same weight in the LIF membrane, so input precision
is the number of time bins.

Timing comes from the components: a component with `stage=` defines a step
of the timeline, run in every activation or once per time bin (inferred from
what the component is attached to, or set with `per=`). When a component
draws current is its power interval, `start` to `end` (default: its own
stage).

A kernel (the weights of one output channel) occupies one column per weight
slice: weights of `weight_bits` are stored in cells of the memory's
`cell_bits` (bit-slicing). Convolutions are unrolled into windows, each
window's input patch driving the kernel rows:
  "parallel":   one copy of the kernel weights per window; all windows at once.
                Copies that fit in a tile share it, packed block-diagonally
                (see mapping.py); columns without weights stay off.
  "sequential": one copy of the kernel weights; the windows are applied one
                after another (column tiles still work in parallel).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Count/activity rules, by the unit an instance belongs to:
TILE_RULES = ("tiles", "physical_rows", "physical_columns", "used_columns", "column_groups")
WINDOW_RULES = ("outputs", "output_bank")   # neurons per window / per tile set's columns
LAYER_RULES = ("one", "fixed")
DATA_RULES = ("spiking_rows",)              # word lines carrying a spike, per activation
ENCODINGS = ("twos_complement", "offset", "differential", "analog")
MODELS = ("static", "crossbar_read", "reference_read", "slice_mirror")
LEVELS = ("activation", "time_bin")         # how often a stage runs
EVENTS = LEVELS + ("output_spike",)
CONV_MAPPINGS = ("parallel", "sequential")
BIN = "bin"                                 # anchors "bin.start" and "bin.end" (= "end")
ACTIVATIONS = "activations"                 # time-bin pseudo-stage: all activations of a bin


@dataclass
class Memory:
    """A memory cell technology: bits per cell and the conductance of each level.

    Levels are spaced linearly in conductance from 1/r_off (level 0) to 1/r_on
    (top level), unless `levels_s` lists every level's conductance (siemens,
    ascending) for nonuniform devices. Analog cells use the continuous range.
    """
    cell_bits: int = 1
    r_on: float | None = None       # lowest resistance (top level)
    r_off: float | None = None      # highest resistance (level 0)
    levels_s: tuple[float, ...] | None = None

    def conductances(self):
        if self.levels_s is not None:
            return tuple(self.levels_s)
        top = 2 ** self.cell_bits - 1
        return tuple(1 / self.r_off + level / top * (1 / self.r_on - 1 / self.r_off)
                     for level in range(top + 1))


@dataclass
class Crossbar:
    memory: Memory
    rows: int = 64                  # physical rows (word lines) per tile
    cols: int = 64                  # physical columns per tile
    v_read: float = 0.2             # read voltage across a selected cell
    active_rows: int | None = None  # rows driven per activation; None = all rows
    reference_columns: bool = False  # analog encoding: one G(0) column per output


@dataclass
class Mapping:
    """How the trained network is mapped onto the crossbars."""
    # Weight bits (None: unquantized weights, analog encoding only).
    weight_bits: int | None = 6
    # Quantization range, symmetric around 0 (see mapping.quantize_weights):
    # "std<k>" = k standard deviations (default "std3"); "max" = the largest
    # |weight|; "mse" = the clip with the least squared error, per layer.
    weight_scaling: str = "std3"
    # twos_complement / offset: bit-sliced integer codes, sign or offset
    # handled after the array; differential: positive and negative columns;
    # analog: one multi-level cell per weight, G linear in the weight value.
    weight_encoding: str = "twos_complement"
    # Convolutions: "parallel" (a weight copy per window) or "sequential".
    conv: str = "parallel"


def scaling_std(scaling):
    """k of a "std<k>" scaling, else None; raises for unknown scalings."""
    if scaling in ("max", "mse"):
        return None
    match = re.fullmatch(r"std(\d+(?:\.\d+)?)", str(scaling))
    if not match or float(match.group(1)) <= 0:
        raise ValueError(f"weight_scaling must be 'max', 'mse' or 'std<k>' (e.g. 'std3'), got {scaling!r}")
    return float(match.group(1))


@dataclass
class Component:
    """A circuit block: `count` instances installed (area); `on` of them
    powered from `start` to `end`.

    Timing: `stage="name"` makes this component define a step of the
    timeline, lasting `stage_ns`, or `op_ns` per operation when each instance
    handles its inputs one after another (an ADC shared by a column group of
    size n converts n columns: n x op_ns). Stages run in the order they are
    defined; `after` (stage names; [] = at the start of its level) and
    `offset_ns` (negative = overlap) place one differently. A stage runs in
    every activation or once per time bin, after all activations: inferred
    from `count` (columns, rows, column groups, tiles: per activation;
    neurons and layer-wide parts: per time bin), or per="activation" /
    per="time_bin".

    Power: `start` / `end` anchors: "stage" (its start as start, its end as
    end), "stage.start", "stage.end", with an offset in ns ("read.start+1",
    "fire.end-0.5"), "bin.start", and "end" (= "bin.end", the end of the time
    bin). Default: the component's own stage. An interval within the
    per-activation stages is powered in every activation (gated per
    activation); one reaching a time-bin stage or the bin's edges is powered
    once per time bin, where a per-activation stage starts with the first
    activation and ends with the last. A component without a stage or
    interval only has area and event costs.

    energy = supply_v * static_ua * powered time      (bias/static current)
           + event_pj * events                        (per activation, time bin or output spike)
    Data-driven models, charged from supply_v while powered, with currents
    computed from the spikes and conductances (per activation; over a time
    bin only with one activation per time bin):
      "crossbar_read"   the cell current of the weight columns;
      "reference_read"  the cell current of the G(0) reference columns;
      "slice_mirror"    current mirrors copying every weight-slice column's
                        current with gain `slice_gains` (per slice, least
                        significant first; default binary: most significant
                        slice 1, the next 1/2, ...) into its neuron.

    count / on: a rule name or {"rule": name, "value": n (fixed),
    "size": n (column_groups), "gated": True}. on="all" repeats `count`.
    "gated": only instances whose tile (tile rules), window ("outputs"), tile
    set ("output_bank") or layer receives at least one input spike in that
    activation or time bin.
    """
    name: str
    count: str | dict = "tiles"
    on: str | dict = "all"
    stage: str | None = None
    stage_ns: float | None = None
    op_ns: float | None = None
    per: str | None = None
    after: str | list | None = None
    offset_ns: float = 0.0
    start: str | None = None
    end: str | None = None
    supply_v: float = 1.1
    static_ua: float = 0.0
    event_pj: float = 0.0
    events: str = "activation"
    area_um2: float = 0.0
    model: str = "static"
    slice_gains: tuple[float, ...] | None = None
    group: str = "periphery"        # reporting label only


@dataclass(frozen=True)
class Stage:
    """A step of the timeline, derived from the component that defines it."""
    name: str
    duration_ns: float
    level: str                      # "activation" or "time_bin"
    after: tuple | None             # None = after the previous stage of its level
    offset_ns: float
    owner: str


@dataclass(frozen=True)
class Power:
    """A component's power interval: its level and (stage, "start"|"end",
    offset_ns) anchors."""
    level: str
    start: tuple
    end: tuple


@dataclass
class Architecture:
    name: str
    crossbar: Crossbar
    mapping: Mapping
    components: list[Component]
    activation_interval_ns: float | None = None  # pipelined activations; None = back to back
    time_bin_interval_ns: float | None = None    # overlapping time bins; None = serial

    def __post_init__(self):
        self.stages, self.power = validate(self)

    @property
    def activation_stages(self):
        return {s.name for s in self.stages if s.level == "activation"}


@dataclass
class Block:
    """A reusable piece of hardware: its components and, for a crossbar
    type, the Crossbar itself."""
    components: list[Component] = field(default_factory=list)
    crossbar: Crossbar | None = None


def compose(name, mapping, parts, *, activation_interval_ns=None, time_bin_interval_ns=None):
    """Assemble an Architecture from exactly one crossbar (a Block, which
    brings the array's components and its stage) plus any Components. Stages
    run in the order their components are given."""
    blocks = [p if isinstance(p, Block) else
              Block(components=[p]) if isinstance(p, Component) else None for p in parts]
    if None in blocks:
        raise ValueError(f"{name}: parts must be a crossbar and Components")
    crossbars = [b.crossbar for b in blocks if b.crossbar is not None]
    if len(crossbars) != 1:
        raise ValueError(f"{name}: compose needs exactly one crossbar, got {len(crossbars)}")
    return Architecture(name, crossbars[0], mapping, [c for b in blocks for c in b.components],
                        activation_interval_ns, time_bin_interval_ns)


def rule_of(spec, *, activity=False):
    """Normalize a count/on rule to {"rule", "value"?, "size"?, "gated"}."""
    rule = {"rule": spec} if isinstance(spec, str) else dict(spec)
    allowed = TILE_RULES + WINDOW_RULES + LAYER_RULES + (DATA_RULES + ("all",) if activity else ())
    if rule.get("rule") not in allowed:
        raise ValueError(f"unknown {'on' if activity else 'count'} rule {spec!r}")
    if rule["rule"] == "fixed" and (isinstance(rule.get("value"), bool)
                                    or not isinstance(rule.get("value"), int) or rule["value"] < 0):
        raise ValueError("fixed rule needs a nonnegative integer 'value'")
    if rule["rule"] == "column_groups":
        _positive_int(rule.get("size"), "column_groups size")
    rule.setdefault("gated", False)
    if not isinstance(rule["gated"], bool) or (rule["gated"] and not activity):
        raise ValueError(f"'gated' is a boolean for 'on' rules only: {spec!r}")
    if set(rule) - {"rule", "value", "size", "gated"}:
        raise ValueError(f"unknown rule fields in {spec!r}")
    return rule


_ANCHOR = re.compile(r"\s*([A-Za-z_]\w*)(?:\.(start|end))?\s*(?:([+-])\s*(\d+(?:\.\d*)?|\.\d+))?\s*")


def parse_anchor(text, default_edge):
    """"fire", "fire.start+1", "read.end-0.5", "bin.start", "end" ->
    (stage, "start"|"end", offset_ns); a bare stage name takes default_edge."""
    match = _ANCHOR.fullmatch(str(text))
    if not match:
        raise ValueError(f"bad anchor {text!r}: use 'stage', 'stage.start' or 'stage.end' "
                         "(optionally + or - ns), 'bin.start' or 'end'")
    stage, edge, sign, value = match.groups()
    if stage == "end" and edge is None:
        stage, edge = BIN, "end"
    offset = (-1 if sign == "-" else 1) * float(value) if value else 0.0
    return stage, edge or default_edge, offset


def stage_level(c):
    """How often a component's stage runs: `per`, else from its count rule."""
    if c.per is not None:
        return c.per
    return "activation" if rule_of(c.count)["rule"] in TILE_RULES else "time_bin"


def operations(c):
    """Operations each instance performs in its stage, one after another: the
    columns of its group for column_groups, else 1."""
    count = rule_of(c.count)
    return count["size"] if count["rule"] == "column_groups" else 1


def _positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


def _number(value, label, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value \
            or value < 0 or (positive and value == 0):
        raise ValueError(f"{label} must be a {'positive' if positive else 'nonnegative'} number")


def _validate_memory(memory):
    _positive_int(memory.cell_bits, "memory: cell_bits")
    if memory.levels_s is None:
        for label in ("r_on", "r_off"):
            _number(getattr(memory, label), f"memory: {label}", positive=True)
        if memory.r_off <= memory.r_on:
            raise ValueError("memory: r_off must exceed r_on")
    else:
        levels = list(memory.levels_s)
        if len(levels) != 2 ** memory.cell_bits:
            raise ValueError("memory: levels_s needs 2^cell_bits entries")
        for level in levels:
            _number(level, "memory: levels_s")
        if levels != sorted(levels) or levels[0] == levels[-1]:
            raise ValueError("memory: levels_s must ascend")


def slices_per_group(arch):
    """Weight-slice columns per sign group (x2 groups for differential)."""
    mp, cell = arch.mapping, arch.crossbar.memory.cell_bits
    if mp.weight_encoding == "analog":
        return 1
    bits = mp.weight_bits - 1 if mp.weight_encoding == "differential" else mp.weight_bits
    return -(-bits // cell)


def _validate_mapping(arch):
    xb, mp = arch.crossbar, arch.mapping
    if not isinstance(mp, Mapping):
        raise ValueError("the mapping must be a Mapping")
    if mp.weight_encoding not in ENCODINGS:
        raise ValueError(f"weight_encoding must be one of {ENCODINGS}")
    if mp.weight_bits is None:
        if mp.weight_encoding != "analog":
            raise ValueError("weight_bits=None (unquantized) needs the analog encoding")
    else:
        _positive_int(mp.weight_bits, "mapping.weight_bits")
        if mp.weight_bits < 2:
            raise ValueError("signed weights need weight_bits >= 2")
    scaling_std(mp.weight_scaling)
    if xb.reference_columns and mp.weight_encoding != "analog":
        raise ValueError("reference_columns applies to the analog encoding only")
    if mp.conv not in CONV_MAPPINGS:
        raise ValueError(f"mapping.conv must be one of {CONV_MAPPINGS}")


def _stage_of(c):
    """The Stage a component defines."""
    if not isinstance(c.stage, str) or not re.fullmatch(r"[A-Za-z_]\w*", c.stage) \
            or c.stage in (BIN, ACTIVATIONS, "end"):
        raise ValueError(f"{c.name}: stage names are identifiers other than "
                         f"'{BIN}', '{ACTIVATIONS}' and 'end', got {c.stage!r}")
    if (c.stage_ns is None) == (c.op_ns is None):
        raise ValueError(f"{c.name}: stage {c.stage!r} needs either stage_ns or op_ns")
    duration = c.stage_ns if c.stage_ns is not None else c.op_ns * operations(c)
    _number(duration, f"{c.name}: stage duration")
    if c.per is not None and c.per not in LEVELS:
        raise ValueError(f"{c.name}: per must be one of {LEVELS}")
    after = c.after
    if after is not None:
        after = (after,) if isinstance(after, str) else tuple(after)
    if isinstance(c.offset_ns, bool) or not isinstance(c.offset_ns, (int, float)):
        raise ValueError(f"{c.name}: offset_ns must be a number")
    return Stage(c.stage, float(duration), stage_level(c), after, float(c.offset_ns), c.name)


def _power_of(c, levels):
    """The Power interval of a component, or None if it draws no current."""
    if c.start is None and c.end is None and c.stage is None:
        return None
    if c.stage is None and (c.start is None or c.end is None):
        raise ValueError(f"{c.name}: give both 'start' and 'end' (it has no stage of its own)")
    start = parse_anchor(c.start if c.start is not None else c.stage, "start")
    end = parse_anchor(c.end if c.end is not None else c.stage, "end")
    for stage, _, _ in (start, end):
        if stage not in levels and stage not in (BIN, ACTIVATIONS):
            raise ValueError(f"{c.name}: unknown stage {stage!r} in start/end; the stages are "
                             f"{', '.join(levels) or 'none'} (plus 'bin' and 'end')")
    level = "activation" if levels.get(start[0]) == levels.get(end[0]) == "activation" else "time_bin"
    return Power(level, start, end)


def validate(arch):
    """Check the architecture; returns its stages and its components' power
    intervals."""
    xb = arch.crossbar
    if not isinstance(xb.memory, Memory):
        raise ValueError("crossbar.memory must be a Memory")
    _validate_memory(xb.memory)
    for label in ("rows", "cols"):
        _positive_int(getattr(xb, label), f"crossbar.{label}")
    if xb.active_rows is not None:
        _positive_int(xb.active_rows, "crossbar.active_rows")
        if xb.active_rows > xb.rows:
            raise ValueError("active_rows cannot exceed rows")
    _number(xb.v_read, "crossbar.v_read", positive=True)
    _validate_mapping(arch)
    for label in ("activation_interval_ns", "time_bin_interval_ns"):
        if getattr(arch, label) is not None:
            _number(getattr(arch, label), label, positive=True)

    names, stages = set(), []
    for c in arch.components:
        if not c.name or c.name in names:
            raise ValueError(f"component names must be unique: {c.name!r}")
        names.add(c.name)
        rule_of(c.count)
        if c.stage is not None:
            stage = _stage_of(c)
            if stage.name in {s.name for s in stages}:
                raise ValueError(f"{c.name}: stage {stage.name!r} is already defined by "
                                 f"{next(s.owner for s in stages if s.name == stage.name)}")
            stages.append(stage)
    if not any(s.level == "activation" for s in stages):
        raise ValueError("a per-activation stage is required (the crossbar's read)")
    levels = {s.name: s.level for s in stages}
    for s in stages:
        for dep in s.after or ():
            if levels.get(dep, "time_bin" if dep == ACTIVATIONS else None) != s.level:
                raise ValueError(f"stage {s.name}: 'after' must name stages of the same "
                                 f"level ({s.level}), got {dep!r}")

    power = {}
    for c in arch.components:
        if c.model not in MODELS or c.events not in EVENTS:
            raise ValueError(f"{c.name}: model must be one of {MODELS}, events one of {EVENTS}")
        on = rule_of(c.on, activity=True)
        for label in ("supply_v", "static_ua", "event_pj", "area_um2"):
            _number(getattr(c, label), f"{c.name}.{label}")
        power[c.name] = p = _power_of(c, levels)
        if (c.static_ua or c.model != "static") and p is None:
            raise ValueError(f"{c.name}: static or data-driven current needs a stage or "
                             "'start' and 'end'")
        if on["rule"] in DATA_RULES and p is not None and p.level != "activation":
            raise ValueError(f"{c.name}: {on['rule']} counts spikes per activation; power it "
                             "within the per-activation stages")
        if on["rule"] in DATA_RULES and c.event_pj and c.events == "time_bin":
            raise ValueError(f"{c.name}: {on['rule']} events are counted per activation")
        if c.model == "reference_read" and not xb.reference_columns:
            raise ValueError(f"{c.name}: reference_read needs crossbar.reference_columns")
        if c.slice_gains is not None:
            if c.model != "slice_mirror" or len(c.slice_gains) != slices_per_group(arch):
                raise ValueError(f"{c.name}: slice_gains needs model 'slice_mirror' and one "
                                 f"gain per weight slice ({slices_per_group(arch)})")
            for gain in c.slice_gains:
                _number(gain, f"{c.name}.slice_gains")
    return stages, power
