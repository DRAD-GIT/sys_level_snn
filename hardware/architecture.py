"""Description of a CIM macro: crossbar, mapping, components and their timing.

Units: resistance ohm, voltage V, current uA, time ns, event energy pJ, area
um^2 per installed instance.

A spiking layer runs one time bin after another. Within a time bin the
crossbar is activated (a group of rows driven, its columns read) once per row
phase (`active_rows` smaller than the tile height) and, with the "sequential"
conv mapping, once per convolution window: an **activation**. Input spikes of
every time bin carry the same weight in the LIF membrane, so input precision
is the number of time bins.

Timing comes from the components: a component with `time_ns` is a step of
the timeline, named after the component, run in every activation or once per
time bin (inferred from what the component is attached to, or set with
`per=`). `at` places a step, `when` sets when a component draws current
(default: during its own step).

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
TILE_RULES = ("tiles", "physical_rows", "physical_columns", "used_columns", "column_groups",
              "used_column_groups")
WINDOW_RULES = ("outputs", "output_bank")   # neurons per window / per tile set's columns
LAYER_RULES = ("one", "fixed")
DATA_RULES = ("spiking_rows",)              # word lines carrying a spike, per activation
ENCODINGS = ("twos_complement", "offset", "differential", "analog")
MODELS = ("static", "crossbar_read", "reference_read", "slice_mirror")
LEVELS = ("activation", "time_bin")         # how often a step runs
EVENTS = LEVELS + ("output_spike",)
CONV_MAPPINGS = ("parallel", "sequential")
# Descriptive specifications of an architecture, for reports and the paper
# table only (they change no cost): a display label, technology (nm), supply
# (V), storage device, cell precision, bit-cell, R_High/R_Low (kOhm), sensing
# mode, accumulation (rows summed per read).
SPEC_KEYS = ("label", "tech", "supply", "device", "cell", "bitcell", "r_ratio", "sensing",
             "accumulation")
COLUMN_PLACEMENTS = ("interleaved", "contiguous")
BIN = "bin"                                 # the whole time bin: "bin.start", "bin.end"
ACTIVATIONS = "activations"                 # all activations of a time bin, from its start


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
    # Where a tile's weight columns sit among its column groups (e.g. the
    # columns sharing an ADC or a driver): "interleaved" (column j in group
    # j mod groups, so the groups share them evenly) or "contiguous" (from
    # column 0 on, filling one group after another).
    columns: str = "interleaved"


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
    """A circuit block: `count` instances installed (area), `powered` of them
    drawing current during `when`.

    Timing: a component with `time_ns` is a step of the timeline, named after
    the component, lasting `time_ns` (one operation of every instance at
    once). serial=True (column_groups counts only): each instance converts
    its group's weight columns one after another, `time_ns` each, so the
    step lasts as long as the fullest group: with Mapping(columns=
    "interleaved") a tile's weight columns are dealt to its groups in turn,
    ceil(weight columns in the fullest tile / groups per tile) conversions
    (32 weight columns, 8 groups of 8: 4); with "contiguous" they fill the
    groups one after another, min(group size, weight columns) (here 8).
    A step runs in every activation or once per time bin: inferred from
    `count` (columns, rows, column groups, tiles: per activation; neurons and
    layer-wide parts: per time bin), or per="activation" / per="time_bin". By default a step
    starts when all steps of its level defined before it have ended (a
    time-bin step: also after all activations); `at` starts it at an anchor
    instead.

    Names and anchors: a bare name is a window ("cells": the step; "bin":
    the time bin; "activations": all activations of the bin); an anchor is a
    point, always with its edge: "name.start" or "name.end", with an offset
    in ns ("cells.start+1", "lif.end-0.5"); "bin.start", "bin.end",
    "activations.start", "activations.end". A per-activation step can only
    be anchored to per-activation steps; seen from the time bin, a
    per-activation step starts with the first activation and ends with the
    last.

    Power: `when` = a window ("vi": during that step; "bin": the whole time
    bin) or a (from, to) pair of anchors (("cells.start", "lif.end")).
    Default: the component's own step. An interval within the
    per-activation steps is powered in every activation (gated per
    activation); one reaching a time-bin step or the bin's edges is powered
    once per time bin. A component without a step or `when` only has area
    and event costs.

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

    count / powered: a rule name or {"rule": name, "value": n (fixed),
    "size": n (column groups), "gated": True}. powered="all" repeats `count`;
    powered="used_column_groups": only the groups holding a weight column
    (by Mapping.columns), its size taken from a column_groups count.
    "gated": only instances whose tile (tile rules), window ("outputs"), tile
    set ("output_bank") or layer receives at least one input spike in that
    activation or time bin.
    """
    name: str
    count: str | dict = "tiles"
    powered: str | dict = "all"
    time_ns: float | None = None
    serial: bool = False
    at: str | None = None
    per: str | None = None
    when: str | tuple | None = None
    supply_v: float = 1.1
    static_ua: float = 0.0
    event_pj: float = 0.0
    events: str = "activation"
    area_um2: float = 0.0
    model: str = "static"
    slice_gains: tuple[float, ...] | None = None
    group: str = "periphery"        # reporting label only


@dataclass(frozen=True)
class Step:
    """A step of the timeline, from the component of the same name."""
    name: str
    time_ns: float                  # per operation
    serial_size: int | None         # column group converted one column at a time, else None
    level: str                      # "activation" or "time_bin"
    at: tuple | None                # (name, "start"|"end", offset_ns); None = after earlier steps


@dataclass(frozen=True)
class Power:
    """A component's power interval: its level and (name, "start"|"end",
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
    specs: dict = field(default_factory=dict)    # descriptive only (SPEC_KEYS), no cost

    def __post_init__(self):
        self.steps, self.power = validate(self)


@dataclass
class Block:
    """A reusable piece of hardware: its components and, for a crossbar
    type, the Crossbar itself."""
    components: list[Component] = field(default_factory=list)
    crossbar: Crossbar | None = None


def compose(name, mapping, parts, *, activation_interval_ns=None, time_bin_interval_ns=None,
            specs=None):
    """Assemble an Architecture from exactly one crossbar (a Block, which
    brings the array's components and its step) plus any Components. Steps
    run in the order their components are given, unless placed with `at`.
    specs: descriptive specifications for reports, {key: value} with keys from
    SPEC_KEYS (e.g. {"tech": 40, "bitcell": "2T1R"}); they change no cost."""
    blocks = [p if isinstance(p, Block) else
              Block(components=[p]) if isinstance(p, Component) else None for p in parts]
    if None in blocks:
        raise ValueError(f"{name}: parts must be a crossbar and Components")
    crossbars = [b.crossbar for b in blocks if b.crossbar is not None]
    if len(crossbars) != 1:
        raise ValueError(f"{name}: compose needs exactly one crossbar, got {len(crossbars)}")
    return Architecture(name, crossbars[0], mapping, [c for b in blocks for c in b.components],
                        activation_interval_ns, time_bin_interval_ns, dict(specs or {}))


def rule_of(spec, *, activity=False):
    """Normalize a count/powered rule to {"rule", "value"?, "size"?, "gated"}."""
    rule = {"rule": spec} if isinstance(spec, str) else dict(spec)
    allowed = TILE_RULES + WINDOW_RULES + LAYER_RULES + (DATA_RULES + ("all",) if activity else ())
    if rule.get("rule") not in allowed:
        raise ValueError(f"unknown {'powered' if activity else 'count'} rule {spec!r}")
    if rule["rule"] == "fixed" and (isinstance(rule.get("value"), bool)
                                    or not isinstance(rule.get("value"), int) or rule["value"] < 0):
        raise ValueError("fixed rule needs a nonnegative integer 'value'")
    if rule["rule"] in ("column_groups", "used_column_groups"):
        _positive_int(rule.get("size"), f"{rule['rule']} size")
    rule.setdefault("gated", False)
    if not isinstance(rule["gated"], bool) or (rule["gated"] and not activity):
        raise ValueError(f"'gated' is a boolean for 'powered' rules only: {spec!r}")
    if set(rule) - {"rule", "value", "size", "gated"}:
        raise ValueError(f"unknown rule fields in {spec!r}")
    return rule


_ANCHOR = re.compile(r"\s*([A-Za-z_]\w*)\.(start|end)\s*(?:([+-])\s*(\d+(?:\.\d*)?|\.\d+))?\s*")


def parse_anchor(text):
    """"cells.start", "lif.end-0.5", "bin.end" -> (name, "start"|"end",
    offset_ns). The edge is always given: a bare name is a window, not a
    point."""
    match = _ANCHOR.fullmatch(text) if isinstance(text, str) else None
    if not match:
        raise ValueError(f"bad anchor {text!r}: a point is 'name.start' or 'name.end' "
                         "(optionally + or - ns), e.g. 'cells.start', 'lif.end-1', 'bin.end'")
    name, edge, sign, value = match.groups()
    offset = (-1 if sign == "-" else 1) * float(value) if value else 0.0
    return name, edge, offset


def powered_rule(c):
    """A component's normalized powered rule: "all" repeats its count (with
    its gating); used_column_groups without a size takes its count's."""
    count = rule_of(c.count)
    spec = {"rule": c.powered} if isinstance(c.powered, str) else dict(c.powered)
    if spec.get("rule") == "used_column_groups" and "size" not in spec \
            and count["rule"] == "column_groups":
        spec["size"] = count["size"]
    on = rule_of(spec, activity=True)
    return dict(count, gated=on["gated"]) if on["rule"] == "all" else on


def step_level(c):
    """How often a component's step runs: `per`, else from its count rule."""
    if c.per is not None:
        return c.per
    return "activation" if rule_of(c.count)["rule"] in TILE_RULES else "time_bin"


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
    if mp.columns not in COLUMN_PLACEMENTS:
        raise ValueError(f"mapping.columns must be one of {COLUMN_PLACEMENTS}")


def _step_of(c):
    """The Step (timeline step) a timed component defines."""
    if not re.fullmatch(r"[A-Za-z_]\w*", c.name) or c.name in (BIN, ACTIVATIONS):
        raise ValueError(f"{c.name}: a component with time_ns is a step named after it: its "
                         f"name must be an identifier other than '{BIN}' and '{ACTIVATIONS}'")
    _number(c.time_ns, f"{c.name}.time_ns")
    if c.per is not None and c.per not in LEVELS:
        raise ValueError(f"{c.name}: per must be one of {LEVELS}")
    size = None
    if c.serial:
        count = rule_of(c.count)
        if count["rule"] != "column_groups":
            raise ValueError(f"{c.name}: serial=True needs a column_groups count (the columns "
                             "each instance converts one after another)")
        size = count["size"]
    at = None if c.at is None else parse_anchor(c.at)
    return Step(c.name, float(c.time_ns), size, step_level(c), at)


def _point(anchor, levels, label):
    """Check an anchor's name; returns the level of the step it names
    (None for the time bin's own anchors)."""
    name, edge, _ = anchor
    if name == BIN or name == ACTIVATIONS:
        return None
    if name not in levels:
        raise ValueError(f"{label}: unknown step {name!r}; the steps are "
                         f"{', '.join(levels) or 'none'} (plus '{BIN}' and '{ACTIVATIONS}'; "
                         "a step is a component with time_ns)")
    return levels[name]


def _power_of(c, levels):
    """The Power interval of a component, or None if it draws no current."""
    when = c.when
    if when is None:
        if c.time_ns is None:
            return None
        when = c.name
    if isinstance(when, str):          # a window: that step (or "bin", "activations")
        if not re.fullmatch(r"\s*[A-Za-z_]\w*\s*", when):
            raise ValueError(f"{c.name}: when={when!r}: a single string is a window (a step "
                             "name or 'bin'); give an interval as a (from, to) pair of anchors, "
                             "e.g. ('cells.start', 'lif.end')")
        when = (f"{when.strip()}.start", f"{when.strip()}.end")
    if not isinstance(when, (tuple, list)) or len(when) != 2:
        raise ValueError(f"{c.name}: when must be a window (a step name or 'bin') or a "
                         f"(from, to) pair of anchors, got {c.when!r}")
    start, end = parse_anchor(when[0]), parse_anchor(when[1])
    found = [_point(a, levels, f"{c.name}.when") for a in (start, end)]
    level = "activation" if found == ["activation", "activation"] else "time_bin"
    return Power(level, start, end)


def validate(arch):
    """Check the architecture; returns its steps and its components' power
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
    if not isinstance(arch.specs, dict):
        raise ValueError(f"{arch.name}: specs must be a dict")
    unknown = set(arch.specs) - set(SPEC_KEYS)
    if unknown:
        raise ValueError(f"{arch.name}: unknown specs {sorted(unknown)}; use {', '.join(SPEC_KEYS)}")
    for key, value in arch.specs.items():
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ValueError(f"{arch.name}: specs[{key!r}] must be a string or a number")

    names, steps = set(), []
    for c in arch.components:
        if not isinstance(c.name, str) or not c.name or c.name in names:
            raise ValueError(f"component names must be unique: {c.name!r}")
        names.add(c.name)
        rule_of(c.count)
        if not isinstance(c.serial, bool):
            raise ValueError(f"{c.name}: serial must be True or False")
        if c.time_ns is not None:
            steps.append(_step_of(c))
        elif c.at is not None or c.serial or c.per is not None:
            raise ValueError(f"{c.name}: at, serial and per describe a step: give it time_ns")
    if not any(s.level == "activation" for s in steps):
        raise ValueError("a per-activation step is required (the crossbar's read)")
    levels = {s.name: s.level for s in steps}
    for s in steps:
        if s.at is None:
            continue
        target = _point(s.at, levels, f"{s.name}.at")
        if s.level == "activation" and target != "activation":
            raise ValueError(f"{s.name}: a per-activation step can only start at a "
                             f"per-activation step, got at={s.at[0]!r}")
        if s.at[0] == BIN and s.at[1] == "end":
            raise ValueError(f"{s.name}: a step cannot start at the end of the time bin")

    power = {}
    for c in arch.components:
        if c.model not in MODELS or c.events not in EVENTS:
            raise ValueError(f"{c.name}: model must be one of {MODELS}, events one of {EVENTS}")
        powered = powered_rule(c)
        for label in ("supply_v", "static_ua", "event_pj", "area_um2"):
            _number(getattr(c, label), f"{c.name}.{label}")
        power[c.name] = p = _power_of(c, levels)
        if (c.static_ua or c.model != "static") and p is None:
            raise ValueError(f"{c.name}: static or data-driven current needs time_ns or when")
        if powered["rule"] in DATA_RULES and p is not None and p.level != "activation":
            raise ValueError(f"{c.name}: {powered['rule']} counts spikes per activation; power "
                             "it within the per-activation steps")
        if powered["rule"] in DATA_RULES and c.event_pj and c.events == "time_bin":
            raise ValueError(f"{c.name}: {powered['rule']} events are counted per activation")
        if c.model == "reference_read" and not xb.reference_columns:
            raise ValueError(f"{c.name}: reference_read needs crossbar.reference_columns")
        if c.slice_gains is not None:
            if c.model != "slice_mirror" or len(c.slice_gains) != slices_per_group(arch):
                raise ValueError(f"{c.name}: slice_gains needs model 'slice_mirror' and one "
                                 f"gain per weight slice ({slices_per_group(arch)})")
            for gain in c.slice_gains:
                _number(gain, f"{c.name}.slice_gains")
    return steps, power
