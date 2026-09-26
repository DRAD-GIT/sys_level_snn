"""Description of a CIM macro: crossbar, precision, mapping, timeline, components.

Units: resistance ohm, voltage V, current uA, time ns, event energy pJ, area
um^2 per installed instance.

A spiking layer runs one timestep per SNN time bin. Within a timestep the
crossbar is read (read-level stages) once per row phase (`active_rows` smaller
than the tile height) and, with the "sequential" conv mapping, once per
convolution window. Input spikes of every time bin carry the same weight in
the LIF membrane, so input precision is the number of time bins.

Weights of `weight_bits` are stored in cells of `cell_bits` (bit-slicing: more
columns per weight). Convolutions are unrolled into windows:
  "sequential": one copy of the kernel weights; the windows are applied one
                after another (column tiles still work in parallel);
  "parallel":   one copy of the kernel weights per window; all windows at once.
"""
from dataclasses import dataclass, field

# Count/activity rules, by the unit an instance belongs to:
TILE_RULES = ("tiles", "physical_rows", "physical_columns", "used_columns", "column_groups")
WINDOW_RULES = ("outputs", "output_bank")   # neurons / output positions of a window
LAYER_RULES = ("one", "fixed")
DATA_RULES = ("spiking_rows",)              # word lines carrying a spike, per read
ENCODINGS = ("twos_complement", "offset", "differential", "analog")
MODELS = ("static", "crossbar_read", "reference_read")
EVENTS = ("read", "timestep", "output_spike")
LEVELS = ("read", "timestep")
MAPPINGS = ("sequential", "parallel")


@dataclass
class Crossbar:
    rows: int = 64                  # physical rows (word lines) per tile
    cols: int = 64                  # physical columns per tile
    cell_bits: int = 1              # bits stored per cell
    r_on: float = 20e3              # lowest resistance (highest cell level)
    r_off: float = 200e3            # highest resistance (cell level 0)
    v_read: float = 0.2             # read voltage across a selected cell
    active_rows: int | None = None  # rows enabled per read; None = all rows
    reference_columns: bool = False  # analog encoding: one G(0) column per output


@dataclass
class Precision:
    # None: unquantized weights (analog encoding only).
    weight_bits: int | None = 4
    # twos_complement / offset: bit-sliced integer codes, sign or offset
    # handled after the array; differential: positive and negative columns;
    # analog: one multi-level cell per weight, G linear in the weight value.
    weight_encoding: str = "twos_complement"


@dataclass
class Stage:
    """A step of the timeline. level "read" repeats every crossbar read;
    "timestep" runs once per time bin, where the block of all reads of that
    time bin is the pseudo-stage "reads".

    after: stages (same level) this one waits for. None = the previous stage
    of its level (serial); [] = the start of its level (parallel).
    offset_ns: start relative to the latest `after` end; negative = overlap.
    """
    name: str
    duration_ns: float
    level: str = "read"
    after: list[str] | None = None
    offset_ns: float = 0.0


@dataclass
class Component:
    """A circuit block: `count` instances installed (area); `on` of them
    powered during the `during` stages ("timestep" = the whole time bin).

    energy = supply_v * static_ua * powered time      (bias/static current)
           + event_pj * events                        (per read, time bin or output spike)
    model "crossbar_read" / "reference_read": the array's cell current,
    computed from the spikes and conductances, drawn from supply_v during
    its single read-level stage.

    count / on: a rule name or {"rule": name, "value": n (fixed),
    "size": n (column_groups), "gated": True}. on="all" repeats `count`.
    "gated": only instances whose tile (tile rules), window (window rules) or
    layer receives at least one input spike in that read or time bin.
    """
    name: str
    count: str | dict = "tiles"
    on: str | dict = "all"
    during: list[str] = field(default_factory=list)
    supply_v: float = 1.1
    static_ua: float = 0.0
    event_pj: float = 0.0
    events: str = "read"
    area_um2: float = 0.0
    model: str = "static"
    group: str = "periphery"        # reporting label only


@dataclass
class Architecture:
    name: str
    crossbar: Crossbar
    precision: Precision
    stages: list[Stage]
    components: list[Component]
    conv_mapping: str = "sequential"
    read_interval_ns: float | None = None      # pipelined reads; None = back to back
    timestep_interval_ns: float | None = None  # overlapping time bins; None = serial

    def __post_init__(self):
        validate(self)

    @property
    def read_stages(self):
        return {s.name for s in self.stages if s.level == "read"}


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


def _positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


def _number(value, label, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value \
            or value < 0 or (positive and value == 0):
        raise ValueError(f"{label} must be a {'positive' if positive else 'nonnegative'} number")


def validate(arch):
    xb, pr = arch.crossbar, arch.precision
    for label in ("rows", "cols", "cell_bits"):
        _positive_int(getattr(xb, label), f"crossbar.{label}")
    if xb.active_rows is not None:
        _positive_int(xb.active_rows, "crossbar.active_rows")
        if xb.active_rows > xb.rows:
            raise ValueError("active_rows cannot exceed rows")
    for label in ("r_on", "r_off", "v_read"):
        _number(getattr(xb, label), f"crossbar.{label}", positive=True)
    if xb.r_off <= xb.r_on:
        raise ValueError("r_off must exceed r_on")
    if pr.weight_encoding not in ENCODINGS:
        raise ValueError(f"weight_encoding must be one of {ENCODINGS}")
    if pr.weight_bits is None:
        if pr.weight_encoding != "analog":
            raise ValueError("weight_bits=None (unquantized) needs the analog encoding")
    else:
        _positive_int(pr.weight_bits, "precision.weight_bits")
        if pr.weight_bits < 2:
            raise ValueError("signed weights need weight_bits >= 2")
    if xb.reference_columns and pr.weight_encoding != "analog":
        raise ValueError("reference_columns applies to the analog encoding only")
    if arch.conv_mapping not in MAPPINGS:
        raise ValueError(f"conv_mapping must be one of {MAPPINGS}")

    names = set()
    for stage in arch.stages:
        if not stage.name or stage.name in names or stage.name in ("reads", "timestep"):
            raise ValueError(f"stage names must be unique and not 'reads'/'timestep': {stage.name!r}")
        if stage.level not in LEVELS:
            raise ValueError(f"stage {stage.name}: level must be one of {LEVELS}")
        _number(stage.duration_ns, f"stage {stage.name} duration_ns")
        names.add(stage.name)
    if not arch.read_stages:
        raise ValueError("at least one read-level stage is required")
    for label in ("read_interval_ns", "timestep_interval_ns"):
        if getattr(arch, label) is not None:
            _number(getattr(arch, label), label, positive=True)

    component_names = set()
    for c in arch.components:
        if not c.name or c.name in component_names:
            raise ValueError(f"component names must be unique: {c.name!r}")
        component_names.add(c.name)
        if c.model not in MODELS or c.events not in EVENTS:
            raise ValueError(f"{c.name}: model must be one of {MODELS}, events one of {EVENTS}")
        rule_of(c.count)
        on = rule_of(c.on, activity=True)
        for label in ("supply_v", "static_ua", "event_pj", "area_um2"):
            _number(getattr(c, label), f"{c.name}.{label}")
        for stage in c.during:
            if stage != "timestep" and stage not in names:
                raise ValueError(f"{c.name}: unknown stage {stage!r}")
        read_level = bool(c.during) and set(c.during) <= arch.read_stages
        if c.static_ua and not c.during:
            raise ValueError(f"{c.name}: static current needs 'during' stages")
        if on["rule"] in DATA_RULES and c.during and not read_level:
            raise ValueError(f"{c.name}: {on['rule']} counts spikes per read; use read-level stages")
        if on["rule"] in DATA_RULES and c.event_pj and c.events == "timestep":
            raise ValueError(f"{c.name}: {on['rule']} events are counted per read")
        if c.model != "static" and (len(c.during) != 1 or not read_level):
            raise ValueError(f"{c.name}: {c.model} needs exactly one read-level stage")
        if c.model == "reference_read" and not xb.reference_columns:
            raise ValueError(f"{c.name}: reference_read needs crossbar.reference_columns")
