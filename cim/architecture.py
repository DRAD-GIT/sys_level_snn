"""Modular description of a CIM macro: crossbar, precision, timeline, components.

Units: resistance ohm, voltage V, current uA, time ns, energy pJ (events),
area um^2 per installed instance.

Execution hierarchy for one inference:
    timestep (T per inference; SNN time bins)
      read (reads per timestep = input slices x row phases)
Input slices: an input of `input_bits` applied `input_bits_per_read` bits at a
time (1 = binary word lines, i.e. bit-serial). Row phases: rows enabled per
read (`active_rows`) smaller than the tile height.
Weight slices: a `weight_bits` weight stored in cells of `cell_bits` each.
"""
from dataclasses import dataclass, field

# Instance rules. "Phased" counts (powered per read) replace the row-tile
# count by the per-tile phase counts, so partially filled tiles stop early.
RULES = ("one", "fixed", "tiles", "physical_rows", "physical_columns", "used_columns",
         "column_groups", "output_bank", "outputs")
ENCODINGS = ("twos_complement", "offset", "differential", "analog")
MODELS = ("static", "crossbar_read", "reference_read")
EVENTS = ("read", "timestep", "output_spike")
LEVELS = ("read", "timestep")


@dataclass
class Crossbar:
    rows: int = 64                 # physical rows per tile
    cols: int = 64                 # physical columns per tile
    cell_bits: int = 1             # bits stored per cell
    r_on: float = 20e3             # lowest resistance (highest cell level)
    r_off: float = 200e3           # highest resistance (cell level 0)
    v_read: float = 0.2            # read voltage across a selected cell
    active_rows: int | None = None  # rows enabled per read; None = all rows
    reference_columns: bool = False  # analog encoding: offset-reference array


@dataclass
class Precision:
    weight_bits: int = 4
    # twos_complement / offset: bit-sliced codes, signs/offset handled after
    # the array; differential: separate positive and negative columns;
    # analog: one multi-level cell per weight, float weights mapped linearly.
    weight_encoding: str = "twos_complement"
    input_bits: int = 1            # bits per input value per timestep (1 = spikes)
    input_bits_per_read: int = 1   # word-line / DAC resolution


@dataclass
class Stage:
    """A step of the timeline. level "read" repeats every crossbar read;
    "timestep" runs once per timestep, where the whole block of reads appears
    as the pseudo-stage "reads".

    after: stages (same level) this one waits for. None = the previous stage
    of its level (serial); [] = the start of its level (parallel with others).
    offset_ns: start relative to the latest `after` end; negative = overlap.
    """
    name: str
    duration_ns: float
    level: str = "read"
    after: list[str] | None = None
    offset_ns: float = 0.0


@dataclass
class Component:
    """A circuit block, replicated `count` times (area) with `on` of those
    powered during the `during` stages ("timestep" = the whole timestep).

    Energy = supply_v * static_ua * powered time  (static/bias current)
           + event_pj * events                    (e.g. per conversion or spike)
    model "crossbar_read" / "reference_read": the array's cell current,
    computed from the actual inputs and conductances, drawn from supply_v
    during its single read-level stage.
    count/on: rule name or {"rule": ..., "value"/"size": ...}; on "all" = count.
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
    group: str = "periphery"       # reporting label only


@dataclass
class Architecture:
    name: str
    crossbar: Crossbar
    precision: Precision
    stages: list[Stage]
    components: list[Component]
    read_interval_ns: float | None = None      # pipelined reads; None = back to back
    timestep_interval_ns: float | None = None  # overlapping timesteps; None = serial

    def __post_init__(self):
        validate(self)


def _positive_int(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


def _number(value, label, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value \
            or value < 0 or (positive and value == 0):
        raise ValueError(f"{label} must be a {'positive' if positive else 'nonnegative'} number")


def rule_of(spec):
    """Normalize a count/on rule to a dict."""
    rule = {"rule": spec} if isinstance(spec, str) else dict(spec)
    if rule.get("rule") not in RULES + ("all",):
        raise ValueError(f"unknown rule {spec!r}")
    if rule["rule"] == "fixed":
        if isinstance(rule.get("value"), bool) or not isinstance(rule.get("value"), int) \
                or rule["value"] < 0:
            raise ValueError("fixed rule needs a nonnegative integer 'value'")
    if rule["rule"] == "column_groups":
        _positive_int(rule.get("size"), "column_groups size")
    if set(rule) - {"rule", "value", "size"}:
        raise ValueError(f"unknown rule fields in {spec!r}")
    return rule


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
    for label in ("weight_bits", "input_bits", "input_bits_per_read"):
        _positive_int(getattr(pr, label), f"precision.{label}")
    if pr.weight_encoding == "differential" and pr.weight_bits < 2:
        raise ValueError("differential encoding needs weight_bits >= 2 (sign + magnitude)")
    if xb.reference_columns and pr.weight_encoding != "analog":
        raise ValueError("reference_columns applies to the analog encoding only")

    names = set()
    for stage in arch.stages:
        if not stage.name or stage.name in names or stage.name in ("reads", "timestep"):
            raise ValueError(f"stage names must be unique and not 'reads'/'timestep': {stage.name!r}")
        if stage.level not in LEVELS:
            raise ValueError(f"stage {stage.name}: level must be one of {LEVELS}")
        _number(stage.duration_ns, f"stage {stage.name} duration_ns")
        names.add(stage.name)
    if not any(s.level == "read" for s in arch.stages):
        raise ValueError("at least one read-level stage is required")
    for label in ("read_interval_ns", "timestep_interval_ns"):
        if getattr(arch, label) is not None:
            _number(getattr(arch, label), label, positive=True)

    component_names = set()
    for c in arch.components:
        if not c.name or c.name in component_names:
            raise ValueError(f"component names must be unique: {c.name!r}")
        component_names.add(c.name)
        if c.model not in MODELS:
            raise ValueError(f"{c.name}: model must be one of {MODELS}")
        if c.events not in EVENTS:
            raise ValueError(f"{c.name}: events must be one of {EVENTS}")
        rule_of(c.count), rule_of(c.on)
        if rule_of(c.count)["rule"] == "all":
            raise ValueError(f"{c.name}: 'all' is only valid for 'on'")
        for label in ("supply_v", "static_ua", "event_pj", "area_um2"):
            _number(getattr(c, label), f"{c.name}.{label}")
        for stage in c.during:
            if stage != "timestep" and stage not in names:
                raise ValueError(f"{c.name}: unknown stage {stage!r}")
        if c.static_ua and not c.during:
            raise ValueError(f"{c.name}: static current needs 'during' stages")
        if c.model != "static":
            stages = [s for s in arch.stages if s.name in c.during]
            if len(c.during) != 1 or len(stages) != 1 or stages[0].level != "read":
                raise ValueError(f"{c.name}: {c.model} needs exactly one read-level stage")
            if c.model == "reference_read" and not xb.reference_columns:
                raise ValueError(f"{c.name}: reference_read needs crossbar.reference_columns")
