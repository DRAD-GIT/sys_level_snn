"""Declarative physical components and an explicit serial critical-path schedule.

Config ``components`` is a list of objects patched by unique ``name`` onto the
architecture defaults (new names require all fields). Fields: name, model,
group, count, activity, timing, params. Nested count/activity/params objects are
shallow-merged for overrides; timing replaces the old value. No executable
expressions or implicit parameter references are accepted in JSON.

Groups: input_periphery, crossbar, output_periphery, lif.
count: {rule: one|fixed|tiles|physical_rows|physical_columns|logical_columns|
        output_bank|logical_outputs|column_groups, [columns_per_group: int], [value: int]}.
fixed uses value as an explicit per-layer count; row-phase activity repeats it
for every layer-wide phase (use tile rules for tile-local early completion).
Counts are installed instances; physical dimensions include tile padding.
logical_columns includes row-tile replicas; logical_outputs does not.
column_groups rounds columns_per_group UP separately in every physical tile.
activity: same rule/options plus frequency: stage (default)|image|bin|row_phase.
Activity describes powered instance-events, not installed instances. For stage
and row_phase, partially occupied row tiles stop after their last read phase.
image/bin charge once per image/bin respectively for each selected stage's
unit duration, irrespective of stage repetitions. These fixed models do not
infer activity from spikes.
timing: list of stage names, or "through_lif" (all stages through lif inclusive).
params for fixed_current: supply_v, current_ua, area_um2 (per installed unit).
conv_data_read/conv_reference_read: supply_v, area_um2; these use actual summed
input/G current, not a fixed current. c3_column is an explicitly FIXED macro
current model with the same params as fixed_current (no voltage transfer).

Config ``schedule`` is null for defaults or a full ordered list of objects:
{name: str, duration_ns: nonnegative number, repeat: image|bin|row_phase,
 owner: component name}. Each stage contributes latency exactly once, assigned
to owner only for backwards-compatible flat component latency reporting.
Components attached to a stage run in parallel. Exactly one final ``lif`` stage
with repeat=bin is required: membrane power integrates preceding stages and
spikes emit once per bin. Defaults: read/subtract/lif or read/VI/lif. To insert
an operation supply the complete schedule and attach components via timing.

Python extensions: register_model(name, evaluator, validator). Validator takes
params and must reject unknown/invalid fields. Evaluator(spec, context,
installed_count, powered_ns) returns energy in nJ. It must be pure and additive
across batches. Common area uses params['area_um2']; returned energy is checked
finite/nonnegative. Register before constructing/loading a config. No eval.
"""
from copy import deepcopy
from dataclasses import dataclass, fields
import math

GROUPS = ("input_periphery", "crossbar", "output_periphery", "lif")
RULES = {"one", "fixed", "tiles", "physical_rows", "physical_columns", "logical_columns",
         "output_bank", "logical_outputs", "column_groups"}
FREQUENCIES = {"stage", "image", "bin", "row_phase"}
MODELS = {}


def number(value, label, *, positive=False, integer=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or (value <= 0 if positive else value < 0)
            or (integer and not isinstance(value, int))):
        raise ValueError(f"{label} must be a finite {'positive' if positive else 'nonnegative'}"
                         f" {'integer' if integer else 'number'}")


def register_model(name, evaluator, validator):
    """Register a new electrical evaluator; duplicate names are rejected."""
    if not isinstance(name, str) or not name or name in MODELS:
        raise ValueError(f"Invalid or duplicate model: {name}")
    if not callable(evaluator) or not callable(validator):
        raise ValueError("Model evaluator and validator must be callable")
    MODELS[name] = (evaluator, validator)


def _params(params, keys):
    if set(params) != set(keys):
        raise ValueError(f"Model parameters must be exactly {sorted(keys)}")
    for key, value in params.items():
        number(value, key, positive=key == "supply_v")


def _fixed(spec, context, installed, powered_ns):
    p = spec["params"]
    return p["supply_v"] * p["current_ua"] * powered_ns * 1e-6


def _read(spec, context, installed, powered_ns):
    stages = selected_stages(spec, context.schedule)
    if len(stages) != 1 or stages[0]["repeat"] != "row_phase":
        raise ValueError("Conventional read models require one row_phase stage")
    current = context.data_current_a if spec["model"] == "conv_data_read" else context.reference_current_a
    return current * spec["params"]["supply_v"] * stages[0]["duration_ns"]


for _name in ("fixed_current", "c3_column"):
    register_model(_name, _fixed, lambda p: _params(p, ("supply_v", "current_ua", "area_um2")))
for _name in ("conv_data_read", "conv_reference_read"):
    register_model(_name, _read, lambda p: _params(p, ("supply_v", "area_um2")))


@dataclass
class EvaluationContext:
    config: object
    images: int
    bins_per_image: int
    row_tile_phases: list[int]
    column_tiles: int
    logical_outputs: int
    schedule: list[dict]
    data_current_a: float = 0.0
    reference_current_a: float = 0.0

    @property
    def phases(self):
        return max(self.row_tile_phases)

    def instances(self, rule, *, phased=False):
        """Installed count, or sum of powered instance counts over row phases."""
        rows = sum(self.row_tile_phases) if phased else len(self.row_tile_phases)
        global_phases = self.phases if phased else 1
        c = self.config
        return {
            "one": lambda: global_phases,
            "fixed": lambda: rule["value"] * global_phases,
            "tiles": lambda: rows * self.column_tiles,
            "physical_rows": lambda: rows * self.column_tiles * c.xbar_row,
            "physical_columns": lambda: rows * self.column_tiles * c.xbar_col,
            "logical_columns": lambda: rows * self.logical_outputs,
            "output_bank": lambda: global_phases * self.column_tiles * c.xbar_col,
            "logical_outputs": lambda: global_phases * self.logical_outputs,
            "column_groups": lambda: rows * self.column_tiles * math.ceil(c.xbar_col / rule["columns_per_group"]),
        }[rule["rule"]]()

    def powered_ns(self, spec):
        activity = spec["activity"]
        total = 0.0
        for stage in selected_stages(spec, self.schedule):
            frequency = activity.get("frequency", "stage")
            repeat = stage["repeat"] if frequency == "stage" else frequency
            events = self.images * (1 if repeat == "image" else self.bins_per_image)
            total += self.instances(activity, phased=repeat == "row_phase") * events * stage["duration_ns"]
        return total

    def stage_ns(self, stage):
        return (stage["duration_ns"] * self.images
                * (1 if stage["repeat"] == "image" else self.bins_per_image)
                * (self.phases if stage["repeat"] == "row_phase" else 1))


def selected_stages(spec, schedule):
    if spec["timing"] == "through_lif":
        return schedule[:next(i for i, s in enumerate(schedule) if s["name"] == "lif") + 1]
    return [s for s in schedule if s["name"] in spec["timing"]]


def default_specs(c):
    conventional = hasattr(c, "vread")
    specs = []

    def fixed(name, group, count, activity, timing, current, area, supply=None,
              model="fixed_current"):
        specs.append(dict(name=name, model=model, group=group,
                          count={"rule": count}, activity={"rule": activity}, timing=timing,
                          params=dict(supply_v=c.vdd if supply is None else supply,
                                      current_ua=current, area_um2=area)))

    if conventional:
        for name, model in (("crossbar", "conv_data_read"), ("reference_array", "conv_reference_read")):
            if name == "reference_array" and not c.reference_array:
                continue
            specs.append(dict(name=name, model=model, group="crossbar", count={"rule": "tiles"},
                              activity={"rule": "tiles"}, timing=["read"],
                              params=dict(supply_v=c.vdd, area_um2=c.xbar_area)))
        fixed("DA", "output_periphery", "physical_columns", "logical_columns", ["read"], c.DA_curr, c.DA_area)
        schedule = [dict(name="read", duration_ns=c.xbar_lat, repeat="row_phase", owner="crossbar")]
        if c.reference_array:
            fixed("reference_subtractor", "output_periphery", "output_bank", "logical_outputs",
                  ["subtract"], c.ref_sub_curr, c.ref_sub_area)
            schedule.append(dict(name="subtract", duration_ns=c.ref_sub_lat, repeat="row_phase", owner="reference_subtractor"))
    else:
        fixed("column", "crossbar", "physical_columns", "logical_columns", ["read"], c.col_curr, c.col_area, model="c3_column")
        fixed("column_driver", "input_periphery", "column_groups", "column_groups", ["read"], c.driver_curr, c.driver_area)
        specs[-1]["count"]["columns_per_group"] = c.driver_part
        specs[-1]["activity"]["columns_per_group"] = c.driver_part
        fixed("VI", "output_periphery", "physical_columns", "logical_columns", ["VI"], c.VI_curr, c.VI_area, c.VI_supp)
        schedule = [dict(name="read", duration_ns=c.col_lat, repeat="row_phase", owner="column"),
                    dict(name="VI", duration_ns=c.VI_lat, repeat="row_phase", owner="VI")]
    fixed("LIF", "lif", "output_bank", "logical_outputs", "through_lif", c.lif_curr, c.lif_area)
    schedule.append(dict(name="lif", duration_ns=c.lif_lat, repeat="bin", owner="LIF"))
    return specs, schedule


def _rule(rule, activity=False):
    if (not isinstance(rule, dict) or not isinstance(rule.get("rule"), str)
            or rule.get("rule") not in RULES):
        raise ValueError(f"Unknown count/activity rule: {rule}")
    allowed = {"rule"} | ({"frequency"} if activity else set())
    if rule["rule"] == "fixed":
        allowed.add("value")
        number(rule.get("value"), "fixed count value", integer=True)
    if rule["rule"] == "column_groups":
        allowed.add("columns_per_group")
        number(rule.get("columns_per_group"), "columns_per_group", positive=True, integer=True)
    if set(rule) - allowed:
        raise ValueError(f"Unknown count/activity fields: {set(rule) - allowed}")
    if activity and (not isinstance(rule.get("frequency", "stage"), str)
                     or rule.get("frequency", "stage") not in FREQUENCIES):
        raise ValueError("Unknown activity frequency")


def resolve_components(c):
    defaults, schedule = default_specs(c)
    if not isinstance(c.components, list):
        raise ValueError("components must be a list")
    specs = {s["name"]: s for s in defaults}
    seen = set()
    keys = {"name", "model", "group", "count", "activity", "timing", "params"}
    for patch in c.components:
        if not isinstance(patch, dict) or not isinstance(patch.get("name"), str) or not patch["name"]:
            raise ValueError("Each component requires a nonempty name")
        name = patch["name"]
        if name in seen or set(patch) - keys:
            raise ValueError(f"Duplicate component or unknown fields: {name}")
        seen.add(name)
        spec = deepcopy(specs.get(name, {}))
        for key, value in patch.items():
            if key in ("count", "activity", "params") and isinstance(value, dict):
                spec[key] = {**spec.get(key, {}), **deepcopy(value)}
            else:
                spec[key] = deepcopy(value)
        specs[name] = spec
    schedule = deepcopy(schedule if c.schedule is None else c.schedule)
    if not isinstance(schedule, list) or not schedule:
        raise ValueError("schedule must be a nonempty list")
    stage_names = set()
    for stage in schedule:
        if not isinstance(stage, dict) or set(stage) != {"name", "duration_ns", "repeat", "owner"}:
            raise ValueError("Stage requires name, duration_ns, repeat, owner")
        name = stage["name"]
        if not isinstance(name, str) or not name or name in stage_names:
            raise ValueError("Stage names must be unique nonempty strings")
        stage_names.add(name)
        number(stage["duration_ns"], "duration_ns")
        if (stage["repeat"] not in ("image", "bin", "row_phase")
                or not isinstance(stage["owner"], str) or stage["owner"] not in specs):
            raise ValueError("Unknown stage repeat or owner")
    if schedule[-1]["name"] != "lif" or schedule[-1]["repeat"] != "bin":
        raise ValueError("Final stage must be lif, repeated once per bin")
    for spec in specs.values():
        if (set(spec) != keys or spec.get("group") not in GROUPS
                or not isinstance(spec.get("model"), str) or spec.get("model") not in MODELS):
            raise ValueError(f"Incomplete component, unknown model or group: {spec.get('name')}")
        _rule(spec["count"])
        _rule(spec["activity"], activity=True)
        timing = spec["timing"]
        if timing != "through_lif" and (not isinstance(timing, list) or not timing
                or any(not isinstance(t, str) or t not in stage_names for t in timing)
                or len(set(timing)) != len(timing)):
            raise ValueError("timing must reference unique existing stages or through_lif")
        if not isinstance(spec["params"], dict):
            raise ValueError("params must be an object")
        MODELS[spec["model"]][1](spec["params"])
        number(spec["params"].get("area_um2"), "area_um2")
        if spec["model"] in ("conv_data_read", "conv_reference_read"):
            stages = selected_stages(spec, schedule)
            if not hasattr(c, "vread") or len(stages) != 1 or stages[0]["repeat"] != "row_phase":
                raise ValueError("Conventional read models require conventional hardware and one row_phase stage")
        if spec["model"] == "c3_column" and hasattr(c, "vread"):
            raise ValueError("c3_column requires C3 hardware")
    return list(specs.values()), schedule


def validate_config(c):
    for f in fields(c):
        name, value = f.name, getattr(c, f.name)
        if name in ("components", "schedule"):
            continue
        if name in ("temporal_map", "reference_array"):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean")
        elif name == "active_rows" and value is None:
            continue
        else:
            positive = name in ("min_res", "max_res", "vdd", "vread", "VI_supp", "xbar_row", "xbar_col", "active_rows", "driver_part")
            number(value, name, positive=positive,
                   integer=name in ("xbar_row", "xbar_col", "active_rows", "driver_part"))
    if c.max_res <= c.min_res:
        raise ValueError("max_res must exceed min_res")
    if c.active_rows is not None and c.active_rows > c.xbar_row:
        raise ValueError("active_rows must not exceed xbar_row")
    return resolve_components(c)


def evaluate_components(c, context):
    # Local import avoids coupling the registry to inference or tensor libraries.
    from clean_hardware import ComponentMetrics
    specs, schedule = validate_config(c)
    context.schedule = schedule
    result = {}
    for spec in specs:
        installed = context.instances(spec["count"])
        energy = MODELS[spec["model"]][0](spec, context, installed, context.powered_ns(spec))
        number(energy, f"{spec['name']} energy")
        result[spec["name"]] = ComponentMetrics(
            energy_nj=float(energy), area_mm2=installed * spec["params"]["area_um2"] * 1e-6,
            count=installed, group=spec["group"], model=spec["model"])
    for stage in schedule:
        result[stage["owner"]].latency_us += context.stage_ns(stage) * 1e-3
    return result
