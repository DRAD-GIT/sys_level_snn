# SNN inference and modular CIM hardware estimation

This directory evaluates trained **N-MNIST LeNet** and **IBM DVS-Gesture** SNNs on conventional current-mode CIM and C3CIM macro-cost models. It runs software inference, observes the inputs of weighted layers, and estimates hardware metrics for those layers. Hardware estimation does **not** replace the software forward pass or simulate analog classification accuracy.

## 1. Repository layout

```text
run.py                 <- MAIN SCRIPT: define both hardware configs here and run
models/                SNN network + dataset classes, and their SLAYER parameter YAMLs
  base.py              shared base classes and ModelSpec (checkpoint, YAML, layers to profile)
  nmnist.py            N-MNIST LeNet + dataset loader, SPEC
  gesture.py           DVS-Gesture network + dataset loader, SPEC
  nmnist.yaml, gesture.yaml
pretrained/            trained checkpoints (nmnist_lenet.pt, gesture.pt)
datasets/              place the datasets here (see each folder's README)
  N-MNIST/
  DVS_Gesture/
hardware/              CIM hardware cost models
  configs.py           ConvHardwareConfig / C3HardwareConfig defaults, JSON loader
  components.py        component registry, counting/activity rules, stage schedule
  mapping.py           quantization, weight->conductance mapping, tile geometry
  estimators.py        per-layer energy/latency/area for each architecture
  metrics.py           result containers and text summaries
  examples/            JSON override examples (see section 9)
evaluation/            pipeline used by run.py
  runner.py            load model, run inference, hook layers, accumulate metrics
  report.py            text log, JSON and CSV export, final metrics table
tests/                 hand-calculation and regression tests
```

## 2. Quick start

Install Python, PyTorch, NumPy, PyYAML (`requirements.txt`) and `slayerSNN`, including its compiled backend. Put the datasets in `datasets/` (see `datasets/*/README.md`). Then:

1. Open `run.py`. Choose `MODEL`, `BATCH_SIZE`, `MAX_BATCHES` and `WEIGHT_BITS` in **RUN SETTINGS**. Edit the numbers in `CONVENTIONAL` and `C3CIM`. Every field has its unit in a comment.
2. Run it:

```bash
python run.py                                   # uses the settings in run.py
python run.py --model gesture                   # override a run setting once
python run.py --model nmnist -b 12 --batches -1 # full N-MNIST test set (slow)
python run.py -n 4                              # quantize weights to 4 bits first
python run.py --c3-config hardware/examples/c3_driver_override.json   # C3 config from JSON

python -m unittest discover -s tests -v         # regression tests
```

The console, `logs/<model>_B<batch>_N<batches>_b<bits>.txt`, a JSON per architecture and `logs/comparison_summary.csv` all receive the results. The run ends with a **FINAL METRICS** table: energy and latency per inference, power, TOPS/W, area and GOPS/mm² for each architecture.

Remove an entry from `ARCHITECTURES` in `run.py` to skip that architecture. `-n 0` (default) keeps the trained weights; `-n 4` quantizes the software weights before inference. `-k` controls the N-MNIST standard-deviation range and `-d` controls decimal rounding of quantization levels. Quantization can change accuracy and layer activity; it is not merely an energy scaling factor. Gesture batch size is capped at two. Mapping is spatially parallel; temporal mapping is not supported.

The checkpoints are full pickled modules that were saved when the model classes lived in `demo/nets/`. `models.load_pretrained` maps those old import paths to `models/`, so the checkpoints load unchanged. Only load trusted checkpoints, because loading uses `torch.load(weights_only=False)`. Dataset paths in the YAMLs are resolved from the repository root, whatever your working directory. Edit the YAMLs to point elsewhere (absolute paths work; keep the trailing slash on directories). To add a model, write its classes and a `SPEC` in a new `models/<name>.py` and register it in `models/__init__.py`.

## 3. End-to-end flow

```text
Checkpoint + dataset + quantization options
                    |
           Software SNN inference
                    |
        Hook each convolution/dense input
                    |
        Layer shape, weights, spike activity
                    |
             Tile mapping / row phases
                    |
       +------------+-------------+
       |                          |
Conventional CIM              C3CIM macro model
spikes + conductances         fixed column-source current
-> physical read current      + shared driver + VI costs
       |                          |
       +------------+-------------+
                    |
        Configured component evaluators
                    |
        input_periphery | crossbar
        output_periphery | lif
                    |
        Image-weighted layer aggregation
                    |
        Layer/network logs + JSON + CSV
```

Only weighted convolution and dense modules are profiled as VMM layers. Pooling/dropout and the rest of the software network are not separately charged as hardware blocks. LIF hardware is accounted for explicitly. Hardware activity currently interprets any nonzero hooked input as a binary spike. No activation cache is persisted: changing hardware parameters reruns inference.

## 4. Physical mapping: installed resources are not activity

Let:

- `K` = flattened input vector length, e.g. input channels × kernel height × kernel width.
- `O` = total logical outputs in the spatially unrolled layer.
- `R`, `C` = physical tile row and column capacities.
- `A` = rows enabled per read (`active_rows`, or `R` if omitted).
- `Nr = ceil(K/R)`, `Nc = ceil(O/C)`; installed data tiles = `Nr × Nc`.
- `p_i = ceil(valid_rows_in_row_tile_i / A)`.
- `P = max(p_i)`; `S = sum(p_i)`.

All tiles for a layer are assumed available and operate concurrently. Row phases within a tile are sequential. Each input image/time bin executes sequentially; no inter-image or inter-layer pipeline overlap is credited. Tile currents for matching outputs are assumed to combine ideally without wire/routing/extra accumulator costs.

```text
Example: K=96, O=2, tile=64x64, active_rows=8

  Tile A: 64 valid rows          Tile B: 32 valid rows
  8 phases                      4 phases
  2 used column positions       2 used column positions
          |                               |
          +--- matching output paths -----+
                          |
              2 logical combined outputs
                          |
                       LIF stage
```

This example has two physical data tiles. Each has 64 physical column slots, but only two useful columns. The default LIF allocation reserves a **64-position output bank**, one LIF per physical combined column position, with only two useful outputs powered. It does **not** mean one LIF is multiplexed across 32 distinct outputs. Row-tile contributions for the *same* output integrate into the same membrane. The partially filled second tile finishes after four phases, while the layer takes eight phases.

Area counts installed hardware, including unused capacity where specified. Energy uses the separately configured activity rule. Thus a component can have 128 installed column instances but only `O × S = 24` active column-phase events per time bin. This distinction is essential for partially occupied tiles.

## 5. Conventional versus C3CIM electrical behavior

### Conventional current-mode CIM

Weights are mapped affinely to positive conductance, including zero in the mapping range:

```text
G(w) = alpha*w + G(0)
I_data = Vread * sum(x_i * G(w_i))
I_reference = Vread * sum(x_i * G(0))
I_signed = I_data - I_reference
```

Both data and reference currents dissipate energy. The default physical reference array costs a second set of array tiles, operating in parallel with the data array. Subtraction circuitry has separately configurable cost. Its scalar defaults are zero and represent ideal subtraction, not measured zero-cost hardware. Signed weights require offset-reference handling.

Each row is read once per time bin regardless of how many phases are needed. Therefore splitting reads into eight phases does **not** multiply the total resistive read energy by eight. It does increase repeated peripheral costs and LIF active duration. The conventional source-energy convention inherited by the estimator is `Vdd × I_read × read_duration`; this is a supply-side estimate, not the same quantity as array-only dissipation `Vread × I_read`.

### C3CIM: constant-current drive, voltage-domain MAC, VI output

The intended physical signal path is:

```text
Input-controlled programmed resistance network
                    |
       Constant-current-driven column
                    |
             MAC voltage per tile
                    |
          Per-tile VI conversion
                    |
          Combine matching currents
                    |
          LIF membrane integration
```

**Implemented limitation:** the C3 evaluator estimates macro costs from configured column, driver, VI, and LIF currents. It does not yet calculate the voltage-domain MAC or VI output current from actual inputs/resistances. No transfer function has been invented. Resistance parameters must not be mistaken for an implemented C3 electrical simulation. A future electrical plugin needs the circuit equation/lookup for MAC voltage, VI transfer, voltage limits, signed-weight correction, and relevant signal-dependent power.

`col_curr=0.1` means **0.1 microampere = 100 nA per active column**. `driver_part=32` means one constant-current driver/bias block is shared by 32 physical columns **within a tile**. `driver_curr=11.87` is overhead current per shared block, not the 100 nA column current. With 64 columns, each tile has two driver blocks. Changing `driver_part` changes driver count, area, and overhead energy; it does not serialize columns or change the specified per-column current.

The reported power assumes these two currents describe non-overlapping contributions. If a measured macro current already includes column-source current or VI power, do not add the same contribution again as another component.

## 6. Units and aggregation

| Quantity | Configuration/internal unit | Reported unit |
|---|---|---|
| Voltage | V | V |
| Resistance | ohm | ohm |
| Component current | microampere | microampere |
| Component area | square micrometre | square millimetre |
| Stage duration | ns | microsecond |
| Energy | nJ | nJ per inference |

For fixed-current circuits, `E[nJ] = V[V] × I[uA] × powered-instance-time[ns] × 1e-6`. Area is installed count × area per instance. Latency is determined by the schedule, not by summing the active duration of overlapping components.

Image energy, latency, and operations are accumulated then divided by the number of processed images, not the number of batches. Area is physical installed area and is not divided by image count. Layer energy/latency/area are summed for the sequential network model. Reported `average_power_mw = mean_energy_nj / mean_latency_us`; `TOPS/W = operations / energy_nj × 1e-3`, using **two operations per MAC**. This is a ratio of aggregate quantities, not an unweighted average of layer efficiencies. Hardware latency excludes dataset acquisition time and host/GPU wall-clock runtime.

Mean conventional read current is summed across physical read paths and averaged over row phases and simulation bins. C3 read current is the configured column-source current weighted by active column phases. Neither should be confused with signed output current or total supply current of every peripheral. The network current field sums layer means; it is not a simultaneous network-current measurement.

## 7. Worked C3 example derived from the existing classes

Use `K=96`, `O=2`, `R=C=64`, `A=8`, and one image with one simulation bin. The scalar defaults originate in the existing hardware classes:

| Component | Parameters |
|---|---|
| Column | 0.1 uA, 1.1 V, 50 ns, 4.27 um² per column |
| Shared driver | 11.87 uA, 1.1 V, one per 32 columns, 86.36 um² |
| VI | 24.3 uA, 1 V, 10 ns, 29.79 um² |
| LIF | 6 uA, 1.1 V, final latency 2 ns, 86.79 um² |

Here `p=[8,4]`, `P=8`, `S=12`. Column/VI active-instance phases are `2×12=24`. Driver active-instance phases are `2 drivers per row tile × 12=24`. The 64-position combined output bank has two useful LIFs active during the 482 ns computation.

| Component | Installed instances | Area (um²) | Energy per bin (nJ) |
|---|---:|---:|---:|
| Column | 128 | 546.56 | 0.000132 |
| Shared driver | 4 | 345.44 | 0.0156684 |
| VI | 128 | 3813.12 | 0.005832 |
| LIF | 64 | 5554.56 | 0.0063624 |
| **Total** | — | **10259.68** | **0.0279948** |

Latency is `8×(50+10)+2 = 482 ns`, not eight times the LIF final delay. Area is `0.01025968 mm²`. For fixed currents and unchanged geometry, 300 simulation bins multiply latency and energy by 300, but leave area unchanged. Actual model layers have different shapes; this hand example is not a measured model result.

The original legacy scripts and this estimator are **not numerically identical**: the refactored accounting adds row phases/reference costs, explicitly includes LIF integration power, uses two operations per MAC, and distinguishes fixed hardware area from repeated-image costs.

### Matching conventional hand example

For the same `K=96`, `O=2`, tile geometry and one bin, assume all input spikes are one, all weights are +1, and the explicit mapping range is [-1,+1]. Use the conventional defaults `min_res=2000`, `max_res=200000`, `vread=0.1`, `vdd=1.1`, read time 4.5 ns, final LIF time 2 ns, and zero-cost ideal subtraction.

`Gmax=500 uS`, `G(0)=252.5 uS`. Summed across the rows/outputs, data read current is 9.6 mA and reference current is 4.848 mA **summed over all read phases**. Average read-phase current is their sum divided by eight; these are not simultaneously enabled full-vector currents in the phased implementation.

| Component | Installed instances | Energy per bin (nJ) |
|---|---:|---:|
| Data array | 2 tiles | 0.04752 |
| Reference array | 2 tiles | 0.0239976 |
| DA | 128 column circuits | 0.00072468 |
| Reference subtraction | 64 output positions | 0 (ideal default) |
| LIF | 64 output positions | 0.0005016 |
| **Total** | — | **0.07274388** |

Latency is `8×4.5+2=38 ns`; area is `(4×136.67 + 128×30.22 + 64×86.79) um² = 0.0099694 mm²`. The explicit mapping range is important: the inference path normally derives the range from each layer's weights/quantization levels, so a synthetic all-positive layer need not use [-1,+1] automatically.

## 8. Results and comparison tables

Text logs under `logs/` show layer/network totals and component breakdowns. JSON exports preserve numeric precision and configuration identity; the CSV `logs/comparison_summary.csv` stores comparison-ready rows keyed by model, architecture, quantization, configuration and evaluated sample count. Both architectures are added per model run, so running both models creates their four comparison rows. Repeating the same configuration/sample count replaces that row. Separate runs can update the CSV safely using its lock.

Use the CSV's energy per inference, latency per inference, average power, TOPS/W, and inferences/J for comparisons, while preserving sample count and configuration identity. One-image smoke tests validate execution only; they are **not** dataset-average benchmark results. Existing text log names can be overwritten by another run with the same model/batch/precision settings. JSON/CSV configuration IDs distinguish hardware configurations.

## 9. Component configuration: edit numbers without editing inference

### File responsibilities

| File | Responsibility |
|---|---|
| `run.py` | Hardware definitions and run settings; calls the pipeline |
| `evaluation/runner.py` | Load model/data, run inference, hooks, averaging |
| `evaluation/report.py` | Text log, JSON/CSV export, final table |
| `hardware/configs.py` | Hardware scalar defaults and JSON loading |
| `hardware/mapping.py` | Weight quantization/mapping, tile geometry |
| `hardware/estimators.py` | Architecture electrical inputs per layer |
| `hardware/metrics.py` | Metric containers, aggregation, summaries |
| `hardware/components.py` | Component registry, counting/activity rules, default architecture composition, schedule validation/evaluation |
| `hardware/examples/c3_driver_override.json` | C3 shared-driver count override |
| `hardware/examples/c3_added_stage.json` | C3 input buffer plus a new serial stage |
| `tests/` | Hand calculations, configuration validation, plugins, batching, stages, export regression tests |

You can put `components` and `schedule` directly into the configs in `run.py`, or keep them in a JSON file passed with `--conv-config`/`--c3-config`. A JSON file **replaces** that architecture's `run.py` config: omitted fields fall back to the class defaults, not to the values in `run.py`.

Each hardware config accepts scalar legacy parameters plus optional `components` and `schedule`. Omitted values use the dataclass defaults. Components are named entries, organized into **four reporting groups**: `input_periphery`, `crossbar`, `output_periphery`, `lif`. Changing a group label only changes reporting: it does not change placement, replication, activity, or timing.

The built-in composition is:

| Architecture | Group | Component names | Model |
|---|---|---|---|
| Conventional | input_periphery | None by default | Add measured circuitry if not already included elsewhere |
| Conventional | crossbar | `crossbar`, `reference_array` | Actual summed data/reference currents |
| Conventional | output_periphery | `DA`, `reference_subtractor` | Fixed current |
| Conventional | lif | `LIF` | Fixed current while integrating and emitting |
| C3 | input_periphery | `column_driver` | Shared driver overhead, fixed current |
| C3 | crossbar | `column` | `c3_column`: fixed 100 nA per active column by default |
| C3 | output_periphery | `VI` | Fixed conversion-stage supply current |
| C3 | lif | `LIF` | Fixed current while integrating and emitting |

The group assignment for the C3 bias driver is organizational; it is not a claim that this circuit physically sits on the row-input side.

### Component fields

A **new** component specifies all these fields:

```json
{
  "name": "extra_bias",
  "model": "fixed_current",
  "group": "input_periphery",
  "count": {"rule": "tiles"},
  "activity": {"rule": "tiles", "frequency": "stage"},
  "timing": ["read"],
  "params": {"supply_v": 1.1, "current_ua": 2.0, "area_um2": 5.0}
}
```

- `name`: unique identity used for overrides, reports, and schedule attribution.
- `model`: registered Python evaluator. Built-ins are `fixed_current`, `c3_column`, `conv_data_read`, `conv_reference_read`.
- `group`: one of the four group names.
- `count`: installed-instance rule, used for area.
- `activity`: powered-instance rule and event frequency, used for energy.
- `timing`: stage names during which it is powered, or `"through_lif"`.
- `params`: electrical/area parameters. Fixed-current models require exactly `supply_v`, `current_ua`, `area_um2`. Conventional read models require `supply_v`, `area_um2`, and obtain physical current from the architecture evaluator.

Using the same name as a default component patches it rather than adding a duplicate. `count`, `activity`, and `params` are shallow-merged with that default; other fields replace their previous values. New names require a complete specification. Explicit component parameters override their legacy scalar-derived defaults. For example, overriding `column_driver.params.current_ua` takes precedence over `driver_curr` for that component. Exported `resolved_components` shows the actual values evaluated.

### Exact counting rules (no ambiguous per-column shorthand)

| Rule | Installed instances | Meaning |
|---|---:|---|
| `one` | 1 | One circuit for the evaluated layer |
| `fixed` | Explicit `value` | User-specified number of circuits per layer |
| `tiles` | `Nr × Nc` | One circuit per tile |
| `physical_rows` | `Nr × Nc × R` | All physical row slots, including padding |
| `physical_columns` | `Nr × Nc × C` | All physical tile-column slots |
| `logical_columns` | `Nr × O` | Occupied column slots across row tiles |
| `output_bank` | `Nc × C` | Padded column positions after combining row-tile outputs |
| `logical_outputs` | `O` | Used output positions after combination |
| `column_groups` | `Nr × Nc × ceil(C / columns_per_group)` | Shared blocks; group rounding is separate in each tile |

To supply exact numbers instead of derived counts, use e.g. `"count":{"rule":"fixed","value":5}` and `"activity":{"rule":"fixed","value":3}`: five installed circuits, three powered during each selected event. With row-phase repetition those three run in every layer-wide phase; use tile rules for automatic tile-local early completion. Values must be nonnegative integers.

For `column_groups`, supply a positive integer `columns_per_group` in both count and activity when both share the same grouping. **Installed count and activity are intentionally independent**; changing one does not silently change the other.

With `frequency: "row_phase"`, rules that refer to row-tile resources replace `Nr` by `S=sum(p_i)` to obtain total powered-instance events. Combined-output rules multiply by `P=max(p_i)` instead. A `physical_rows` activity rule therefore powers every physical row circuit of an active tile each phase; it does not imply only enabled input rows consume power. If a circuit instead gates individual row drivers according to spikes, it needs a suitable electrical/activity evaluator rather than this fixed-current rule.

### Activity frequency

| Frequency | Events per selected stage |
|---|---|
| `stage` (default) | Follow the selected stage's `repeat` |
| `image` | One event per image, using that stage's unit duration |
| `bin` | One event per simulation bin |
| `row_phase` | Events per read phase, accounting for early completion of partial row tiles |

Selecting two timing stages charges energy for both. `frequency` overrides affect energy **only**, not the schedule's latency. They do not automatically detect silent input spikes. The default conventional read models are an exception to fixed-current activity: they use the actual summed input/conductance current and charge each row once, not once again for every phase.

### Example A: change the shared C3 driver count

`hardware/examples/c3_driver_override.json` contains:

```json
{
  "active_rows": 8,
  "components": [{
    "name": "column_driver",
    "count": {"columns_per_group": 16},
    "activity": {"columns_per_group": 16},
    "params": {"current_ua": 11.87, "area_um2": 86.36}
  }]
}
```

```bash
python run.py --c3-config hardware/examples/c3_driver_override.json
```

For the worked 96-input example, this doubles installed drivers from four to eight, and doubles driver area/energy. It leaves column-source current, VI/LIF costs, and read latency unchanged. An equivalent simple configuration is `{"active_rows":8,"driver_part":16}`; the explicit component form also lets you replace the evaluator or independently tune driver electrical parameters.

### Example B: add circuitry without increasing latency

Add this entry to `components=[...]` of either config in `run.py` (the C3 config has it as a commented example), or save it as `extra_bias.json` and pass it with `--conv-config` or `--c3-config`:

```json
{
  "active_rows": 8,
  "components": [{
    "name": "extra_bias",
    "model": "fixed_current",
    "group": "input_periphery",
    "count": {"rule": "tiles"},
    "activity": {"rule": "tiles"},
    "timing": ["read"],
    "params": {"supply_v": 1.1, "current_ua": 2.0, "area_um2": 5.0}
  }]
}
```

It runs concurrently with the existing read stage. For the C3 hand example it adds two installed instances, 10 um², and `1.1×2×12×50×1e-6=0.00132 nJ` per bin. **Latency does not change**. Choose another group if desired; the arithmetic remains identical.

### Example C: add a component that adds a serial stage

`hardware/examples/c3_added_stage.json` adds an input buffer and replaces the default schedule:

```json
{
  "active_rows": 8,
  "components": [{
    "name": "input_buffer",
    "model": "fixed_current",
    "group": "input_periphery",
    "count": {"rule": "tiles"},
    "activity": {"rule": "tiles", "frequency": "stage"},
    "timing": ["buffer"],
    "params": {"supply_v": 1.1, "current_ua": 2.0, "area_um2": 5.0}
  }],
  "schedule": [
    {"name":"buffer", "duration_ns":5, "repeat":"bin", "owner":"input_buffer"},
    {"name":"read", "duration_ns":50, "repeat":"row_phase", "owner":"column"},
    {"name":"VI", "duration_ns":10, "repeat":"row_phase", "owner":"VI"},
    {"name":"lif", "duration_ns":2, "repeat":"bin", "owner":"LIF"}
  ]
}
```

```bash
python run.py --model gesture --c3-config hardware/examples/c3_added_stage.json
```

For the hand example, the buffer adds 5 ns per bin and 10 um². The LIF remains powered during the buffer stage because its timing is `through_lif`, so there is also extra LIF energy. No inference or reporting code needs changing.

### Schedule semantics

`schedule: null` uses scalar-derived architecture defaults. A custom list **replaces the whole schedule**, and its explicit durations override scalar latency fields for evaluation. Each stage contains `name`, `duration_ns`, `repeat` (`image`, `bin`, or `row_phase`), and `owner`.

All components attached to a stage consume energy concurrently. The stage's delay enters the critical path exactly once; its owner receives that incremental latency in the component breakdown. Ownership is accounting attribution, not an additional power source. Stages are a serial, no-overlap cost model, not a waveform simulator or arbitrary pipelined dependency graph. Repeated read/conversion stages represent the work of all row phases; their summed duration gives `P×(read+conversion)`.

The final stage must be `lif`, repeated per bin. The default membrane is active through all preceding stages and its emission stage. Default schedule owners are `crossbar`/`reference_subtractor`/`LIF` for conventional CIM and `column`/`VI`/`LIF` for C3. Drivers and the conventional reference array add power and area without duplicating stage latency. A component with zero attributed latency is **not** a zero-duration powered component.

Unknown component fields, models, groups, stage references, invalid counts, and nonfinite/negative electrical costs are rejected. Configuration does not evaluate strings as Python expressions.

## 10. Adding a new electrical evaluator in Python

Fixed current is not appropriate for every block. Register a custom evaluator when energy depends on a measured lookup, signal or custom equation. Example: a characterized energy per powered nanosecond:

```python
from hardware.components import register_model, number
from hardware import C3HardwareConfig


def validate_characterized(params):
    if set(params) != {"area_um2", "energy_nj_per_instance_ns"}:
        raise ValueError("Unexpected characterized-model parameters")
    for key, value in params.items():
        number(value, key)


def characterized_energy(spec, context, installed_count, powered_ns):
    return spec["params"]["energy_nj_per_instance_ns"] * powered_ns


register_model("characterized", characterized_energy, validate_characterized)
# Register BEFORE constructing/loading a config that references this model.
config = C3HardwareConfig(components=[{
    "name": "measured_bias",
    "model": "characterized",
    "group": "input_periphery",
    "count": {"rule": "tiles"},
    "activity": {"rule": "tiles"},
    "timing": ["read"],
    "params": {"area_um2": 5, "energy_nj_per_instance_ns": 0.000002}
}])
```

The evaluator returns finite, nonnegative **batch-total energy in nJ**. Common logic calculates area from installed count and `area_um2`, and attributes latency from the schedule. `powered_ns` already includes powered-instance count, repetitions, and batch size; do not multiply those again. Evaluators should be additive over batches and not retain global inference state. Duplicate model registration is rejected. To use it from `run.py`, put the registration near the top of `run.py`, before `CONVENTIONAL`/`C3CIM` are constructed. JSON does not automatically import arbitrary Python modules.

The context exposes configuration, image/bin counts, row-tile phases, column-tile count, logical-output count, resolved schedule, and conventional summed data/reference currents. **It does not currently expose raw per-row spike/resistance tensors, calculate C3 voltage, or propagate output signals between arbitrary plugins.** A future voltage-domain C3 evaluator needs an extension of the architecture/context signal preparation as well as a registered cost model. This interface is extensible accounting, not a circuit simulator.

To add an entirely new architecture, provide its config/default component composition and electrical context preparation, then add its config type to `ESTIMATORS` in `evaluation/runner.py`. Reuse the component evaluator, counting, schedule and reporting framework. The current two default architectures are selected by their configuration type/fields; supplying a new architecture is a Python extension, not just changing an architecture name in JSON.

## 11. Scope and caveats

- Ideal cross-tile current combination; no routed-wire losses, compliance/settling checks, routing or communication costs unless explicitly modeled.
- No analog accuracy simulation, ADC quantization/nonidealities, noise, or signed C3 signal correction.
- No automatic scaling of per-tile area when tile geometry changes: provide measurements for the actual macro.
- LIF is modeled powered during integration and a final emission stage; substitute measured activity/power if the circuit gates differently.
- Voltage/current/conductance and full-waveform circuit behavior are architecture-specific, not consequences of grouping a component.
- The pipeline supports the two existing model topologies, including their known kernel/padding assumptions; it is not a general arbitrary-PyTorch graph mapper.
