# SNN inference and modular CIM hardware estimation

This repository runs trained spiking neural networks (**N-MNIST LeNet** and **IBM DVS-Gesture**) in software, records the input spikes of every weighted layer, and estimates the energy, latency and area of those layers on compute-in-memory (CIM) hardware that you describe as a set of modular components. Hardware estimation uses the real spike activity; it does not simulate analog nonidealities or their effect on accuracy.

## 1. Repository layout

```text
run.py                 MAIN SCRIPT: model, architectures to evaluate, metric switches
architectures/         hardware parts
  crossbars.py         crossbar types: conv_xbar (current-mode), c3cim_xbar (constant-current)
hardware/              the hardware model
  architecture.py      Memory, Crossbar, Precision, Stage, Component, compose()
  mapping.py           layer -> windows, tiles, weight slices; spike activity
  timeline.py          stage placement: serial, parallel, overlapping, pipelined
  engine.py            evaluate_layer: energy / latency / area of one layer
models/                SNN networks, datasets and their neuron/simulation YAMLs
  srm.py               plain-PyTorch SRM spiking layers (replaces slayerSNN)
  events.py            event-file readers and spike binning
pretrained/            trained weights (nmnist_lenet.pth, gesture.pth)
datasets/              place the datasets here (see each folder's README)
evaluation/            pipeline used by run.py
  runner.py            inference per weight precision, per-layer hardware evaluation
  probes.py            layer input spikes and LIF output spike counts
  report.py            metric switches, text report, JSON and CSV export
  software.py          prediction, loss, accuracy
examples/              run_dense_layer.py: run.py for one random dense layer, checked by hand
tools/                 checkpoint conversion and slayerSNN verification scripts
tests/                 reference-model, hand-calculation and pipeline tests
```

## 2. Quick start

Install Python 3.10+, PyTorch, NumPy and PyYAML (`pip install -r requirements.txt`); nothing needs compiling, and it runs on CPU or GPU. Put the datasets in `datasets/` (see `datasets/*/README.md`). Then edit `run.py`:

- `MODEL`, `BATCH_SIZE`, `MAX_BATCHES`: what to run.
- `ARCHITECTURES`: which designs to evaluate, each composed from a crossbar (with its memory cells) and your own stages and components (see section 3).
- `METRICS`: switch each reported metric on or off (accuracy, energy, latency, power, area, TOPS/W, pJ per synaptic operation, per-layer results, per-component breakdown).

```bash
python run.py                                   # settings in run.py
python run.py --model gesture                   # override the model once
python run.py --model nmnist -b 12 --batches -1 # full N-MNIST test set (slow)
python -m unittest discover -s tests            # all tests
```

Results are printed and saved under `logs/`: a JSON per architecture (with the full architecture description) and `logs/comparison_summary.csv`, one row per model, architecture, configuration and sample count. Only enabled metrics are reported.

Architectures are grouped by weight precision. Each group runs the network with its weights quantized as that hardware stores them (symmetric uniform, `weight_bits`), so the reported accuracy and the spike activity that drives the energy both belong to that precision. Gesture batch size is capped at two.

The weight files are plain tensors (`torch.load(weights_only=True)`); `models.load_pretrained` checks that the YAML's neuron parameters still reproduce the neuron kernels stored with them. Dataset paths in the YAMLs are resolved from the repository root. To add a model, write its classes and a `SPEC` in `models/<name>.py` and register it in `models/__init__.py`.

## 3. Defining hardware

An architecture is composed from one **crossbar type** (with its memory cells as parameters) and any **Stages** and **Components** you define; nothing else is built in, so every peripheral circuit and neuron is described the same generic way.

| Part | Where | What it is |
|---|---|---|
| crossbar (exactly one) | `architectures/crossbars.py` | memory cells: `cell_bits` and `r_on`/`r_off` (levels linear in conductance) or `levels_s` (every level's conductance, for nonuniform cells). `conv_xbar`: current-mode, cell current G x v_read into each column, charged from `cell_supply_v` (VDD when an OTA derives v_read from the supply, v_read when the source line is driven directly); `c3cim_xbar`: constant-current columns with shared drivers. Each adds the `"read"` stage and the array's own costs |
| stages | `hardware.Stage` | the timeline: what repeats every read, what runs once per time bin, serial / parallel / overlapping |
| components | `hardware.Component` | any circuit: your name, how many are installed (`count`), how many are powered (`on`, optionally spike-`gated`), when (`during` stages or a `window` between stage edges), and its static current and/or event energy |

`compose` puts them together with the weight precision and the conv mapping. The 1-bit RRAM design in `run.py`:

```python
RRAM_1BIT_XBAR = compose(
    "rram_1bit_conv_xbar",
    Precision(weight_bits=4, weight_encoding="twos_complement"),
    [
        crossbars.conv_xbar(cell_bits=1, r_on=20e3, r_off=200e3,   # 1-bit RRAM cells
                            rows=64, cols=64, v_read=0.2, read_ns=5.0, cell_supply_v=VDD),
        Stage("fire", 2.0, level="timestep"),                  # once per time bin, after the reads
        Component("sl_ota", count="physical_columns",          # one per column,
                  on={"rule": "used_columns", "gated": True},  # on when its tile gets a spike
                  during="read", supply_v=VDD, static_ua=10.0),
        Component("lif", count="outputs", during="fire",       # one per output neuron
                  supply_v=VDD, static_ua=10.0),
    ],
    conv_mapping="sequential",
)
```

To keep the OTA on until the neuron has fired, write `during=["read", "fire"]`; for an exact interval, `window=(("read", "start", 1.0), ("fire", "end", -1.0))`. The same components can sit on a `c3cim_xbar`, and the cells are changed in the crossbar call. `examples/run_dense_layer.py` is a complete worked example.

Units: ohm, V, uA, ns, pJ (event energy), um^2 per installed instance.

**Crossbar**: the memory, tile size, read voltage, `active_rows` (rows enabled per read; fewer than `rows` splits a read into row phases), and `reference_columns` (analog encoding: one G(0) column per output).

**Precision**: `weight_bits` (None = unquantized, analog only) and `weight_encoding`:

| Encoding | Columns per weight | Cells |
|---|---|---|
| `twos_complement` | ceil(weight_bits / cell_bits) | bit slices of the two's-complement code |
| `offset` | ceil(weight_bits / cell_bits) | bit slices of code + 2^(bits-1) |
| `differential` | 2 x ceil((weight_bits - 1) / cell_bits) | magnitude slices, positive and negative columns |
| `analog` | 1 | G linear in the weight over [-max\|w\|, max\|w\|]; G(0) at the midpoint |

A cell at level L has the memory's level-L conductance. How slice currents are weighted and combined after the array is part of the component list (e.g. `slice_mirror`).

Inputs are binary spikes. Spikes of every time bin are integrated with equal weight in the LIF, so input precision is the number of time bins: each time bin is one **timestep**.

**Conv mapping**: each kernel (one output channel) occupies K = in_channels x kh x kw rows of one column per weight slice (a 3x3x3 kernel with 4-bit weights on 1-bit cells: 27 rows x 4 columns). A layer is unrolled into windows (one per output pixel, stride and padding included; a dense layer has one window), and a window's input patch drives the kernel rows. One weight copy holds all kernels in the fewest tiles: K rows x (out_channels x columns per weight), split into row tiles x column tiles.

| `conv_mapping` | Weight copies | Reads per time bin |
|---|---|---|
| `sequential` | 1: all kernels once; the windows are applied one after another | windows x row phases |
| `parallel` | one per window: every window computed at once | row phases of a tile set |

With `parallel`, copies that fit in a tile share it, packed block-diagonally: as many as fit both the tile's rows and its columns (e.g. 18-row x 16-column copies: 3 per 64x64 tile, so 25 windows take 9 tiles). Each copy has its own rows (its window's inputs) and columns (its outputs). The cells of a copy's rows in the other copies' columns hold level 0 and conduct G(level 0) x v_read on every spiking row, which is charged to the array (and to the slice mirrors and reference columns of those columns). Columns without weights are off, with every component counted on them (`used_columns`). A copy larger than a tile gets its own tiles. The windows read together are a **slot**: one window per read with `sequential`, one tile set's copies with `parallel`. With `active_rows`, a packed tile reads the rows of all its copies in phases, so packing can add reads.

**Stages** (timeline): `level="read"` stages repeat every read; `level="timestep"` stages run once per time bin, where the whole block of reads is the pseudo-stage `"reads"` (which the first timestep stage follows by default). Placement: `after=None` follows the previous stage of the level (serial), `after=[]` starts with the level (parallel), `after=["x", "y"]` waits for those, and a negative `offset_ns` overlaps the start with the end of the dependency. `read_interval_ns` / `timestep_interval_ns` pipeline consecutive reads / time bins.

**Components**: each has an installed `count` (area), a powered `on` rule (default `"all"` = the count rule), when it is powered, and costs. When it is powered is either the stages it is powered `during` (one or several; overlaps counted once; `"timestep"` = the whole time bin) or a `window` between two stage edges, e.g. `window=(("read", "start", 1.0), ("fire", "end", -1.0))` = from 1 ns after the read starts to 1 ns before the fire step ends (a read-level stage starts with the first read of the time bin and ends with the last). Costs:

- `static_ua` x `supply_v` x powered time, for every powered instance;
- `event_pj` per event: per powered instance per read (`events="read"`), per time bin (`"timestep"`), or per LIF output spike of the layer (`"output_spike"`);
- data-driven models, drawn from `supply_v` while powered (per read in read stages; with one read per time bin they can also stay on through time-bin stages or a window), with currents computed from the spikes and conductances: `"crossbar_read"` (weight columns), `"reference_read"` (G(0) reference columns) and `"slice_mirror"` (current mirrors copying each weight-slice column with gain `slice_gains`; default binary, most significant slice x1, the next x1/2, ...).

Count and on rules, with the unit each instance belongs to:

| Rule | Unit | Instances per unit |
|---|---|---|
| `tiles` | row tile of a tile set | column tiles |
| `physical_rows` / `physical_columns` | row tile of a tile set | column tiles x tile rows / cols |
| `used_columns` | row tile of a tile set | out_channels x columns per weight, per copy in it |
| `column_groups` (`size`) | row tile of a tile set | column tiles x ceil(cols / size) |
| `outputs` | window | out_channels (neurons; installed for every window) |
| `output_bank` | tile set (slot) | column tiles x cols |
| `one` / `fixed` (`value`) | layer | 1 / value |
| `spiking_rows` (on only) | read | word lines carrying a spike, one per column tile |

`{"rule": ..., "gated": True}` powers an instance only when its unit receives at least one input spike in that read (read-level components) or time bin (whole-bin components). Without gating, every valid unit is powered: a partially filled row tile stops after its last row phase.

Leave unknown values at 0 (e.g. areas) and switch the metric off. A different memory is a change of the crossbar's cell parameters; a new circuit is a `Component` (and a `Stage` if it takes time of its own).

## 4. How costs are computed

For each layer and batch (`hardware/engine.py`):

1. **Mapping**: windows, weight copies, copies per tile, tiles, row phases and reads per time bin (`mapping.layer_geometry`).
2. **Activity**: every window's input patch is extracted from the spikes; per row, the number of spikes; the spikes that leak into packed copies' columns; per read and time bin, which row tiles, slots, windows and the layer receive a spike (`mapping.spike_activity`).
3. **Timeline**: stage start/end times, reads per time bin, latency per inference = (T - 1) x timestep interval + timestep span (`timeline.build_timeline`). Latency follows the schedule and does not depend on the data.
4. **Energy** per component: static energy = supply x current x (powered time per read or time bin) x (powered instances summed over all reads or time bins); event energy; data-driven energy = supply x (for each column slice: v_read x sum over spikes of that row's conductance plus the level-0 leak of packed copies, times its mirror gain for `slice_mirror`) x powered time.

Per inference: energy is divided by the number of evaluated samples; area counts installed instances once. Network totals add the layers, which run one after another. Reported metrics: energy (nJ), latency (us), power = energy / latency (mW), area (mm^2), TOPS/W = 2 x dense MACs / energy (every input in every time bin, zeros included), and pJ per synaptic operation (SOP = an input spike reaching one output neuron; the event-driven SNN figure).

## 5. Verification

`tests/test_hardware.py` holds a deliberately literal reference model: it builds every window's patch by hand, walks every sample, time bin, window, row tile and row phase, decides for each unit whether it is powered, and sums every cell's current. The engine must match it for both conv mappings, stride and padding, partial tiles, row phases, linear 1-bit and nonuniform 2-bit memories, all four encodings, slice mirrors with binary and custom gains, and every rule (gated or not). Hand calculations cover the 1-bit RRAM design's dense layer cell by cell (cells, gated OTAs, binary mirrors, comparators on for the fire step), and two compositions defined in the test (a conventional analog crossbar with reference columns and DAs, and a C3CIM crossbar with VI converters) reproduce this project's earlier worked examples exactly (K = 96 rows, 2 outputs, 8 active rows, one time bin):

| Architecture | Energy (nJ) | Latency (ns) | Area (um^2) |
|---|---:|---:|---:|
| `c3cim` | 0.0279948 | 482 | 10259.68 |
| `conventional` | 0.07274388 | 38 | 9969.4 |

Timeline tests cover serial, parallel, overlapping and pipelined stages, powered time as the union of overlapping stages, and a component that would serve two pipelined reads at once. Pipeline tests cover the layer probes, the runner and the metric switches.

## 6. Spiking-neuron implementation (no slayerSNN)

The networks were trained with the slayerSNN (SLAYER PyTorch) framework, which is no longer maintained and needs a compiled CUDA extension. `models/srm.py` reimplements the parts inference needs in plain PyTorch, following slayerSNN's computations exactly:

- `psp`: causal filtering with the SRM alpha kernel, times `Ts`, accumulated in slayerSNN's tap order with fused multiply-adds (its CUDA kernels are built with `-use_fast_math`, which fuses them the same way).
- `spike`: fire when the membrane reaches `theta`, then add the refractory kernel from the spike step onwards. Spikes have amplitude `1/Ts`.
- `conv` and `dense`: `Conv3d` layers applied per time step.
- `pool`: a per-channel window sum scaled by `1.1 × theta`, including SLAYER's padding of odd sizes.
- Event reading and binning (`models/events.py`): polarity shifted to start at 0, round-half-to-even time bins, and a bin holding any event set to `1/Ts`.

The kernels generated from the YAMLs match the kernels stored in the trained weights bit for bit. `tests/test_srm.py` checks the layers against literal transcriptions of slayerSNN's CUDA loops. This is an inference implementation only: it has no surrogate gradients, so it cannot train.

**Verification against the original framework.** Two steps are still to do on a machine that has slayerSNN and the datasets:

```bash
python tools/export_slayer_reference.py --model nmnist --samples 20 --full
python tools/export_slayer_reference.py --model gesture --samples 22
```

Copy the resulting `reference/*_slayer.pt` files into this repository. `python tools/compare_slayer_reference.py reference/*.pt` then reports, for every profiled layer, how many spike entries differ and from which time step. It also compares predicted classes, full-test-set accuracy (with `--full`) and the effect on the hardware energy. `tests/test_slayer_reference.py` runs automatically once reference files exist. Expect identical spikes almost everywhere: slayerSNN's CUDA kernels round float32 sums in a different order from PyTorch, so a membrane potential within rounding of the threshold can occasionally flip a spike.

The original pickled checkpoints (commit `7f020a7`, `pretrained/*.pt`) needed slayerSNN to load. `tools/convert_checkpoints.py` produced the current `.pth` files from them without slayerSNN.

## 7. Scope and caveats

- Costs come from the component list: anything not listed (routing, buffers, control) is not charged.
- No analog nonidealities: IR drop, device variation, noise, compliance and settling limits are not modelled, and accuracy comes from the software network.
- Latency follows the fixed schedule; reads without spikes still take their time (only power is gated).
- Only weighted conv and dense layers are mapped; pooling runs in software and is not charged.
- `outputs` installs a neuron for every output (channel x window), also with the sequential mapping; use `output_bank` or `fixed` for time-multiplexed neurons.
