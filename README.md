# SNN inference and modular CIM hardware estimation

This repository runs trained spiking neural networks (**N-MNIST LeNet** and **IBM DVS-Gesture**) in software, records the input spikes of every weighted layer, and estimates the energy, latency and area of those layers on compute-in-memory (CIM) hardware that you describe as a set of modular components. Hardware estimation uses the real spike activity; it does not simulate analog nonidealities or their effect on accuracy.

## 1. Repository layout

```text
run.py                 MAIN SCRIPT: model, architectures to evaluate, metric switches
architectures/         hardware definitions, one file per design (ARCH = Architecture(...))
  rram_1bit.py         1-bit RRAM, OTA-held source lines, comparator LIFs
  conventional.py      current-mode CIM, analog cells + reference columns
  c3cim.py             C3CIM macro (fixed currents)
hardware/              the hardware model
  architecture.py      Crossbar, Precision, Stage, Component, Architecture (+ validation)
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
tools/                 checkpoint conversion and slayerSNN verification scripts
tests/                 reference-model, hand-calculation and pipeline tests
```

## 2. Quick start

Install Python 3.10+, PyTorch, NumPy and PyYAML (`pip install -r requirements.txt`); nothing needs compiling, and it runs on CPU or GPU. Put the datasets in `datasets/` (see `datasets/*/README.md`). Then edit `run.py`:

- `MODEL`, `BATCH_SIZE`, `MAX_BATCHES`: what to run.
- `ARCHITECTURES`: which designs from `architectures/` to evaluate (or build one inline).
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

An architecture has five parts. `architectures/rram_1bit.py`:

```python
ARCH = Architecture(
    name="rram_1bit",
    crossbar=Crossbar(rows=64, cols=64, cell_bits=1, r_on=20e3, r_off=200e3, v_read=0.2),
    precision=Precision(weight_bits=4, weight_encoding="twos_complement"),
    conv_mapping="sequential",
    stages=[Stage("read", 5.0),                     # every crossbar read
            Stage("fire", 2.0, level="timestep")],  # once per time bin, after the reads
    components=[
        Component("cells", model="crossbar_read", count="tiles", during=["read"], supply_v=1.1),
        Component("sl_ota", count="physical_columns", on={"rule": "used_columns", "gated": True},
                  during=["read"], supply_v=1.1, static_ua=10.0),
        Component("lif_comparator", count="outputs", during=["timestep"], supply_v=1.1, static_ua=10.0),
    ],
)
```

Units: ohm, V, uA, ns, pJ (event energy), um^2 per installed instance.

**Crossbar**: tile size, bits per cell, `r_on`/`r_off`, read voltage, `active_rows` (rows enabled per read; fewer than `rows` splits a read into row phases), and `reference_columns` (analog encoding: one G(0) column per output).

**Precision**: `weight_bits` (None = unquantized, analog only) and `weight_encoding`:

| Encoding | Columns per weight | Cells |
|---|---|---|
| `twos_complement` | ceil(weight_bits / cell_bits) | bit slices of the two's-complement code |
| `offset` | ceil(weight_bits / cell_bits) | bit slices of code + 2^(bits-1) |
| `differential` | 2 x ceil((weight_bits - 1) / cell_bits) | magnitude slices, positive and negative columns |
| `analog` | 1 | G linear in the weight over [-max\|w\|, max\|w\|]; G(0) at the midpoint |

A cell at level L of 2^cell_bits - 1 has G = G_off + L / (2^cell_bits - 1) x (G_on - G_off). How slice currents are weighted and combined after the array is part of your component list.

Inputs are binary spikes. Spikes of every time bin are integrated with equal weight in the LIF, so input precision is the number of time bins: each time bin is one **timestep**.

**Conv mapping**: a layer is unrolled into windows (one per output pixel; stride and padding included; a dense layer has one window).

| `conv_mapping` | Weight copies | Reads per time bin |
|---|---|---|
| `sequential` | 1: the kernel weights once; windows applied one after another, column tiles in parallel | windows x row phases |
| `parallel` | one per window: all windows at once (maximum resources) | row phases |

Every weight copy holds K = in_channels x kh x kw rows and out_channels x columns-per-weight columns, split into row tiles x column tiles.

**Stages** (timeline): `level="read"` stages repeat every read; `level="timestep"` stages run once per time bin, where the whole block of reads is the pseudo-stage `"reads"` (which the first timestep stage follows by default). Placement: `after=None` follows the previous stage of the level (serial), `after=[]` starts with the level (parallel), `after=["x", "y"]` waits for those, and a negative `offset_ns` overlaps the start with the end of the dependency. `read_interval_ns` / `timestep_interval_ns` pipeline consecutive reads / time bins.

**Components**: each has an installed `count` (area), a powered `on` rule (default `"all"` = the count rule), the stages it is powered `during` (or `"timestep"` for the whole time bin), and costs:

- `static_ua` x `supply_v` x powered time, for every powered instance;
- `event_pj` per event: per powered instance per read (`events="read"`), per time bin (`"timestep"`), or per LIF output spike of the layer (`"output_spike"`);
- `model="crossbar_read"` / `"reference_read"`: the cell current computed from the spikes and conductances, drawn from `supply_v` during its read stage.

Count and on rules, with the unit each instance belongs to:

| Rule | Unit | Instances per unit |
|---|---|---|
| `tiles` | row tile of a weight copy | column tiles |
| `physical_rows` / `physical_columns` | row tile | column tiles x tile rows / cols |
| `used_columns` | row tile | out_channels x columns per weight |
| `column_groups` (`size`) | row tile | column tiles x ceil(cols / size) |
| `outputs` | window | out_channels (neurons; installed for every window) |
| `output_bank` | window (per weight copy) | column tiles x cols |
| `one` / `fixed` (`value`) | layer | 1 / value |
| `spiking_rows` (on only) | read | word lines carrying a spike, one per column tile |

`{"rule": ..., "gated": True}` powers an instance only when its unit receives at least one input spike in that read (read-level components) or time bin (whole-bin components). Without gating, every valid unit is powered: a partially filled row tile stops after its last row phase.

Leave unknown values at 0 (e.g. areas) and switch the metric off; add, remove or rename components freely. A new design is a new file in `architectures/`.

## 4. How costs are computed

For each layer and batch (`hardware/engine.py`):

1. **Mapping**: windows, weight copies, tiles, row phases and reads per time bin (`mapping.layer_geometry`).
2. **Activity**: every window's input patch is extracted from the spikes; per row, the number of spikes; per read and time bin, which row tiles, windows and the layer receive a spike (`mapping.spike_activity`).
3. **Timeline**: stage start/end times, reads per time bin, latency per inference = (T - 1) x timestep interval + timestep span (`timeline.build_timeline`). Latency follows the schedule and does not depend on the data.
4. **Energy** per component: static energy = supply x current x (powered time per read or time bin) x (powered instances summed over all reads or time bins); event energy; array energy = supply x v_read x sum over spikes of the conductances of that row (every column slice) x read time.

Per inference: energy is divided by the number of evaluated samples; area counts installed instances once. Network totals add the layers, which run one after another. Reported metrics: energy (nJ), latency (us), power = energy / latency (mW), area (mm^2), TOPS/W = 2 x dense MACs / energy (every input in every time bin, zeros included), and pJ per synaptic operation (SOP = an input spike reaching one output neuron; the event-driven SNN figure).

## 5. Verification

`tests/test_hardware.py` holds a deliberately literal reference model: it builds every window's patch by hand, walks every sample, time bin, window, row tile and row phase, decides for each unit whether it is powered, and sums every cell's current. The engine must match it for both conv mappings, stride and padding, partial tiles, row phases, 2-bit cells, all four encodings and every rule (gated or not). Hand calculations cover the `rram_1bit` dense layer cell by cell, and the `conventional` and `c3cim` files reproduce this project's earlier worked examples exactly (K = 96 rows, 2 outputs, 8 active rows, one time bin):

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

- Costs come from the component list: anything not listed (routing, buffers, control, the circuit that weights and combines bit-slice currents) is not charged.
- No analog nonidealities: IR drop, device variation, noise, compliance and settling limits are not modelled, and accuracy comes from the software network.
- Latency follows the fixed schedule; reads without spikes still take their time (only power is gated).
- Only weighted conv and dense layers are mapped; pooling runs in software and is not charged.
- `outputs` installs a neuron for every output (channel x window), also with the sequential mapping; use `output_bank` or `fixed` for time-multiplexed neurons.
