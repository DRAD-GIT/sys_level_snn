# SNN inference and modular CIM hardware estimation

This repository runs trained spiking neural networks (**N-MNIST LeNet**, **IBM DVS-Gesture** and a **CIFAR-10 VGG-11** with LIF neurons on rate-coded images) in software, records the input spikes of every weighted layer, and estimates the energy, latency and area of those layers on compute-in-memory (CIM) hardware that you describe as a set of modular components. Hardware estimation uses the real spike activity; it does not simulate analog nonidealities or their effect on accuracy.

## 1. Repository layout

```text
run.py                 MAIN SCRIPT: model and data, weight quantization, hardware, metric switches
literature_macros.py   published RRAM/MRAM macros calibrated to their papers (run.py's LITERATURE)
docs/literature_shortlist.md  shortlisted literature, their reported results and the selection factors
crossbars/             crossbar types, one per file
  conv_xbar.py         current-mode crossbar (cell current G x v_read into each column)
  c3cim_xbar.py        constant-current crossbar with shared column drivers
hardware/              the hardware model
  architecture.py      Memory, Crossbar, Mapping, Component (steps, power windows), compose()
  mapping.py           layer -> windows, tiles, weight slices; spike activity
  timeline.py          step placement: serial, parallel, overlapping, pipelined
  engine.py            evaluate_layer: energy / latency / area of one layer
models/                SNN networks, dataset readers and their neuron/simulation YAMLs
  srm.py               plain-PyTorch SRM spiking layers (replaces slayerSNN)
  events.py            event-file readers and spike binning
  lif.py               trainable LIF spiking layers (surrogate gradient), BatchNorm folding, rate coding
  cifar10.py           CIFAR-10 VGG-11 SNN and the rate-coded test set
pretrained/            trained weights (nmnist_lenet.pth, gesture.pth; cifar10_vgg11.pth from tools/train_cifar10.py)
evaluation/            pipeline used by run.py
  runner.py            evaluate(): recordings -> per-layer hardware evaluation; accuracy sweep
  recording.py         the recorded forward pass: layer inputs and outputs, compressed
  probes.py            layer input spikes and LIF output spikes
  report.py            metric switches, text report, JSON and CSV export
  software.py          predicted class from the output spikes
examples/              run_dense_layer.py: run.py for one random dense layer, checked by hand
tools/                 record.py (record forward passes ahead), accuracy_sweep.py (accuracy vs weight bits),
                       latex_table.py (the paper's comparison table from the results),
                       train_cifar10.py (trains the CIFAR-10 SNN);
                       checkpoint conversion and slayerSNN verification
recordings/            recorded forward passes (committed, so the hardware evaluation runs without the datasets)
tests/                 reference-model, hand-calculation and pipeline tests
```

## 2. Quick start

Install Python 3.8+, PyTorch 1.12+, NumPy and PyYAML (`pip install -r requirements.txt`, or with conda: `conda env create -f environment.yml && conda activate c3cim`); nothing needs compiling, and it runs on CPU or GPU. `run.py` follows the flow of an evaluation, top to bottom:

1. **Models and data**: `MODELS` (the datasets evaluated in one run, default N-MNIST, DVS-Gesture and thermometer-coded CIFAR-10), `DATASET_DIR` (see below), `MAX_SAMPLES` (the first test samples, `-1` = all), `PARALLEL` (samples run at once, per model), `RECORDING_DIR`.
2. **Weight quantization**: `WEIGHT_BITS`, `WEIGHT_SCALING`.
3. **Hardware**: `ARCHITECTURES`, each a crossbar (with its memory cells) and then its components, mapped with the weight quantization of step 2. Start with the **Rules of thumb** in section 3.
4. **Forward pass**: the network runs once per weight quantization and every weighted layer's input spikes and output spikes (all channels, pixels, time bins and samples) are **recorded** (see below). Later runs reuse the recording instead of rerunning inference.
5. **Metrics**: estimated from the recording; `METRICS` switches each one on or off (accuracy, energy, latency, power, area, TOPS/W, pJ per synaptic operation, per-layer results, per-component breakdown).

```bash
python run.py                                   # our work (OURWORK) on every model in MODELS
python run.py --literature                      # also the published macros (literature_macros.COMPARED)
python run.py --table                           # then write the paper's table, commit and push it
python run.py --data /data/neuromorphic         # dataset location for this run
python run.py --model gesture                   # only these models (several allowed)
python run.py --model nmnist --samples 1000     # first 1000 N-MNIST test samples (-1 or no flag: all)
python run.py --rerecord                        # record the forward pass again
python tools/record.py --data /data/neuromorphic  # record both test sets ahead, at run.py's quantization
python -m unittest discover -s tests            # all tests
```

`--samples` means the same in every script: a positive N evaluates or records the first N test samples, `-1` all of them, and leaving it out means all (except `tools/export_slayer_reference.py`, whose layer-by-layer dumps default to 20).

**Datasets** are not stored in the repository. Set `DATASET_DIR` (or `--data`) to a folder holding them. The dataset folder is found by its name, ignoring case, hyphens, underscores and spaces: the one folder starting with `N-MNIST` (e.g. `N_MNIST`, `nmnist_v2`) for N-MNIST, and with `Gesture` or `DVS-Gesture` (e.g. `DVS_Gesture`) for DVS-Gesture. `DATASET_DIR` may also be the dataset folder itself. Inside it, the paths in `models/<model>.yaml` apply:

```text
N-MNIST.../                        Gesture.../
├── Test/     00000.bin, ...       ├── DvsGestureNpy/<trial>/0.npy ... 10.npy
└── Test.txt  "<index> <label>"    └── DvsGesture/trials_to_test.txt  (e.g. user24_led.aedat)
```

CIFAR-10 is the python release (`cifar-10-python.tar.gz` from https://www.cs.toronto.edu/~kriz/cifar-10.html), extracted as is: the folder `cifar-10-batches-py/` (`data_batch_1` ... `data_batch_5`, `test_batch`) in `DATASET_DIR`.

Only the test split is needed (CIFAR-10 training also reads the training batches). Gesture reads `DvsGestureNpy/<trial name without extension>/<class>.npy` for every trial listed in `trials_to_test.txt`.

Results are printed and saved under `logs/`: a JSON per architecture (with the full architecture description) and `logs/comparison_summary.csv`, one row per model, architecture, configuration and sample count. Only enabled metrics are reported. The run ends with a summary: one line per model and architecture (accuracy, energy, latency, power, TOPS/W per inference).

**Paper table**: `python run.py --table` evaluates, then writes the table into the paper repository (`PAPER_REPO` in `run.py`, by default a clone of `C3CIM_journal_TCAS_v1` next to this repository, at `Chapters/comparison_table.tex`), pulls it first, and commits and pushes the table if it changed; its columns are the datasets in `MODELS` with results, its rows the evaluated architectures. It also writes `Chapters/neuromorphic_table.tex`: the comparison with other neuromorphic designs in the style of DS-CIM's Table III (`tools/neuromorphic_table.py`; one column per work with its publication, technology, implementation, and accuracy and energy per sample on each dataset; the literature numbers are in its `WORKS`, ours come from the results). `python tools/latex_table.py` writes `logs/comparison_table.tex` (the hardware comparison table, `\input` it in the paper). It has one row per architecture in `run.py`'s `ARCHITECTURES` that has results (so the published macros only after `run.py --literature`; or `--architectures name ...`), named after it, and last, after a rule, the architectures of `run.py`'s `OURWORK` (our work), each with its N-MNIST and DVS-Gesture results from `comparison_summary.csv` (the architecture's latest configuration, its run with the most samples; a warning if that is not the full test set, and empty cells if it has not been run). Rows of published works can be added with their published numbers in `tools/latex_table.py`'s `LITERATURE` (none by default); published macros that should run our workloads are defined in `literature_macros.py` (see below). The specification columns come from each architecture's definition: `compose(..., specs={"tech": 40, "supply": 1.1, "device": "Resistive", "bitcell": "2T1R", "sensing": "Voltage"})` (keys `label` for the row name, e.g. `r"\textbf{This work}"`, `tech`, `supply`, `device`, `cell`, `bitcell`, `r_ratio`, `sensing`, `accumulation`; they change no cost); cell precision, R_High/R_Low and accumulation (rows summed per read) are otherwise read from the crossbar, and unknown ones stay empty. Units: power mW, latency us and energy uJ per inference, TOPS/W. In each result column the best value of any row (lowest power, latency and energy, highest TOPS/W) is **red**, and our work's values that beat every other row are **bold**. Run `run.py` for each model first (with power, latency, energy and TOPS/W switched on). The table has a column group per dataset: by default every dataset with results in the summary, or `--datasets nmnist gesture cifar10`.

**Published macros** (`literature_macros.py`; `python run.py --literature` evaluates those listed in its `COMPARED` next to our work): DS-CIM (TCAS-I'24, 40 nm SOT-MRAM, simulated), a 28 nm time-domain RRAM macro (SSC-L'25), a 180 nm memristive SNN chip (arXiv'25), a 28 nm source-follower RRAM macro (A-SSCC'25) and a 28 nm 2T2R RRAM macro (ESSERC'24), each defined with `compose` so it runs the same recorded N-MNIST and DVS-Gesture workloads as our designs. Every number is derived from its paper next to it: the paper's energy per operation at its own test condition (a per-block energy table, power at several numbers of driven rows, or TOPS/W or power x time), split by the paper's breakdown, divided among the instances working in that condition, with the array from the paper's cell resistances and read voltage where it gives them. `python literature_macros.py` runs each macro on its paper's test condition and prints the engine's value against the paper's (all within 2.5%; `tests/test_literature_macros.py`). All macros store the same 6-bit weights as our designs (one recording, one accuracy), each in its paper's cell precision limited to 3 bits (`MAX_CELL_BITS`), with the weight split over as many columns as that needs; the calibration check runs them at their papers' own precision. Inputs are binary spikes (a spike drives its row); a paper's energy for n-bit inputs is divided by n; macros without a neuron are charged none; the file states these and every other assumption.

**CIFAR-10** (`models/cifar10.py`, `models/lif.py`): VGG-11 (conv 64-M-128-M-256-256-M-512-512-M-512-512-M, fc 10; 3x3 convolutions, 2x2 max pooling), the network Han et al. (TCAS-I'22) evaluate on CIFAR-10, with leaky integrate-and-fire neurons (`v = beta v + I`, fire at `theta`, soft reset; `models/cifar10.yaml`: `tSample` 16 time steps, `beta` 0.9, `theta` 1). Each image is rate-coded: every pixel channel spikes at each time step with probability equal to its intensity (a fixed seed per test image, so recordings are reproducible). So every weighted layer, the first included, receives binary spikes and maps onto the crossbars, and max pooling keeps them binary. There is no published SLAYER checkpoint for it; train it once on a GPU:

```bash
python tools/train_cifar10.py --amp             # 200 epochs; saves pretrained/cifar10_vgg11.pth
python tools/train_cifar10.py --amp --resume    # continue an interrupted run
```

**DVS Gesture downscaled to 16x16** (`--model gesture16`, `models/gesture16.py`, `models/gesture16.yaml`): the version DS-CIM and ReckOn evaluate, for a like-for-like comparison: each 8x8 block of the sensor is one input (2 polarities x 16 x 16 = 512 binary inputs), 10 classes (the "other gestures" class left out), time steps of 30 ms over a gesture's first 2.4 s (80 steps; chosen from a sweep of steps and windows, 90.42% float), and a fully connected 512-512-10 SNN with LIF neurons (268k weights). Train it with `python tools/train_event_snn.py --model gesture16` (or `sbatch slurm/train_event_snn.sbatch`; it reads the dataset once and caches it, then trains in minutes on one GPU), then record and evaluate it with `python run.py --model gesture16`. Once it has results, the neuromorphic comparison table uses it for our Gesture entry.

**N-MNIST downscaled to 17x17** (`--model nmnist17`, `models/nmnist17.py`): the input ANP-I uses, 2x2 sensor blocks (2 x 17 x 17 = 578 binary inputs), the first 100 ms in 10 ms steps (10 steps), a fully connected 578-512-10 SNN with LIF neurons. Train it with `python tools/train_event_snn.py --model nmnist17` (or `MODEL=nmnist17 sbatch slurm/train_event_snn.sbatch`); once it has results, the neuromorphic comparison table uses it for our N-MNIST entry.

**Thermometer-coded CIFAR-10** (`--model cifar10_thermo`, `models/cifar10_thermo.yaml`): the same VGG-11, with each pixel channel turned into 8 binary input channels (channel i on where the intensity exceeds (i+1)/9), the same frame at every one of 8 time steps. Every layer still receives binary spikes (the first has 24 input channels, 216 crossbar rows), but without rate coding's randomness, so fewer time steps are needed. Train it with `python tools/train_cifar10.py --model cifar10_thermo --amp` (or `MODEL=cifar10_thermo sbatch slurm/train_cifar10.sbatch`); it is evaluated like the others (`python run.py --model cifar10_thermo`) and fills the CIFAR-10 columns of the table with `--datasets nmnist gesture cifar10_thermo`.

On a Slurm cluster: `mkdir -p logs && sbatch slurm/train_cifar10.sbatch` (edit its `#SBATCH` lines and environment first; `DATA=/path sbatch ...` sets the dataset folder). Submitting it again continues from the last finished epoch.

Training uses BatchNorm after every convolution, an arctan surrogate gradient, random crop and flip, SGD with a cosine learning rate and a mean-squared error on the output firing rates; the best test epoch is saved with BatchNorm folded into the convolutions' weights and biases (the bias is a constant input current to each neuron, not a crossbar row), and the script ends by evaluating the saved file exactly as `run.py` loads it. Then `python run.py --model cifar10` evaluates it like the other models; check its accuracy at 6 bits first with `python tools/accuracy_sweep.py --model cifar10 --bits 6 float`.

**Accuracy vs weight precision** without any hardware estimation: `python tools/accuracy_sweep.py` runs both test sets with the weights quantized to 2, 3, 4, 5, 6 and 8 bits and in float (`--model`, `--bits 3 4 float`, `--data`, `--samples 1000`), reading each test set once, and saves `logs/weight_quantization.csv`. `--scaling max mse std3` compares quantization ranges side by side (see **Mapping**). `--sensitivity` also quantizes one layer at a time (the others float) to show which layers limit the accuracy, and `--layer-bits SF1=8 SF2=float` fixes named layers' bit widths for mixed precision. The accuracy so far is printed every `--every` samples (default 1000) and at the end.

**Recordings** (`recordings/<model>_<bits>b_<scaling>/`, one per model and weight quantization): every weighted layer's input spikes, bit-packed (1 bit per entry, time innermost) and zlib-compressed, which shrinks them about 10x (the 6-bit `"std3"` recordings of the full test sets: about 150 MB for N-MNIST, 15 KiB per sample, and 44 MB for DVS-Gesture); each layer's output spike count per sample (full output spikes too with `tools/record.py --full-outputs`); the predictions and labels. Hardware only needs the output counts: a layer's output is not the next layer's input (pooling and another spiking step lie between), and no metric uses the full output trains. A recording stores a fingerprint of the checkpoint, the neuron YAML, the SNN code and the quantization; if any of them changes, or more samples are asked for than were recorded, it is recorded again instead of reused. The hardware evaluation reads only the recording, never the dataset, and runs on the GPU when there is one: the recording is unpacked there, and the spike activity (which row groups of which windows hold a spike, in every activation and time bin) is computed as a convolution of the spike frames with 0/1 masks of the row groups, which is exact on binary spikes.

Architectures are grouped by weight precision. Each group runs the network with its weights quantized as that hardware stores them (symmetric uniform, `weight_bits` and `weight_scaling`), so the reported accuracy and the spike activity that drives the energy both belong to that precision. Samples are evaluated in parallel in batches set per model (`batch_size` in the model's `SPEC`: N-MNIST 50, DVS-Gesture 2, which needs about 0.5 GB of GPU memory per sample); `--parallel N` (or `PARALLEL` in `run.py`) overrides it for one run. The hardware evaluation takes one recorded file per step by default; `--parallel N` merges consecutive files into steps of at least N samples (whole files). On an RTX 2080 Ti the full N-MNIST hardware evaluation took 21.4 s with one file (50 samples) per step and 17.4 s with 250 samples per step; 500 ran out of GPU memory. `--parallel` only affects speed and memory, not the results; lower it if a GPU runs out of memory.

The weight files are plain tensors (`torch.load(weights_only=True)`); `models.load_pretrained` checks that the YAML's neuron parameters still reproduce the neuron kernels stored with them. To add a model, write its classes and a `SPEC` in `models/<name>.py` and register it in `models/__init__.py`.

## 3. Defining hardware

An architecture is composed from one **crossbar type** (with its memory cells as parameters), the **Mapping** of the network onto it, and any **Components** you define. Nothing else is built in: every peripheral circuit and neuron is a component, and the timeline comes from the components that take time.

| Part | Where | What it is |
|---|---|---|
| crossbar (exactly one) | `crossbars/` | memory cells: `cell_bits` and `r_on`/`r_off` (levels linear in conductance) or `levels_s` (every level's conductance, for nonuniform cells). `conv_xbar`: current-mode, cell current G x v_read into each column, charged from `cell_supply_v` (VDD when an OTA derives v_read from the supply, v_read when the source line is driven directly); `c3cim_xbar`: constant-current columns with shared drivers. Each brings the array's per-activation step (`time_ns`; `"cells"` for `conv_xbar`, `"column_source"` for `c3cim_xbar`) and its own costs |
| mapping | `hardware.Mapping` | how the network goes onto the crossbars: `weight_bits`, `weight_scaling`, `weight_encoding`, `conv`, `columns` |
| components | `hardware.Component` | any circuit: your name, how many are installed (`count`), how many are powered (`powered`, optionally spike-`gated`), optionally its time (`time_ns`, which makes it a step of the timeline; `serial`, `at`, `per`), when it draws current (`when`; default: during its own step), and its static current and/or event energy |

### Rules of thumb

**Building a design**
- One crossbar, then your components, in a list: `compose(name, Mapping(...), [crossbar, Component(...), ...])`. The crossbar brings its own components: `conv_xbar` the array `"cells"`, `c3cim_xbar` the `"column_source"`s and `"column_driver"`s.
- Every circuit you add is a `Component` with a unique name; `count` is how many are installed (area), `powered` how many draw current.

**Time: `time_ns` or not?**
- A component that **takes time others must wait for** (a read, an ADC conversion, a neuron update) gets `time_ns`. It becomes a **step** of the timeline, named after the component, and sets the latency.
- A component that is **just switched on while other steps run** (a bias, an OTA, a driver) gets **no** `time_ns`, only `when`. It has no window of its own, so `when` must name real steps; with a current but neither `time_ns` nor `when`, the engine stops with an error.
- Steps run **one after another** in the order given. To run one **alongside** another, place it: `at="cells.start"`. Adding a step without `at` makes every time bin longer.
- `serial=True` (ADC-like, `column_groups` count): each instance converts its group's weight columns one by one, so the step lasts `time_ns` x the fullest group.
- How often a step runs is inferred from `count`: columns, rows, column groups, tiles **per activation**; neurons and layer-wide parts **once per time bin**. `per=` overrides it.

**Power: `when`**
- Default: the component's own step. Otherwise `when="cells"` (during that step), `when="bin"` (the whole time bin) or `when=("cells.start", "lif.end")` (from one point to another).
- `when` only changes **energy**, never the timeline; `time_ns` and `at` decide the timeline. A step's `time_ns` still matters with a wider `when`: it moves the anchors the window is built from.
- A window inside the per-activation steps is charged **in every activation**; one reaching a time-bin step or the bin's edges is charged **once per time bin**.
- The array's data-driven current (`"cells"`) can only stay on past its own step with **one activation per time bin**.

**Names and anchors (one rule)**
- A **bare name is a window**: `"cells"`, `"bin"`, `"activations"`.
- A **point always carries its edge**: `"cells.start"`, `"lif.end"`, `"bin.end"`, optionally with an offset in ns (`"cells.start+1"`). Only steps (components with `time_ns`) and `bin` / `activations` have edges.
- Prefer the step you mean over `bin.end`: `"lif.end"` stays right when steps are added after the neuron.

**How many are powered: `powered`**
- `"all"` (default): every installed instance; `"used_columns"`: only columns holding weights; `"used_column_groups"`: only the groups (ADCs, drivers) holding a weight column; add `"gated": True` to power an instance only when its tile / window receives a spike.
- `Mapping(columns=...)` decides where weight columns sit among shared groups: `"interleaved"` spreads them (short serial steps, every group used), `"contiguous"` packs them (empty groups can stay off, longer serial steps).
- `active_rows` (crossbar) splits a tile's rows into row phases, one activation each; the default drives all rows at once.

**Running**
- Inference runs once per weight quantization (`weight_bits`, `weight_scaling`) and is recorded; changing hardware, `columns`, timing or power only re-runs the fast hardware evaluation. Designs with the same quantization share a recording.
- `--samples N` = the first N test samples, `-1` or no flag = all; `--parallel N` = samples processed at once (recording batch, evaluation step).

A 1-bit RRAM design:

```python
RRAM_1BIT_XBAR = compose(
    "rram_1bit_conv_xbar",
    Mapping(weight_bits=6, weight_scaling="std3", weight_encoding="twos_complement",
            conv="parallel"),
    [
        crossbars.conv_xbar(cell_bits=1, r_on=20e3, r_off=200e3, rows=64, cols=64,
                            v_read=0.2, cell_supply_v=VDD, time_ns=5.0),   # step "cells"
        Component("sl_ota", count="physical_columns",               # one per column,
                  powered={"rule": "used_columns", "gated": True},  # on when its tile gets a spike
                  when="cells",                                     # during the read
                  supply_v=VDD, static_ua=10.0),
        Component("lif", count="outputs",                           # one per output neuron
                  time_ns=2.0,                                      # step: once per time bin, after the reads
                  supply_v=VDD, static_ua=10.0),                    # powered during its step
    ],
)
```

To keep the OTA on until the neuron has fired, write `when=("cells.start", "lif.end")`; with offsets, `when=("cells.start+1", "lif.end-1")`. The same components can sit on a `c3cim_xbar`, and the cells are changed in the crossbar call. `examples/run_dense_layer.py` is a complete worked example.

A tile with 8 of its 64 rows driven at a time, a current-source driver per 32 columns switching on with the read, and one 3-bit ADC shared by every 8 columns (6 ns per conversion, 10 uA):

```python
crossbars.conv_xbar(..., rows=64, cols=64, active_rows=8, time_ns=5.0),   # "cells": 8 activations per time bin
Component("driver", count={"rule": "column_groups", "size": 32},         # all its columns at once
          time_ns=3.0, at="cells.start", supply_v=VDD, static_ua=11.87), # starts with the read
Component("adc", count={"rule": "column_groups", "size": 8},             # 8 ADCs per tile
          time_ns=6.0, serial=True,                                       # its columns one after another
          powered="used_column_groups",                                   # only ADCs with weight columns
          supply_v=VDD, static_ua=10.0),                                  # after the read (and driver)
Component("lif", count="outputs", time_ns=2.0, supply_v=VDD, static_ua=10.0),
```

With 60 weight columns in the tile, the report prints each layer's timeline as `cells 5 ns (per activation, x8) | driver 3 ns (per activation, x8) | adc 48 ns (per activation, x8) | lif 2 ns (per time bin) = 426 ns per time bin`. A serial step converts only the columns that hold weights, and where they sit is `Mapping(columns=...)`. With 32 weight columns in a 64-column tile:

| `columns` | Weight columns | ADCs of 8 | ADC step | Drivers of 32 |
|---|---|---|---|---|
| `"interleaved"` (default) | dealt to the groups in turn (column j to group j mod groups) | all 8 convert 4 each | 4 x 6 = 24 ns | both drive 16 |
| `"contiguous"` | columns 0-31, filling one group after another | 4 convert 8 each, 4 hold none | 8 x 6 = 48 ns | one drives 32, one none |

`powered="used_column_groups"` powers only the groups holding a weight column (the size comes from the `column_groups` count); with `powered="all"` (the default) every group of a tile is powered for the whole step. Interleaving halves the ADC step here, and with every ADC powered it also halves their energy; contiguous placement can switch off the empty groups' drivers. `c3cim_xbar`'s drivers are powered this way.

Units: ohm, V, uA, ns, pJ (event energy), um^2 per installed instance.

**Crossbar**: the memory, tile size, read voltage, `active_rows` (rows driven per activation; fewer than `rows` splits the rows into phases, one activation each), and `reference_columns` (analog encoding: one G(0) column per output).

**Mapping**: `weight_bits` (default 6; None = unquantized, analog only), `weight_scaling` (default `"std3"`), `weight_encoding`, `conv` (default `"parallel"`, see **Conv mapping**) and `columns` (default `"interleaved"`: where a tile's weight columns sit among the column groups that share an ADC, a driver, ...; see above). Weights are quantized per layer, symmetric around 0 (codes -(2^(b-1)-1) ... 2^(b-1)-1; weights beyond the range saturate), with the range from `weight_scaling`:

| `weight_scaling` | Range (clip) |
|---|---|
| `"max"` | the largest \|weight\|: nothing clipped; outliers stretch the steps |
| `"mse"` | the clip with the least squared quantization error, searched per layer (200 steps up to the largest \|weight\|; never worse than `"max"` in that error) |
| `"std<k>"`; default `"std3"` | k standard deviations of the layer's weights (never beyond the largest \|weight\|) |

The trained weights are centred on zero (|mean| < 0.14 standard deviations in every layer), so no offset is used. Lower weight error does not guarantee higher accuracy: compare the scalings with `tools/accuracy_sweep.py`. Measured test accuracy (%) with every layer quantized (N-MNIST: first 1,024 test samples; DVS-Gesture: all 264, where one sample is 0.38 points):

| Weights | N-MNIST max | N-MNIST mse | N-MNIST std3 | Gesture max | Gesture mse | Gesture std3 |
|---|---:|---:|---:|---:|---:|---:|
| 4-bit | 25.9 | 10.6 | 29.1 | 66.3 | 79.5 | 76.9 |
| 6-bit | 43.5 | 49.0 | 95.5 | 84.1 | 83.3 | 87.5 |
| 8-bit | 95.0 | 95.1 | 95.9 | 86.4 | 86.7 | 87.5 |
| float | 96.2 | | | 86.4 | | |

On the full test sets, 6-bit `"std3"` gives 96.80% on N-MNIST (float 97.77%) and 87.50% on DVS-Gesture (float 86.36%; the difference is 3 of 264 samples). With `"max"`, a few large weights in N-MNIST's dense layers (up to 13 standard deviations) set the step, so most of their weights round to 0 and the output layer's firing collapses or saturates; `"std3"` is the most consistent choice, hence the 6-bit `"std3"` default.

Encodings:

| Encoding | Columns per weight | Cells |
|---|---|---|
| `twos_complement` | ceil(weight_bits / cell_bits) | bit slices of the two's-complement code |
| `offset` | ceil(weight_bits / cell_bits) | bit slices of code + 2^(bits-1) |
| `differential` | 2 x ceil((weight_bits - 1) / cell_bits) | magnitude slices, positive and negative columns |
| `analog` | 1 | G linear in the weight over [-max\|w\|, max\|w\|]; G(0) at the midpoint |

A cell at level L has the memory's level-L conductance. How slice currents are weighted and combined after the array is part of the component list (e.g. `slice_mirror`).

Inputs are binary spikes. Spikes of every time bin are integrated with equal weight in the LIF, so input precision is the number of time bins.

**Conv mapping**: each kernel (one output channel) occupies K = in_channels x kh x kw rows of one column per weight slice (a 3x3x3 kernel with 4-bit weights on 1-bit cells: 27 rows x 4 columns). A layer is unrolled into windows (one per output pixel, stride and padding included; a dense layer has one window), and a window's input patch drives the kernel rows. One weight copy holds all kernels in the fewest tiles: K rows x (out_channels x columns per weight), split into row tiles x column tiles.

| `Mapping(conv=...)` | Weight copies | Activations per time bin |
|---|---|---|
| `parallel` (default) | one per window: every window computed at once, each output with its own columns | row phases of a tile set |
| `sequential` | 1: all kernels once; the windows are applied one after another (the outputs share columns) | windows x row phases |

With `parallel`, copies that fit in a tile share it, packed block-diagonally: as many as fit both the tile's rows and its columns (e.g. 18-row x 16-column copies: 3 per 64x64 tile, so 25 windows take 9 tiles). Each copy has its own rows (its window's inputs) and columns (its outputs). The cells of a copy's rows in the other copies' columns hold level 0 and conduct G(level 0) x v_read on every spiking row, which is charged to the array (and to the slice mirrors and reference columns of those columns). Columns without weights are off, with every component counted on them (`used_columns`). A copy larger than a tile gets its own tiles. The windows activated together are a **slot**: one window per activation with `sequential`, one tile set's copies with `parallel`. With `active_rows`, a packed tile drives the rows of all its copies in phases, so packing can add activations.

**Activations and steps** (timeline): an **activation** is one drive of a set of rows and read of the columns: one row phase, and with `sequential` also one window. A time bin holds activations x (the per-activation steps), then the per-time-bin steps:

```
| cells | adc | cells | adc | ... | cells | adc |  lif  |
 \________ per activation, x activations ______/ per time bin
```

A component with `time_ns` is a **step** of the timeline, named after the component. `time_ns` is the time of one operation: every instance works on all its inputs at once, so the step lasts `time_ns`. With `serial=True` (`column_groups` counts only) each instance converts its group's weight columns one after another, and the step lasts as long as the fullest group: ceil(weight columns in the fullest tile / groups per tile) x `time_ns` with interleaved columns, min(group size, weight columns) x `time_ns` with contiguous ones; empty columns are skipped. How often a step runs is inferred from `count`: columns, rows, column groups and tiles work **per activation**; neurons (`outputs`, `output_bank`) and layer-wide parts (`one`, `fixed`) **per time bin**. `per="activation"` or `per="time_bin"` overrides it (e.g. a per-output accumulator that adds every row phase).

**Placement**: a step starts when every step of its level given before it has ended (a per-time-bin step also waits for all activations), so the steps run one after another in the order of the components. `at=` starts a step at an anchor instead: `"cells.start"` (in parallel with the read), `"adc.end-1"` (1 ns before the ADC ends, overlapping it), `"bin.start"` (a per-time-bin step at the start of the time bin, alongside the activations). A step always lasts its own duration: `at` moves it, it never stretches it. Anchors may name a step given later; steps placed at each other in a cycle are an error. A per-activation step can only be placed at per-activation steps. `compose(..., activation_interval_ns=..., time_bin_interval_ns=...)` pipelines consecutive activations / time bins.

**Names and anchors** (one rule): a bare name is a **window**, `"cells"` (that step), `"bin"` (the time bin), `"activations"` (all activations of the time bin); an **anchor** is a **point** and always carries its edge: `"name.start"`, `"name.end"`, with an offset in ns (`"cells.start+1"`, `"lif.end-0.5"`); `"bin.start"`, `"bin.end"` (the time bin); `"activations.start"`, `"activations.end"` (all activations of the time bin). Seen from the time bin, a per-activation step starts with the first activation and ends with the last.

**Power** (when a component draws current, `when=`): a window, `"cells"` (during that step) or `"bin"` (the whole time bin), or a `(from, to)` pair of anchors (`when=("cells.start", "lif.end")`: from the read's start to the neuron's end). The default is the component's own step; a component with neither a step nor `when` only has area and event costs. `when` never moves anything on the timeline: `at` decides when a component works, `when` when it is switched on. A window within the per-activation steps is powered **in every activation** and gated per activation; one reaching a per-time-bin step or the bin's edges is powered **once per time bin**. One window per component: a component powered in two separate windows is two components.

**Costs** of a component:

- `static_ua` x `supply_v` x powered time, for every powered instance;
- `event_pj` per event: per powered instance per activation (`events="activation"`), per time bin (`"time_bin"`), or per LIF output spike of the layer (`"output_spike"`);
- data-driven models, drawn from `supply_v` while powered (per activation; over a whole time bin only with one activation per time bin), with currents computed from the spikes and conductances: `"crossbar_read"` (weight columns), `"reference_read"` (G(0) reference columns) and `"slice_mirror"` (current mirrors copying each weight-slice column with gain `slice_gains`; default binary, most significant slice x1, the next x1/2, ...).

Count and powered rules, with the unit each instance belongs to:

| Rule | Unit | Instances per unit |
|---|---|---|
| `tiles` | row tile of a tile set | column tiles |
| `physical_rows` / `physical_columns` | row tile of a tile set | column tiles x tile rows / cols |
| `used_columns` | row tile of a tile set | out_channels x columns per weight, per copy in it |
| `column_groups` (`size`) | row tile of a tile set | column tiles x ceil(cols / size) |
| `used_column_groups` (`size`) | row tile of a tile set | the column groups holding a weight column (by `Mapping.columns`) |
| `outputs` | window | out_channels (neurons; installed for every window) |
| `output_bank` | tile set (slot) | column tiles x cols |
| `one` / `fixed` (`value`) | layer | 1 / value |
| `spiking_rows` (powered only) | activation | word lines carrying a spike, one per column tile |

`{"rule": ..., "gated": True}` powers an instance only when its unit receives at least one input spike in that activation (per-activation power) or time bin (per-time-bin power). Without gating, every valid unit is powered: a partially filled row tile stops after its last row phase.

Leave unknown values at 0 (e.g. areas) and switch the metric off. A different memory is a change of the crossbar's cell parameters; a new circuit is a `Component` (with `time_ns` if it takes time of its own).

## 4. How costs are computed

For each layer and batch (`hardware/engine.py`):

1. **Mapping**: windows, weight copies, copies per tile, tiles, row phases and activations per time bin (`mapping.layer_geometry`).
2. **Activity** (on the GPU if available): per row, the number of spikes (the frames' sum, unfolded once); the spikes that leak into packed copies' columns; per activation and time bin, which row tiles, slots, windows and the layer receive a spike (`mapping.spike_activity`).
3. **Timeline**: step start/end times (serial steps from the layer's weight columns per tile), activations per time bin, latency per inference = (T - 1) x time-bin interval + time-bin span (`timeline.build_timeline`). Latency follows the schedule and does not depend on the data.
4. **Energy** per component: static energy = supply x current x (powered time per activation or time bin) x (powered instances summed over all activations or time bins); event energy; data-driven energy = supply x (for each column slice: v_read x sum over spikes of that row's conductance plus the level-0 leak of packed copies, times its mirror gain for `slice_mirror`) x powered time.

Per inference: energy is divided by the number of evaluated samples; area counts installed instances once. Network totals add the layers, which run one after another. Reported metrics: energy (nJ), latency (us), power = energy / latency (mW), area (mm^2), TOPS/W = 2 x dense MACs / energy (every input in every time bin, zeros included), and pJ per synaptic operation (SOP = an input spike reaching one output neuron; the event-driven SNN figure).

## 5. Verification

`tests/test_hardware.py` holds a deliberately literal reference model: it builds every window's patch by hand, walks every sample, time bin, window, row tile and row phase, decides for each unit whether it is powered, and sums every cell's current. The engine must match it for both conv mappings, stride and padding, partial tiles, row phases, linear 1-bit and nonuniform 2-bit memories, all four encodings, slice mirrors with binary and custom gains, and every rule (gated or not). Hand calculations cover the 1-bit RRAM design's dense layer cell by cell (cells, gated OTAs, binary mirrors, comparators on for the LIF step), and two compositions defined in the test (a conventional analog crossbar with reference columns and DAs, and a C3CIM crossbar with VI converters) reproduce this project's earlier worked examples exactly (K = 96 rows, 2 outputs, 8 active rows, one time bin):

| Architecture | Energy (nJ) | Latency (ns) | Area (um^2) |
|---|---:|---:|---:|
| `c3cim` | 0.0279948 | 482 | 10259.68 |
| `conventional` | 0.07274388 | 38 | 9969.4 |

A further hand calculation covers a tile with row phases and ADCs shared by column groups (`serial=True`): 8 activations x (5 + 8 x 6 ns) + 2 ns = 426 ns per time bin with 60 weight columns, and 8 x (5 + 4 x 6) + 2 ns with 32 (interleaved, 4 conversions per ADC) against 8 x (5 + 8 x 6) + 2 ns contiguous, with the ADCs and drivers holding weight columns (8 and 2 interleaved, 4 and 1 contiguous); the reference model places every tile's weight columns one by one and is checked with both placements. Timeline tests cover serial, parallel (`at`), overlapping and pipelined steps, per-time-bin steps at the bin's start and at per-activation steps, cycles, power windows between step edges with offsets (including a step referenced before it is defined), and a component that would serve two pipelined activations at once; composition tests cover the inferred and overridden step frequencies, the anchor syntax and the `when` forms. Pipeline tests cover the layer probes, the runner and the metric switches.

## 6. Spiking-neuron implementation (no slayerSNN)

The networks were trained with the slayerSNN (SLAYER PyTorch) framework, which is no longer maintained and needs a compiled CUDA extension. `models/srm.py` reimplements the parts inference needs in plain PyTorch, following slayerSNN's computations exactly:

- `psp`: causal filtering with the SRM alpha kernel, times `Ts`, accumulated in slayerSNN's tap order with fused multiply-adds (its CUDA kernels are built with `-use_fast_math`, which fuses them the same way).
- `spike`: fire when the membrane reaches `theta`, then add the refractory kernel from the spike step onwards. Spikes have amplitude `1/Ts`.
- `conv` and `dense`: `Conv3d` layers applied per time step.
- `pool`: a per-channel window sum scaled by `1.1 × theta`, including SLAYER's padding of odd sizes.
- Event reading and binning (`models/events.py`): polarity shifted to start at 0, round-half-to-even time bins, and a bin holding any event set to `1/Ts`.

The kernels generated from the YAMLs match the kernels stored in the trained weights bit for bit. `tests/test_srm.py` checks the layers against literal transcriptions of slayerSNN's CUDA loops. This is an inference implementation only: it has no surrogate gradients, so it cannot train.

**Verification against the original framework.** slayerSNN (built from source with CUDA 12.8 and PyTorch 2.8 on an RTX 2080 Ti) ran both networks (detailed recordings of the first 20 N-MNIST and 5 DVS-Gesture test samples, and predictions on both whole test sets); `SRMLayer` was then run on the same inputs:

| Check | N-MNIST | DVS-Gesture |
|---|---|---|
| input spikes from the event readers | identical, 20/20 | identical, 5/5 |
| every layer's input spikes | 0 of 24.5 M differ (SC1-SC3, SF1, SF2) | 0 of 63.1 M differ (SC1, SC2, SF1, SF2) |
| output spikes and predicted classes | identical, 20/20 | identical, 5/5 |
| hardware energy from either set of spikes | identical | identical |
| full test set, batched on the GPU | 97.77% vs 97.77%, same prediction on 10000/10000 | 86.36% vs 86.36%, same prediction on 264/264 |

To repeat or extend the check on a machine with slayerSNN:

```bash
python tools/export_slayer_reference.py --model nmnist --data /path/to/datasets --samples 20 --full
python tools/export_slayer_reference.py --model gesture --data /path/to/datasets --samples 5
python tools/compare_slayer_reference.py reference/nmnist_slayer.pt --full --data /path/to/datasets
```

The export records slayerSNN's layer inputs, output spikes and predictions in `reference/<model>_slayer.pt`; the comparison feeds the same inputs to `SRMLayer` and reports, per layer, how many spike entries differ and from which time step, the predicted classes, the full-test-set accuracy (`--full`) and the effect on the hardware energy. `tests/test_slayer_reference.py` runs automatically once reference files are in `reference/`. slayerSNN's `setup.py` does not install with current pip: build it with `python setup.py build_ext --inplace`, link `src` as `slayerSNN` and add the folder to the Python path; it needs `numpy<2`.

The original pickled checkpoints (commit `7f020a7`, `pretrained/*.pt`) needed slayerSNN to load. `tools/convert_checkpoints.py` produced the current `.pth` files from them without slayerSNN.

## 7. Scope and caveats

- Costs come from the component list: anything not listed (routing, buffers, control) is not charged.
- No analog nonidealities: IR drop, device variation, noise, compliance and settling limits are not modelled, and accuracy comes from the software network.
- Latency follows the fixed schedule; activations without spikes still take their time (only power is gated).
- A component has one power interval; a circuit powered in two separate intervals is described as two components.
- Only weighted conv and dense layers are mapped; pooling runs in software and is not charged.
- `outputs` installs a neuron for every output (channel x window), also with the sequential mapping; use `output_bank` or `fixed` for time-multiplexed neurons.
