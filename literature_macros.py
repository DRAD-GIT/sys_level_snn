"""Published resistive-memory CIM macros, modelled in the engine so that they
run the same recorded workloads (N-MNIST, DVS Gesture) as our designs.
run.py evaluates them with its ARCHITECTURES (list LITERATURE).

    python literature_macros.py      # checks every macro against its paper

Each macro is calibrated to its own paper: the paper's reported numbers (a
per-block energy table, or its TOPS/W or power with its power/energy
breakdown) are turned into per-instance component energies and powers, and
the check at the bottom runs each macro on the paper's own test condition and
compares the engine's energy with the paper's. The derivation of every number
is written next to it.

How the numbers are derived, for every macro:
  1. The reference energy: the paper's energy per operation of its whole
     array (a matrix-vector multiplication, MVM) at a stated test condition,
     from a per-block energy table, or from TOPS/W (1 TOPS/W = 1 op/pJ; ops =
     2 x MACs, the convention of all five papers), or power x time.
  2. It is split by the paper's own energy/power breakdown (a pie chart or
     table) into the blocks of the macro.
  3. Each block's share is divided among the instances that work in that test
     condition (the driven rows, the used columns, the neurons) and becomes an
     event energy (pJ per instance per operation), or, for a block that draws
     current for as long as it is on, a static current during its window.
  4. The array (component "cells") uses the paper's cell resistances and read
     voltage; the engine computes its current from the spikes and the stored
     weights. Where the paper's array energy differs from V_read x I x t (the
     read bias changes the resistance, or the paper's "array" share includes
     clamps or mirrors), cell_supply_v is the effective rail that makes the
     array reproduce the paper's array energy at the test condition.
  5. Timing: the read time is the paper's read pulse (or read phase); steps
     that the paper runs serially (e.g. a shared ADC) are serial steps.

What differs from the papers (applies to all):
  * Inputs are our networks' binary spikes, one per input per time bin, not
    the papers' multi-bit (latency / dual-spike / bit-serial) inputs. Energies
    of blocks that scale with the input pulse length are scaled to one 1-bit
    pulse (stated per macro).
  * Weights: every macro stores run.py's 6-bit weights (WEIGHT_BITS), so all
    rows share one recorded forward pass and one accuracy. Each keeps its
    paper's cell precision (at most MAX_CELL_BITS = 3 bits) and weight layout
    (differential or two's complement), and splits the 6-bit weight over as
    many columns as that needs. The component energies are per row, column,
    weight or neuron, so a weight on more columns than in the paper costs
    proportionally more. check() runs each macro at its paper's own weight
    precision (native()).
  * Macros without a neuron (the ANN macros) are charged no neuron: the table
    shows their array + periphery only.
  * The mapping (conv="parallel", contiguous columns) is the same as our
    designs', so every macro runs the same workload.
  * specs["label"]: a dagger marks simulated (not measured) papers.
"""
import dataclasses

import crossbars
from hardware import Architecture, Component, Mapping, Memory, compose

WEIGHT_BITS = 6           # as in run.py: all rows share its recorded forward pass
WEIGHT_SCALING = "std3"   # as in run.py
MAX_CELL_BITS = 3         # cells store at most 3 bits, whatever the device allows


def mapping(encoding, weight_bits=WEIGHT_BITS):
    return Mapping(weight_bits=weight_bits, weight_scaling=WEIGHT_SCALING,
                   weight_encoding=encoding, conv="parallel", columns="contiguous")


# ============================================================================
# 1. DS-CIM: Fu et al., "DS-CIM: A 40nm Asynchronous Dual-Spike Driven, MRAM
#    CIM Macro for SNN", IEEE TCAS-I 71(4), 2024. SIMULATED.
#    Zotero R9PBCTVH. Reports DVS Gesture: 90.00%, 729.35 nJ, 40.46 us.
# ----------------------------------------------------------------------------
# Test condition of Table I ("DS-CIM macro resources needed for 512 1bIn-
# signed 4bW computation", 40 nm, VDD 0.9 V): one 512 x 1 input vector, 1-bit
# inputs of 10 ns pulse width, 512 x 512 array, 50% input sparsity (256
# spiking rows) and 50% weight sparsity (taken as half of the cells in the
# parallel, low-resistance state). Table I energies for that vector:
#   AER module (128)          4.374 pJ
#   WL driver (512)           146.8 fJ
#   MRAM array (512 x 512)    0.74 nJ
#   readout circuit (512)     27.65 pJ
#   preprocessing (64)        0.49 nJ
#   DS-neuron (64)            0.21 nJ
#   total                     1.47 nJ
# Weights: the paper's signed 4-bit weight is 8 binary MRAM cells, 4 positive
# + 4 negative columns (preprocessing circuit, Fig. 9(a)): differential
# encoding with binary cells (cell_bits=1, within the 3-bit limit). Paper:
# 4 magnitude bits -> 8 columns per weight, 64 weights per 512 columns (its
# 64 preprocessing circuits and neurons). Here: 6-bit weights, 5 magnitude
# bits -> 10 columns per weight, 51 weights per 512 columns.
DS_CIM = compose(
    "ds_cim_tcas24",
    mapping("differential"),
    [
        # Cells: Rp = 89.15 kOhm, Rap = 157.14 kOhm (measured at 0.1 V, text
        # with Fig. 4); V_read = 0.17 V (Sec. V-A); one 10 ns read pulse per
        # spike.
        # V_read x I x t for the test vector: 256 rows x 512 columns x the mean
        # of 1/Rp and 1/Rap = 1.152 S, x 0.17 V x 10 ns = 0.333 nJ, below the
        # paper's 0.74 nJ (the resistance falls at the higher read bias, Fig.
        # 4(a)). cell_supply_v = 0.74 nJ / (1.152 S x 0.17 V x 10 ns) = 0.378 V
        # reproduces 0.74 nJ.
        crossbars.conv_xbar(cell_bits=1, r_on=89.15e3, r_off=157.14e3, rows=512, cols=512,
                            v_read=0.17, cell_supply_v=0.378, time_ns=10.0),
        # AER: 4.374 pJ for the 256 spiking rows = 17.09 fJ per input spike.
        Component("aer", count="physical_rows", powered="spiking_rows", event_pj=0.01709),
        # WL driver: 146.8 fJ / 256 spiking rows = 0.573 fJ per input spike.
        Component("wl_driver", count="physical_rows", powered="spiking_rows",
                  event_pj=0.000573),
        # Readout (BL clamp): 27.65 pJ / 512 columns = 54.0 fJ per column per
        # read; off when no word line of the tile is on (Event_en of the
        # readout circuit).
        Component("readout", count="physical_columns",
                  powered={"rule": "used_columns", "gated": True}, event_pj=0.0540),
        # Preprocessing: a scaling current mirror per column, combining a
        # weight's columns (Fig. 9(a)): 0.49 nJ / 512 columns = 0.957 pJ per
        # column per read.
        Component("preprocessing", count="physical_columns",
                  powered={"rule": "used_columns", "gated": True}, event_pj=0.957),
        # DS-neuron: 0.21 nJ / 64 = 3.281 pJ per neuron per time bin in which
        # its inputs spike (asynchronous: it fires within the read, no step).
        Component("ds_neuron", count="outputs", powered={"rule": "outputs", "gated": True},
                  events="time_bin", event_pj=3.281),
    ],
    specs={"label": r"TCAS-I'24 DS-CIM$^\dagger$", "tech": 40, "supply": 0.9,
           "device": "SOT-MRAM", "bitcell": "1T1MTJ", "sensing": "Current (time)"},
)

# ============================================================================
# 2. PCM: Khwa et al., "A 40-nm, 2M-Cell, 8b-Precision, Hybrid SLC-MLC PCM
#    Computing-in-Memory Macro with 20.5 - 65.0TOPS/W for Tiny-AI Edge
#    Devices", ISSCC 2022, paper 11.3. Measured. Zotero DEZEP3DX.
# ----------------------------------------------------------------------------
# The paper gives no power breakdown, only per-precision totals (Fig. 11.3.7).
# We calibrate at its 1-bit-input point, the one closest to our spikes:
#   1bIN-2bW-5bOUT: access time 3.25 ns, 3.9 TOPS, 261 TOPS/W (0.85 V,
#   hybrid SLC-MLC, input reordering (IN-R), 55% input sparsity).
# Array (Fig. 11.3.2, chip summary): sub-banks of 256 rows x 1024 columns of
# 1T1R PCM; 8 word lines are driven at once (1-bit inputs, 8 accumulations);
# each column is read by a current-controlled voltage-metric block, a fixed
# current I_M through the column's 8 cells (so the read energy does not
# depend on the stored data), then a 4-bit voltage-swing-remapping sense
# amplifier (VSR-VSA), shared by 8 columns through an 8-to-1 MUX, and a
# digital shifter and adder.
# Energy per sensed column: one read of a column is 8 rows x a 2-bit weight
# (one MLC cell) = 8 MACs = 16 ops; 16 / 261 = 61.3 fJ per column read,
# counting every input (the paper's ops are dense). IN-R skips a group of 8
# inputs when all are zero: at 55% sparsity 0.55 x 256 = 140.8 zero inputs
# fill 17 of the 32 groups, so 15 of 32 groups are read, and the energy of
# a read group is 61.3 fJ x 32 / 15 = 130.8 fJ per column. The engine skips
# a row phase (8 rows) that holds no spike in the same way (gated).
# Timing: 3.25 ns per read of a column (the access time); the 8 columns of
# a VSA are read one after another (8-to-1 MUX): 26 ns per group of 8 rows.
# Weights: two's complement across neighbouring cells, as in the paper; 2-bit
# MLC cells (cell_bits=2, within the 3-bit limit): 6-bit weights -> 3 columns
# per weight (the paper's 8-bit weights use 2 SLC + 3 MLC cells).
PCM_KHWA = compose(
    "pcm_khwa_isscc22",
    mapping("twos_complement"),
    [
        # Cells: 256 x 1024, 8 rows per activation. The paper gives no PCM
        # resistances; the cell current is the fixed bias I_M, charged in
        # "sense" below, so the cells are not charged (cell_supply_v=0) and
        # r_on / r_off (1 k / 1 M) only satisfy the cell model. The word-line
        # settling is inside the 3.25 ns access time (step "sense").
        crossbars.conv_xbar(cell_bits=2, r_on=1e3, r_off=1e6, rows=256, cols=1024,
                            active_rows=8, v_read=0.1, cell_supply_v=0.0, time_ns=0.0),
        # Column read (I_M bias + VSR-VSA + shifter/adder): 130.8 fJ per used
        # column in every row phase that holds a spike.
        Component("sense", count="physical_columns",
                  powered={"rule": "used_columns", "gated": True}, event_pj=0.1308),
        # VSA timing: one per 8 columns (8-to-1 MUX), 3.25 ns per column, one
        # column after another.
        Component("vsa", count={"rule": "column_groups", "size": 8},
                  powered="used_column_groups", time_ns=3.25, serial=True),
    ],
    specs={"label": r"ISSCC'22 PCM~\cite{khwa_40-nm_2022}", "tech": 40, "supply": 0.85,
           "device": "PCM", "cell": "2 (MLC)", "bitcell": "1T1R", "r_ratio": "--",
           "sensing": "Voltage"},
)

# ============================================================================
# 3. Tempo-CIM: Jiang et al., "Tempo-CIM: A RRAM Compute-in-Memory
#    Neuromorphic Accelerator with Area-Efficient LIF Neuron and Split-Train-
#    Merged-Inference Algorithm for Edge AI Applications", IEEE JETCAS 13(4),
#    2023. Chip measured; power breakdown and efficiency post-layout simulated.
#    Zotero 5TY7UBXG.
# ----------------------------------------------------------------------------
# The paper gives two efficiencies for "4b input, 8b weight" (68.51 TOPS/W
# in the abstract, 117.9 in Fig. 17 and the conclusion); we calibrate to the
# array's native 4-bit weights, Fig. 17 (chip summary, 0.9 V core):
# 4bIN-4bW: 5.17 GOPS, 224.8 TOPS/W.
#   array: 256 rows x 64 columns (16 Kb), in banks of 64 rows x 4 columns;
#   a signed 4-bit weight uses 4 binary cells of a bank's row, one per bit
#   (architecture overview) -> 16 outputs, 64 rows summed per output.
#   ops per MVM = 2 x 256 x 16 = 8192 -> 8192 / 224.8 = 36.44 pJ per MVM,
#   in 8192 / 5.17 GOPS = 1584.5 ns per MVM.
#   power breakdown (Fig. 18(b), simulated at 0.9 V): array 35.2% = 12.83 pJ,
#   encoder 28.8% = 10.49 pJ, digital 24.3% = 8.855 pJ, neuron 11.7% =
#   4.263 pJ.
# Single-spike latency coding: each input is one pulse of one unit delay,
# whose position in a window of 16 units encodes the 4-bit value; the unit
# delay is 1584.5 / 16 = 99 ns (the paper tests its neuron with 100 ns
# pulses). A 1-bit spike is one such pulse, so the per-pulse energies hold.
# Weights: two's complement in binary cells (cell_bits=1), as in the paper;
# paper: 4-bit weights in 4 columns; here: 6-bit weights in 6 columns.
TEMPO_CIM = compose(
    "tempo_cim_jetcas23",
    mapping("twos_complement"),
    [
        # Cells: HRS 600 kOhm, LRS 20 kOhm (measured, text with Fig. 10); one
        # 99 ns pulse per spike; 64 rows per bank (one tile = 64 rows x 64
        # columns).
        # V_read is not given: with the cells' energy alone (cell_supply_v =
        # v_read), the array share 12.83 pJ for all 256 rows x 64 columns with
        # half the cells in LRS (sum G = 0.4232 S) over 99 ns needs
        # v_read = sqrt(12.83 pJ / (0.4232 S x 99 ns)) = 17.5 mV (an effective
        # value: the paper does not give the read bias or the conducting time).
        crossbars.conv_xbar(cell_bits=1, r_on=20e3, r_off=600e3, rows=64, cols=64,
                            v_read=0.0175, cell_supply_v=0.0175, time_ns=99.0),
        # Encoder (latency coder + WL/SL driver per row): 10.49 pJ / 256
        # rows = 41.0 fJ per input pulse.
        Component("encoder", count="physical_rows", powered="spiking_rows", event_pj=0.0410),
        # Digital control: 8.855 pJ per MVM of the 4 tiles of 64 x 64 =
        # 2.214 pJ per tile per read.
        Component("digital", count="tiles", powered={"rule": "tiles", "gated": True},
                  event_pj=2.214),
        # Charge-pump LIF: 4.263 pJ / 16 neurons = 266.5 fJ per neuron per
        # time bin in which its inputs spike.
        Component("lif", count="outputs", powered={"rule": "outputs", "gated": True},
                  events="time_bin", event_pj=0.2665),
    ],
    specs={"label": r"JETCAS'23 Tempo-CIM", "tech": 40, "supply": 0.9, "device": "RRAM",
           "bitcell": "1T1R", "sensing": "Charge (LIF)"},
)

# ============================================================================
# 4. Memristive SNN: fully integrated memristive SNN with a 128 x 24 memristor
#    array and analog SRM neurons on a 180 nm CMOS chip (DVS Gesture 93.06%,
#    N-MNIST 94.73%, 101.05 TSOPS/W). Measured, with simulated layers.
#    Zotero Q5FRQAVH.
# ----------------------------------------------------------------------------
# The paper gives the average power during DVS Gesture inference, 43.83 mW
# (1.94 uJ per sample over 44.31 us), and its energy breakdown (Fig. S13a):
#   neuron 61.78% = TIA pair 55.76% + soma 6.02%, array 10.51%, WL driver
#   2.35%, BL driver 2.47%, SL driver 8.37%, controller 0.00%, I/O 13.97%,
#   others 0.56%.
# It gives no spike counts, so the blocks are modelled by their average power
# (share x 43.83 mW), per instance of the paper's network (Fig. 4b): a fully
# connected 800-480-120-11 SNN (128 x 128 x 2 input, 6-pixel padding, 7 x 7
# pooling -> 20 x 20 x 2 = 800 inputs) on PEs of 128 rows x 24 columns with
# 12 neurons (differential pairs):
#   neurons = 480 + 120 + 11 = 611
#   PEs = ceil(800/128) x 480/12 + ceil(480/128) x 120/12 + 1 = 280 + 40 + 1 = 321
# Supply: the paper gives none; 1.8 V (180 nm core) is assumed; the currents
# below are power / 1.8 V, so the energies do not depend on it.
# Time: inputs at a temporal resolution of 100 ns (a 30 us sample is 300
# steps): one time bin = 100 ns. The paper's 14.31 us circuit latency after
# the input is not added.
# Weights: the device holds 15 evenly spaced conductance states within
# 150 uS (Fig. 3b), taken as 10..150 uS in 10 uS steps, plus the off state
# (~0 uS, below the 4 uS stuck-off limit): 16 levels, a 4-bit cell. Limited
# to 3 bits: 8 levels, every second programmed state, 0, 20, ..., 140 uS.
# Signed weights are differential pairs (Methods): 6-bit weights, 5 magnitude
# bits in 3-bit cells -> 2 columns per sign, 4 per weight, 6 outputs per 24
# columns (the paper: 4-bit cells, 2 columns per weight, 12 outputs; so
# twice the PEs, each with the per-PE power below).
MEMRISTIVE_SNN = compose(
    "memristive_snn_180nm",
    mapping("differential"),
    [
        # Cells: 0.2 V read pulses (retention test, text with Fig. 3); their
        # energy is inside the array's average power below, so the cell
        # current itself is not charged again (cell_supply_v=0).
        crossbars.conv_xbar(cell_bits=MAX_CELL_BITS, levels_s=tuple(i * 20e-6 for i in range(8)),
                            rows=128, cols=24, v_read=0.2, cell_supply_v=0.0, time_ns=100.0),
        # Per PE (tile), always on (average power over the inference):
        #   array 10.51% x 43.83 mW / 321 = 14.35 uW -> 7.97 uA
        #   WL driver 2.35% -> 3.21 uW -> 1.78 uA
        #   BL driver 2.47% -> 3.37 uW -> 1.87 uA
        #   SL driver 8.37% -> 11.43 uW -> 6.35 uA
        #   others 0.56% -> 0.765 uW -> 0.425 uA
        Component("array", count="tiles", when="bin", supply_v=1.8, static_ua=7.97),
        Component("wl_driver", count="tiles", when="bin", supply_v=1.8, static_ua=1.78),
        Component("bl_driver", count="tiles", when="bin", supply_v=1.8, static_ua=1.87),
        Component("sl_driver", count="tiles", when="bin", supply_v=1.8, static_ua=6.35),
        Component("others", count="tiles", when="bin", supply_v=1.8, static_ua=0.425),
        # Per neuron, always on:
        #   TIA pair 55.76% x 43.83 mW / 611 = 40.0 uW -> 22.2 uA
        #   soma 6.02% -> 4.32 uW -> 2.40 uA
        #   I/O 13.97% (output spikes measured off-chip) -> 10.0 uW -> 5.57 uA
        Component("tia", count="outputs", when="bin", supply_v=1.8, static_ua=22.2),
        Component("soma", count="outputs", when="bin", supply_v=1.8, static_ua=2.40),
        Component("io", count="outputs", when="bin", supply_v=1.8, static_ua=5.57),
    ],
    specs={"label": r"Memristive SNN (180nm)", "tech": 180, "supply": "--", "device": "RRAM",
           "cell": "3", "bitcell": "1T1R", "r_ratio": "50/7.14",
           "sensing": "Current (TIA)"},
)

# ============================================================================
# 5. TD-CIM: Wei et al., "A 28-nm Static-Power-Free Fully Parallel RRAM-Based
#    TD CIM Macro With 1982 TOPS/W/Bit for Edge Applications", IEEE SSC-L 8,
#    2025. Measured. Zotero SB7696M8.
# ----------------------------------------------------------------------------
# Test condition (Sec. III-A, Fig. 7, Table I): 320 x 128 time-domain (TD)
# cells, all 320 inputs in parallel, 1-bit inputs, ternary weights,
# 0.6 / 0.75 / 0.55 V analog / digital / array supplies, SAR clock 333 MHz:
# 1251 TOPS/W, 0.341 TOPS, total power 0.27 mW.
#   ops per MVM = 2 x 320 x 128 = 81920 -> 81920 / 1251 = 65.48 pJ per MVM
#   (check: 0.27 mW x 240.2 ns = 64.9 pJ), in 81920 / 0.341 TOPS = 240.2 ns.
#   measured power breakdown (Fig. 7): input driver (DIN) 36% = 23.57 pJ,
#   array 28% = 18.34 pJ, TDC 34% = 22.26 pJ, others 2% = 1.310 pJ.
# A TD cell is a pair of delay-buffer (DB) cells, each a 1T1R RRAM on one of
# the chain's two lines (DBP / DBN): the columns below are DB-cell columns,
# 2 x 128 = 256, and a ternary weight is one DB column of each sign.
# Weights: the paper maps a signed m-bit weight to m TD cells of a column;
# we use its ternary TD cell as a sign pair of binary DB cells (cell_bits=1)
# with bit-sliced magnitude (differential): 6-bit weights, 5 magnitude
# columns per sign -> 10 DB columns per weight, shifted-and-added after the
# array as in the paper (Sec. III-A).
# Timing: each LPOSC-TDC serves 8 chains (16 DB columns) with one 8-bit SAR
# ADC at 333 MHz: 8 x 3 ns = 24 ns per chain, 12 ns per DB column; 8 chains
# = 192 ns. The rest of the 240.2 ns MVM, 48.2 ns, is the TD evaluation and
# sampling phase (step "cells").
TD_CIM = compose(
    "td_cim_sscl25",
    mapping("differential"),
    [
        # Cells: RH / RL = 500 / 50 kOhm (the device values of Fig. 4). The TD
        # cells draw no DC current ("static-power-free"): the cell current is
        # not charged (cell_supply_v=0); the array's switching energy is the
        # event below. v_read: the array supply, 0.55 V (no effect).
        crossbars.conv_xbar(cell_bits=1, r_on=50e3, r_off=500e3, rows=320, cols=256,
                            v_read=0.55, cell_supply_v=0.0, time_ns=48.2),
        # Array: 18.34 pJ / 256 DB columns = 71.6 fJ per DB column per MVM
        # (every cell of a chain switches whatever its input, so per column,
        # not per spike); skipped when the tile gets no spike.
        Component("array", count="physical_columns",
                  powered={"rule": "used_columns", "gated": True}, event_pj=0.0716),
        # Input driver: 23.57 pJ / 320 rows = 73.7 fJ per row per MVM (it
        # launches the rising edge on every row).
        Component("input_driver", count="physical_rows",
                  powered={"rule": "physical_rows", "gated": True}, event_pj=0.0737),
        # TDC energy: 22.26 pJ / 256 DB columns = 87.0 fJ per DB column converted.
        Component("tdc", count="physical_columns",
                  powered={"rule": "used_columns", "gated": True}, event_pj=0.0870),
        # TDC timing: one TDC per 16 DB columns (8 chains), 12 ns per DB
        # column, one column after another.
        Component("tdc_conversion", count={"rule": "column_groups", "size": 16},
                  powered="used_column_groups", time_ns=12.0, serial=True),
        # Others: 1.310 pJ per MVM.
        Component("others", count="tiles", powered={"rule": "tiles", "gated": True},
                  event_pj=1.310),
    ],
    specs={"label": r"SSCL'25 TD-CIM", "tech": 28, "supply": "0.55/0.6/0.75",
           "device": "RRAM", "cell": "ternary", "bitcell": "1T1R (DB cell)",
           "sensing": "Time (TDC)"},
)

LITERATURE_MACROS = [DS_CIM, PCM_KHWA, TEMPO_CIM, MEMRISTIVE_SNN, TD_CIM]


# ============================================================================
# Calibration check: each macro on its paper's test condition
# ============================================================================
def native(arch, weight_bits, memory=None):
    """`arch` with its paper's own weight precision (and memory cell), as
    calibrated."""
    crossbar = arch.crossbar if memory is None else dataclasses.replace(arch.crossbar,
                                                                        memory=memory)
    return Architecture(arch.name, crossbar,
                        dataclasses.replace(arch.mapping, weight_bits=weight_bits),
                        arch.components, specs=arch.specs)


def check():
    """Run every macro on its paper's test condition and return
    [(macro, what, engine value, paper value)]."""
    import torch
    from hardware import evaluate_layer

    def dense(arch, weights, spikes):
        return evaluate_layer(arch, spikes[None, :, None, None, :], weights[:, :, None, None])

    results = []
    # DS-CIM: Table I vector: 256 of 512 inputs spike; 64 weights of +-15
    # (all 4 magnitude cells in P on one sign, half the cells in P).
    w = torch.tensor([15, -15] * 32).repeat(512, 1).T
    s = (torch.arange(512) % 2 == 0).float()[:, None]
    cost = dense(native(DS_CIM, 5), w, s)
    results.append((DS_CIM, "Table I energy (nJ)", cost.energy_nj, 1.47))
    # Tempo-CIM: all 256 rows, 16 weights with half their bits 1 (5 = 0101,
    # -6 = 1010); paper: 36.44 pJ per MVM (224.8 TOPS/W at 5.17 GOPS).
    w = torch.tensor([5, -6] * 8).repeat(256, 1).T
    cost = dense(native(TEMPO_CIM, 4), w, torch.ones(256, 1))
    results.append((TEMPO_CIM, "energy per MVM (pJ)", 1e3 * cost.energy_nj, 8192 / 224.8))
    # TD-CIM: the paper's ternary weights (one DB column per sign, 128 x 2 =
    # 256 DB columns), all 320 inputs: 65.48 pJ and 240.2 ns per MVM.
    w = torch.tensor([1, -1] * 64).repeat(320, 1).T
    cost = dense(native(TD_CIM, 2), w, torch.ones(320, 1))
    results.append((TD_CIM, "energy per MVM (pJ)", 1e3 * cost.energy_nj, 81920 / 1251))
    results.append((TD_CIM, "time per MVM (ns)", cost.latency_ns, 81920 / 0.341e3))
    # Memristive SNN: the paper's 800-480-120-11 network, 300 time bins of
    # 100 ns (a 30 us sample), 4-bit cells: average power 43.83 mW.
    paper = native(MEMRISTIVE_SNN, 5, Memory(4, levels_s=tuple(i * 10e-6 for i in range(16))))
    energy = latency = 0.0
    for n_in, n_out in ((800, 480), (480, 120), (120, 11)):
        cost = dense(paper, torch.ones(n_out, n_in, dtype=torch.long),
                     torch.ones(n_in, 300))
        energy, latency = energy + cost.energy_nj, cost.latency_ns
    results.append((MEMRISTIVE_SNN, "average power (mW)", energy / latency * 1e3, 43.83))
    # PCM: 1-bit inputs, 2-bit weights (one MLC cell per weight, 1024
    # weights), 15 of the 32 groups of 8 inputs spiking (IN-R at 55%
    # sparsity): the paper's 261 TOPS/W counts every input.
    w = torch.ones(1024, 256, dtype=torch.long)
    s = (torch.arange(256) < 15 * 8).float()[:, None]
    cost = dense(native(PCM_KHWA, 2), w, s)
    results.append((PCM_KHWA, "TOPS/W (1bIN-2bW)", 2 * cost.macs / cost.energy_nj * 1e-3, 261))
    return results


if __name__ == "__main__":
    for arch, what, engine, paper in check():
        print(f"{arch.name:<24} {what:<38} engine {engine:10.4g}   paper {paper:10.4g}   "
              f"({100 * (engine / paper - 1):+.1f}%)")
