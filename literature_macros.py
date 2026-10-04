"""Published resistive-memory CIM macros, modelled in the engine so that they
run the same recorded workloads as our designs. `python run.py --literature`
evaluates the ones listed in COMPARED (at the end of this file) next to our work.

    python literature_macros.py      # checks every macro against its paper

  1. DS_CIM          Fu et al., "DS-CIM: A 40nm Asynchronous Dual-Spike Driven, MRAM
                     Compute-In-Memory Macro for Spiking Neural Network", TCAS-I 2024
  2. TD_CIM          Wei et al., "A 28-nm Static-Power-Free Fully Parallel RRAM-Based TD
                     CIM Macro With 1982 TOPS/W/Bit for Edge Applications", SSC-L 2025
  3. MEMRISTIVE_SNN  Wang et al., "Fully Integrated Memristive Spiking Neural Network with
                     Analog Neurons for High-Speed Event-Based Data Processing", arXiv 2025
  4. ASSCC25_SF      Li et al., "A 28nm RRAM-Based 17.0 TOPS/mm2 and 89.7 TOPS/W Compute-
                     In-Memory Macro Enabled by Source-Follower Cell and IR-Drop-Freed
                     Array", A-SSCC 2025
  5. ESSERC24_RRAM   Yao et al., "A 28 nm RRAM-Based 81.1 TOPS/mm2/bit Compute-In-Memory
                     Macro with Uniform and Linear 64 Read Channels under 512 4-bit
                     Inputs", ESSERC 2024

Each macro is calibrated to its own paper: the paper's reported numbers (a
per-block energy table, power at several parallelisms, or its TOPS/W or power
with its power/energy breakdown) are turned into per-instance component
energies and powers, and check() runs each macro on the paper's own test
condition and compares the engine's result with the paper's. The derivation
of every number is written next to it.

How the numbers are derived, for every macro:
  1. The reference energy: the paper's energy per operation of its array (a
     matrix-vector multiplication, MVM) at a stated test condition, from a
     per-block energy table, measured power x time per operation, or TOPS/W
     (1 TOPS/W = 1 op/pJ; ops = 2 x MACs, the convention of all the papers).
  2. It is split by the paper's own breakdown (a pie chart, a table, or power
     measured at several numbers of driven rows: a per-row and a fixed part).
  3. Each part is divided among the instances that work in that test
     condition (driven rows, used columns, neurons) and becomes an event
     energy per instance, or, for a block drawing current while on, a static
     current during its window.
  4. The array ("cells") uses the paper's resistances and read voltage where
     it gives them (DS-CIM); elsewhere the array's measured energy is charged
     through the components and the cells are not charged (cell_supply_v=0).
  5. Timing: the paper's read pulse or clock cycles; converters shared by
     several columns convert them one after another (serial steps).

Rules for all macros:
  * Weights: run.py's 6-bit weights (WEIGHT_BITS), so every row shares one
    recorded forward pass and one accuracy. Each macro keeps its paper's cell
    precision (at most MAX_CELL_BITS = 3 bits) and weight layout, and splits
    the 6-bit weight over as many columns as that needs; the component
    energies are per row, column or neuron, so a weight on more columns than
    in the paper costs proportionally more. check() runs each macro at its
    paper's own weight precision (native()).
  * Inputs are binary spikes: a spike drives its row, a zero does not (the
    cells on a row are on or off). A paper's energy for n-bit inputs is
    normalised to 1-bit inputs by dividing it by n, as in the papers' own
    TOPS/W/bit figures; timing is kept at the paper's.
  * Macros without a neuron are charged no neuron: the table shows their
    array and periphery only.
  * The mapping (conv="parallel", contiguous columns) is our designs', so
    every macro runs the same workload.
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
# 1. MRAM: DS-CIM, Fu et al., "DS-CIM: A 40nm Asynchronous Dual-Spike Driven, MRAM
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
# 2. RRAM: TD-CIM, Wei et al., "A 28-nm Static-Power-Free Fully Parallel RRAM-Based
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

# ============================================================================
# 3. RRAM: Memristive SNN, fully integrated memristive SNN with a 128 x 24 memristor
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
# 4. RRAM: Li et al., "A 28nm RRAM-Based 17.0 TOPS/mm2 and 89.7 TOPS/W
#    Compute-In-Memory Macro Enabled by Source-Follower Cell and IR-Drop-Freed
#    Array", IEEE A-SSCC 2025, paper 25.4. Measured. Zotero 8FCZIK6X.
# ----------------------------------------------------------------------------
# Macro (Fig. 2, Fig. 7): 512 input rows x 512 output channels of 2T2R
# source-follower (SF) cells, each output a differential pair of source lines
# (SL+ / SL-): 1024 RRAM columns. 512 input buffers (RAB), 512 ADCs (WC-ADC).
# 28 nm, VDD = 0.9 V (analog), VDDP = 1.2 V (ADC clamps).
# The SF cell's transistor works in saturation with the input as an analog
# word-line voltage: its current is set by the WL voltage and the RRAM, not
# by V_read / R, and the paper gives no RRAM resistances. So the array is
# charged from the measured power, not from conductances (cell_supply_v=0).
#
# Operating point (Figs. 5, 6, 7): 512 x 512 parallelism, 4-bit inputs, 2-bit
# weights, 9-bit outputs, 50% weight sparsity: 195 mW, 89.7 TOPS/W, 17.5 TOPS.
# Timing: 15 ADC clock cycles per MVM, 4 for the array to settle and 11 for
# the SAR conversion (Fig. 4 text); at the 500 MHz maximum clock, 30 ns.
#   check: ops per MVM = 2 x 512 x 512 = 524288; / 30 ns = 17.5 TOPS (Fig. 7);
#   195 mW x 30 ns = 5.85 nJ per MVM -> 524288 / 5850 pJ = 89.6 TOPS/W.
# Power vs input parallelism (driven rows), 512 outputs (Fig. S2):
#     rows   total    array          periphery
#       64   67.2 mW  52% = 34.94    48% = 32.26
#      128   87.0     57% = 49.59    43% = 37.41
#      256  126.6     67% = 84.82    33% = 41.78
#      512  195.0     77% = 150.15   23% = 44.85
# A straight-line fit over the four points (within 2.6% for the array, 7.2%
# for the periphery) splits each into a per-driven-row and a fixed part;
# x 30 ns per MVM:
#     array      0.2589 mW/row -> 7.768 pJ per driven row per MVM (all 1024
#                cells on its word line conduct); 17.73 mW fixed -> 0.532 nJ
#                per MVM over 1024 columns = 0.519 pJ per column
#     periphery  0.0255 mW/row -> 0.764 pJ per driven row (input DAC, MUX and
#                buffer); 32.96 mW fixed -> 0.989 nJ per MVM over 512 ADCs
#                = 1.931 pJ per ADC = 0.966 pJ per column
# 1-bit input normalisation: our inputs are spikes, a row either driven or
# not (its cells on or off), as the paper's lower parallelisms leave rows
# undriven. The energies above are for 4-bit inputs; as in the paper's own
# 1-bit-normalised efficiency (717.6 TOPS/W/bit = 89.7 x 4 input bits x 2
# weight bits), each is divided by the 4 input bits:
#     array      1.942 pJ per driven row, 0.1298 pJ per column
#     periphery  0.191 pJ per driven row, 0.2415 pJ per column (ADC)
# Timing is kept at 30 ns (1-bit inputs do not shorten the conversion).
# Weights: the SF 2T2R cell holds -1 / 0 / +1 (Fig. 5), a differential pair
# of binary cells (cell_bits=1); 6-bit weights -> 5 magnitude columns per
# sign, 10 columns per weight, 102 weights per 1024 columns.
ASSCC25_SF = compose(
    "sf_rram_asscc25",
    mapping("differential"),
    [
        # Cells: 512 rows x 1024 columns, all rows at once; settling 4 cycles
        # x 2 ns = 8 ns. No resistances in the paper: r_on / r_off only
        # satisfy the cell model (cell_supply_v=0: no cell-current charge).
        crossbars.conv_xbar(cell_bits=1, r_on=10e3, r_off=100e3, rows=512, cols=1024,
                            v_read=0.9, cell_supply_v=0.0, time_ns=8.0),
        # Array, per driven row: 1.942 pJ per spike (its whole word line).
        Component("array_row", count="physical_rows", powered="spiking_rows",
                  event_pj=1.942),
        # Array, fixed part: 0.1298 pJ per used column per MVM.
        Component("array_column", count="physical_columns",
                  powered={"rule": "used_columns", "gated": True}, event_pj=0.1298),
        # Input DAC + MUX + ring-amplifier buffer: 0.191 pJ per driven row.
        Component("input_buffer", count="physical_rows", powered="spiking_rows",
                  event_pj=0.191),
        # WC-ADC, one per SL+/SL- pair (2 columns): 0.483 pJ per conversion =
        # 0.2415 pJ per used column; 11 cycles x 2 ns = 22 ns after settling.
        Component("adc", count="physical_columns",
                  powered={"rule": "used_columns", "gated": True}, time_ns=22.0,
                  event_pj=0.2415),
    ],
    specs={"label": r"A-SSCC'25 SF-RRAM", "tech": 28, "supply": "0.9/1.2",
           "device": "RRAM", "cell": "ternary", "bitcell": "2T2R (SF)", "r_ratio": "--",
           "sensing": "Current (SAR)"},
)

# ============================================================================
# 5. RRAM: Yao et al., "A 28nm RRAM-Based 81.1 TOPS/mm2/bit Compute-In-Memory
#    Macro with Uniform and Linear 64 Read Channels under 512 4-bit Inputs",
#    ESSERC 2024, pp. 577-580. Measured. Zotero PMB6SNCI.
# ----------------------------------------------------------------------------
# Macro (Figs. 1, 2, 9): 512 x 512 2T2R RRAM array, 28 nm HPC+, 512 Kb. A
# signed 4-bit weight is one 2T2R cell (WP / WN), each device at one of 8
# conductance levels (3 bits). Inputs are differential voltages on the bit
# lines (BLP / BLN = VCM +/- 0..150 mV) from R2R DACs through BL buffers; the
# 2T2R's SL current is read by one of 64 dual-loop-clamping SAR ADCs (8
# columns share an ADC): 512 inputs x 64 outputs per calculation.
# Operating point (Figs. 7, 9, 10): 650 MHz, 4bIN/4bW/8bO, VDDC 0.9 V, VDDP
# 1.2 V, 50% weight sparsity: one calculation 23.08 ns, 2.84 TOPS.
#   check: ops per calculation = 2 x 512 x 64 = 65536; / 23.08 ns = 2.84 TOPS.
# Power vs driven rows (Fig. 7): 128: 77.05, 256: 88.93, 384: 100.23,
# 512: 111.43 mW. A straight-line fit (within 0.3%): 0.08941 mW per driven
# row + 65.80 mW fixed; x 23.08 ns per calculation:
#   per driven row  2.064 pJ   = the BL buffers, which also carry the cell
#                   current (per-row part at 512 rows, 45.8 mW, matches
#                   Fig. 9's "Buffer" 42.8% x 115.7 mW = 49.5 mW)
#   fixed           1518.7 pJ per calculation = 23.73 pJ per ADC channel,
#                   split as Fig. 9's other shares: ADC 31.8%, DAC 16.8%,
#                   digital & driver 8.6% (65.8 mW vs Fig. 9's 66.2 mW)
#   (111.43 mW x 23.08 ns = 2.57 nJ -> 25.5 TOPS/W, in the paper's
#   19.3-39.3 TOPS/W range, Fig. 10)
# 1-bit input normalisation: our inputs are spikes, a BL either driven or not
# (the cells on its row on or off). The paper's energy is for 4-bit inputs;
# as in its own 1-bit-normalised efficiency (x input bits), each energy is
# divided by 4: 0.5159 pJ per driven row; per ADC channel 5.932 pJ = per
# device column (WP or WN) 2.966 pJ: ADC 1.649, DAC 0.871, digital 0.446.
# Timing kept at the paper's 23.08 ns per conversion (1-bit inputs do not
# shorten the 8-bit SAR conversion).
# Weights: differential (WP / WN) in 3-bit cells (8 levels, within the
# 3-bit limit): 6-bit weights -> 5 magnitude bits = 2 columns per sign, 4
# device columns per weight. Device columns: 512 2T2R x 2 = 1024 per tile.
ESSERC24_RRAM = compose(
    "rram_esserc24",
    mapping("differential"),
    [
        # Cells: 512 rows x 1024 device columns, all rows at once. The cell
        # current is in the BL buffers' measured power, so it is not charged
        # again (cell_supply_v=0); the paper gives only the array's load
        # range (1.1-10 mS), not the device levels: r_on / r_off only
        # satisfy the cell model. v_read: the 150 mV input swing.
        crossbars.conv_xbar(cell_bits=3, r_on=10e3, r_off=100e3, rows=512, cols=1024,
                            v_read=0.15, cell_supply_v=0.0, time_ns=0.0),
        # BL buffer (+ cell current), per driven row: 0.5159 pJ per spike.
        Component("bl_buffer", count="physical_rows", powered="spiking_rows",
                  event_pj=0.5159),
        # ADC, DAC and digital, per used device column (half an ADC channel).
        Component("adc", count="physical_columns",
                  powered={"rule": "used_columns", "gated": True}, event_pj=1.649),
        Component("dac", count="physical_columns",
                  powered={"rule": "used_columns", "gated": True}, event_pj=0.871),
        Component("digital", count="physical_columns",
                  powered={"rule": "used_columns", "gated": True}, event_pj=0.446),
        # ADC timing: 64 ADCs, 16 device columns (8 2T2R columns) each, one
        # 2T2R after another: 23.08 ns per 2T2R = 11.54 ns per device column.
        Component("adc_conversion", count={"rule": "column_groups", "size": 16},
                  powered="used_column_groups", time_ns=11.54, serial=True),
    ],
    specs={"label": r"ESSERC'24 RRAM", "tech": 28, "supply": "0.9/1.2", "device": "RRAM",
           "bitcell": "2T2R", "r_ratio": "--", "sensing": "Current (SAR)"},
)

LITERATURE_MACROS = [DS_CIM, TD_CIM, MEMRISTIVE_SNN, ASSCC25_SF, ESSERC24_RRAM]

# The macros `python run.py --literature` evaluates next to our work (edit to
# choose; every macro above stays defined and checked against its paper).
# The RRAM macros of the neuromorphic comparison table (DS-CIM is there with
# its own reported numbers; the memristive SNN's 180 nm is left out).
COMPARED = [ESSERC24_RRAM, TD_CIM, ASSCC25_SF]


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
    """[(macro, what, engine, paper)] at each paper's own condition (energies
    x the input bits, to undo the 1-bit normalisation)."""
    import torch
    from hardware import evaluate_layer

    def dense(arch, weights, spikes):
        return evaluate_layer(arch, spikes[None, :, None, None, :], weights[:, :, None, None])

    r = []
    w = torch.tensor([1, -1] * 64).repeat(320, 1).T
    c = dense(native(TD_CIM, 2), w, torch.ones(320, 1))
    r += [(TD_CIM, "pJ per MVM", 1e3 * c.energy_nj, 81920 / 1251),
          (TD_CIM, "ns per MVM", c.latency_ns, 81920 / 0.341e3)]
    p = native(ESSERC24_RRAM, 4)
    p = dataclasses.replace(p, mapping=dataclasses.replace(p.mapping, columns="interleaved"))
    w = torch.tensor([7, -7] * 32).repeat(512, 1).T
    for rows, mw in ((128, 77.05), (256, 88.93), (384, 100.23), (512, 111.43)):
        c = dense(p, w, (torch.arange(512) < rows).float()[:, None])
        r.append((ESSERC24_RRAM, f"mW at {rows} rows (x4)", 4e3 * c.energy_nj / c.latency_ns, mw))
    r.append((ESSERC24_RRAM, "ns per calculation", c.latency_ns, 23.08))
    p, w = native(ASSCC25_SF, 2), torch.tensor([1, -1] * 256).repeat(512, 1).T
    for rows, mw in ((64, 67.2), (128, 87.0), (256, 126.6), (512, 195.0)):
        c = dense(p, w, (torch.arange(512) < rows).float()[:, None])
        r.append((ASSCC25_SF, f"mW at {rows} rows (x4)", 4e3 * c.energy_nj / c.latency_ns, mw))
    r.append((ASSCC25_SF, "ns per MVM", c.latency_ns, 30.0))
    w = torch.tensor([15, -15] * 32).repeat(512, 1).T
    c = dense(native(DS_CIM, 5), w, (torch.arange(512) % 2 == 0).float()[:, None])
    r.append((DS_CIM, "nJ, Table I vector", c.energy_nj, 1.4722))
    paper = native(MEMRISTIVE_SNN, 5, Memory(4, levels_s=tuple(i * 10e-6 for i in range(16))))
    energy = 0.0
    for n_in, n_out in ((800, 480), (480, 120), (120, 11)):
        c = dense(paper, torch.ones(n_out, n_in, dtype=torch.long), torch.ones(n_in, 300))
        energy += c.energy_nj
    r.append((MEMRISTIVE_SNN, "mW, DVS Gesture network", energy / c.latency_ns * 1e3, 43.83))
    return r


if __name__ == "__main__":
    for arch, what, engine, paper in check():
        print(f"{arch.name:<22} {what:<26} engine {engine:9.4g}  paper {paper:9.4g}  "
              f"({100 * (engine / paper - 1):+.1f}%)")
