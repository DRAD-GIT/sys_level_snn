# Literature shortlist

Selected from the authors' Zotero library (399 items, all screened; the
resistive-memory and SNN-hardware candidates read in full). Numbers are as
reported by each paper.

## A. Comparison by reported dataset results (current approach)

Each work's own results on N-MNIST, IBM DVS Gesture and CIFAR-10, compared
with our designs evaluated on the same datasets (as in DS-CIM's Table III).
Only rows on the same dataset *and* input version are directly comparable;
TOPS/W and energy per synaptic operation carry across datasets better than
energy per inference.

| Work | Technology, silicon | N-MNIST | DVS Gesture | CIFAR-10 |
|---|---|---|---|---|
| Wang et al., "Fully Integrated Memristive Spiking Neural Network with Analog Neurons for High-Speed Event-Based Data Processing", arXiv 2025 (preprint) | RRAM, 180 nm, measured (part simulated) | 94.73%, 1.78 uJ | 93.06%, 1.94 uJ, 44.31 us, 43.83 mW (incl. I/O) | - |
| Zhang et al., "ANP-I: A 28-nm 1.5-pJ/SOP Asynchronous Spiking Neural Network Processor Enabling Sub-0.1-uJ/Sample On-Chip Learning for Edge-AI Applications", IEEE JSSC 2024 | digital, 28 nm, measured (0.56 V, 40 MHz) | 96.0%, 343 nJ (input downscaled to 2x17x17) | 92.0%, 3.9 uJ (16x16, 10 classes) | - |
| Frenkel & Indiveri, "ReckOn: A 28nm Sub-mm2 Task-Agnostic Spiking Recurrent Neural Network Processor Enabling On-Chip Learning over Second-Long Timescales", ISSCC 2022 | digital, 28 nm, measured (0.5 V, 13 MHz) | - | 87.3%, 46.1 uJ, 77 uW (16x16, 10 classes) | - |
| Fu et al., "DS-CIM: A 40nm Asynchronous Dual-Spike Driven, MRAM Compute-In-Memory Macro for Spiking Neural Network", IEEE TCAS-I 2024 | SOT-MRAM, 40 nm, simulated | - | 90.00%, 729 nJ, 40.46 us (16x16, 10 classes) | - |
| Amir et al., "A Low Power, Fully Event-Based Gesture Recognition System", CVPR 2017 (TrueNorth) | digital, 28 nm, measured | - | 96.5%, < 200 mW, 105 ms latency (full input, 11 classes) | - |
| Han et al., "Efficient Discrete Temporal Coding Spike-Driven In-Memory Computing Macro for Deep Neural Network Based on Nonvolatile Memory", IEEE TCAS-I 2022 | ReRAM, 65 nm, simulated | - | - | ~88% (VGG-11, 6-bit temporal coding), 21.74 uJ, 14.12 TOPS/W |
| Kim et al., "Neuro-CIM: ADC-Less Neuromorphic Computing-in-Memory Processor With Operation Gating/Stopping and Digital-Analog Networks", IEEE JSSC 2023 | SRAM, measured | - | - | 92.1%, 0.72 uJ (4b in / 4b w, converted SNN) |
| Yan et al., "RRAM-based Spiking Nonvolatile Computing-In-Memory Processing Engine with Precision-Configurable In Situ Nonlinear Activation", VLSI 2019 | RRAM, 150 nm, measured | - | - | 95.9% *relative* to its binarized software net; 0.257 pJ/MAC |

Caveats for the table: DS-CIM, ANP-I and ReckOn use DVS Gesture downscaled to
16x16 with 10 classes (ours: 128x128, 11 classes); ANP-I's N-MNIST is
downscaled to 2x17x17 (ours: 2x34x34). The memristive SNN chip is an arXiv
preprint (check for a published version before citing).

Also in the library, MNIST only (not used): SPOON (ISCAS 2020), ODIN
(TBioCAS 2018), MorphIC (TBioCAS 2019), Tempo-CIM (JETCAS 2023), SPIRIT (IEDM
2019), SiOx-RRAM SNN (AICAS 2020), 3 nm SRAM SNN (DAC 2024), SNN chip
(Sensors 2021). Not SNN: Liu et al., analog ReRAM 78.4 TOPS/W (ISSCC 2020,
MLP on MNIST).

## B. Macro modelling in the engine (literature_macros.py)

Ranked by array-level numbers given (cell R/G, read voltage, read time, array
energy separate from the periphery). Definitions, derivations and checks
against each paper are in literature_macros.py.

| # | Work | Array numbers | Silicon | In run.py |
|---|---|---|---|---|
| 1 | DS-CIM (TCAS-I 2024), SOT-MRAM 40 nm | 4/4 | simulated | yes |
| 2 | Wei et al., "A 28-nm Static-Power-Free Fully Parallel RRAM-Based TD CIM Macro With 1982 TOPS/W/Bit for Edge Applications", SSC-L 2025 | 3.5/4 | measured | defined, not evaluated |
| 3 | Memristive SNN (arXiv 2025), RRAM 180 nm | 3/4 | measured | yes |
| 4 | Li et al., "A 28nm RRAM-Based 17.0 TOPS/mm2 and 89.7 TOPS/W Compute-In-Memory Macro Enabled by Source-Follower Cell and IR-Drop-Freed Array", A-SSCC 2025 | 2.5/4 | measured | yes |
| 5 | Yao et al., "A 28 nm RRAM-Based 81.1 TOPS/mm2/bit Compute-In-Memory Macro with Uniform and Linear 64 Read Channels under 512 4-bit Inputs", ESSERC 2024 | 2/4 | measured | yes |

Further candidates read (not modelled): Li et al., 40-nm MLC-RRAM CIM macro
(JSSC 2022); Liu et al., 28 nm 576K RRAM CIM macro (J. Semicond. 2025); Wang
et al., near-threshold memristive CIM engine (Nat. Commun. 2025); Khwa et al.,
40-nm hybrid SLC-MLC PCM CIM macro (ISSCC 2022, compared in our DATE'25
paper); Singh et al., 115.1 TOPS/W CIM with ring-oscillator ADC (AICAS 2023,
compared in our DATE'25 paper). Excluded at the authors' request: Jung et al.,
MRAM crossbar (Nature 2022).

## Selection factors

1. Scope: hardware that computes (macro, chip, processor), not devices,
   simulators or reviews; SNN for the dataset comparison, analog CIM for the
   macro modelling; RRAM, MRAM or PCM for the resistive-memory set (FeFET and
   flash excluded); not our own work.
2. Evidence: measured over simulated; peer-reviewed over preprints; internally
   consistent numbers; recent and small nodes preferred.
3. Dataset comparison: results on N-MNIST, DVS Gesture or CIFAR-10 with
   accuracy and energy or power per inference; the same dataset version as
   ours (or footnoted); comparable metrics.
4. Macro modelling: array-level numbers; calibration data (per-block table,
   several measured operating points); native 1-bit inputs; the engine
   reproduces the paper's numbers.
5. Practical: continuity with our DATE'25 comparisons; a mix of devices; the
   PDF available for checking.
