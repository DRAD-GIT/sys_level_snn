"""Comparison with other neuromorphic designs, in the style of DS-CIM's Table III
(Fu et al., TCAS-I 2024): one column per work, rows for its publication,
technology and implementation, the datasets it evaluated, its input and each
metric. Every dataset cell has a line per dataset in a fixed order (N-MNIST,
IBM DVS Gesture, CIFAR-10), so a dataset's numbers sit at the same height in
every column (blank line: not evaluated; "--": not reported).

    python tools/neuromorphic_table.py               # -> logs/neuromorphic_table.tex
    python tools/neuromorphic_table.py --out paper/Chapters/neuromorphic_table.tex

The literature columns hold each paper's reported numbers (WORKS below; edit
them there), ordered by publication date, with energy, latency, power and
TOPS/W scaled to our 40 nm node with DeepScaleTool (see SCALE_TO_NM). Values a
paper does not report but its other numbers give are computed and marked:
power or latency from energy and the other (a), TOPS/W from our network's
operations on that dataset and the paper's energy (b, done in make_table). Our
columns are filled from run.py's results (logs/comparison_summary.csv): for
every architecture of run.py's OURWORK, its accuracy, energy, latency, power
and TOPS/W per inference on each dataset (the latest configuration, its run
with the most samples). Our values that beat every other work in their row
are bold. Sources and caveats are in docs/literature_shortlist.md.

Include the output with \\input{neuromorphic_table}; the preamble needs
booktabs, makecell and adjustbox.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import latex_table  # noqa: E402

# The datasets: (key, run.py models of ours, first with results used; row label).
DATASETS = (("nmnist", ("nmnist17", "nmnist"), "N-MNIST"),
            ("gesture", ("gesture16", "gesture"), "IBM DVS Gesture"),
            ("cifar10", ("cifar10_thermo",), "CIFAR-10"))
# Our input per model (the "Input" rows).
OUR_INPUTS = {"nmnist17": r"$2\times17\times17$", "nmnist": r"$2\times34\times34$",
              "gesture16": r"$16\times16$", "gesture": r"$128\times128$",
              "cifar10_thermo": r"$32\times32$"}
# The metric rows, in order: (key, row label, better: +1 higher, -1 lower).
METRICS = (("accuracy", "Accuracy", +1), ("energy", "Energy / sample", -1),
           ("energy_per_step", "Energy / step", -1),
           ("latency", "Latency / sample", -1),
           ("tops_per_w", "TOPS/W", +1))
# Our time steps per model (the "Energy / step" row: energy per sample / steps).
OUR_STEPS = {"nmnist17": 10, "nmnist": 300, "gesture16": 80, "cifar10_thermo": 8}
# Bits per operation (input, weight) for the normalised TOPS/W (TOPS/W x input
# bits x weight bits), only where a work reports its own TOPS/W.
OUR_BITS = (1, 6)
# Rows where our columns share one cell when their values are the same (one
# recorded network, same inputs).
MERGED_ROWS = ("Publication", "Technology", "Synaptic memory", "Bit-cell", "Input bits", "Input encoding",
               "Cell precision", "Weight precision", "Datasets", "Input", "Accuracy")

# Technology scaling (DeepScaleTool: Sarangi and Baas, ISCAS 2021,
# https://sourceforge.net/projects/deepscaletool/, DeepScaleTool.xlsm of
# 03/02/2021). Its macros give, per node, energy, delay and power relative to
# 65 nm; the scaling factor from a node to a target node is value(node) /
# value(target), and a metric at the target node is the reported value
# divided by that factor (energy and power), the delay likewise, and TOPS/W
# multiplied by the energy factor. The tool supports 130 to 7 nm only (its
# macros map any other node to 7 nm's values), so works at other nodes are
# left as reported and marked. Accuracy is not scaled.
SCALE_TO_NM = 40          # every work's energy, latency, power and TOPS/W at this node
DEEPSCALE = {             # node: (energy, delay, power), relative to 65 nm
    130: (2.52, 1.96, 1.28), 90: (1.51, 1.31, 1.15), 65: (1.0, 1.0, 1.0), 45: (0.63, 0.81, 0.78),
    40: (0.55, 0.76, 0.73), 32: (0.44, 0.70, 0.63), 28: (0.37, 0.67, 0.56), 22: (0.30, 0.62, 0.48),
    14: (0.19, 0.60, 0.32), 10: (0.15, 0.57, 0.26), 7: (0.11, 0.53, 0.21)}

# Each work's numbers, per dataset it evaluated: input (LaTeX), accuracy as
# a LaTeX string, energy per sample in nJ, latency per sample in us, power in
# mW, TOPS/W; a metric not reported is left out and shows as --, and TOPS/W
# left out is computed in make_table (mark b). "marks" puts a footnote mark
# on a value (a: computed from the paper's other numbers, derivation next to
# it). "node" (nm) is used for scaling; "date" (year, month) orders the
# columns. Footnote marks refer to FOOTNOTES.
WORKS = [
    {"name": "ANP-I", "publication": "JSSC'24", "date": (2024, 8), "node": 28,
     "technology": "28nm", "memory": "SRAM", "bitcell": "--", "encoding": "Events", "in_bits": "1-bit",
     "precision": "8/10-bit", "cell": "1-bit",
     "impl": r"Digital",
     "results": {"nmnist": {"input": r"$2\times17\times17$", "accuracy": r"96.0\%",
                            "energy": 343},
                 "gesture": {"input": r"$14\times14$", "accuracy": r"92.0\%", "energy": 3900}}},
    # DS-CIM Table II: 40.46 us per Gesture sample -> power 735.35 nJ / 40.46 us.
    {"name": "DS-CIM", "publication": "TCAS-I'24", "date": (2024, 4), "node": 40,
     "technology": "40nm", "memory": "SOT-MRAM", "bitcell": "1T1MTJ", "encoding": "Dual-spike", "in_bits": "--",
     "precision": "Signed 4-bit", "cell": "1-bit",
     "impl": "Mixed signal",
     "results": {"gesture": {"input": r"$16\times16$", "accuracy": r"90.00\%", "energy": 735.35,
                             "latency": 40.46, "power": 735.35 / 40.46,
                             "marks": {"power": "a"}}}},
    # 181 uJ per inference; 100 time steps of 50 us, pipelined -> 5 ms per
    # inference, power 181 uJ / 5 ms.
    {"name": "Dorzhigulov et al.", "publication": "Front. Neurosci.'23", "date": (2023, 7), "node": 130,
     "technology": "130nm", "memory": "RRAM", "bitcell": "1T1R", "encoding": "Rate", "in_bits": "--",
     "precision": "2-bit", "cell": "--",
     "impl": "Mixed signal",
     "results": {"cifar10": {"input": r"$32\times32$", "accuracy": r"61.74\%", "energy": 181000,
                             "latency": 100 * 50.0, "power": 181000 / (100 * 50.0),
                             "marks": {"power": "a"}}}},
    # ISSCC'24 30.2, Fig. 30.2.7 summary: inference energy 25.9 uJ (Gesture), 3.8 uJ
    # (N-MNIST); running power 524 uW (N-MNIST) and 834 uW (Gesture, JSSC'26 Table I)
    # -> latency = energy / power. 4 time steps (N-MNIST), 16 (Gesture, Fig. 30.2.5
    # and the JSSC version) -> energy per step. Accuracy evaluated in software with
    # the measured IMC linearity.
    {"name": "Liu et al.", "publication": "ISSCC'24", "date": (2024, 2), "node": 22,
     "technology": "22nm", "memory": "SRAM", "bitcell": "6T + IMC cell", "encoding": "Events", "in_bits": "1-bit",
     "precision": "4/8-bit", "cell": "1-bit",
     "impl": "Digital IMC",
     "results": {"nmnist": {"input": "--", "accuracy": r"97\%", "energy": 3800,
                            "power": 0.524, "latency": 3800 / 0.524,
                            "energy_per_step": 3800 / 4,
                            "marks": {"latency": "a", "energy_per_step": "e"}},
                 "gesture": {"input": "--", "accuracy": r"94\%", "energy": 25900,
                             "power": 0.834, "latency": 25900 / 0.834,
                             "energy_per_step": 25900 / 16,
                             "marks": {"latency": "a", "energy_per_step": "e"}}}},
    # TCAS-I'25 (Sun et al., Nanjing Univ.), TSMC 28nm post-layout (Design Compiler, IC
    # Compiler), not fabricated: 1.1 uJ per inference = 41.5 mW x 28.2 us, 216.9 TOPS/W
    # (Table X). Binary weights (8-bit input layer); 16 input time steps, 4 after its
    # temporal-pooling layer, so no single energy per step.
    {"name": "Sun et al.", "publication": "TCAS-I'25", "date": (2025, 8), "node": 28,
     "technology": "28nm (post-layout)", "memory": "SRAM", "bitcell": "--", "encoding": "Events", "in_bits": "1-bit",
     "precision": "Binary", "cell": "1-bit",
     "impl": "Digital",
     "results": {"gesture": {"input": r"$2\times32\times32$", "accuracy": r"95.49\%", "energy": 1100,
                             "latency": 28.2, "power": 41.5, "tops_per_w": 216.9}}},
    # VGG-11 (Table III): 21.74 uJ, 73 ns per layer -> 11 layers x 73 ns per
    # image, power 21.74 uJ over it. One pass (single-spike temporal coding):
    # energy per step = energy per sample (e).
    {"name": "Han et al.", "publication": "TCAS-I'22", "date": (2022, 11), "node": 65,
     "technology": "65nm", "memory": "RRAM", "bitcell": "1T1R", "encoding": "Temporal (delay)", "in_bits": "6-bit",
     "precision": "--", "cell": "5-bit (32 levels)",
     "impl": "Mixed signal",
     "results": {"cifar10": {"input": r"$32\times32$", "accuracy": r"88\%", "energy": 21740,
                             "energy_per_step": 21740,
                             "latency": 11 * 0.073, "power": 21740 / (11 * 0.073),
                             "tops_per_w": 14.12, "bits": (6, 5),
                             "marks": {"power": "a", "energy_per_step": "e"}}}},
]

# Works kept out of the table for now (same format as WORKS): move one into
# WORKS to show it.
SET_ASIDE = [
    {"name": "SPOON", "publication": "ISCAS'20", "date": (2020, 10), "node": 28,
     "technology": "28nm FDSOI", "memory": "SRAM", "bitcell": "--", "encoding": "TTFS", "in_bits": "1-bit",
     "precision": "8-bit", "cell": "1-bit",
     "impl": r"Digital",
     "results": {"nmnist": {"input": r"$2\times34\times34$", "accuracy": r"93.8\%",
                            "energy": 665}}},
    # Fig. 6: 35 nJ per step (inference, 0.5 V, 13 MHz), 1318 steps per Gesture
    # sample on average -> 46.1 uJ per sample (d); at 77 uW -> latency (a).
    {"name": "ReckOn", "publication": "ISSCC'22", "date": (2022, 2), "node": 28,
     "technology": "28nm FDSOI", "memory": "SRAM", "bitcell": "--", "encoding": "Events", "in_bits": "1-bit",
     "precision": "8-bit", "cell": "1-bit",
     "impl": r"Digital",
     "results": {"gesture": {"input": r"$16\times16$", "accuracy": r"87.3\%", "energy": 35 * 1318,
                             "energy_per_step": 35, "power": 0.077, "latency": 35 * 1318 / 0.077,
                             "marks": {"energy": "d", "latency": "a"}}}},
]

# Our columns: run.py architecture name -> column heading.
OUR_NAMES = {"c3cim_xbar": "C3CIM", "c3cim_op_xbar": "C3CIM-OP"}
OUR_SPECS = {"publication": r"\textbf{This work}", "node": 40, "technology": "40nm", "memory": "RRAM",
             "bitcell": "2T1R",
             "encoding": "Rate", "in_bits": "1-bit", "precision": "6-bit", "cell": "1-bit", "impl": "Mixed signal"}

FOOTNOTES = [
    f"Energy, latency and TOPS/W scaled to {SCALE_TO_NM}nm with DeepScaleTool; "
    "accuracy as reported.",
    r"$^{a}$ Computed from the reported energy and power: latency = energy / power.",
    r"$^{b}$ Computed as this work's operations on that dataset (its network, all synaptic "
    r"operations) divided by the reported energy per sample.",
    r"$^{c}$ TOPS/W $\times$ input bits $\times$ weight bits, for reported TOPS/W "
    r"(Han et al.: 6-bit inputs, its 5-bit cell as the weight; this work: 1-bit spikes, 6-bit weights).",
    r"$^{d}$ Computed from the reported energy per step $\times$ average steps per sample "
    r"(ReckOn: 35\,nJ $\times$ 1318 steps).",
    r"$^{e}$ Computed as energy per sample / time steps (Liu et al.: 4 for N-MNIST, 16 for "
    r"Gesture; Han et al.: 1, one single-spike pass; this work: 10, 80 and 8).",
    r"$^{\P}$ Outside DeepScaleTool's range (130--7nm): as reported, not scaled.",
]
CAPTION = "Comparison with other neuromorphic designs"
LABEL = "table:neuromorphic"


def scale(metric, value, node, target=SCALE_TO_NM):
    """A metric reported at `node` (nm) at the target node with DeepScaleTool's
    factors; None if a node is outside its range."""
    if node not in DEEPSCALE or target not in DEEPSCALE:
        return None
    energy, delay, power = (DEEPSCALE[node][i] / DEEPSCALE[target][i] for i in range(3))
    factor = {"energy": energy, "latency": delay, "power": power, "energy_per_step": energy, "tops_per_w": 1 / energy,
              "norm_tops_per_w": 1 / energy}[metric]
    return value / factor


def _unit(value, units):
    """value in the first unit of `units` [(name, factor to the next)] -> LaTeX, 3 digits."""
    for name, step in units:
        if abs(value) < step or name == units[-1][0]:
            return f"{value:#.3g}".rstrip(".") + f"\\,{name}"
        value /= step


FORMATS = {"energy": lambda v: _unit(v, [("nJ", 1e3), (r"\textmu J", 1e3), ("mJ", 1e3)]),
           "energy_per_step": lambda v: _unit(v, [("nJ", 1e3), (r"\textmu J", 1e3), ("mJ", 1e3)]),
           "latency": lambda v: _unit(v, [(r"\textmu s", 1e3), ("ms", 1e3), ("s", 1)]),
           "power": lambda v: _unit(v, [(r"\textmu W", 1e3), ("mW", 1e3), ("W", 1)]),
           "tops_per_w": lambda v: f"{v:#.3g}".rstrip("."),
           "norm_tops_per_w": lambda v: f"{v:#.3g}".rstrip(".")}


def scaled(metric, value, node):
    """A metric value at SCALE_TO_NM (as reported where the node is out of
    DeepScaleTool's range), for comparing works."""
    if metric == "accuracy":
        return float(str(value).replace("\\%", "").replace("%", ""))
    result = scale(metric, value, node)
    return value if result is None else result


def cell(metric, value, node, mark=""):
    """One metric value as LaTeX: scaled to SCALE_TO_NM where it is a number,
    with its footnote mark."""
    sup = f"$^{{{mark}}}$" if mark else ""
    if metric not in FORMATS or not isinstance(value, (int, float)):
        return f"{value}{sup}"
    result = scale(metric, value, node)
    if result is None:
        return FORMATS[metric](value * 1e3 if metric == "power" else value) + r"$^{\P}$" + sup
    return FORMATS[metric](result * 1e3 if metric == "power" else result) + sup


def our_values(row, model):
    """A comparison_summary.csv row -> {metric: value} (energy nJ, latency us, power mW)."""
    return {"input": OUR_INPUTS.get(model, model), "accuracy": f"{float(row['accuracy']):.2f}\\%",
            **{m: float(row[m]) for m in ("energy", "latency", "power", "tops_per_w")},
            "norm_tops_per_w": float(row["tops_per_w"]) * OUR_BITS[0] * OUR_BITS[1],
            **({"energy_per_step": float(row["energy"]) / OUR_STEPS[model]} if model in OUR_STEPS else {})}


def our_columns(runs, architectures, warn=print):
    """Our columns from run.py's results (the latest configuration of each
    architecture, its run with the most samples), per dataset with results."""
    columns = []
    for name in architectures:
        results = {}
        for key, choices, dataset in DATASETS:
            model = next((m for m in choices if runs.get((m, name))), None)
            if model is None:
                warn(f"warning: no results for {name} on {' or '.join(choices)}: run python run.py "
                     f"--model {choices[0]}")
                continue
            rows = runs[(model, name)]
            config = rows[-1]["configuration"]
            best = max((r for r in rows if r["configuration"] == config), key=lambda r: int(r["samples"]))
            if int(best["samples"]) < latex_table.FULL_TEST_SET.get(model, 0):
                warn(f"warning: {name} on {model}: {best['samples']} of "
                     f"{latex_table.FULL_TEST_SET[model]} test samples")
            results[key] = our_values(best, model)
        columns.append({"name": OUR_NAMES.get(name, name.replace("_", r"\_")), **OUR_SPECS,
                        "date": (9999, 0), "results": results, "ours": True})
    return columns


def add_computed_tops_per_w(works, ours):
    """TOPS/W of works that do not report it: our network's operations on the
    dataset (our TOPS/W x energy) over the work's energy per sample (mark b)."""
    ops = {}                                 # operations per sample, per dataset
    for column in ours:
        for key, r in column["results"].items():
            ops.setdefault(key, r["tops_per_w"] * r["energy"] * 1e3)    # ops/pJ x pJ
    out = []
    for work in works:
        results = {}
        for key, r in work["results"].items():
            r = dict(r, marks=dict(r.get("marks", {})))
            if "tops_per_w" in r and "bits" in r:          # reported TOPS/W only
                r["norm_tops_per_w"] = r["tops_per_w"] * r["bits"][0] * r["bits"][1]
            if "tops_per_w" not in r and "energy" in r and key in ops:
                r["tops_per_w"] = ops[key] / (r["energy"] * 1e3)
                r["marks"]["tops_per_w"] = "b"
            results[key] = r
        out.append(dict(work, results=results))
    return out


def _lines(values):
    """One cell with a line per dataset, in DATASETS order: a makecell whose
    blank lines keep every dataset at the same height in every column."""
    return r"\makecell{" + r" \\ ".join(v if v else r"\phantom{0}" for v in values) + "}"


def build_table(columns):
    n = len(columns)
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
             f"\\caption{{{CAPTION}}}", f"\\label{{{LABEL}}}",
             r"\renewcommand{\arraystretch}{1.15}",
             r"\begin{adjustbox}{max width=\textwidth}",
             f"\\begin{{tabular}}{{@{{}}l*{{{n}}}{{c}}@{{}}}}", r"\toprule"]

    ours = [i for i, c in enumerate(columns) if c.get("ours")]

    def merge_ours(cells):
        """Our columns' cells as one cell across them where they are the same."""
        if len(ours) > 1 and ours == list(range(ours[0], ours[-1] + 1)) and \
                len({cells[i] for i in ours}) == 1:
            cells = cells[:ours[0]] + [f"\\multicolumn{{{len(ours)}}}{{c}}{{{cells[ours[0]]}}}"] + \
                cells[ours[-1] + 1:]
        return cells

    def row(label, cells):
        if label in MERGED_ROWS:
            cells = merge_ours(cells)
        lines.append(f"\\textbf{{{label}}} & " + " & ".join(cells) + r" \\")

    row("Name", [f"\\textbf{{{c['name']}}}" for c in columns])
    row("Publication", [c["publication"] for c in columns])
    lines.append(r"\midrule")
    for key, label in (("technology", "Technology"), ("memory", "Synaptic memory"),
                       ("bitcell", "Bit-cell"), ("cell", "Cell precision")):
        row(label, [c.get(key, "--") for c in columns])
    lines.append(r"\midrule")
    for key, label in (("in_bits", "Input bits"), ("encoding", "Input encoding"),
                       ("precision", "Weight precision")):
        row(label, [c.get(key, "--") for c in columns])
    lines.append(r"\midrule")
    row("Datasets", [_lines([d if key in c["results"] else "" for key, _, d in DATASETS])
                     for c in columns])
    lines.append(r"\midrule")
    row("Input", [_lines([c["results"][key]["input"] if key in c["results"] else ""
                          for key, _, _ in DATASETS]) for c in columns])
    for metric, label, better in METRICS:
        if not any(metric in r for c in columns if not c.get("ours") for r in c["results"].values()):
            continue                              # only ours: no comparison, no row
        lines.append(r"\midrule")
        cells = []
        for c in columns:
            values = []
            for key, _, _ in DATASETS:
                r = c["results"].get(key)
                if r is None:                     # dataset not evaluated
                    values.append("")
                    continue
                if metric not in r:               # not reported
                    values.append("--")
                    continue
                text = cell(metric, r[metric], c["node"], r.get("marks", {}).get(metric, ""))
                others = [scaled(metric, o["results"][key][metric], o["node"]) for o in columns
                          if not o.get("ours") and key in o["results"] and metric in o["results"][key]]
                value = scaled(metric, r[metric], c["node"])
                if c.get("ours") and metric != "accuracy" and others and \
                        all(better * (value - o) > 0 for o in others):
                    text = f"\\textbf{{{text}}}"
                values.append(text)
            cells.append(_lines(values))
        row(label, cells)
    body = "\n".join(lines)
    notes = [n for n in FOOTNOTES if not n.startswith("$^{") or n.split("$", 2)[1] in body]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
              r"\par\smallskip\raggedright\scriptsize",
              " \\\\\n".join(notes), r"\end{table*}", ""]
    return "\n".join(lines)


def make_table(our_architectures, logs_dir, warn=print):
    """The table text: WORKS by date (TOPS/W computed where not reported),
    then a column per name in our_architectures."""
    summary = os.path.join(logs_dir, "comparison_summary.csv")
    if not os.path.exists(summary):
        raise FileNotFoundError(f"no results in {summary}: run python run.py first")
    ours = our_columns(latex_table.load_results(summary), our_architectures, warn)
    works = sorted(add_computed_tops_per_w(WORKS, ours), key=lambda w: w["date"])
    return build_table(works + ours)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logs", default=os.path.join(ROOT, "logs"),
                        help="folder of run.py's results (comparison_summary.csv)")
    parser.add_argument("--out", help="output .tex (default: <logs>/neuromorphic_table.tex)")
    args = parser.parse_args()
    import run
    table = make_table([a.name for a in run.OURWORK], args.logs,
                       warn=lambda text: print(text, file=sys.stderr))
    out = args.out or os.path.join(args.logs, "neuromorphic_table.tex")
    with open(out, "w", encoding="utf-8", newline="\n") as file:
        file.write(table)
    print(table, end="")
    print(f"saved {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
