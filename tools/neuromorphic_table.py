"""Comparison with other neuromorphic designs, in the style of DS-CIM's Table III
(Fu et al., TCAS-I 2024): one column per work, rows for its publication,
technology and implementation, the datasets it evaluated (of N-MNIST, IBM DVS
Gesture and CIFAR-10), and for each metric its values on those datasets in
the same order ("a / b / c"; NA where not reported).

    python tools/neuromorphic_table.py               # -> logs/neuromorphic_table.tex
    python tools/neuromorphic_table.py --out paper/Chapters/neuromorphic_table.tex

The literature columns hold each paper's reported numbers (WORKS below; edit
them there), ordered by publication date, with energy, latency, power and
TOPS/W scaled to our 40 nm node with DeepScaleTool (see SCALE_TO_NM). Our columns are filled from run.py's results
(logs/comparison_summary.csv): for every architecture of run.py's OURWORK, its
accuracy, energy, latency, power and TOPS/W per inference on each dataset
(the latest configuration, its run with the most samples). Sources and caveats are in
docs/literature_shortlist.md.

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

# Our datasets: (run.py model, name in the table).
DATASETS = (("nmnist", "N-MNIST"), ("gesture", "IBM DVS Gesture"), ("cifar10_thermo", "CIFAR-10"))
# The metric rows, in order: (key, row label).
METRICS = (("accuracy", "Accuracy"), ("energy", "Energy / sample"), ("latency", "Latency / sample"),
           ("power", "Power"), ("tops_per_w", "TOPS/W"))

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

# Each work's reported numbers, per dataset it evaluated (of N-MNIST, IBM DVS
# Gesture and CIFAR-10): accuracy as a LaTeX string, energy per sample in nJ,
# latency per sample in us, power in mW, TOPS/W; a metric not reported is left
# out and shows as NA. "node" (nm) is used for scaling; "date" (year, month)
# orders the columns. Footnote marks refer to FOOTNOTES.
WORKS = [
    {"name": "ANP-I", "publication": "JSSC'24", "date": (2024, 8), "node": 28,
     "technology": "28nm", "memory": "SRAM", "impl": r"Digital$^{\ddagger}$",
     "results": {r"N-MNIST$^{\dagger}$": {"accuracy": r"96.0\%", "energy": 343},
                 r"IBM DVS Gesture$^{*}$": {"accuracy": r"92.0\%", "energy": 3900}}},
    {"name": "ReckOn", "publication": "ISSCC'22", "date": (2022, 2), "node": 28,
     "technology": "28nm FDSOI", "memory": "SRAM", "impl": r"Digital$^{\ddagger}$",
     "results": {r"IBM DVS Gesture$^{*}$": {"accuracy": r"87.3\%", "energy": 46100, "power": 0.077}}},
    {"name": "DS-CIM", "publication": "TCAS-I'24", "date": (2024, 4), "node": 40,
     "technology": "40nm", "memory": "SOT-MRAM", "impl": "Mixed signal",
     "results": {r"IBM DVS Gesture$^{*}$": {"accuracy": r"90.00\%", "energy": 729.3,
                                            "latency": 40.46}}},
    {"name": "Han et al.", "publication": "TCAS-I'22", "date": (2022, 11), "node": 65,
     "technology": "65nm", "memory": "ReRAM", "impl": "Mixed signal",
     "results": {"CIFAR-10": {"accuracy": r"88\%", "energy": 21740, "tops_per_w": 14.12}}},
    {"name": "Neuro-CIM", "publication": "JSSC'23", "date": (2023, 10), "node": 28,
     "technology": "28nm", "memory": "SRAM", "impl": "Mixed signal",
     "results": {"CIFAR-10": {"accuracy": r"92.1\%", "energy": 720}}},
    {"name": "Yan et al.", "publication": "VLSI'19", "date": (2019, 6), "node": 150,
     "technology": "150nm", "memory": "RRAM", "impl": "Mixed signal",
     "results": {"CIFAR-10": {"accuracy": r"95.9\%$^{\S}$"}}},
]

# Our columns: run.py architecture name -> column heading.
OUR_NAMES = {"c3cim_xbar": "C3CIM", "c3cim_op_xbar": "C3CIM-OP"}
OUR_SPECS = {"publication": r"\textbf{This work}", "node": 40, "technology": "40nm", "memory": "RRAM",
             "impl": "Mixed signal"}

FOOTNOTES = [
    f"Energy, latency, power and TOPS/W scaled to {SCALE_TO_NM}nm with DeepScaleTool; "
    "accuracy as reported.",
    r"$^{*}$ Downscaled to $16\times16$, 10 classes (ours: $128\times128$, 11 classes).",
    r"$^{\dagger}$ Downscaled to $2\times17\times17$ (ours: $2\times34\times34$).",
    r"$^{\ddagger}$ Embedded on-chip learning.",
    r"$^{\S}$ Relative to its binarized software network.",
    r"$^{\P}$ Outside DeepScaleTool's range (130--7nm): as reported, not scaled.",
    r"This work: 6-bit weights; CIFAR-10 on VGG-11 with thermometer-coded binary inputs (8 time steps).",
]
CAPTION = "Comparison with other neuromorphic designs"
LABEL = "table:neuromorphic"


def scale(metric, value, node, target=SCALE_TO_NM):
    """A metric reported at `node` (nm) at the target node with DeepScaleTool's
    factors; None if a node is outside its range."""
    if node not in DEEPSCALE or target not in DEEPSCALE:
        return None
    energy, delay, power = (DEEPSCALE[node][i] / DEEPSCALE[target][i] for i in range(3))
    factor = {"energy": energy, "latency": delay, "power": power, "tops_per_w": 1 / energy}[metric]
    return value / factor


def _unit(value, units):
    """value in the first unit of `units` [(name, factor to the next)] -> LaTeX, 3 digits."""
    for name, step in units:
        if abs(value) < step or name == units[-1][0]:
            return f"{value:#.3g}".rstrip(".") + f"\\,{name}"
        value /= step


FORMATS = {"energy": lambda v: _unit(v, [("nJ", 1e3), (r"$\mu$J", 1e3), ("mJ", 1e3)]),
           "latency": lambda v: _unit(v, [(r"$\mu$s", 1e3), ("ms", 1e3), ("s", 1)]),
           "power": lambda v: _unit(v, [(r"$\mu$W", 1e3), ("mW", 1e3), ("W", 1)]),
           "tops_per_w": lambda v: f"{v:#.3g}".rstrip(".")}


def cell(metric, value, node):
    """One metric value as LaTeX: scaled to SCALE_TO_NM where it is a number."""
    if metric not in FORMATS or not isinstance(value, (int, float)):
        return value
    scaled = scale(metric, value, node)
    if scaled is None:
        return FORMATS[metric](value * 1e3 if metric == "power" else value) + r"$^{\P}$"
    return FORMATS[metric](scaled * 1e3 if metric == "power" else scaled)


def our_values(row):
    """A comparison_summary.csv row -> {metric: value} (energy nJ, latency us, power mW)."""
    return {"accuracy": f"{float(row['accuracy']):.2f}\\%",
            **{m: float(row[m]) for m in ("energy", "latency", "power", "tops_per_w")}}


def our_columns(runs, architectures, warn=print):
    """Our columns from run.py's results (the latest configuration of each
    architecture, its run with the most samples), per dataset with results."""
    columns = []
    for name in architectures:
        results = {}
        for model, dataset in DATASETS:
            rows = runs.get((model, name))
            if not rows:
                warn(f"warning: no results for {name} on {model}: run python run.py --model {model}")
                continue
            config = rows[-1]["configuration"]
            best = max((r for r in rows if r["configuration"] == config), key=lambda r: int(r["samples"]))
            if int(best["samples"]) < latex_table.FULL_TEST_SET.get(model, 0):
                warn(f"warning: {name} on {model}: {best['samples']} of "
                     f"{latex_table.FULL_TEST_SET[model]} test samples")
            results[dataset] = our_values(best)
        columns.append({"name": OUR_NAMES.get(name, name.replace("_", r"\_")), **OUR_SPECS,
                        "date": (9999, 0), "results": results, "ours": True})
    return columns


def _stack(values):
    """Values of one cell, one per dataset: "a", or a makecell of "a" / "/ b" / ..."""
    if len(values) == 1:
        return values[0]
    return r"\makecell{" + r" \\ ".join([values[0]] + [f"/ {v}" for v in values[1:]]) + "}"


def build_table(columns):
    n = len(columns)
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
             f"\\caption{{{CAPTION}}}", f"\\label{{{LABEL}}}",
             r"\renewcommand{\arraystretch}{1.15}",
             r"\begin{adjustbox}{max width=\textwidth}",
             f"\\begin{{tabular}}{{@{{}}l*{{{n}}}{{c}}@{{}}}}", r"\toprule"]

    def row(label, cells):
        lines.append(f"\\textbf{{{label}}} & " + " & ".join(cells) + r" \\")

    row("Name", [f"\\textbf{{{c['name']}}}" for c in columns])
    row("Publication", [c["publication"] for c in columns])
    lines.append(r"\midrule")
    for key, label in (("technology", "Technology"), ("memory", "Synaptic memory"),
                       ("impl", "Implementation")):
        row(label, [c[key] for c in columns])
    lines.append(r"\midrule")
    row("Datasets", [_stack(list(c["results"])) for c in columns])
    for key, label in METRICS:
        lines.append(r"\midrule")
        cells = []
        for c in columns:
            values = [cell(key, r[key], c["node"]) if key in r else "NA" for r in c["results"].values()]
            if all(v == "NA" for v in values):
                cells.append("NA")
            else:
                text = _stack(values)
                cells.append(f"\\textbf{{{text}}}" if c.get("ours") and len(values) == 1 else text)
        row(label, cells)
    body = "\n".join(lines)
    notes = [n for n in FOOTNOTES if not n.startswith("$^{") or n.split("$", 2)[1] in body]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
              r"\par\smallskip\raggedright\scriptsize",
              " \\\\\n".join(notes), r"\end{table*}", ""]
    return "\n".join(lines)


def make_table(our_architectures, logs_dir, warn=print):
    """The table text: WORKS, then a column per name in our_architectures."""
    summary = os.path.join(logs_dir, "comparison_summary.csv")
    if not os.path.exists(summary):
        raise FileNotFoundError(f"no results in {summary}: run python run.py first")
    works = sorted(WORKS, key=lambda w: w["date"])          # oldest first, ours last
    return build_table(works + our_columns(latex_table.load_results(summary), our_architectures, warn))


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
