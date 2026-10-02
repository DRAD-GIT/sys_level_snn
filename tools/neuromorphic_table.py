"""Comparison with other neuromorphic designs, in the style of DS-CIM's Table III
(Fu et al., TCAS-I 2024): one column per work, rows for its publication,
technology and implementation, the datasets it evaluated (of N-MNIST, IBM DVS
Gesture and CIFAR-10), and for each metric its values on those datasets in
the same order ("a / b / c"; NA where not reported).

    python tools/neuromorphic_table.py               # -> logs/neuromorphic_table.tex
    python tools/neuromorphic_table.py --out paper/Chapters/neuromorphic_table.tex

The literature columns hold each paper's reported numbers (WORKS below, as
LaTeX strings; edit them there). Our columns are filled from run.py's results
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

# Each work's reported numbers, per dataset it evaluated (of N-MNIST, IBM DVS
# Gesture and CIFAR-10), as written in the paper (LaTeX strings; a metric not
# reported is left out and shows as NA). Footnote marks refer to FOOTNOTES.
WORKS = [
    {"name": "Memristive SNN", "publication": r"arXiv'25$^{\P}$", "technology": "180nm",
     "memory": "RRAM", "impl": "Mixed signal",
     "results": {"N-MNIST": {"accuracy": r"94.73\%", "energy": r"1.78\,$\mu$J"},
                 "IBM DVS Gesture": {"accuracy": r"93.06\%", "energy": r"1.94\,$\mu$J",
                                     "latency": r"44.31\,$\mu$s", "power": r"43.83\,mW"}}},
    {"name": "ANP-I", "publication": "JSSC'24", "technology": "28nm",
     "memory": "SRAM", "impl": r"Digital$^{\ddagger}$",
     "results": {r"N-MNIST$^{\dagger}$": {"accuracy": r"96.0\%", "energy": "343\\,nJ"},
                 r"IBM DVS Gesture$^{*}$": {"accuracy": r"92.0\%", "energy": r"3.9\,$\mu$J"}}},
    {"name": "ReckOn", "publication": "ISSCC'22", "technology": "28nm FDSOI",
     "memory": "SRAM", "impl": r"Digital$^{\ddagger}$",
     "results": {r"IBM DVS Gesture$^{*}$": {"accuracy": r"87.3\%", "energy": r"46.1\,$\mu$J",
                                            "power": r"77\,$\mu$W"}}},
    {"name": "DS-CIM", "publication": "TCAS-I'24", "technology": "40nm",
     "memory": "SOT-MRAM", "impl": "Mixed signal",
     "results": {r"IBM DVS Gesture$^{*}$": {"accuracy": r"90.00\%", "energy": "729.3\\,nJ",
                                            "latency": r"40.46\,$\mu$s"}}},
    {"name": "TrueNorth", "publication": "CVPR'17", "technology": "28nm",
     "memory": "SRAM", "impl": "Digital",
     "results": {"IBM DVS Gesture": {"accuracy": r"96.5\%", "energy": r"18.8\,mJ$^{\parallel}$",
                                     "latency": "105\\,ms", "power": "178.8\\,mW"}}},
    {"name": "Han et al.", "publication": "TCAS-I'22", "technology": "65nm",
     "memory": "ReRAM", "impl": "Mixed signal",
     "results": {"CIFAR-10": {"accuracy": r"88\%", "energy": r"21.74\,$\mu$J",
                              "tops_per_w": "14.12"}}},
    {"name": "Neuro-CIM", "publication": "JSSC'23", "technology": "28nm",
     "memory": "SRAM", "impl": "Mixed signal",
     "results": {"CIFAR-10": {"accuracy": r"92.1\%", "energy": r"0.72\,$\mu$J"}}},
    {"name": "Yan et al.", "publication": "VLSI'19", "technology": "150nm",
     "memory": "RRAM", "impl": "Mixed signal",
     "results": {"CIFAR-10": {"accuracy": r"95.9\%$^{\S}$"}}},
]

# Our columns: run.py architecture name -> column heading.
OUR_NAMES = {"c3cim_xbar": "C3CIM", "c3cim_op_xbar": "C3CIM-OP"}
OUR_SPECS = {"publication": r"\textbf{This work}", "technology": "40nm", "memory": "RRAM",
             "impl": "Mixed signal"}

FOOTNOTES = [
    r"$^{*}$ Downscaled to $16\times16$, 10 classes (ours: $128\times128$, 11 classes).",
    r"$^{\dagger}$ Downscaled to $2\times17\times17$ (ours: $2\times34\times34$).",
    r"$^{\ddagger}$ Embedded on-chip learning.",
    r"$^{\S}$ Relative to its binarized software network.",
    r"$^{\P}$ Preprint.",
    r"$^{\parallel}$ Reported power $\times$ latency.",
    r"This work: 6-bit weights; CIFAR-10 on VGG-11 with thermometer-coded binary inputs (8 time steps).",
]
CAPTION = "Comparison with other neuromorphic designs"
LABEL = "table:neuromorphic"


def _unit(value, units):
    """value in the first unit of `units` [(name, factor to the next)] -> LaTeX, 3 digits."""
    for name, step in units:
        if abs(value) < step or name == units[-1][0]:
            return f"{value:#.3g}".rstrip(".") + f"\\,{name}"
        value /= step


def our_values(row):
    """A comparison_summary.csv row -> {metric: LaTeX} (energy nJ, latency us, power mW)."""
    return {"accuracy": f"{float(row['accuracy']):.2f}\\%",
            "energy": _unit(float(row["energy"]), [("nJ", 1e3), (r"$\mu$J", 1e3), ("mJ", 1e3)]),
            "latency": _unit(float(row["latency"]), [(r"$\mu$s", 1e3), ("ms", 1e3), ("s", 1)]),
            "power": _unit(float(row["power"]), [("mW", 1e3), ("W", 1)]),
            "tops_per_w": f"{float(row['tops_per_w']):#.3g}".rstrip(".")}


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
                        "results": results, "ours": True})
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
            values = [r.get(key, "NA") for r in c["results"].values()]
            if all(v == "NA" for v in values):
                cells.append("NA")
            else:
                cell = _stack(values)
                cells.append(f"\\textbf{{{cell}}}" if c.get("ours") and len(values) == 1 else cell)
        row(label, cells)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
              r"\par\smallskip\raggedright\scriptsize",
              " \\\\\n".join(FOOTNOTES), r"\end{table*}", ""]
    return "\n".join(lines)


def make_table(our_architectures, logs_dir, warn=print):
    """The table text: WORKS, then a column per name in our_architectures."""
    summary = os.path.join(logs_dir, "comparison_summary.csv")
    if not os.path.exists(summary):
        raise FileNotFoundError(f"no results in {summary}: run python run.py first")
    return build_table(WORKS + our_columns(latex_table.load_results(summary), our_architectures, warn))


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
