"""Comparison with other neuromorphic designs, in the style of DS-CIM's Table III
(Fu et al., TCAS-I 2024): one column per work, rows for its publication,
technology and implementation, then its accuracy and energy per sample on
N-MNIST, IBM DVS Gesture and CIFAR-10.

    python tools/neuromorphic_table.py               # -> logs/neuromorphic_table.tex
    python tools/neuromorphic_table.py --out paper/Chapters/neuromorphic_table.tex

The literature columns hold each paper's reported numbers (WORKS below, as
LaTeX strings; edit them there). Our columns are filled from run.py's results
(logs/comparison_summary.csv): for every architecture of run.py's OURWORK, its
accuracy and energy per inference on each dataset (the latest configuration,
its run with the most samples). Sources and caveats are in
docs/literature_shortlist.md.

Include the output with \\input{neuromorphic_table}; the preamble needs
booktabs and adjustbox.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import latex_table  # noqa: E402

# Datasets (rows), in order: (key in the literature numbers, run.py model, heading).
DATASETS = (("nmnist", "nmnist", "N-MNIST"),
            ("gesture", "gesture", "IBM DVS Gesture"),
            ("cifar10", "cifar10_thermo", "CIFAR-10"))

# Reported numbers, as written in each paper (LaTeX strings; "--" = not
# reported). Footnote marks refer to FOOTNOTES.
WORKS = [
    {"name": "Memristive SNN", "publication": r"arXiv'25$^{\P}$", "technology": "180nm",
     "memory": "RRAM", "impl": "Mixed signal", "silicon": "Measured",
     "accuracy": {"nmnist": r"94.73\%", "gesture": r"93.06\%"},
     "energy": {"nmnist": r"1.78\,$\mu$J", "gesture": r"1.94\,$\mu$J"}},
    {"name": "ANP-I", "publication": "JSSC'24", "technology": "28nm",
     "memory": "SRAM", "impl": r"Digital$^{\ddagger}$", "silicon": "Measured",
     "accuracy": {"nmnist": r"96.0\%$^{\dagger}$", "gesture": r"92.0\%$^{*}$"},
     "energy": {"nmnist": r"343\,nJ$^{\dagger}$", "gesture": r"3.9\,$\mu$J$^{*}$"}},
    {"name": "ReckOn", "publication": "ISSCC'22", "technology": "28nm FDSOI",
     "memory": "SRAM", "impl": r"Digital$^{\ddagger}$", "silicon": "Measured",
     "accuracy": {"gesture": r"87.3\%$^{*}$"},
     "energy": {"gesture": r"46.1\,$\mu$J$^{*}$"}},
    {"name": "DS-CIM", "publication": "TCAS-I'24", "technology": "40nm",
     "memory": "SOT-MRAM", "impl": "Mixed signal", "silicon": "Simulated",
     "accuracy": {"gesture": r"90.00\%$^{*}$"},
     "energy": {"gesture": r"729.3\,nJ$^{*}$"}},
    {"name": "TrueNorth", "publication": "CVPR'17", "technology": "28nm",
     "memory": "SRAM", "impl": "Digital", "silicon": "Measured",
     "accuracy": {"gesture": r"96.5\%"},
     "energy": {"gesture": r"18.8\,mJ$^{\parallel}$"}},
    {"name": "Han et al.", "publication": "TCAS-I'22", "technology": "65nm",
     "memory": "ReRAM", "impl": "Mixed signal", "silicon": "Simulated",
     "accuracy": {"cifar10": r"88\%"},
     "energy": {"cifar10": r"21.74\,$\mu$J"}},
    {"name": "Neuro-CIM", "publication": "JSSC'23", "technology": "28nm",
     "memory": "SRAM", "impl": "Mixed signal", "silicon": "Measured",
     "accuracy": {"cifar10": r"92.1\%"},
     "energy": {"cifar10": r"0.72\,$\mu$J"}},
    {"name": "Yan et al.", "publication": "VLSI'19", "technology": "150nm",
     "memory": "RRAM", "impl": "Mixed signal", "silicon": "Measured",
     "accuracy": {"cifar10": r"95.9\%$^{\S}$"},
     "energy": {}},
]

# Our columns: run.py architecture name -> column heading.
OUR_NAMES = {"c3cim_xbar": "C3CIM", "c3cim_op_xbar": "C3CIM-OP"}
OUR_SPECS = {"publication": r"\textbf{This work}", "technology": "40nm", "memory": "RRAM",
             "impl": "Mixed signal", "silicon": "Simulated"}

FOOTNOTES = [
    r"$^{*}$ IBM DVS Gesture downscaled to $16\times16$, 10 classes (ours: $128\times128$, 11 classes).",
    r"$^{\dagger}$ N-MNIST downscaled to $2\times17\times17$ (ours: $2\times34\times34$).",
    r"$^{\ddagger}$ Embedded on-chip learning.",
    r"$^{\S}$ Relative to its binarized software network.",
    r"$^{\P}$ Preprint.",
    r"$^{\parallel}$ Reported power (178.8\,mW) $\times$ latency (105\,ms).",
    r"This work: 6-bit weights; CIFAR-10 on VGG-11 with thermometer-coded binary inputs (8 time steps).",
]
CAPTION = "Comparison with other neuromorphic designs"
LABEL = "table:neuromorphic"


def energy_text(nj):
    """Energy per inference in nJ -> LaTeX with a fitting unit (3 significant digits)."""
    for limit, unit, div in ((1e3, "nJ", 1.0), (1e6, r"$\mu$J", 1e3), (float("inf"), "mJ", 1e6)):
        if nj < limit:
            return f"{nj / div:#.3g}".rstrip(".") + f"\\,{unit}"


def our_columns(runs, architectures, warn=print):
    """Our columns from run.py's results: {heading, specs..., accuracy, energy}."""
    columns = []
    for name in architectures:
        column = {"name": OUR_NAMES.get(name, name.replace("_", r"\_")), **OUR_SPECS,
                  "accuracy": {}, "energy": {}}
        for key, model, _ in DATASETS:
            rows = runs.get((model, name))
            if not rows:
                warn(f"warning: no results for {name} on {model}: run python run.py --model {model}")
                continue
            config = rows[-1]["configuration"]
            best = max((r for r in rows if r["configuration"] == config), key=lambda r: int(r["samples"]))
            if int(best["samples"]) < latex_table.FULL_TEST_SET.get(model, 0):
                warn(f"warning: {name} on {model}: {best['samples']} of "
                     f"{latex_table.FULL_TEST_SET[model]} test samples")
            column["accuracy"][key] = f"{float(best['accuracy']):.2f}\\%"
            column["energy"][key] = energy_text(float(best["energy"]))
        columns.append(column)
    return columns


def build_table(columns):
    n = len(columns)
    ours = [i for i, c in enumerate(columns) if c.get("publication") == OUR_SPECS["publication"]]
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
             f"\\caption{{{CAPTION}}}", f"\\label{{{LABEL}}}",
             r"\begin{adjustbox}{max width=\textwidth}",
             f"\\begin{{tabular}}{{@{{}}l*{{{n}}}{{c}}@{{}}}}", r"\toprule"]

    def row(label, cells, bold=False):
        cells = [f"\\textbf{{{c}}}" if bold else c for c in cells]
        return f"\\textbf{{{label}}} & " + " & ".join(cells) + r" \\"

    lines.append(row("Name", [c["name"] for c in columns], bold=True))
    lines.append(row("Publication", [c["publication"] for c in columns]))
    lines.append(r"\midrule")
    for key, label in (("technology", "Technology"), ("memory", "Synaptic memory"),
                       ("impl", "Implementation"), ("silicon", "Silicon")):
        lines.append(row(label, [c[key] for c in columns]))
    for quantity, label in (("accuracy", "Accuracy"), ("energy", "Energy / sample")):
        lines.append(r"\midrule")
        lines.append(f"\\multicolumn{{{n + 1}}}{{@{{}}l}}{{\\textbf{{{label}}}}} \\\\")
        for key, _, heading in DATASETS:
            cells = [c[quantity].get(key, "--") for c in columns]
            cells = [f"\\textbf{{{v}}}" if i in ours and v != "--" else v for i, v in enumerate(cells)]
            lines.append(f"\\quad {heading} & " + " & ".join(cells) + r" \\")
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
