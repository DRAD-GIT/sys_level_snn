"""Comparison with other neuromorphic designs, in the style of DS-CIM's Table III
(Fu et al., TCAS-I 2024): one column per work, rows for its publication,
technology and implementation, then, for the input and each metric, one row
per dataset (N-MNIST, IBM DVS Gesture, CIFAR-10), so a dataset's numbers line
up across the columns ("--": not evaluated; NA: not reported).

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
OUR_INPUTS = {"nmnist17": r"$2\times17\times17^{\dagger}$", "nmnist": r"$2\times34\times34$",
              "gesture16": r"$16\times16^{*}$", "gesture": r"$128\times128$",
              "cifar10_thermo": r"$32\times32$"}
# The metric rows, in order: (key, row label, better: +1 higher, -1 lower).
METRICS = (("accuracy", "Accuracy", +1), ("energy", "Energy / sample", -1),
           ("latency", "Latency / sample", -1), ("power", "Power", -1),
           ("tops_per_w", "TOPS/W", +1))

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
# mW, TOPS/W; a metric not reported is left out and shows as NA, and TOPS/W
# left out is computed in make_table (mark b). "marks" puts a footnote mark
# on a value (a: computed from the paper's other numbers, derivation next to
# it). "node" (nm) is used for scaling; "date" (year, month) orders the
# columns. Footnote marks refer to FOOTNOTES.
WORKS = [
    {"name": "ANP-I", "publication": "JSSC'24", "date": (2024, 8), "node": 28,
     "technology": "28nm", "memory": "SRAM", "impl": r"Digital$^{\ddagger}$",
     "results": {"nmnist": {"input": r"$2\times17\times17^{\dagger}$", "accuracy": r"96.0\%",
                            "energy": 343},
                 "gesture": {"input": r"$14\times14^{\S}$", "accuracy": r"92.0\%", "energy": 3900}}},
    # DS-CIM Table II: 40.46 us per Gesture sample -> power 735.35 nJ / 40.46 us.
    {"name": "DS-CIM", "publication": "TCAS-I'24", "date": (2024, 4), "node": 40,
     "technology": "40nm", "memory": "SOT-MRAM", "impl": "Mixed signal",
     "results": {"gesture": {"input": r"$16\times16^{*}$", "accuracy": r"90.00\%", "energy": 735.35,
                             "latency": 40.46, "power": 735.35 / 40.46,
                             "marks": {"power": "a"}}}},
    # 181 uJ per inference; 100 time steps of 50 us, pipelined -> 5 ms per
    # inference, power 181 uJ / 5 ms.
    {"name": "Dorzhigulov et al.", "publication": "Front. Neurosci.'23", "date": (2023, 7), "node": 130,
     "technology": "130nm", "memory": "RRAM", "impl": "Mixed signal",
     "results": {"cifar10": {"input": r"$32\times32$", "accuracy": r"61.74\%", "energy": 181000,
                             "latency": 100 * 50.0, "power": 181000 / (100 * 50.0),
                             "marks": {"energy": r"\#", "latency": "a", "power": "a"}}}},
    {"name": "SPOON", "publication": "ISCAS'20", "date": (2020, 10), "node": 28,
     "technology": "28nm FDSOI", "memory": "SRAM", "impl": r"Digital$^{\ddagger}$",
     "results": {"nmnist": {"input": r"$2\times34\times34^{\|}$", "accuracy": r"93.8\%",
                            "energy": 665}}},
    # 46.1 uJ per Gesture sample at 77 uW (inference, 0.5 V, 13 MHz) ->
    # latency 46.1 uJ / 77 uW.
    {"name": "ReckOn", "publication": "ISSCC'22", "date": (2022, 2), "node": 28,
     "technology": "28nm FDSOI", "memory": "SRAM", "impl": r"Digital$^{\ddagger}$",
     "results": {"gesture": {"input": r"$16\times16^{*}$", "accuracy": r"87.3\%", "energy": 46100,
                             "power": 0.077, "latency": 46100 / 0.077, "marks": {"latency": "a"}}}},
    # VGG-11 (Table III): 21.74 uJ, 73 ns per layer -> 11 layers x 73 ns per
    # image (c), power 21.74 uJ over it.
    {"name": "Han et al.", "publication": "TCAS-I'22", "date": (2022, 11), "node": 65,
     "technology": "65nm", "memory": "RRAM", "impl": "Mixed signal",
     "results": {"cifar10": {"input": r"$32\times32$", "accuracy": r"88\%", "energy": 21740,
                             "latency": 11 * 0.073, "power": 21740 / (11 * 0.073),
                             "tops_per_w": 14.12, "marks": {"latency": "c", "power": "a"}}}},
]

# Our columns: run.py architecture name -> column heading.
OUR_NAMES = {"c3cim_xbar": "C3CIM", "c3cim_op_xbar": "C3CIM-OP"}
OUR_SPECS = {"publication": r"\textbf{This work}", "node": 40, "technology": "40nm", "memory": "RRAM",
             "impl": "Mixed signal"}

FOOTNOTES = [
    f"Energy, latency, power and TOPS/W scaled to {SCALE_TO_NM}nm with DeepScaleTool; "
    "accuracy as reported.",
    r"$^{*}$ Downscaled to $16\times16$, 10 classes.",
    r"$^{\dagger}$ Downscaled to $2\times17\times17$, first 100\,ms.",
    r"$^{\ddagger}$ Embedded on-chip learning.",
    r"$^{\S}$ Downscaled to $14\times14$ with five temporal filters, 10 classes.",
    r"$^{\|}$ $2\times34\times34$, first saccade, one spike per pixel.",
    r"$^{\#}$ Estimated energy of the crossbar array and neural periphery; 2-bit weights.",
    r"$^{a}$ Computed from the reported values: power = energy / latency, or latency = energy / power.",
    r"$^{b}$ Computed as this work's operations on that dataset (its network, all synaptic "
    r"operations) divided by the reported energy per sample.",
    r"$^{c}$ Computed as 11 layers $\times$ the reported 73\,ns per layer (VGG-11).",
    r"$^{\P}$ Outside DeepScaleTool's range (130--7nm): as reported, not scaled.",
    r"This work: 6-bit weights; N-MNIST on a 578-512-10 SNN (10 time steps of 10\,ms); "
    r"IBM DVS Gesture on a 512-512-10 SNN (80 time steps of 30\,ms); "
    r"CIFAR-10 on VGG-11 with thermometer-coded binary inputs (8 time steps).",
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
            **{m: float(row[m]) for m in ("energy", "latency", "power", "tops_per_w")}}


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
            if "tops_per_w" not in r and "energy" in r and key in ops:
                r["tops_per_w"] = ops[key] / (r["energy"] * 1e3)
                r["marks"]["tops_per_w"] = "b"
            results[key] = r
        out.append(dict(work, results=results))
    return out


def build_table(columns):
    n = len(columns)
    lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize",
             f"\\caption{{{CAPTION}}}", f"\\label{{{LABEL}}}",
             r"\renewcommand{\arraystretch}{1.15}",
             r"\begin{adjustbox}{max width=\textwidth}",
             f"\\begin{{tabular}}{{@{{}}ll*{{{n}}}{{c}}@{{}}}}", r"\toprule"]

    def row(label, sub, cells):
        head = f"\\textbf{{{label}}}" if label else ""
        lines.append(f"{head} & {sub} & " + " & ".join(cells) + r" \\")

    row("Name", "", [f"\\textbf{{{c['name']}}}" for c in columns])
    row("Publication", "", [c["publication"] for c in columns])
    lines.append(r"\midrule")
    for key, label in (("technology", "Technology"), ("memory", "Synaptic memory"),
                       ("impl", "Implementation")):
        row(label, "", [c[key] for c in columns])
    lines.append(r"\midrule")
    for i, (key, _, dataset) in enumerate(DATASETS):
        row("Input" if i == 0 else "", dataset,
            [c["results"][key]["input"] if key in c["results"] else "--" for c in columns])
    for metric, label, better in METRICS:
        lines.append(r"\midrule")
        for i, (key, _, dataset) in enumerate(DATASETS):
            others = [scaled(metric, c["results"][key][metric], c["node"]) for c in columns
                      if not c.get("ours") and key in c["results"] and metric in c["results"][key]]
            cells = []
            for c in columns:
                r = c["results"].get(key)
                if r is None:
                    cells.append("--")
                    continue
                if metric not in r:
                    cells.append("NA")
                    continue
                text = cell(metric, r[metric], c["node"], r.get("marks", {}).get(metric, ""))
                value = scaled(metric, r[metric], c["node"])
                if c.get("ours") and others and all(better * (value - o) > 0 for o in others):
                    text = f"\\textbf{{{text}}}"
                cells.append(text)
            row(label if i == 0 else "", dataset, cells)
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
