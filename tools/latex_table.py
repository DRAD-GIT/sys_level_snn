"""Hardware comparison table for the paper, filled from the evaluation results.

    python tools/latex_table.py                       # -> logs/comparison_table.tex
    python tools/latex_table.py --out paper/table.tex --logs logs

The table holds the LITERATURE rows (published numbers; none by default),
then one row per architecture of run.py's ARCHITECTURES that has results
(or per --architectures),
named after it, and last, after a rule, the architectures of run.py's
OURWORK. Every architecture row gets its results per dataset (N-MNIST, DVS
Gesture and, if run, CIFAR-10; --datasets to choose) from
logs/comparison_summary.csv, written by run.py. For each model it uses the
architecture's latest configuration (the last run of it, so a design changed
since then is not mixed in) and, of that, the run with the most samples; a
run on fewer samples than the whole test set, or a missing run (empty cells),
is reported with a warning. An architecture's hardware specifications come
from its definition: those given in compose(..., specs={...}) (and its row
name, specs["label"]), else, for cell precision, R_High/R_Low and
accumulation (rows summed per read), read from the crossbar; unknown ones
stay empty. Units: power mW, latency us and energy uJ per inference,
TOPS/W. In every result column the best value of any row (lowest power,
latency and energy, highest TOPS/W) is red, and a value of our work is bold
where it beats every row that is not our work (literature and other
architectures).

Include the output with \\input{comparison_table}; the preamble needs
booktabs, multirow, makecell, adjustbox and xcolor with [table] (colortbl).
The table defines its N-MNIST shade itself (\\providecolor{nmband}), so an
earlier \\definecolor{nmband}{...} in the preamble still takes precedence.
"""
import argparse
import csv
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from hardware import architecture  # noqa: E402

# ============================================================================
# THE TABLE
# ============================================================================
# Literature rows (none by default). specs: Tech. (nm), Supply (V), Storage
# device, Cell/precision, Bit-cell, R_High/R_Low (kOhm), Sensing mode,
# Accumulation (LaTeX allowed); "nmnist" / "gesture" / "cifar10" = (power mW,
# latency us, energy uJ, TOPS/W); None = empty cells. For example:
#   {"work": r"ISSCC'22~\cite{khwa_40-nm_2022}",
#    "specs": ["40", "0.9", "PCM", "1", "1T1R", "--", "Voltage", "8"],
#    "nmnist": (396, 22.9, 9.1, 25.4), "gesture": (1970.4, 88.5, 174.3, 25.6)},
#   {"work": r"AICAS'23~\cite{singh_1151_2023}",
#    "specs": ["40", "1.1", "Resistive", "1", "1T1R", "200/2", "Current", "64"],
#    "nmnist": (531.6, 9.8, 5.2, 44.4), "gesture": (1040.2, 37.7, 39.2, 113.8)},
LITERATURE = []

# The specification columns, in order: an architecture's specs (all but its label).
SPEC_KEYS = tuple(key for key in architecture.SPEC_KEYS if key != "label")
# Datasets that can have a column group, in table order, with their headings.
MODELS = ("nmnist", "gesture", "gesture16", "cifar10", "cifar10_thermo")
HEADINGS = {"nmnist": "N-MNIST", "gesture": "IBM DVS128 Gesture",
            "gesture16": r"IBM DVS Gesture ($16\times16$)", "cifar10": "CIFAR-10",
            "cifar10_thermo": "CIFAR-10"}
CAPTION_NAMES = {"nmnist": "N-MNIST", "gesture": "IBM-Gesture", "gesture16": "IBM-Gesture",
                 "cifar10": "CIFAR-10",
                 "cifar10_thermo": "CIFAR-10"}
FULL_TEST_SET = {"nmnist": 10000, "gesture": 264, "gesture16": 240, "cifar10": 10000,
                 "cifar10_thermo": 10000}
DEFAULT_MODELS = ("nmnist", "gesture")
# ============================================================================

CAPTION = "Hardware comparison of deploying SNN models trained on the %s datasets"
LABEL = "table:comp"
# The best value per column: min for power, latency, energy; max for TOPS/W.
BEST = (min, min, min, max)

COLUMNS = (r"\makecell{Power\\(mW)}", r"\makecell{Latency\\($\mu$s)}",
           r"\makecell{Energy\\($\mu$J)}", "TOPS/W")


def caption(models):
    names = [CAPTION_NAMES[m] for m in models]
    return CAPTION % (names[0] if len(names) == 1 else
                      ", ".join(names[:-1]) + " and " + names[-1])


def header(models, caption_text, label):
    """Table head with one 4-column result group per dataset; every other
    group (from the first) is shaded."""
    n, last = len(models), 9 + 4 * len(models)
    groups = " ".join("*{4}{>{\\columncolor{nmband}}c}" if i % 2 == 0 else "cccc"
                      for i in range(n))
    rules = "".join(f"\\cmidrule({'l' if i == n - 1 else 'lr'}){{{10 + 4 * i}-{13 + 4 * i}}}"
                    for i in range(n))
    names = "\n".join(f" & \\multicolumn{{4}}{{c}}{{\\textit{{{HEADINGS[m]}}}}}" for m in models)
    columns = "\n".join(" & " + " & ".join(COLUMNS) for _ in models)
    return (r"""\providecolor{nmband}{gray}{0.92}  % N-MNIST column shade (kept if already defined)
\begin{table*}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.25}
\setlength{\aboverulesep}{0pt}
\setlength{\belowrulesep}{0pt}
""" + f"\\caption{{{caption_text}}}\n\\label{{{label}}}\n" + r"""\begin{adjustbox}{max width=\textwidth}
""" + f"\\begin{{tabular}}{{@{{}}l cccccccc {groups}@{{}}}}\n" + r"""\toprule
 & \multicolumn{8}{c}{\multirow{2}{*}{\textbf{Hardware specifications}}}
""" + f" & \\multicolumn{{{4 * n}}}{{c}}{{\\textbf{{Evaluation results}}}} \\\\\n"
            + f"\\cmidrule(l){{10-{last}}}\n & \\multicolumn{{8}}{{c}}{{}}\n{names} \\\\\n"
            + f"\\cmidrule(r){{2-9}}{rules}\n" + r"""\textbf{Work}
 & \makecell{Tech.\\(nm)} & \makecell{Supply\\(V)} & \makecell{Storage\\device} & \makecell{Cell/\\precision}
 & Bit-cell & \makecell{$R_\mathrm{High}/R_\mathrm{Low}$\\(k$\Omega$)} & \makecell{Sensing\\mode} & \makecell{Accumu-\\lation}
""" + columns + " \\\\\n\\midrule\n")


FOOTER = r"""\bottomrule
\end{tabular}
\end{adjustbox}
\end{table*}
"""


def load_results(summary_csv):
    """{(model, architecture): [csv rows]} from comparison_summary.csv."""
    runs = {}
    with open(summary_csv, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            runs.setdefault((row["model"], row["architecture"]), []).append(row)
    return runs


def pick_run(runs, model, architecture, warn=print):
    """The architecture's latest configuration, and of it the run with the
    most samples; returns (power mW, latency us, energy uJ, TOPS/W)."""
    rows = runs.get((model, architecture))
    if not rows:
        warn(f"warning: no results for {architecture} on {model} (empty cells): run "
             f"python run.py --model {model} with it in ARCHITECTURES")
        return None
    config = rows[-1]["configuration"]
    best = max((r for r in rows if r["configuration"] == config), key=lambda r: int(r["samples"]))
    missing = [m for m in ("power", "latency", "energy", "tops_per_w") if not best.get(m)]
    if missing:
        raise ValueError(f"{architecture} on {model}: metrics {missing} were switched off in "
                         "run.py's METRICS when it ran")
    samples = int(best["samples"])
    if samples < FULL_TEST_SET.get(model, 0):
        warn(f"warning: {architecture} on {model}: {samples} of {FULL_TEST_SET[model]} test "
             "samples (run it with --samples -1 for the full test set)")
    return (float(best["power"]), float(best["latency"]), float(best["energy"]) / 1000,
            float(best["tops_per_w"]))


def number(value):
    """One decimal (none if it is 0: 93, 22.9), two significant digits below 1."""
    if abs(value) >= 1 or value == 0:
        text = f"{value:.1f}"
        return text[:-2] if text.endswith(".0") else text
    return f"{value:.2g}"


def _kilo(ohm):
    return f"{ohm / 1e3:g}"


def derived_specs(arch):
    """The specifications an Architecture tells: cell precision, R_High/R_Low
    (kOhm) and accumulation (rows summed per read)."""
    memory, xb = arch.crossbar.memory, arch.crossbar
    specs = {"cell": "analog" if arch.mapping.weight_encoding == "analog" else str(memory.cell_bits),
             "accumulation": str(xb.active_rows or xb.rows)}
    if memory.r_on and memory.r_off:
        specs["r_ratio"] = f"{_kilo(memory.r_off)}/{_kilo(memory.r_on)}"
    return specs


def architecture_rows(architectures, ourwork=()):
    """One table row per Architecture, named specs["label"] or after it
    (LaTeX-escaped), with its specs, completed by those read from its
    crossbar: first the others, then (after a rule) those in `ourwork`
    (Architectures or names)."""
    ours = {a if isinstance(a, str) else a.name for a in ourwork}
    names = {a.name for a in architectures}
    architectures = list(architectures) + [a for a in ourwork
                                           if not isinstance(a, str) and a.name not in names]
    ordered = [a for a in architectures if a.name not in ours] + \
              [a for a in architectures if a.name in ours]
    rows = []
    for i, arch in enumerate(ordered):
        values = {**derived_specs(arch), **{k: str(v) for k, v in arch.specs.items()}}
        first_ours = arch.name in ours and (i == 0 or ordered[i - 1].name not in ours)
        rows.append({"work": values.get("label", arch.name.replace("_", r"\_")),
                     "specs": [values.get(key, "") for key in SPEC_KEYS],
                     "architecture": arch.name, "ours": arch.name in ours,
                     "midrule": i == 0 or first_ours})
    return rows


def row_values(row, runs, warn=print, models=DEFAULT_MODELS):
    """The row's values per dataset in `models`, each 4 floats or None: an
    architecture's results, or a literature row's published numbers."""
    name = row.get("architecture")
    return [pick_run(runs, model, name, warn) if name else row.get(model) for model in models]


def build_table(rows, runs, warn=print, models=DEFAULT_MODELS):
    """The LaTeX table for `rows` with results from `runs` (load_results), a
    column group per dataset in `models`. Per result column: the best value
    of any row in red; a value of a row with "ours" in bold where it beats
    every row without it."""
    values = [row_values(row, runs, warn, models) for row in rows]
    # Compared as printed, so equal printed values are marked alike.
    best, best_other = {}, {}
    for m in range(len(models)):
        for k, pick in enumerate(BEST):
            column = [float(number(v[m][k])) for v in values if v[m] is not None]
            others = [float(number(v[m][k])) for row, v in zip(rows, values)
                      if v[m] is not None and not row.get("ours")]
            if len(column) > 1:
                best[m, k] = pick(column)
            if others:
                best_other[m, k] = pick(others)
    lines = [header(models, caption(models), LABEL)]
    for i, (row, row_vals) in enumerate(zip(rows, values)):
        if row.get("midrule") and i > 0:        # the header already ends with a rule
            lines.append("\\midrule\n")
        comment = f" % {row['comment']}" if row.get("comment") else ""
        if not row.get("specs") and all(v is None for v in row_vals):   # an empty row
            lines.append(f"{row['work']}{comment}\n{' &' * (8 + 4 * len(models))} \\\\\n")
            continue
        specs = list(row.get("specs") or []) + [""] * (8 - len(row.get("specs") or []))
        cells = []
        for m, v in enumerate(row_vals):
            for k in range(4):
                if v is None:
                    cells.append("")
                    continue
                text = number(v[k])
                value, other = float(text), best_other.get((m, k))
                if row.get("ours") and other is not None and value != other \
                        and BEST[k](value, other) == value:        # beats every other row
                    text = f"\\textbf{{{text}}}"
                if best.get((m, k)) == value:                    # best of the column
                    text = f"\\textcolor{{red}}{{{text}}}"
                cells.append(text)
        groups = "\n".join(f" & {' & '.join(cells[4 * m:4 * m + 4])}" for m in range(len(models)))
        lines.append(f"{row['work']}{comment}\n"
                     f" & {' & '.join(specs)}\n"
                     f"{groups} \\\\\n")
    lines.append(FOOTER)
    # Empty cells as "& &", as written by hand; no trailing spaces.
    table = re.sub(r"(?<=&) +(?=&|\\\\)", " ", "".join(lines))
    return re.sub(r" +\n", "\n", table)


def make_table(architectures, ourwork, logs_dir, models=None, warn=print):
    """The LaTeX table from <logs_dir>/comparison_summary.csv: the LITERATURE
    rows, then a row per architecture (those in `ourwork` last), with a
    column group per dataset in `models` (default: every dataset with
    results or literature numbers)."""
    summary = os.path.join(logs_dir, "comparison_summary.csv")
    if not os.path.exists(summary):
        raise FileNotFoundError(f"no results in {summary}: run python run.py first")
    runs = load_results(summary)
    rows = LITERATURE + architecture_rows(architectures, ourwork)
    models = models or [m for m in MODELS
                        if any(key[0] == m for key in runs) or any(r.get(m) for r in rows)]
    return build_table(rows, runs, warn=warn, models=models or list(DEFAULT_MODELS))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logs", default=os.path.join(ROOT, "logs"),
                        help="folder of run.py's results (comparison_summary.csv)")
    parser.add_argument("--out", help="output .tex (default: <logs>/comparison_table.tex)")
    parser.add_argument("--architectures", nargs="+",
                        help="names of run.py architectures to include (default: ARCHITECTURES; "
                             "OURWORK is always included)")
    parser.add_argument("--datasets", nargs="+", choices=MODELS,
                        help="dataset column groups, in this order (default: every dataset "
                             "with results or literature numbers)")
    args = parser.parse_args()
    import run
    architectures = run.ARCHITECTURES
    if args.architectures:
        known = {a.name: a for name in dir(run) for a in [getattr(run, name)]
                 if hasattr(a, "crossbar") and hasattr(a, "mapping")}
        missing = [n for n in args.architectures if n not in known]
        if missing:
            parser.error(f"not defined in run.py: {missing}; defined: {sorted(known)}")
        architectures = [known[n] for n in args.architectures]
    summary = os.path.join(args.logs, "comparison_summary.csv")
    if not os.path.exists(summary):
        parser.error(f"no results in {summary}: run python run.py first (or pass --logs)")
    ourwork = getattr(run, "OURWORK", [])
    ours = {a.name for a in ourwork}
    evaluated = {architecture for _, architecture in load_results(summary)}
    if not args.architectures:     # leave out macros not evaluated (python run.py --literature)
        architectures = [a for a in architectures if a.name in ours or a.name in evaluated]
    table = make_table(architectures, ourwork, args.logs, args.datasets,
                       warn=lambda text: print(text, file=sys.stderr))
    out = args.out or os.path.join(args.logs, "comparison_table.tex")
    with open(out, "w", encoding="utf-8") as file:
        file.write(table)
    print(table, end="")
    print(f"saved {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
