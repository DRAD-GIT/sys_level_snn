"""Hardware comparison table for the paper, filled from the evaluation results.

    python tools/latex_table.py                       # -> logs/comparison_table.tex
    python tools/latex_table.py --out paper/table.tex --logs logs

The table holds the LITERATURE rows (published numbers) and then, after a
rule, one row per architecture of run.py's ARCHITECTURES (or --architectures),
named after it, with its N-MNIST and DVS-Gesture results from
logs/comparison_summary.csv, written by run.py. For each model it uses the
architecture's latest configuration (the last run of it, so a design changed
since then is not mixed in) and, of that, the run with the most samples; a
run on fewer samples than the whole test set, or a missing run (empty cells),
is reported with a warning. Of an architecture's hardware specifications,
cell precision, R_High/R_Low and accumulation (rows summed per read) are
read from it; the others stay empty unless set in OURS. Units: power mW, latency us and energy uJ per
inference, TOPS/W. A value of this work that is the best of its column
(lowest power, latency and energy, highest TOPS/W) is set in bold (BOLD).

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

# ============================================================================
# THE TABLE
# ============================================================================
# Literature rows. specs: Tech. (nm), Supply (V), Storage device,
# Cell/precision, Bit-cell, R_High/R_Low (kOhm), Sensing mode, Accumulation
# (LaTeX allowed, e.g. \textbf{...}); "nmnist" / "gesture" = (power mW,
# latency us, energy uJ, TOPS/W); None = empty cells.
LITERATURE = [
    {"work": r"ISSCC'22~\cite{khwa_40-nm_2022}",
     "specs": ["40", "0.9", "PCM", "1", "1T1R", "--", "Voltage", "8"],
     "nmnist": (396, 22.9, 9.1, 25.4), "gesture": (1970.4, 88.5, 174.3, 25.6)},
    {"work": r"AICAS'23~\cite{singh_1151_2023}",
     "specs": ["40", "1.1", "Resistive", "1", "1T1R", "200/2", "Current", "64"],
     "nmnist": (531.6, 9.8, 5.2, 44.4), "gesture": (1040.2, 37.7, 39.2, 113.8)},
    {"work": r"Venue'YY~[X]", "comment": "placeholder work 1"},
    {"work": r"Venue'YY~[Y]", "comment": "placeholder work 2"},
]

# Your architectures follow, one row each, named after the architecture.
# Per architecture you may set its row's name ("work") and any specification
# the architecture does not tell: "tech", "supply", "device", "bitcell",
# "sensing" (and override the derived "cell", "r_ratio", "accumulation").
OURS = {
    # "c3cim_op_xbar": {"work": r"\textbf{This work}", "tech": "40", "supply": "1.1",
    #                   "device": "Resistive", "bitcell": r"\textbf{2T1R}", "sensing": "Voltage"},
}
SPEC_KEYS = ("tech", "supply", "device", "cell", "bitcell", "r_ratio", "sensing", "accumulation")
MODELS = ("nmnist", "gesture")
FULL_TEST_SET = {"nmnist": 10000, "gesture": 264}
# Bold the best value of each result column: "results" = only in your
# architectures' rows, "all" = in any row, None = never.
BOLD = "results"
# ============================================================================

CAPTION = "Hardware comparison of deploying SNN models trained on the N-MNIST and IBM-Gesture datasets"
LABEL = "table:comp"
# The best value per column: min for power, latency, energy; max for TOPS/W.
BEST = (min, min, min, max)

HEADER = r"""\providecolor{nmband}{gray}{0.92}  %% N-MNIST column shade (kept if already defined)
\begin{table*}[t]
\centering
\footnotesize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.25}
\setlength{\aboverulesep}{0pt}
\setlength{\belowrulesep}{0pt}
\caption{%(caption)s}
\label{%(label)s}
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{@{}l cccccccc *{4}{>{\columncolor{nmband}}c} cccc@{}}
\toprule
 & \multicolumn{8}{c}{\multirow{2}{*}{\textbf{Hardware specifications}}}
 & \multicolumn{8}{c}{\textbf{Evaluation results}} \\
\cmidrule(l){10-17}
 & \multicolumn{8}{c}{}
 & \multicolumn{4}{c}{\textit{N-MNIST}}
 & \multicolumn{4}{c}{\textit{IBM DVS128 Gesture}} \\
\cmidrule(r){2-9}\cmidrule(lr){10-13}\cmidrule(l){14-17}
\textbf{Work}
 & \makecell{Tech.\\(nm)} & \makecell{Supply\\(V)} & \makecell{Storage\\device} & \makecell{Cell/\\precision}
 & Bit-cell & \makecell{$R_\mathrm{High}/R_\mathrm{Low}$\\(k$\Omega$)} & \makecell{Sensing\\mode} & \makecell{Accumu-\\lation}
 & \makecell{Power\\(mW)} & \makecell{Latency\\($\mu$s)} & \makecell{Energy\\($\mu$J)} & TOPS/W
 & \makecell{Power\\(mW)} & \makecell{Latency\\($\mu$s)} & \makecell{Energy\\($\mu$J)} & TOPS/W \\
\midrule
"""
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


def architecture_rows(architectures, ours=None):
    """One table row per Architecture, named after it (LaTeX-escaped), with
    derived specifications, overridden or completed by ours[name]."""
    rows = []
    for i, arch in enumerate(architectures):
        extra = dict((ours or {}).get(arch.name, {}))
        unknown = set(extra) - set(SPEC_KEYS) - {"work"}
        if unknown:
            raise ValueError(f"OURS[{arch.name!r}]: unknown keys {sorted(unknown)}; use 'work' "
                             f"or {', '.join(SPEC_KEYS)}")
        specs = {**derived_specs(arch), **extra}
        rows.append({"work": extra.get("work", arch.name.replace("_", r"\_")),
                     "specs": [specs.get(key, "") for key in SPEC_KEYS],
                     "architecture": arch.name, "midrule": i == 0})
    return rows


def row_values(row, runs, warn=print):
    """(nmnist values, gesture values) of a row; values are 4 floats or None."""
    architecture = row.get("architecture")
    values = []
    for model in MODELS:
        name = architecture.get(model) if isinstance(architecture, dict) else architecture
        values.append(pick_run(runs, model, name, warn) if name else row.get(model))
    return values


def build_table(rows, runs, bold=BOLD, warn=print):
    """The LaTeX table for `rows` with results from `runs` (load_results)."""
    values = [row_values(row, runs, warn) for row in rows]
    # Best per result column (as printed, so ties in the table are all bold).
    best = {}
    for m in range(len(MODELS)):
        for k, pick in enumerate(BEST):
            column = [float(number(v[m][k])) for v in values if v[m] is not None]
            if len(column) > 1:
                best[m, k] = pick(column)
    lines = [HEADER % {"caption": CAPTION, "label": LABEL}]
    for row, row_vals in zip(rows, values):
        if row.get("midrule"):
            lines.append("\\midrule\n")
        comment = f" % {row['comment']}" if row.get("comment") else ""
        if not row.get("specs") and all(v is None for v in row_vals):   # an empty row
            lines.append(f"{row['work']}{comment}\n{' &' * 16} \\\\\n")
            continue
        specs = list(row.get("specs") or []) + [""] * (8 - len(row.get("specs") or []))
        bold_row = bold == "all" or (bold == "results" and row.get("architecture"))
        cells = []
        for m, v in enumerate(row_vals):
            for k in range(4):
                if v is None:
                    cells.append("")
                    continue
                text = number(v[k])
                if bold_row and best.get((m, k)) == float(text):
                    text = f"\\textbf{{{text}}}"
                cells.append(text)
        lines.append(f"{row['work']}{comment}\n"
                     f" & {' & '.join(specs)}\n"
                     f" & {' & '.join(cells[:4])}\n"
                     f" & {' & '.join(cells[4:])} \\\\\n")
    lines.append(FOOTER)
    # Empty cells as "& &", as written by hand; no trailing spaces.
    table = re.sub(r"(?<=&) +(?=&|\\\\)", " ", "".join(lines))
    return re.sub(r" +\n", "\n", table)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--logs", default=os.path.join(ROOT, "logs"),
                        help="folder of run.py's results (comparison_summary.csv)")
    parser.add_argument("--out", help="output .tex (default: <logs>/comparison_table.tex)")
    parser.add_argument("--architectures", nargs="+",
                        help="names of run.py architectures to include (default: ARCHITECTURES)")
    args = parser.parse_args()
    sys.path.insert(0, ROOT)
    import run
    architectures = run.ARCHITECTURES
    if args.architectures:
        known = {a.name: a for name in dir(run) for a in [getattr(run, name)]
                 if hasattr(a, "crossbar") and hasattr(a, "mapping")}
        missing = [n for n in args.architectures if n not in known]
        if missing:
            parser.error(f"not defined in run.py: {missing}; defined: {sorted(known)}")
        architectures = [known[n] for n in args.architectures]
    runs = load_results(os.path.join(args.logs, "comparison_summary.csv"))
    rows = LITERATURE + architecture_rows(architectures, OURS)
    table = build_table(rows, runs, warn=lambda text: print(text, file=sys.stderr))
    out = args.out or os.path.join(args.logs, "comparison_table.tex")
    with open(out, "w", encoding="utf-8") as file:
        file.write(table)
    print(table, end="")
    print(f"saved {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
