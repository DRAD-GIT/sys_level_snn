"""tools/latex_table.py: which run fills a row, units, formatting, bold."""
import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import latex_table  # noqa: E402

FIELDS = ["model", "architecture", "configuration", "samples", "energy", "latency", "power",
          "tops_per_w"]


def summary(rows):
    folder = tempfile.mkdtemp()
    path = os.path.join(folder, "comparison_summary.csv")
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    return latex_table.load_results(path)


class LatexTableTests(unittest.TestCase):
    def test_latest_configuration_and_most_samples(self):
        runs = summary([
            ["nmnist", "ours", "old", 10000, 9000, 50, 1, 10],     # an earlier design
            ["nmnist", "ours", "new", 10000, 1400, 93, 15.3, 161.7],
            ["nmnist", "ours", "new", 100, 1000, 93, 11, 200],      # a later short check
        ])
        warnings = []
        values = latex_table.pick_run(runs, "nmnist", "ours", warnings.append)
        self.assertEqual(values, (15.3, 93.0, 1.4, 161.7))         # energy nJ -> uJ
        self.assertEqual(warnings, [])
        runs = summary([["gesture", "ours", "a", 20, 1, 1, 1, 1]])
        latex_table.pick_run(runs, "gesture", "ours", warnings.append)
        self.assertIn("20 of 264", warnings[0])
        self.assertIsNone(latex_table.pick_run(runs, "nmnist", "ours", warnings.append))
        self.assertIn("no results for ours on nmnist", warnings[-1])

    def test_number_format(self):
        self.assertEqual([latex_table.number(v) for v in (396, 22.94, 93.0, 0.0234, 0)],
                         ["396", "22.9", "93", "0.023", "0"])

    def test_rows_and_bold(self):
        runs = summary([["nmnist", "ours", "c", 10000, 1400, 93, 15.3, 161.7],
                        ["gesture", "ours", "c", 264, 25900, 359.6, 72, 172.3]])
        rows = [{"work": "A", "specs": ["40"], "nmnist": (396, 22.9, 9.1, 25.4),
                 "gesture": (1970.4, 88.5, 174.3, 25.6)},
                {"work": "Empty", "comment": "placeholder"},
                {"work": "Ours", "midrule": True, "specs": ["40"], "architecture": "ours"}]
        table = latex_table.build_table(rows, runs, warn=lambda _: None)
        self.assertIn(" & \\textbf{15.3} & 93 & \\textbf{1.4} & \\textbf{161.7}\n", table)
        self.assertIn(" & 396 & \\textbf{22.9}", latex_table.build_table(rows, runs, bold="all",
                                                                        warn=lambda _: None))
        self.assertIn(" & 396 & 22.9 & 9.1 & 25.4\n", table)       # literature: never bold
        self.assertIn("Empty % placeholder\n" + " &" * 16 + " \\\\\n", table)
        self.assertIn("\\midrule\nOurs\n", table)
        self.assertNotIn(" \n", table)


    def test_one_row_per_architecture(self):
        import crossbars
        from hardware import Component, Mapping, compose
        rram = compose("rram_xbar", Mapping(), [
            crossbars.conv_xbar(r_on=20e3, r_off=200e3), Component("lif", count="outputs", time_ns=2.0)])
        c3cim = compose("c3cim_op_xbar", Mapping(), [
            crossbars.c3cim_xbar(r_on=2e3, r_off=20e3, active_rows=16),
            Component("lif", count="outputs", time_ns=2.0)])
        rows = latex_table.architecture_rows(
            [rram, c3cim], {"c3cim_op_xbar": {"work": "This work", "tech": "40", "sensing": "Voltage"}})
        self.assertEqual([(r["work"], r["specs"], r["midrule"]) for r in rows],
                         [("rram\\_xbar", ["", "", "", "1", "", "200/20", "", "64"], True),
                          ("This work", ["40", "", "", "1", "", "20/2", "Voltage", "16"], False)])
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            latex_table.architecture_rows([rram], {"rram_xbar": {"voltage": "1"}})
        runs = summary([["nmnist", "rram_xbar", "c", 10000, 1400, 93, 15.3, 161.7]])
        warnings = []
        table = latex_table.build_table(rows, runs, warn=warnings.append)
        self.assertIn("rram\\_xbar\n & & & & 1 & & 200/20 & & 64\n & 15.3 & 93 & 1.4 & 161.7\n"
                      " & & & & \\\\\n", table)
        self.assertEqual(len(warnings), 3)          # rram on gesture, c3cim on both


if __name__ == "__main__":
    unittest.main()
