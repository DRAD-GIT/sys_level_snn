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

    def test_red_best_and_bold_wins(self):
        runs = summary([["nmnist", "ours", "c", 10000, 1400, 93, 15.3, 161.7],
                        ["gesture", "ours", "c", 264, 25900, 359.6, 72, 172.3]])
        rows = [{"work": "A", "specs": ["40"], "nmnist": (396, 22.9, 9.1, 25.4),
                 "gesture": (1970.4, 88.5, 174.3, 25.6)},
                {"work": "Empty", "comment": "placeholder"},
                {"work": "Ours", "midrule": True, "specs": ["40"], "architecture": "ours",
                 "ours": True}]
        table = latex_table.build_table(rows, runs, warn=lambda _: None)
        # Ours beats A in power, energy and TOPS/W (bold, and best: red), not latency.
        self.assertIn(" & \\textcolor{red}{\\textbf{15.3}} & 93 & \\textcolor{red}{\\textbf{1.4}}"
                      " & \\textcolor{red}{\\textbf{161.7}}\n", table)
        self.assertIn(" & 396 & \\textcolor{red}{22.9} & 9.1 & 25.4\n", table)   # best, not ours
        self.assertIn("Empty % placeholder\n" + " &" * 16 + " \\\\\n", table)
        self.assertIn("\\midrule\nOurs\n", table)
        self.assertNotIn(" \n", table)
        self.assertNotIn("\\midrule\n\\midrule", table)

    def test_our_work_last_and_bold_only_against_others(self):
        import crossbars
        from hardware import Component, Mapping, compose

        def arch(name):
            return compose(name, Mapping(), [crossbars.conv_xbar(r_on=1e3, r_off=1e4),
                                             Component("lif", count="outputs", time_ns=2.0)])
        base, a, b = arch("base"), arch("a"), arch("b")
        rows = latex_table.architecture_rows([a, base, b], ourwork=[a, b])
        self.assertEqual([(r["work"], r["ours"], r["midrule"]) for r in rows],
                         [("base", False, True), ("a", True, True), ("b", True, False)])
        runs = summary([["nmnist", "base", "c", 10000, 3000, 10, 30, 20],
                        ["nmnist", "a", "c", 10000, 2000, 20, 20, 30],     # beats base in 3
                        ["nmnist", "b", "c", 10000, 1000, 30, 10, 40]])    # the best in 3
        table = latex_table.build_table(rows, runs, warn=lambda _: None)
        self.assertIn("base\n & & & & 1 & & 10/1 & & 64\n & 30 & \\textcolor{red}{10} & 3 & 20\n",
                      table)
        self.assertIn(" & \\textbf{20} & 20 & \\textbf{2} & \\textbf{30}\n", table)
        self.assertIn(" & \\textcolor{red}{\\textbf{10}} & 30 & \\textcolor{red}{\\textbf{1}}"
                      " & \\textcolor{red}{\\textbf{40}}\n", table)

    def test_one_row_per_architecture(self):
        import crossbars
        from hardware import Component, Mapping, compose
        rram = compose("rram_xbar", Mapping(), [
            crossbars.conv_xbar(r_on=20e3, r_off=200e3), Component("lif", count="outputs", time_ns=2.0)])
        c3cim = compose("c3cim_op_xbar", Mapping(), [
            crossbars.c3cim_xbar(r_on=2e3, r_off=20e3, active_rows=16),
            Component("lif", count="outputs", time_ns=2.0)],
            specs={"label": "This work", "tech": 40, "sensing": "Voltage"})
        rows = latex_table.architecture_rows([rram, c3cim])
        self.assertEqual([(r["work"], r["specs"], r["midrule"]) for r in rows],
                         [("rram\\_xbar", ["", "", "", "1", "", "200/20", "", "64"], True),
                          ("This work", ["40", "", "", "1", "", "20/2", "Voltage", "16"], False)])
        self.assertEqual(latex_table.LITERATURE, [])                 # none by default
        with self.assertRaisesRegex(ValueError, "unknown specs"):
            compose("x", Mapping(), [crossbars.conv_xbar(r_on=1e3, r_off=1e4)],
                    specs={"voltage": 1})
        # Given specs override the ones read from the crossbar.
        own = compose("own", Mapping(), [crossbars.conv_xbar(r_on=20e3, r_off=200e3)],
                      specs={"r_ratio": "200/20 (set)"})
        self.assertEqual(latex_table.architecture_rows([own])[0]["specs"][5], "200/20 (set)")
        runs = summary([["nmnist", "rram_xbar", "c", 10000, 1400, 93, 15.3, 161.7]])
        warnings = []
        table = latex_table.build_table(rows, runs, warn=warnings.append)
        self.assertIn("rram\\_xbar\n & & & & 1 & & 200/20 & & 64\n & 15.3 & 93 & 1.4 & 161.7\n"
                      " & & & & \\\\\n", table)
        self.assertEqual(len(warnings), 3)          # rram on gesture, c3cim on both


    def test_three_datasets(self):
        runs = summary([["nmnist", "ours", "c", 10000, 1400, 93, 15.3, 161.7],
                        ["cifar10", "ours", "c", 10000, 9000, 500, 18, 120]])
        rows = [{"work": "Ours", "specs": ["40"], "architecture": "ours", "ours": True},
                {"work": "Empty"}]
        models = ("nmnist", "gesture", "cifar10")
        table = latex_table.build_table(rows, runs, warn=lambda _: None, models=models)
        self.assertIn("CIFAR-10", table)
        self.assertIn("\\multicolumn{12}{c}{\\textbf{Evaluation results}}", table)
        self.assertIn(" & 15.3 & 93 & 1.4 & 161.7\n & & & &\n & 18 & 500 & 9 & 120 \\\\\n", table)
        self.assertIn("Empty\n" + " &" * 20 + " \\\\\n", table)


if __name__ == "__main__":
    unittest.main()
