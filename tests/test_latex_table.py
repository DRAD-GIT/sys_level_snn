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
        with self.assertRaisesRegex(ValueError, "no results"):
            latex_table.pick_run(runs, "nmnist", "ours")

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


if __name__ == "__main__":
    unittest.main()
