"""run.py --table: the neuromorphic comparison table is written into the paper repository, committed
and pushed, and an unchanged table makes no commit."""
import csv
import os
import subprocess
import tempfile
import unittest

import run


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout


class PublishTableTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.tmp.name
        self.remote = os.path.join(root, "remote.git")
        self.paper = os.path.join(root, "paper")
        git(root, "init", "-q", "--bare", "-b", "main", self.remote)
        git(root, "clone", "-q", self.remote, self.paper)
        for key, value in (("user.name", "Test"), ("user.email", "test@example.com")):
            git(self.paper, "config", key, value)
        with open(os.path.join(self.paper, "main.tex"), "w") as file:
            file.write("paper\n")
        git(self.paper, "add", "main.tex")
        git(self.paper, "commit", "-q", "-m", "start")
        git(self.paper, "push", "-q", "-u", "origin", "main")
        self.logs = os.path.join(root, "logs")
        os.makedirs(self.logs)
        fields = ["model", "architecture", "configuration", "samples", "accuracy", "energy",
                  "latency", "power", "tops_per_w"]
        with open(os.path.join(self.logs, "comparison_summary.csv"), "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(fields)
            for arch in run.OURWORK:
                writer.writerow(["nmnist", arch.name, "c", 10000, 96.8, 8709, 52.5, 165.9, 52.35])
                writer.writerow(["cifar10_thermo", arch.name, "c", 10000, 88.28, 38990, 2.52, 15470,
                                 67.77])

    def tearDown(self):
        self.tmp.cleanup()

    def test_written_committed_and_pushed_once(self):
        out = run.publish_table(repo=self.paper, logs_dir=self.logs)
        self.assertEqual(out, os.path.join(self.paper, run.PAPER_TABLE))
        with open(out) as file:
            table = file.read()
        pushed = git(self.remote, "show", "main:Chapters/neuromorphic_table.tex")
        self.assertEqual(pushed, table)
        self.assertIn(r"\makecell{N-MNIST \\ \phantom{0} \\ CIFAR-10}", table)   # aligned lines
        self.assertIn(r"\textbf{96.80\%}", table)                      # our accuracy wins
        self.assertIn(r"\multicolumn{2}{c}{\makecell{\textbf{96.80\%}", table)  # one accuracy cell
        self.assertNotIn(r"\dagger", table)                            # no input footnotes
        self.assertIn(r"39.0\,$\mu$J", table)                          # our CIFAR-10 energy
        self.assertNotIn(r"\textbf{Power}", table)                     # no power row
        self.assertIn(r"$^{b}$", table)                                 # TOPS/W from our operations
        self.assertIn("DS-CIM", table)                                  # published works
        self.assertNotIn("comparison_table.tex", git(self.remote, "ls-tree", "-r", "main"))
        commits = git(self.remote, "rev-list", "--count", "main")
        run.publish_table(repo=self.paper, logs_dir=self.logs)          # unchanged: no new commit
        self.assertEqual(git(self.remote, "rev-list", "--count", "main"), commits)

    def test_missing_paper_repository(self):
        with self.assertRaisesRegex(SystemExit, "no git clone"):
            run.publish_table(repo=os.path.join(self.tmp.name, "none"), logs_dir=self.logs)


if __name__ == "__main__":
    unittest.main()


class DeepScaleTests(unittest.TestCase):
    def test_factors_match_the_tool(self):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        "tools"))
        import neuromorphic_table as nt
        # The paper's example: 45 -> 32 nm power factor 1.238 (100 mW -> 80.775 mW).
        self.assertAlmostEqual(nt.scale("power", 100.0, 45, target=32), 100 / (0.78 / 0.63))
        self.assertAlmostEqual(0.78 / 0.63, 1.238, places=3)
        self.assertAlmostEqual(nt.scale("energy", 21740, 65), 21740 * 0.55)        # Han et al.
        self.assertAlmostEqual(nt.scale("tops_per_w", 14.12, 65), 14.12 / 0.55)
        self.assertEqual(nt.scale("energy", 5.0, 40), 5.0)                          # same node
        self.assertIsNone(nt.scale("energy", 1780, 180))                            # out of range
        names = [w["name"] for w in sorted(nt.WORKS, key=lambda w: w["date"])]
        self.assertEqual(names, ["SPOON", "ReckOn", "Dorzhigulov et al.", "DS-CIM", "ANP-I"])  # by date
        self.assertEqual([w["name"] for w in nt.SET_ASIDE], ["Han et al."])
        self.assertAlmostEqual(nt.scale("energy", 181000, 130), 181000 * 0.55 / 2.52)
        self.assertNotIn("TrueNorth", names)
