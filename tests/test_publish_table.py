"""run.py --table: the table is written into the paper repository, committed
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
        out = run.publish_table(run.OURWORK, ["nmnist", "gesture", "cifar10_thermo"],
                                repo=self.paper, logs_dir=self.logs)
        self.assertEqual(out, os.path.join(self.paper, run.PAPER_TABLE))
        with open(out) as file:
            table = file.read()
        self.assertIn("CIFAR-10", table)
        self.assertNotIn("IBM DVS128 Gesture", table)        # no gesture results: no columns
        pushed = git(self.remote, "show", "main:Chapters/comparison_table.tex")
        self.assertEqual(pushed, table)
        neuromorphic = git(self.remote, "show", "main:Chapters/neuromorphic_table.tex")
        self.assertIn(r"\makecell{96.80\% \\ / 88.28\%}", neuromorphic)   # our accuracies
        self.assertIn(r"/ 39.0\,$\mu$J}", neuromorphic)                     # our CIFAR-10 energy
        self.assertIn("DS-CIM", neuromorphic)
        commits = git(self.remote, "rev-list", "--count", "main")
        run.publish_table(run.OURWORK, ["nmnist", "cifar10_thermo"], repo=self.paper,
                          logs_dir=self.logs)                # unchanged: no new commit
        self.assertEqual(git(self.remote, "rev-list", "--count", "main"), commits)

    def test_missing_paper_repository(self):
        with self.assertRaisesRegex(SystemExit, "no git clone"):
            run.publish_table(run.OURWORK, ["nmnist"], repo=os.path.join(self.tmp.name, "none"),
                              logs_dir=self.logs)


if __name__ == "__main__":
    unittest.main()
