"""Checkpoints, YAMLs and dataset paths resolve inside this repository."""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class LocalAssetTests(unittest.TestCase):
    def test_load_pretrained_from_any_working_directory(self):
        # Fresh interpreter in /tmp: repo-relative paths must not depend on
        # the working directory, and nothing may need slayerSNN.
        code = r'''
import os
import sys
root = sys.argv[1]
sys.path.insert(0, root)
import models
for name in ("nmnist", "gesture"):
    spec = models.get_spec(name)
    for relative in (spec.checkpoint, spec.params_yaml):
        assert os.path.isfile(spec.path(relative)), relative
    model = models.load_pretrained(spec)
    for layer in spec.layers:
        assert getattr(model, layer).weight is not None
    params = models.load_params(spec.path(spec.params_yaml))
    for field in ("dir_test", "list_test"):
        path = params["training"]["path"][field]
        assert not os.path.isabs(path) and path.startswith("datasets/"), path
assert "slayerSNN" not in sys.modules
'''
        result = subprocess.run([sys.executable, '-c', code, str(ROOT)],
                                cwd='/tmp', capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_run_script_settings(self):
        import run
        from evaluation.report import METRIC_NAMES
        names = [arch.name for arch in run.ARCHITECTURES]
        self.assertEqual(len(names), len(set(names)))
        self.assertLessEqual(set(run.METRICS), set(METRIC_NAMES))

    def test_dataset_placeholders_exist(self):
        for name in ("N-MNIST", "DVS_Gesture"):
            self.assertTrue((ROOT / "datasets" / name / "README.md").is_file())


if __name__ == '__main__':
    unittest.main()
