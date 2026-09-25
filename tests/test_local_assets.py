"""Checkpoints, YAMLs and dataset paths resolve inside this repository."""
import os
import subprocess
import sys
import unittest
from pathlib import Path

from hardware import C3HardwareConfig, ConvHardwareConfig

ROOT = Path(__file__).resolve().parents[1]


class LocalAssetTests(unittest.TestCase):
    def test_load_pretrained_from_any_working_directory(self):
        # Run in a fresh interpreter from /tmp so the legacy pickle aliases
        # and repo-relative paths are exercised without test-process state.
        code = r'''
import os
import sys
root = sys.argv[1]
sys.path.insert(0, root)
import models
import slayerSNN as snn
for name in ("nmnist", "gesture"):
    spec = models.get_spec(name)
    for relative in (spec.checkpoint, spec.params_yaml):
        assert os.path.isfile(spec.path(relative)), relative
    model = models.load_pretrained(spec)
    module = sys.modules[type(model).__module__]
    assert module.__name__ == "models." + name, module.__name__
    assert os.path.commonpath([root, module.__file__]) == root
    for layer, _ in spec.layers:
        assert getattr(model, layer).weight is not None
    params = snn.params(spec.path(spec.params_yaml))
    for field in ("dir_test", "list_test"):
        path = params["training"]["path"][field]
        assert not os.path.isabs(path) and path.startswith("datasets/"), path
'''
        result = subprocess.run([sys.executable, '-c', code, str(ROOT)],
                                cwd='/tmp', capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_run_script_hardware_matches_original_examples(self):
        # run.py spells out the values of the former example_conv.json /
        # example_c3.json: class defaults with eight active rows.
        import run
        self.assertEqual(run.CONVENTIONAL, ConvHardwareConfig(active_rows=8))
        self.assertEqual(run.C3CIM, C3HardwareConfig(active_rows=8))

    def test_dataset_placeholders_exist(self):
        for name in ("N-MNIST", "DVS_Gesture"):
            self.assertTrue((ROOT / "datasets" / name / "README.md").is_file())


if __name__ == '__main__':
    unittest.main()
