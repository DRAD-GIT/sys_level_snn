"""Checkpoints and YAMLs resolve inside this repository; datasets are found
from a user-given folder."""
import os
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
        assert not os.path.isabs(path) and not path.startswith("datasets"), path
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

    def test_dataset_folder_lookup(self):
        import tempfile
        import models
        nmnist, gesture = models.get_spec("nmnist"), models.get_spec("gesture")
        with tempfile.TemporaryDirectory() as data:
            for name in ("N-MNIST_v1", "DVS_Gesture", "other"):
                os.mkdir(os.path.join(data, name))
            self.assertEqual(models.find_dataset(nmnist, data), os.path.join(data, "N-MNIST_v1"))
            self.assertEqual(models.find_dataset(gesture, data), os.path.join(data, "DVS_Gesture"))
            direct = os.path.join(data, "N-MNIST_v1")
            self.assertEqual(models.find_dataset(nmnist, direct), direct)   # the folder itself
            os.mkdir(os.path.join(data, "gesture_copy"))
            with self.assertRaisesRegex(FileNotFoundError, "expected one folder"):
                models.find_dataset(gesture, data)                          # ambiguous
            with self.assertRaisesRegex(FileNotFoundError, "found none"):
                models.find_dataset(nmnist, os.path.join(data, "other"))
        with self.assertRaisesRegex(ValueError, "DATASET_DIR"):
            models.find_dataset(nmnist, None)


if __name__ == '__main__':
    unittest.main()
