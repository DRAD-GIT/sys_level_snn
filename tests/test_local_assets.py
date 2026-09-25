"""Ensure checkpoints/configs can load using only local model assets."""
import os
import subprocess
import sys
import unittest
from pathlib import Path


class LocalAssetTests(unittest.TestCase):
    def test_load_assets_without_legacy_directory_access(self):
        root = Path(__file__).resolve().parents[1]
        code = r'''
import os
import sys
root = sys.argv[1]
sys.path.insert(0, root)
def reject_legacy(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        if 'Final SNN script' in os.fsdecode(args[0]):
            raise AssertionError('Unexpected legacy asset access: ' + os.fsdecode(args[0]))
sys.addaudithook(reject_legacy)
import combined_infer as infer
import torch
import slayerSNN as snn
assert infer.asset_dir == os.path.join(root, 'models')
for directory, checkpoint, yaml in (
    ('NMNIST_SNN', 'nmnist-lenet_net2.pt', 'nmnist.yaml'),
    ('Gesture_SNN', 'gesture-do_net4.pt', 'gesture.yaml'),
):
    model = torch.load(os.path.join(infer.asset_dir, directory, checkpoint),
                       map_location='cpu', weights_only=False)
    assert model is not None
    params = snn.params(os.path.join(infer.asset_dir, directory, yaml))
    for field in ('dir_test', 'list_test'):
        path = os.path.join(root, params['training']['path'][field])
        assert 'Final SNN script' not in path
        assert not os.path.isabs(params['training']['path'][field])
    module = sys.modules[type(model).__module__]
    assert os.path.commonpath([root, module.__file__]) == root
'''
        result = subprocess.run([sys.executable, '-c', code, str(root)],
                                cwd='/tmp', capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
