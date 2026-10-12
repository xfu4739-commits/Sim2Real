from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest

import torch


_TEST_SCRIPT = Path(__file__).resolve().parents[1] / 'test.py'
_SPEC = spec_from_file_location('evaluation_script', _TEST_SCRIPT)
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
initialize_model = _MODULE.initialize_model


class EvaluationGuardTest(unittest.TestCase):
    def test_evaluation_rejects_disabled_checkpoint_before_model_creation(self):
        cfg = {'checkpoint': {'checkpoint': False}}

        with self.assertRaisesRegex(ValueError, 'randomly initialized model'):
            initialize_model(cfg, torch.device('cpu'))

    def test_evaluation_rejects_missing_checkpoint_before_model_creation(self):
        cfg = {
            'checkpoint': {
                'checkpoint': True,
                'checkpoint_path': 'does-not-exist.pth',
            }
        }

        with self.assertRaisesRegex(FileNotFoundError, 'checkpoint not found'):
            initialize_model(cfg, torch.device('cpu'))


if __name__ == '__main__':
    unittest.main()
