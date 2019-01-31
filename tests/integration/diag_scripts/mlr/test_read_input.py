"""Tests for reading data."""
import os

import mock
import pytest
import yaml

from esmvaltool.diag_scripts.mlr.models import MLRModel


EXCEPTIONS = {
    'ValueError': ValueError,
    'TypeError': TypeError,
}


class SimplifiedMLRModel(MLRModel):
    """Test class to avoid calling the base class `__init__` method."""

    def __init__(self, cfg):
        """Very simplified constructor of the base class."""
        self._cfg = cfg
        self._data = {}
        self._datasets = {}
        self.classes = {}


with open(os.path.join(os.path.dirname(__file__), 'configs',
                       'test_load_input_datasets.yml')) as file_:
    CONFIG = yaml.safe_load(file_)

@pytest.mark.parametrize('data', CONFIG['test_load_input_datasets'])
def test_load_input_datasets(data):
    """Test loading of input datasets."""
    cfg = data['cfg']
    output = data['output']
    mlr_model = SimplifiedMLRModel(cfg)

    # Load input datasets
    if 'EXCEPTION' in output:
        exc = output['EXCEPTION']
        with pytest.raises(EXCEPTIONS[exc['type']]) as exc_info:
            mlr_model._load_input_datasets(**cfg.get('metadata', {}))
        assert exc.get('value', '') in exc_info.value
    else:
        mlr_model._load_input_datasets(**cfg.get('metadata', {}))
        assert mlr_model._datasets == output
