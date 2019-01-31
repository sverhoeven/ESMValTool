"""Generate config files which can be used as input for tests."""
import os

import yaml

from .test_read_input import SimplifiedMLRModel
from esmvaltool.diag_scripts.mlr.models import MLRModel


# BASIC_INFO = {
#     'standard_name':





@pytest.mark.parametrize('data', CONFIG['test_load_input_datasets'])
def test_load_input_datasets(data):
    """Test loading of input datasets."""
    cfg = data['cfg']
    output = data['output']
    mlr_model = SimplifiedMLRModel(cfg)

    # Load input datasets
    if isinstance(output, str):
        with pytest.raises(EXCEPTIONS[output]):
            mlr_model._load_input_datasets(**cfg.get('metadata', {}))
    else:
        mlr_model._load_input_datasets(**cfg.get('metadata', {}))
        assert mlr_model._datasets == output
