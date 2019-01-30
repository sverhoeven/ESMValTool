"""Tests for the module :mod:`esmvaltool.diag_scripts.mlr.models`."""
import os

import mock
import pytest
import yaml

from esmvaltool.diag_scripts.mlr.models import MLRModel


# Load test configuration
with open(os.path.join(os.path.dirname(__file__), 'config.yml')) as file_:
    CONFIG = yaml.safe_load(file_)


@mock.patch('esmvaltool.diag_scripts.mlr.models.logger', autospec=True)
class TestMLRModel():
    """Tests for the base class."""

    args = CONFIG['args']
    kwargs = CONFIG['kwargs']

    def test_register_mlr_model(self, mock_logger):
        """Test registering subclass."""
        assert MLRModel._MODELS == {}

        @MLRModel.register_mlr_model('test_model')
        class MyMLRModel(MLRModel):
            """Subclass of `MLRModel`."""

            pass

        assert MLRModel._MODELS == {'test_model': MyMLRModel}
        mock_logger.debug.assert_called_once()
        MLRModel._MODELS = {}

    @mock.patch('esmvaltool.diag_scripts.mlr.models._load_mlr_models')
    @mock.patch('esmvaltool.diag_scripts.mlr.models.MLRModel.__init__',
                autospec=True)
    def test_create(self, mock_mlr_model_init, mock_load_mlr_models,
                    mock_logger):
        """Test creating subclasses."""
        # No subclasses
        assert MLRModel._MODELS == {}
        mock_mlr_model_init.return_value = None
        MLRModel.create('test_model', *self.args, **self.kwargs)
        mock_load_mlr_models.assert_called()
        mock_logger.error.assert_called()
        mock_mlr_model_init.assert_called_with(mock.ANY, *self.args,
                                               **self.kwargs)

        # Wrong subclass
        @MLRModel.register_mlr_model('test_model')
        class MyMLRModel(MLRModel):
            """Subclass of `MLRModel`."""

            pass
        MLRModel.create('another_test_model', *self.args, **self.kwargs)
        mock_load_mlr_models.assert_called()
        mock_logger.warning.assert_called()
        mock_mlr_model_init.assert_called_with(mock.ANY, *self.args,
                                               **self.kwargs)
        MLRModel._MODELS = {}

        # Right subclass
        @MLRModel.register_mlr_model('test_model')
        class MyMLRModel1(MLRModel):
            """Subclass of `MLRModel`."""

            pass
        MLRModel.create('test_model', *self.args, **self.kwargs)
        mock_load_mlr_models.assert_called()
        mock_logger.info.assert_called()
        mock_mlr_model_init.assert_called_with(mock.ANY, *self.args,
                                               **self.kwargs)
        MLRModel._MODELS = {}


# Tests for data processing

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
