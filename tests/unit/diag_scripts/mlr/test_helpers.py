"""Unit tests for the module :mod:`esmvaltool.diag_scripts.mlr`."""
import mock

from esmvaltool.diag_scripts.mlr import _load_mlr_models


@mock.patch('esmvaltool.diag_scripts.mlr.importlib', autospec=True)
@mock.patch('esmvaltool.diag_scripts.mlr.os.listdir', autospec=True)
def test_load_mlr_models(mock_listdir, mock_importlib):
    """Test for loading mlr models."""
    models = ['__pycache__', 'test.py', '42.py', '__init__.py']
    mock_listdir.return_value = models
    _load_mlr_models()
    modules = ['esmvaltool.diag_scripts.mlr.models.{}'.format(mod) for mod in
               ['test', '42']]
    calls = [mock.call(module) for module in modules]
    mock_importlib.import_module.assert_has_calls(calls)
