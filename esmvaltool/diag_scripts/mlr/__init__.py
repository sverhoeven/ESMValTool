"""Convenience functions for MLR diagnostics."""

import importlib
import logging
import os

from esmvaltool.diag_scripts.shared import io

logger = logging.getLogger(os.path.basename(__file__))

NECESSARY_KEYS = io.NECESSARY_KEYS + [
    'tag',
    'var_type',
]
VAR_TYPES = [
    'feature',
    'label',
    'prediction_input',
]


def datasets_have_mlr_attributes(datasets, log_level='debug', mode=None):
    """Check (MLR) attributes of `datasets`.

    Parameters
    ----------
    datasets : list of dict
        Datasets to check.
    log_level : str, optional (default: 'debug')
        Verbosity level of the logger.
    mode : str, optional (default: None)
        Checking mode, possible values: `'only_missing'` (only check if
        attributes are missing), `'only_var_type'` (check only `var_type`) or
        `None` (check both).

    Returns
    -------
    bool
        `True` if all required attributes are available, `False` if not.

    """
    output = True
    for dataset in datasets:
        if mode != 'only_var_type':
            for key in NECESSARY_KEYS:
                if key not in dataset:
                    getattr(logger, log_level)(
                        "Dataset %s does not have necessary (MLR) attribute "
                        "'%s'", dataset, key)
                    output = False
        if mode != 'only_missing' and dataset.get('var_type') not in VAR_TYPES:
            getattr(logger, log_level)(
                "Dataset %s has invalid var_type '%s', must be one of %s",
                dataset, dataset.get('var_type'), VAR_TYPES)
            output = False
    return output


def write_cube(cube, attributes, path):
    """Write cube with all necessary information for MLR models.

    Parameters
    ----------
    cube : iris.cube.Cube
        Cube which should be written.
    attributes : dict
        Attributes for the cube (needed for MLR models).
    path : str
        Path to the new file.
    cfg : dict
        Diagnostic script configuration.

    """
    if not datasets_have_mlr_attributes([attributes], log_level='warning'):
        logger.warning("Cannot write %s", path)
        return
    io.metadata_to_netcdf(cube, attributes)


def _load_mlr_models():
    """Load MLR models from :mod:`esmvaltool.diag_scripts.mlr.models`."""
    current_path = os.path.dirname(os.path.realpath(__file__))
    models_path = os.path.join(current_path, 'models')
    for (root, _, model_files) in os.walk(models_path):
        for model_file in model_files:
            rel_path = ('' if root == models_path else os.path.relpath(
                root, models_path))
            module = os.path.join(rel_path, os.path.splitext(model_file)[0])
            try:
                importlib.import_module(
                    'esmvaltool.diag_scripts.mlr.models.{}'.format(
                        module.replace(os.sep, '.')))
            except ImportError:
                pass
