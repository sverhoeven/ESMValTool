"""Convenience functions for MLR diagnostics."""

import logging
import os

import numpy as np
from sklearn.impute import SimpleImputer

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


class Imputer():
    """Expand `sklearn.impute.SimpleImputer` class to remove missing data."""

    def __init__(self, *args, **kwargs):
        """Initialize imputer."""
        self.imputer = SimpleImputer(*args, **kwargs)
        self.missing_values = self.imputer.missing_values
        self.strategy = self.imputer.strategy
        self.fill_value = self.imputer.fill_value
        self.statistics_ = None
        self.mask_ = None

    def fit(self, x_data):
        """Fit imputer."""
        self._check_array_types(x_data)
        if self.strategy == 'remove':
            return
        self.imputer.fit(x_data.filled(np.nan))

    def transform(self, x_data, y_data=None):
        """Transform data."""
        self._check_array_types(x_data, y_data)
        if self.strategy == 'remove':
            if x_data.mask.shape == ():
                mask = np.full(x_data.shape[0], False)
            else:
                mask = np.any(x_data.mask, axis=1)
            new_x_data = x_data.filled()[~mask]
            new_y_data = None if y_data is None else y_data.filled()[~mask]
            n_imputes = x_data.shape[0] - new_x_data.shape[0]
            self.mask_ = mask
        else:
            new_x_data = self.imputer.transform(x_data.filled(np.nan))
            new_y_data = None if y_data is None else y_data.filled()
            n_imputes = np.count_nonzero(x_data != new_x_data)
            self.statistics_ = self.imputer.statistics_
        return (new_x_data, new_y_data, n_imputes)

    def _check_array_types(self, x_data, y_data=None):
        """Check types of input arrays."""
        if y_data is None:
            arrays_to_check = (x_data, )
        else:
            arrays_to_check = (x_data, y_data)
        for array in arrays_to_check:
            if not np.ma.isMaskedArray(array):
                raise TypeError("Class {} only accepts masked arrays".format(
                    self.__class__.__name__))


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
