"""Convenience functions for MLR diagnostics."""

import logging
import os

from esmvaltool.diag_scripts.shared import io
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

logger = logging.getLogger(os.path.basename(__file__))

NECESSARY_KEYS = io.NECESSARY_KEYS + [
    'tag',
    'var_type',
]
VAR_TYPES = [
    'feature',
    'label',
    'prediction_input',
    'prediction_output',
]


class AdvancedPipeline(Pipeline):
    """Expand `sklearn.pipeline.Pipeline` class."""

    def transform_only(self, x_data):
        """Only perform `transform` steps of Pipeline."""
        for (_, transformer) in self.steps[:-1]:
            x_data = transformer.transform(x_data)
        return x_data


class AdvancedTransformedTargetRegressor(TransformedTargetRegressor):
    """Expand `sklearn.compose.TransformedTargetRegressor` class."""

    def predict(self, x_data, return_std=False, return_cov=False):
        """Expand `predict()` method."""
        if return_std and return_cov:
            logger.warning(
                "Cannot return standard deviation and full covariance matrix "
                "for prediction, returning only standard deviation")
            return_cov = False
        y_pred = super().predict(x_data)
        scale = self.transformer_.scale_
        if return_std:
            (_, y_std) = self.regressor_.predict(x_data, return_std=True)
            if scale is not None:
                y_std *= scale
            return (y_pred, y_std)
        if return_cov:
            (_, y_cov) = self.regressor_.predict(x_data, return_cov=True)
            if scale is not None:
                y_cov *= scale**2
            return (y_pred, y_cov)
        return y_pred


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
