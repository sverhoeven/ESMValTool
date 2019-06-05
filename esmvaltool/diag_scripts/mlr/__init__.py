"""Convenience functions for MLR diagnostics."""

import logging
import os

from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

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
    'prediction_output',
]


class AdvancedPipeline(Pipeline):
    """Expand `sklearn.pipeline.Pipeline` class."""

    def fit_transformers_only(self, x_data, y_data, **fit_kwargs):
        """Fit only `transform` steps of Pipeline."""
        transformers_kwargs = {}
        for (param_name, param_val) in fit_kwargs.items():
            step = param_name.split('__')[0]
            if step in self.steps[:-1]:
                transformers_kwargs[param_name] = param_val
        return self._fit(x_data, y_data, **transformers_kwargs)

    def transform_only(self, x_data):
        """Only perform `transform` steps of Pipeline."""
        for (_, transformer) in self.steps[:-1]:
            x_data = transformer.transform(x_data)
        return x_data


class AdvancedTransformedTargetRegressor(TransformedTargetRegressor):
    """Expand `sklearn.compose.TransformedTargetRegressor` class."""

    def fit(self, x_data, y_data, **fit_kwargs):
        """Expand `fit()` method to accept kwargs."""
        y_data = check_array(y_data,
                             accept_sparse=False,
                             force_all_finite=True,
                             ensure_2d=False,
                             dtype='numeric')
        self._training_dim = y_data.ndim

        # Process kwargs
        (_, regressor_kwargs) = self._get_fit_kwargs(fit_kwargs)

        # Transformers are designed to modify X which is 2D, modify y_data
        # FIXME: Transformer does NOT use transformer_kwargs
        if y_data.ndim == 1:
            y_2d = y_data.reshape(-1, 1)
        else:
            y_2d = y_data
        self._fit_transformer(y_2d)

        # Transform y and convert back to 1d array if needed
        y_trans = self.transformer_.transform(y_2d)
        if y_trans.ndim == 2 and y_trans.shape[1] == 1:
            y_trans = y_trans.squeeze(axis=1)

        # Perform linear regression if regressor is not given
        if self.regressor is None:
            self.regressor_ = LinearRegression()
        else:
            self.regressor_ = clone(self.regressor)

        # Fit regressor with kwargs
        self.regressor_.fit(x_data, y_trans, **regressor_kwargs)
        return self

    def fit_transformer_only(self, y_data, **fit_kwargs):
        """Fit only `transformer` step."""
        y_data = check_array(y_data,
                             accept_sparse=False,
                             force_all_finite=True,
                             ensure_2d=False,
                             dtype='numeric')
        self._training_dim = y_data.ndim

        # Process kwargs
        (_, _) = self._get_fit_kwargs(fit_kwargs, verbose=False)

        # Transformers are designed to modify X which is 2D, modify y_data
        # FIXME: Transformer does NOT use transformer_kwargs
        if y_data.ndim == 1:
            y_2d = y_data.reshape(-1, 1)
        else:
            y_2d = y_data
        self._fit_transformer(y_2d)

    def predict(self, x_data, always_return_1d=True, **predict_kwargs):
        """Expand `predict()` method to accept kwargs."""
        predict_kwargs = dict(predict_kwargs)
        check_is_fitted(self, "regressor_")

        # Kwargs for returning variance or covariance
        return_var = predict_kwargs.pop('return_var', False)
        return_cov = predict_kwargs.pop('return_cov', False)
        if return_var and return_cov:
            logger.warning(
                "Cannot return variance and full covariance matrix for "
                "prediction, returning only variance")
            return_cov = False

        # Main prediction
        pred = self.regressor_.predict(x_data, **predict_kwargs)
        if pred.ndim == 1:
            pred_trans = self.transformer_.inverse_transform(
                pred.reshape(-1, 1))
        else:
            pred_trans = self.transformer_.inverse_transform(pred)
        squeeze = pred_trans.ndim == 2 and pred_trans.shape[1] == 1
        if not always_return_1d:
            squeeze = squeeze and self._training_dim == 1
        if squeeze:
            pred_trans = pred_trans.squeeze(axis=1)
        if not (return_var or return_cov):
            return pred_trans

        # Return variance or covariance if desired
        scale = self.transformer_.scale_
        if return_var:
            (_, y_err) = self.regressor_.predict(x_data,
                                                 return_std=True,
                                                 **predict_kwargs)
            y_err *= y_err
        else:
            (_, y_err) = self.regressor_.predict(x_data,
                                                 return_cov=True,
                                                 **predict_kwargs)
        if scale is not None:
            y_err *= scale**2
        return (pred_trans, y_err)

    @staticmethod
    def _get_fit_kwargs(fit_kwargs, verbose=True):
        """Separate `transformer` and `regressor` kwargs."""
        transformer_kwargs = {}
        regressor_kwargs = {}
        for (param_name, param_val) in fit_kwargs.items():
            param_split = param_name.split('__', 1)
            if len(param_split) != 2:
                logger.warning(
                    "Fit parameters for 'AdvancedTransformedTargetRegressor' "
                    "have to be given as 'transformer__{param}' or "
                    "'regressor__{param}', got '%s'", param_name)
                continue
            if param_split[0] == 'transformer':
                transformer_kwargs[param_split[1]] = param_val
            elif param_split[0] == 'regressor':
                regressor_kwargs[param_split[1]] = param_val
            else:
                if verbose:
                    logger.warning(
                        "Allowed prefixes for fit parameters given to "
                        "'AdvancedTransformedTargetRegressor' are "
                        "'transformer' and 'regressor', got '%s'",
                        param_split[0])
        # FIXME
        if transformer_kwargs:
            logger.warning(
                "Keyword arguments for transformer of "
                "'AdvancedTransformedTargetRegressor' are not supported yet")
        return (transformer_kwargs, regressor_kwargs)


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
