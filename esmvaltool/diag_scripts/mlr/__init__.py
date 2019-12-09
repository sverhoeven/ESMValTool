"""Convenience functions for MLR diagnostics."""

import logging
import os
import re
from copy import deepcopy
from pprint import pformat

import iris
import numpy as np
from cf_units import Unit
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from yellowbrick.regressor import ResidualsPlot

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
    'prediction_input_error',
    'prediction_output',
    'prediction_output_error',
    'prediction_output_misc',
    'prediction_reference',
    'prediction_residual',
]


def _has_valid_coords(cube, coords):
    """Check if cube has valid coords for calculating weights."""
    for coord_name in coords:
        try:
            coord = cube.coord(coord_name)
        except iris.exceptions.CoordinateNotFoundError:
            return False
        if coord.shape[0] <= 1:
            return False
    return True


class AdvancedPipeline(Pipeline):
    """Expand :class:`sklearn.pipeline.Pipeline`."""

    def fit_transformers_only(self, x_data, y_data, **fit_kwargs):
        """Fit only ``transform`` steps of Pipeline."""
        transformer_steps = [s[0] for s in self.steps[:-1]]
        transformers_kwargs = {}
        for (param_name, param_val) in fit_kwargs.items():
            step = param_name.split('__')[0]
            if step in transformer_steps:
                transformers_kwargs[param_name] = param_val
        if transformers_kwargs:
            logger.debug("Used parameters %s to fit only transformers",
                         transformers_kwargs)
        return self._fit(x_data, y_data, **transformers_kwargs)

    def transform_only(self, x_data):
        """Only perform ``transform`` steps of Pipeline."""
        for (_, transformer) in self.steps[:-1]:
            x_data = transformer.transform(x_data)
        return x_data

    def transform_target_only(self, y_data):
        """Only perform ``transform`` steps of target regressor."""
        reg = self.steps[-1][1]
        if not hasattr(reg, 'transformer_'):
            raise ValueError(
                "Transforming target not possible, final regressor step does "
                "not have necessary 'transformer_' attribute")
        if y_data.ndim == 1:
            y_data = y_data.reshape(-1, 1)
        y_trans = reg.transformer_.transform(y_data)
        if y_trans.ndim == 2 and y_trans.shape[1] == 1:
            y_trans = y_trans.squeeze(axis=1)
        return y_trans


class AdvancedResidualsPlot(ResidualsPlot):
    """Expand :class:`yellowbrick.regressor.ResidualsPlot`."""

    def score(self, X, y=None, train=False, **kwargs):
        """Change sign convention of residuals."""
        score = self.estimator.score(X, y, **kwargs)
        if train:
            self.train_score_ = score
        else:
            self.test_score_ = score

        y_pred = self.predict(X)
        residuals = y - y_pred
        self.draw(y_pred, residuals, train=train)

        return score


class AdvancedTransformedTargetRegressor(TransformedTargetRegressor):
    """Expand :class:`sklearn.compose.TransformedTargetRegressor`."""

    def fit(self, x_data, y_data, **fit_kwargs):
        """Expand :meth:`fit` to accept kwargs."""
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
        """Fit only ``transformer`` step."""
        y_data = check_array(y_data,
                             accept_sparse=False,
                             force_all_finite=True,
                             ensure_2d=False,
                             dtype='numeric')
        self._training_dim = y_data.ndim

        # Process kwargs
        (_, _) = self._get_fit_kwargs(fit_kwargs)

        # Transformers are designed to modify X which is 2D, modify y_data
        # FIXME: Transformer does NOT use transformer_kwargs
        if y_data.ndim == 1:
            y_2d = y_data.reshape(-1, 1)
        else:
            y_2d = y_data
        self._fit_transformer(y_2d)

    def predict(self, x_data, always_return_1d=True, **predict_kwargs):
        """Expand :meth:`predict()` to accept kwargs."""
        predict_kwargs = dict(predict_kwargs)
        check_is_fitted(self)

        # Kwargs for returning variance or covariance
        return_var = predict_kwargs.pop('return_var', False)
        return_cov = predict_kwargs.pop('return_cov', False)
        if return_var and return_cov:
            raise RuntimeError(
                "Cannot return variance (return_cov=True) and full "
                "covariance matrix (return_cov=True) simultaneously")

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

    def _get_fit_kwargs(self, fit_kwargs):
        """Separate ``transformer`` and ``regressor`` kwargs."""
        transformer_kwargs = {}
        regressor_kwargs = {}
        for (param_name, param_val) in fit_kwargs.items():
            param_split = param_name.split('__', 1)
            if len(param_split) != 2:
                raise ValueError(
                    f"Fit parameters for {self.__class__} have to be given as "
                    f"'transformer__...' or 'regressor__...', got "
                    f"'{param_name}'")
            if param_split[0] == 'transformer':
                transformer_kwargs[param_split[1]] = param_val
            elif param_split[0] == 'regressor':
                regressor_kwargs[param_split[1]] = param_val
            else:
                raise ValueError(
                    f"Allowed prefixes for fit parameters given to "
                    f"{self.__class__} are 'transformer' and 'regressor', got "
                    f"'{param_split[0]}' for parameter '{param_name}'")

        # FIXME
        if transformer_kwargs:
            logger.warning(
                "Keyword arguments %s for transformer of %s are not "
                "supported at the moment", transformer_kwargs,
                str(self.__class__))
        return (transformer_kwargs, regressor_kwargs)


def create_alias(dataset, attributes, delimiter='-'):
    """Create alias key of a dataset using a list of attributes.

    Parameters
    ----------
    dataset : dict
        Metadata dictionary representing a single dataset.
    attributes : list of str
        List of attributes used to create the alias.
    delimiter : str, optional (default: '-')
        Delimiter used to separate different attributes in the alias.

    Returns
    -------
    str
        Dataset alias.

    Raises
    ------
    AttributeError
        ``dataset`` does not contain one of the ``attributes``.

    """
    alias = []
    if not attributes:
        raise ValueError(
            "Expected at least one element for attributes, got empty list")
    for attribute in attributes:
        if attribute not in dataset:
            raise AttributeError(
                f"Datset {dataset} does not contain attribute '{attribute}' "
                f"for alias creation")
        alias.append(dataset[attribute])
    return delimiter.join(alias)


def datasets_have_mlr_attributes(datasets, log_level='debug', mode='full'):
    """Check (MLR) attributes of ``datasets``.

    Parameters
    ----------
    datasets : list of dict
        Datasets to check.
    log_level : str, optional (default: 'debug')
        Verbosity level of the logger.
    mode : str, optional (default: 'full')
        Checking mode. Must be one of ``'only_missing'`` (only check if
        attributes are missing), ``'only_var_type'`` (check only `var_type`) or
        ``'full'`` (check both).

    Returns
    -------
    bool
        ``True`` if all required attributes are available, ``False`` if not.

    Raises
    ------
    ValueError
        Invalid value for argument ``mode`` is given.

    """
    output = True
    accepted_modes = ('full', 'only_missing', 'only_var_type')
    if mode not in accepted_modes:
        raise ValueError(
            f"'mode' must be one of {accepted_modes}, got '{mode}'")
    for dataset in datasets:
        if mode != 'only_var_type':
            for key in NECESSARY_KEYS:
                if key not in dataset:
                    getattr(logger, log_level)(
                        "Dataset '%s' does not have necessary (MLR) attribute "
                        "'%s'", dataset, key)
                    output = False
        if mode != 'only_missing' and dataset.get('var_type') not in VAR_TYPES:
            getattr(logger, log_level)(
                "Dataset '%s' has invalid var_type '%s', must be one of %s",
                dataset, dataset.get('var_type'), VAR_TYPES)
            output = False
    return output


def get_absolute_time_units(units):
    """Convert time reference units to absolute ones.

    This function converts reference time units (like ``'days since YYYY'``) to
    absolute ones (like ``'days'``).

    Parameters
    ----------
    units : cf_units.Unit
        Time units to convert.

    Returns
    -------
    cf_units.Unit
        Absolute time units.

    Raises
    ------
    ValueError
        If conversion failed (e.g. input units are not time units).

    """
    if units.is_time_reference():
        units = Unit(units.symbol.split()[0])
    if not units.is_time():
        raise ValueError(
            f"Cannot convert units '{units}' to reasonable time units")
    return units


def get_all_weights(cube, normalize=False):
    """Get all possible weights of cube.

    Parameters
    ----------
    cube : iris.cube.Cube
        Input cube.
    normalize : bool, optional (default: False)
        Normalize weights with total area and total time range.

    Returns
    -------
    numpy.ndarray
        Area weights.

    """
    cube_str = cube.summary(shorten=True)
    logger.debug("Calculating all weights of cube %s", cube_str)
    weights = np.ones(cube.shape)

    # Area weights
    if _has_valid_coords(cube, ['latitude', 'longitude']):
        area_weights = get_area_weights(cube, normalize=normalize)
        weights *= area_weights

    # Time weights
    if _has_valid_coords(cube, ['time']):
        time_weights = get_time_weights(cube, normalize=normalize)
        weights *= time_weights

    return weights


def get_area_weights(cube, normalize=False):
    """Get area weights of cube.

    Parameters
    ----------
    cube : iris.cube.Cube
        Input cube.
    normalize : bool, optional (default: False)
        Normalize weights with total area.

    Returns
    -------
    numpy.ndarray
        Area weights.

    Raises
    ------
    iris.exceptions.CoordinateNotFoundError
        Cube does not contain the coordinates ``latitude`` and ``longitude``.
    iris.exceptions.CoordinateNotRegularError
        Length of ``latitude`` or ``longitude`` coordinate is smaller than 2.

    """
    cube_str = cube.summary(shorten=True)
    logger.debug("Calculating area weights of cube %s", cube_str)
    for coord_name in ('latitude', 'longitude'):
        try:
            coord = cube.coord(coord_name)
        except iris.exceptions.CoordinateNotFoundError:
            logger.error(
                "Calculation of area weights for cube %s failed, coordinate "
                "'%s' not found", cube_str, coord_name)
            raise
        if coord.shape[0] <= 1:
            raise iris.exceptions.CoordinateNotRegularError(
                f"Calculation of area weights for cube {cube_str} failed, "
                f"coordinate '{coord_name}' has length {coord.shape[0]}, "
                f"needs to be > 1")
        if not coord.has_bounds():
            logger.debug(
                "Guessing bounds of coordinate '%s' of cube %s for area "
                "weights calculation", coord_name, cube_str)
            coord.guess_bounds()
    area_weights = iris.analysis.cartography.area_weights(cube,
                                                          normalize=normalize)
    return area_weights


def get_input_data(cfg, pattern=None, check_mlr_attributes=True):
    """Get input data and check MLR attributes if desired.

    Use ``input_data`` and ancestors to get all relevant input files. Only
    accepts files with all necessary MLR attributes if desired.

    Parameters
    ----------
    cfg : dict
        Recipe configuration.
    pattern : str, optional
        Pattern matched against ancestor files.
    check_mlr_attributes : bool, optional (default: True)
        If ``True``, only returns datasets with valid MLR attributes. If
        ``False``, returns all found datasets.

    Returns
    -------
    list of dict
        List of input datasets.

    Raises
    ------
    ValueError
        No input data found or at least one dataset has invalid attributes.

    """
    logger.debug("Extracting input files")
    input_data = list(cfg['input_data'].values())
    input_data.extend(io.netcdf_to_metadata(cfg, pattern=pattern))
    input_data = deepcopy(input_data)
    if not input_data:
        raise ValueError("No input data found")
    if check_mlr_attributes:
        if not datasets_have_mlr_attributes(input_data, log_level='error'):
            raise ValueError("At least one input dataset does not have valid "
                             "MLR attributes")
    logger.debug("Found files:")
    logger.debug(pformat([d['filename'] for d in input_data]))
    return input_data


def get_squared_error_cube(ref_cube, error_datasets):
    """Get array of squared errors.

    Parameters
    ----------
    ref_cube : iris.cube.Cube
        Reference cube (determines mask, coordinates and attributes of output).
    error_datasets : list of dict
        List of metadata dictionaries where each dictionary represents a single
        dataset.

    Returns
    -------
    iris.cube.Cube
        Cube containing squared errors.

    Raises
    ------
    ValueError
        Shape of a dataset does not match shape of reference cube.

    """
    squared_error_cube = ref_cube.copy()

    # Fill cube with zeros
    squared_error_cube.data = np.ma.array(
        np.full(squared_error_cube.shape, 0.0),
        mask=np.ma.getmaskarray(squared_error_cube.data),
    )

    # Adapt cube metadata
    if 'error' in squared_error_cube.attributes.get('var_type', ''):
        if not squared_error_cube.attributes.get('squared'):
            squared_error_cube.var_name += '_squared'
            squared_error_cube.long_name += ' (squared)'
            squared_error_cube.units = units_power(squared_error_cube.units, 2)
    else:
        if squared_error_cube.attributes.get('squared'):
            squared_error_cube.var_name += '_error'
            squared_error_cube.long_name += ' (error)'
        else:
            squared_error_cube.var_name += '_squared_error'
            squared_error_cube.long_name += ' (squared error)'
            squared_error_cube.units = units_power(squared_error_cube.units, 2)
    squared_error_cube.attributes['squared'] = 1
    squared_error_cube.attributes['var_type'] = 'prediction_output_error'

    # Aggregate errors
    for dataset in error_datasets:
        path = dataset['filename']
        cube = iris.load_cube(path)

        # Check shape
        if cube.shape != ref_cube.shape:
            raise ValueError(
                f"Expected shape {ref_cube.shape} for error cubes, got "
                f"{cube.shape} for dataset '{path}'")

        # Add squared error
        new_data = cube.data
        if not cube.attributes.get('squared'):
            new_data **= 2
        squared_error_cube.data += new_data
        logger.debug("Added '%s' to squared error datasets", path)
    return squared_error_cube


def get_time_weights(cube, normalize=False):
    """Get time weights of cube.

    Parameters
    ----------
    cube : iris.cube.Cube
        Input cube.
    normalize : bool, optional (default: False)
        Normalize weights with total time range.

    Returns
    -------
    numpy.ndarray
        Time weights.

    Raises
    ------
    iris.exceptions.CoordinateNotFoundError
        Cube does not contain the coordinate ``time``.
    iris.exceptions.CoordinateNotRegularError
        Length of ``time`` coordinate is smaller than 2.

    """
    cube_str = cube.summary(shorten=True)
    logger.debug("Calculating time weights of cube %s", cube_str)
    try:
        coord = cube.coord('time')
    except iris.exceptions.CoordinateNotFoundError:
        logger.error(
            "Calculation of time weights for cube %s failed, coordinate "
            "'time' not found", cube_str)
        raise
    if coord.shape[0] <= 1:
        raise iris.exceptions.CoordinateNotRegularError(
            f"Calculation of time weights for cube {cube_str} failed, "
            f"coordinate 'time' has length {coord.shape[0]}, needs to be "
            f"> 1")
    if not coord.has_bounds():
        logger.debug(
            "Guessing bounds of coordinate 'time' of cube %s for time weights "
            "calculation", cube_str)
        coord.guess_bounds()
    time_weights = coord.bounds[:, 1] - coord.bounds[:, 0]
    if normalize:
        time_weights /= np.ma.sum(time_weights)
    new_axis_pos = np.delete(np.arange(cube.ndim), cube.coord_dims('time'))
    for idx in new_axis_pos:
        time_weights = np.expand_dims(time_weights, idx)
    time_weights = np.broadcast_to(time_weights, cube.shape)
    return time_weights


def square_root_metadata(cube):
    """Take the square root of the cube metadata.

    Parameters
    ----------
    cube : iris.cube.Cube
        Cube (will be modified in-place).

    """
    if 'squared_' in cube.var_name:
        cube.var_name = cube.var_name.replace('squared_', '')
    elif '_squared' in cube.var_name:
        cube.var_name = cube.var_name.replace('_squared', '')
    else:
        cube.var_name = 'root_' + cube.var_name
    if 'squared ' in cube.long_name:
        cube.long_name = cube.long_name.replace('squared ', '')
    elif 'Squared ' in cube.long_name:
        cube.long_name = cube.long_name.replace('Squared ', '')
    elif ' squared' in cube.long_name:
        cube.long_name = cube.long_name.replace(' squared', '')
    elif ' Squared' in cube.long_name:
        cube.long_name = cube.long_name.replace(' Squared', '')
    elif ' (squared)' in cube.long_name:
        cube.long_name = cube.long_name.replace(' (squared)', '')
    elif ' (Squared)' in cube.long_name:
        cube.long_name = cube.long_name.replace(' (Squared)', '')
    else:
        cube.long_name = 'Root ' + cube.long_name
    cube.units = cube.units.root(2)
    if cube.attributes.get('squared'):
        cube.attributes.pop('squared')


def units_power(units, power):
    """Raise a :class:`cf_units.Unit` to given power preserving symbols.

    Raise :class:`cf_units.Unit` to given power without expanding it first. For
    example, raising ``'J'`` to the power of 2 (by using ``**2``) gives
    ``'kg2 m4 s-4'``, not ``'W2'``.

    Parameters
    ----------
    units : cf_units.Unit
        Input units.
    power : int
        Desired exponent.

    Returns
    -------
    cf_units.Unit
        Input units raised to given power.

    Raises
    ------
    TypeError
        Argument ``power`` is not :obj:`int`-like.
    ValueError
        Invalid unit given.

    """
    if round(power) != power:
        raise TypeError(
            f"Expected integer-like power for units exponentiation, got "
            f"{power}")
    power = int(power)
    if any([units.is_no_unit(), units.is_unknown()]):
        raise ValueError(
            f"Cannot raise units '{units.name}' to power {power:d}")
    if units.origin is None:
        logger.warning(
            "Symbol-preserving exponentiation of units '%s' is not "
            "supported, origin is not given", units)
        return units**power
    if units.origin.split()[0][0].isdigit():
        logger.warning(
            "Symbol-preserving exponentiation of units '%s' is not "
            "supported yet because of leading numbers", units)
        return units**power
    new_units_list = []
    for split in units.origin.split():
        for elem in split.split('.'):
            if elem[-1].isdigit():
                exp = [int(d) for d in re.findall(r'-?\d+', elem)][0]
                val = ''.join([abc for abc in re.findall(r'[A-Za-z]', elem)])
                new_units_list.append(f'{val}{exp * power}')
            else:
                new_units_list.append(f'{elem}{power}')
    new_units = ' '.join(new_units_list)
    return Unit(new_units)


def write_cube(cube, attributes):
    """Write cube with all necessary information for MLR models.

    Parameters
    ----------
    cube : iris.cube.Cube
        Cube which should be written.
    attributes : dict
        Attributes for the cube (needed for MLR models).

    Raises
    ------
    IOError
        File cannot be written due to invalid attributes.

    """
    if not datasets_have_mlr_attributes([attributes], log_level='error'):
        raise IOError(
            f"Cannot write cube {cube.summary(shorten=True)} using attributes "
            f"attributes")
    io.metadata_to_netcdf(cube, attributes)
