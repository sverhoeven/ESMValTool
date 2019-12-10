#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple postprocessing of MLR model output.

Description
-----------
This diagnostic performs simple postprocessing operations for MLR model output
(mean and error).

Author
------
Manuel Schlund (DLR, Germany)

Project
-------
CRESCENDO

Note
----
Prior to postprocessing, group input datasets according to ``tag`` and
``prediction_name``. For each group, accepts datasets with three different
``var_type``s:
    * ``prediction_output``: **Exactly one** necessary, refers to
      the mean prediction and serves as reference dataset (regarding shape).
    * ``prediction_output_error``: Arbitrary number of error datasets. If not
      given, error calculation is skipped. May be squared errors (marked by the
      attribute ``squared``) or not. In addition, a single covariance dataset
      can be specified (``short_name`` ending with ``_cov``).
    * ``prediction_input``: Dataset used to estimate covariance structure of
      the mean prediction (i.e. matrix of Pearson correlation coefficients) for
      error estimation. At most one dataset allowed. Ignored when no
      ``prediction_output_error`` is given.
Real error calculation (using covariance dataset) and estimation (using dataset
to estimate covariance structure) is only possible if the mean prediction cube
is collapsed completely, i.e. all coordinates are listed for either ``mean`` or
``sum``.

Configuration options in recipe
-------------------------------
add_var_from_cov : bool, optional (default: True)
    Calculate variances from covariance matrix (diagonal elements) and add
    those to (squared) error datasets. Set to ``False`` if variance is already
    given separately in prediction output.
area_weighted : bool, optional (default: True)
    Calculate weighted averages/sums for area (using grid cell boundaries).
    convert_units_to : str, optional
convert_units_to : str, optional
    Convert units of the input data.
ignore : list of dict, optional
    Ignore specific datasets by specifying multiple :obj:`dict`s of metadata.
mean : list of str, optional
    Perform mean over the given coordinates.
pattern : str, optional
    Pattern matched against ancestor file names.
sum : list of str, optional
    Perform sum over the given coordinates.
time_weighted : bool, optional (default: True)
    Calculate weighted averages/sums for time (using grid cell boundaries).

"""

import logging
import os
from copy import deepcopy
from pprint import pformat

import iris
import numpy as np
from cf_units import Unit

from esmvaltool.diag_scripts import mlr
from esmvaltool.diag_scripts.shared import (get_diagnostic_filename,
                                            group_metadata, io, run_diagnostic)

logger = logging.getLogger(os.path.basename(__file__))

OPS = {
    'mean': iris.analysis.MEAN,
    'sum': iris.analysis.SUM,
}


def _calculate_lower_error_bound(cfg, squared_error_cube, basepath):
    """Calculate lower error bound."""
    logger.debug("Calculating lower error bound")
    lower_bound = _collapse_regular_cube(cfg, squared_error_cube, power=2)
    lower_bound.data = np.ma.sqrt(lower_bound.data)
    mlr.square_root_metadata(lower_bound)
    _convert_units(cfg, lower_bound)
    lower_bound.attributes['error_type'] = 'lower_bound'
    new_path = basepath.replace('.nc', '_lower_bound.nc')
    io.iris_save(lower_bound, new_path)
    logger.info("Lower bound of error: %s %s", lower_bound.data,
                lower_bound.units)


def _calculate_real_error(cfg, ref_cube, cov_cube, basepath):
    """Calculate real error using covariance."""
    logger.debug("Calculating real error using covariance")
    real_error = _collapse_covariance_cube(cfg, cov_cube, ref_cube)
    real_error.data = np.ma.sqrt(real_error.data)
    real_error.var_name = cov_cube.var_name.replace('_cov', '_error')
    real_error.long_name = cov_cube.long_name.replace('(covariance)',
                                                      '(error)')
    real_error.units = real_error.units.root(2)
    _convert_units(cfg, real_error)
    real_error.attributes['source'] = ref_cube.attributes['filename']
    real_error.attributes['error_type'] = 'real_error'
    new_path = basepath.replace('.nc', '_real.nc')
    io.iris_save(real_error, new_path)
    logger.info("Real error (using covariance): %s %s", real_error.data,
                real_error.units)


def _calculate_upper_error_bound(cfg, squared_error_cube, basepath):
    """Calculate upper error bound."""
    logger.debug("Calculating upper error bound")
    upper_bound = squared_error_cube.copy()
    upper_bound.data = np.ma.sqrt(upper_bound.data)
    mlr.square_root_metadata(upper_bound)
    upper_bound = _collapse_regular_cube(cfg, upper_bound)
    _convert_units(cfg, upper_bound)
    upper_bound.attributes['error_type'] = 'upper_bound'
    new_path = basepath.replace('.nc', '_upper_bound.nc')
    io.iris_save(upper_bound, new_path)
    logger.info("Upper bound of error: %s %s", upper_bound.data,
                upper_bound.units)


def _convert_units(cfg, cube):
    """Convert units if desired."""
    cfg_settings = cfg.get('convert_units_to')
    if cfg_settings:
        units_to = cfg_settings
        logger.debug("Converting units from '%s' to '%s'", cube.units,
                     units_to)
        try:
            cube.convert_units(units_to)
        except ValueError:
            raise ValueError(
                f"Cannot convert units from '{cube.units}' to '{units_to}'")


def _collapse_covariance_cube(cfg, cov_cube, ref_cube):
    """Collapse covariance cube with using desired operations."""
    (weights, units, coords) = _get_all_weights(cfg, ref_cube)
    if len(coords) < ref_cube.ndim:
        raise ValueError(
            f"Calculating real error using covariance dataset for covariance "
            f"structure estimation ('prediction_input') is only possible if "
            f"all {ref_cube.ndim:d} dimensions of the cube  are collapsed, "
            f"got only {len(coords):d} ({coords})")
    weights = weights.ravel()
    weights = weights[~np.ma.getmaskarray(ref_cube.data).ravel()]
    weights = np.outer(weights, weights)
    cov_cube = cov_cube.collapsed(cov_cube.coords(dim_coords=True),
                                  iris.analysis.SUM,
                                  weights=weights)
    cov_cube.units *= units**2
    return cov_cube


def _collapse_estimated_covariance(squared_error_cube, cov_est_cube, weights):
    """Estimate covariance and collapse cube."""
    logger.debug("Estimating covariance (memory and time intensive)")
    error = np.ma.sqrt(squared_error_cube.data)
    error = np.ma.filled(error, 0.0)
    cov_est = np.ma.array(cov_est_cube.data)
    if cov_est.ndim > 2:
        error = error.reshape(error.shape[0], -1)
        cov_est = cov_est.reshape(cov_est.shape[0], -1)
        weights = weights.reshape(weights.shape[0], -1)

    # Pearson coefficients (= normalized covariance) over both dimensions
    pearson_dim0 = _corrcoef(cov_est, weights=weights)
    pearson_dim1 = _corrcoef(cov_est, rowvar=False, weights=weights)

    # Covariances
    cov_dim0 = (np.einsum('...i,...j->...ij', error, error) *
                np.einsum('...i,...j->...ij', weights, weights) * pearson_dim1)
    cov_dim1 = (np.einsum('i...,j...->...ij', error, error) *
                np.einsum('i...,j...->...ij', weights, weights) * pearson_dim0)

    # Errors over dimensions
    error_dim0 = np.ma.sqrt(np.ma.sum(cov_dim0, axis=(1, 2)))
    error_dim1 = np.ma.sqrt(np.ma.sum(cov_dim1, axis=(1, 2)))

    # Collaps further (all weights are already included in first step)
    cov_order_0 = pearson_dim0 * np.ma.outer(error_dim0, error_dim0)
    cov_order_1 = pearson_dim1 * np.ma.outer(error_dim1, error_dim1)
    error_order_0 = np.ma.sqrt(np.ma.sum(cov_order_0))
    error_order_1 = np.ma.sqrt(np.ma.sum(cov_order_1))
    logger.debug(
        "Found real errors %e and %e after collapsing with different "
        "orderings, using maximum", error_order_0, error_order_1)

    # Ordering of collapsing matters, maximum is used
    return max([error_order_0, error_order_1])


def _collapse_regular_cube(cfg, cube, power=1):
    """Collapse cube with using desired operations."""
    (weights, units, coords) = _get_all_weights(cfg, cube, power=power)
    cube = cube.collapsed(coords, iris.analysis.SUM, weights=weights)
    cube.units *= units
    return cube


def _estimate_real_error(cfg, squared_error_cube, cov_est_dataset, basepath):
    """Estimate real error using estimated covariance."""
    logger.debug(
        "Estimating real error using estimated covariance from "
        "'prediction_input' dataset %s", cov_est_dataset['filename'])
    cov_est_cube = iris.load_cube(cov_est_dataset['filename'])
    if cov_est_cube.shape != squared_error_cube.shape:
        raise ValueError(
            f"Expected identical shapes for 'prediction_input' dataset used "
            f"for covariance structure estimation and error datasets, got "
            f"{cov_est_cube.shape} and {squared_error_cube.shape}, "
            f"respectively")
    if cov_est_cube.ndim < 2:
        raise ValueError(
            f"Expected at least 2D 'prediction_input' dataset for covariance "
            f"structure estimation, got {cov_est_cube.ndim:d}D dataset")

    # Calculate weights
    (weights, units, coords) = _get_all_weights(cfg, squared_error_cube)
    if len(coords) < cov_est_cube.ndim:
        raise ValueError(
            f"Estimating real error using 'prediction_input' dataset for "
            f"covariance structure estimation is only possible if all "
            f"{cov_est_cube.ndim:d} dimensions of the cube are collapsed, got "
            f"only {len(coords):d} ({coords})")

    # Calculate error
    error = _collapse_estimated_covariance(squared_error_cube, cov_est_cube,
                                           weights)

    # Create cube (collapse using dummy operation)
    real_error = squared_error_cube.collapsed(coords, iris.analysis.MEAN)
    real_error.data = error
    mlr.square_root_metadata(real_error)
    real_error.units *= units
    _convert_units(cfg, real_error)
    real_error.attributes['error_type'] = 'estimated_real_error'

    # Save cube
    new_path = basepath.replace('.nc', '_estimated.nc')
    io.iris_save(real_error, new_path)
    logger.info("Estimated real error (using estimated covariance): %s %s",
                real_error.data, real_error.units)


def _corrcoef(array, rowvar=True, weights=None):
    """Fast version of :func:`numpy.ma.corrcoef`."""
    if not rowvar:
        array = array.T
        if weights is not None:
            weights = weights.T
    mean = np.ma.average(array, axis=1, weights=weights).reshape(-1, 1)
    if weights is None:
        sqrt_weights = 1.0
    else:
        sqrt_weights = np.ma.sqrt(weights)
    demean = (array - mean) * sqrt_weights
    res = np.ma.dot(demean, demean.T)
    row_norms = np.ma.sqrt(np.ma.sum(demean**2, axis=1))
    res /= np.ma.outer(row_norms, row_norms)
    return res


def _get_all_weights(cfg, cube, power=1):
    """Get all necessary weights (including norm for mean calculation)."""
    cfg = deepcopy(cfg)
    all_coords = []
    weights = np.ones(cube.shape)
    units = Unit('1')
    for operation in ('sum', 'mean'):
        normalize = (operation == 'mean')
        coords = cfg.get(operation, [])
        all_coords.extend(coords)
        if coords == 'all':
            coords = [c.name() for c in cube.coords(dim_coords=True)]
        if 'latitude' in coords and 'longitude' in coords:
            (area_weights, area_units) = _get_area_weights(cfg,
                                                           cube,
                                                           power=power,
                                                           normalize=normalize)
            if operation == 'sum':
                units *= area_units
            if area_weights is not None:
                weights *= area_weights
            else:
                weights /= _get_normalization_factor(['latitude', 'longitude'],
                                                     cube, normalize)**power
            coords.remove('latitude')
            coords.remove('longitude')
        if 'time' in coords:
            (time_weights, time_units) = _get_time_weights(cfg,
                                                           cube,
                                                           power=power,
                                                           normalize=normalize)
            if operation == 'sum':
                units *= time_units
            if time_weights is not None:
                weights *= time_weights
            else:
                weights /= _get_normalization_factor(['time'], cube,
                                                     normalize)**power
            coords.remove('time')
        weights /= _get_normalization_factor(coords, cube, normalize)**power
    logger.debug("Found coordinates %s to collapse over", all_coords)
    logger.debug("Found units '%s' for weights", units)
    return (weights, units, all_coords)


def _get_area_weights(cfg, cube, power=1, normalize=False):
    """Calculate area weights."""
    area_weights = None
    if cfg.get('area_weighted', True):
        area_weights = mlr.get_area_weights(cube, normalize=normalize)
        area_weights = area_weights**power
    return (area_weights, Unit('m2')**power)


def _get_covariance_dataset(error_datasets, ref_cube):
    """Extract covariance dataset."""
    explanation = ("i.e. dataset with short_name == '*_cov' among "
                   "'prediction_output_error' datasets")
    cov_datasets = []
    other_datasets = []

    # Get covariance dataset(s)
    for dataset in error_datasets:
        if '_cov' in dataset['short_name']:
            cov_datasets.append(dataset)
        else:
            other_datasets.append(dataset)
    if not cov_datasets:
        logger.warning(
            "No covariance dataset (%s) found, calculation of real error not "
            "possible", explanation)
        return (None, other_datasets)
    if len(cov_datasets) > 1:
        raise ValueError(
            f"Expected at most one covariance dataset ({explanation}), got "
            f"{len(cov_datasets):d}")

    # Check shape
    cov_cube = iris.load_cube(cov_datasets[0]['filename'])
    ref_size = np.ma.array(ref_cube.data).compressed().shape[0]
    if cov_cube.shape != (ref_size, ref_size):
        raise ValueError(
            f"Expected shape of covariance dataset to be "
            f"{(ref_size, ref_size)}, got {cov_cube.shape} (after removal of "
            f"all missing values)")
    return (cov_cube, other_datasets)


def _get_new_path(cfg, old_path):
    """Get new path."""
    basename = os.path.splitext(os.path.basename(old_path))[0]
    return get_diagnostic_filename(basename, cfg)


def _get_normalization_factor(coords, cube, normalize):
    """Get normalization constant for calculation of means."""
    norm = 1.0
    if not normalize:
        return norm
    for coord in coords:
        coord_idx = cube.coord_dims(coord)[0]
        norm *= cube.shape[coord_idx]
    return norm


def _get_time_weights(cfg, cube, power=1, normalize=False):
    """Calculate time weights."""
    time_weights = None
    time_units = mlr.get_absolute_time_units(cube.coord('time').units)
    if cfg.get('time_weighted', True):
        time_weights = mlr.get_time_weights(cube, normalize=normalize)
        time_weights = time_weights**power
    return (time_weights, time_units**power)


def check_cfg(cfg):
    """Check options of configuration and catch errors."""
    for operation in ('sum', 'mean'):
        if operation in cfg:
            cfg[operation] = list(set(cfg[operation]))
    for coord in cfg.get('sum', []):
        if coord in cfg.get('mean', []):
            raise ValueError(
                f"Coordinate '{coord.name()}' given in 'sum' and 'mean'")


def postprocess_errors(cfg, ref_cube, error_datasets, cov_estim_datasets):
    """Postprocess errors."""
    logger.info(
        "Postprocessing errors using mean prediction cube %s as reference",
        ref_cube.summary(shorten=True))

    # Extract covariance
    (cov_cube,
     error_datasets) = _get_covariance_dataset(error_datasets, ref_cube)

    # Extract squared errors
    squared_error_cube = mlr.get_squared_error_cube(ref_cube, error_datasets)

    # Extract variance from covariance if desired
    if cfg.get('add_var_from_cov', True) and cov_cube is not None:
        var = np.ma.empty(ref_cube.shape, dtype=ref_cube.dtype)
        mask = np.ma.getmaskarray(ref_cube.data)
        var[mask] = np.ma.masked
        var[~mask] = np.diagonal(cov_cube.data.copy())
        squared_error_cube.data += var
        logger.debug(
            "Added variance calculated from covariance to squared error "
            "datasets")
        if not error_datasets:
            error_datasets = True

    # Extract basename for error cubes
    basename = os.path.splitext(
        os.path.basename(ref_cube.attributes['filename']))[0] + '_error'
    basepath = get_diagnostic_filename(basename, cfg)

    # Lower and upper error bounds
    if error_datasets:
        _calculate_lower_error_bound(cfg, squared_error_cube, basepath)
        _calculate_upper_error_bound(cfg, squared_error_cube, basepath)

        # Estimated real error using estimated covariance
        if cov_estim_datasets:
            _estimate_real_error(cfg, squared_error_cube,
                                 cov_estim_datasets[0], basepath)

    # Real error
    if cov_cube is not None:
        _calculate_real_error(cfg, ref_cube, cov_cube, basepath)


def postprocess_mean(cfg, cube, data):
    """Postprocess mean prediction cube."""
    logger.info("Postprocessing mean prediction cube %s",
                cube.summary(shorten=True))
    cube = _collapse_regular_cube(cfg, cube)
    _convert_units(cfg, cube)
    cube.attributes['source'] = data['filename']
    new_path = _get_new_path(cfg, data['filename'])
    io.iris_save(cube, new_path)
    logger.info("Mean prediction: %s %s", cube.data, cube.units)


def split_datasets(datasets, tag, pred_name):
    """Split datasets into mean and error."""
    grouped_data = group_metadata(datasets, 'var_type')

    # Mean/reference dataset
    mean = grouped_data.get('prediction_output', [])
    if len(mean) != 1:
        raise ValueError(
            f"Expected exactly one 'prediction_output' dataset for tag "
            f"'{tag}' of prediction '{pred_name}', got {len(mean):d}")
    logger.info(
        "Found mean prediction dataset ('prediction_output') for tag '%s' of "
        "prediction '%s': %s (used as reference)", tag, pred_name,
        mean[0]['filename'])

    # Errors
    error = grouped_data.get('prediction_output_error', [])
    if not error:
        logger.warning(
            "No 'prediction_output_error' datasets for tag '%s' of prediction "
            "'%s' found, error calculation not possible (not searching for "
            "'prediction_input' datasets for covariance estimation, either)",
            tag, pred_name)
        cov_estimation = []
    else:
        logger.info(
            "Found error datasets ('prediction_output_error') for tag '%s' of "
            "prediction '%s':", tag, pred_name)
        logger.info(pformat([d['filename'] for d in error]))

        # Estimation for covariance
        cov_estimation = grouped_data.get('prediction_input', [])
        if not cov_estimation:
            logger.warning(
                "No 'prediction_input' dataset for tag '%s' of prediction "
                "'%s' found, real error estimation using estimated covariance "
                "structure not possible", tag, pred_name)
        elif len(cov_estimation) > 1:
            raise ValueError(
                f"Expected at most one 'prediction_input' dataset for tag "
                f"'{tag}' of prediction '{pred_name}', got "
                f"{len(cov_estimation):d}")
        else:
            logger.info(
                "Found 'prediction_input' dataset for covariance structure "
                "estimation for tag '%s' of prediction '%s': %s", tag,
                pred_name, cov_estimation[0]['filename'])

    return (mean[0], error, cov_estimation)


def main(cfg):
    """Run the diagnostic."""
    input_data = mlr.get_input_data(cfg,
                                    pattern=cfg.get('pattern'),
                                    ignore=cfg.get('ignore'))

    # Check cfg
    check_cfg(cfg)

    # Process data
    for (tag, tag_datasets) in group_metadata(input_data, 'tag').items():
        logger.info("Processing tag '%s'", tag)
        grouped_data = group_metadata(tag_datasets, 'prediction_name')
        for (pred_name, datasets) in grouped_data.items():
            logger.info("Processing prediction '%s'", pred_name)
            (dataset, error_datasets,
             cov_estim_datastets) = split_datasets(datasets, tag, pred_name)
            if dataset is None:
                continue

            # Extract cubes
            logger.debug(
                "Loaded mean prediction cube from '%s' (used as reference)",
                dataset['filename'])
            ref_cube = iris.load_cube(dataset['filename'])
            if ref_cube.ndim < 1:
                raise ValueError(
                    f"Postprocessing scalar dataset '{dataset['filename']}' "
                    f"not possible")

            # Process mean prediction
            postprocess_mean(cfg, ref_cube, dataset)

            # Process errors
            postprocess_errors(cfg, ref_cube, error_datasets,
                               cov_estim_datastets)


# Run main function when this script is called
if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
