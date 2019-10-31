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
mean : list of str, optional
    Perform mean over the given coordinates.
pattern : str, optional
    Pattern matched against ancestor files.
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
                                            group_metadata, io, run_diagnostic,
                                            select_metadata)

logger = logging.getLogger(os.path.basename(__file__))

OPS = {
    'mean': iris.analysis.MEAN,
    'sum': iris.analysis.SUM,
}


def _add_squared_error_attributes(cube):
    """Add correct squared error attributes to cube."""
    cube.var_name += '_squared_error'
    cube.long_name += ' (squared error)'
    cube.units = mlr.units_power(cube.units, 2)
    cube.attributes['var_type'] = 'prediction_output_error'


def _calculate_lower_error_bound(cfg, squared_error_cube, basepath):
    """Calculate lower error bound."""
    logger.debug("Calculating lower error bound")
    lower_bound = _collapse_regular_cube(cfg, squared_error_cube, power=2)
    lower_bound.data = np.ma.sqrt(lower_bound.data)
    _square_root_metadata(lower_bound)
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
    if real_error is None:
        logger.warning("Calculating real error using covariance failed")
        return
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
    _square_root_metadata(upper_bound)
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
            logger.warning("Cannot convert units from '%s' to '%s'",
                           cube.units, units_to)


def _collapse_covariance_cube(cfg, cov_cube, ref_cube):
    """Collapse covariance cube with using desired operations."""
    (weights, units, _) = _get_all_weights(cfg, ref_cube)
    weights = weights.ravel()
    weights = weights[~np.ma.getmaskarray(ref_cube.data).ravel()]
    weights = np.outer(weights, weights)
    cov_cube = cov_cube.collapsed(cov_cube.coords(dim_coords=True),
                                  iris.analysis.SUM,
                                  weights=weights)
    cov_cube.units *= units**2
    return cov_cube


def _collapse_estimated_covariance(squared_error_cube, ref_cube, weights):
    """Estimate covariance and collapse cube."""
    logger.debug("Estimating covariance (memory and time intensive)")
    error = np.ma.sqrt(squared_error_cube.data)
    error = np.ma.filled(error, 0.0)
    ref = np.ma.array(ref_cube.data)
    if ref.ndim > 2:
        error = error.reshape(error.shape[0], -1)
        ref = ref.reshape(ref.shape[0], -1)
        weights = weights.reshape(weights.shape[0], -1)

    # Pearson coefficients (= normalized covariance) over both dimensions
    pearson_dim0 = _corrcoef(ref, weights=weights)
    pearson_dim1 = _corrcoef(ref, rowvar=False, weights=weights)

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


def _estimate_real_error(cfg, squared_error_cube, ref_dataset, basepath):
    """Estimate real error using estimated covariance from reference data."""
    logger.debug(
        "Estimating real error estimated covariance from reference dataset %s",
        ref_dataset['filename'])
    ref_cube = iris.load_cube(ref_dataset['filename'])
    if ref_cube.shape != squared_error_cube.shape:
        logger.warning(
            "Expected shape of reference dataset for covariance estimation "
            "to be %s, got %s", squared_error_cube.shape, ref_cube.shape)
        logger.warning(
            "Estimating real error using reference for covariance failed")
        return
    if ref_cube.ndim < 2:
        logger.warning(
            "Reference dataset for covariance estimation is %iD, but needs to "
            "be at least 2D", ref_cube.ndim)
        logger.warning(
            "Estimating real error using reference for covariance failed")
        return

    # Calculate weights
    (weights, units, coords) = _get_all_weights(cfg, squared_error_cube)
    if len(coords) < ref_cube.ndim:
        logger.warning(
            "Estimating real error using reference for covariance is only "
            "possible if all %i dimensions are collapsed, got only %i (%s)",
            ref_cube.ndim, len(coords), coords)
        return

    # Calculate error
    error = _collapse_estimated_covariance(squared_error_cube, ref_cube,
                                           weights)

    # Create cube (collapse using dummy operation)
    real_error = squared_error_cube.collapsed(coords, iris.analysis.MEAN)
    real_error.data = error
    _square_root_metadata(real_error)
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
    """Get all necessary weights (including mean calculation)."""
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
    cov_datasets = []
    other_datasets = []

    # Get covariance dataset(s)
    for dataset in error_datasets:
        if '_cov' in dataset['short_name']:
            cov_datasets.append(dataset)
        else:
            other_datasets.append(dataset)
    if not cov_datasets:
        logger.debug("No covariance dataset found")
        return (None, other_datasets)
    if len(cov_datasets) > 1:
        logger.warning(
            "Got multiple error datasets for covariance, using only first "
            "one ('%s')", cov_datasets[0]['filename'])

    # Check shape
    cov_cube = iris.load_cube(cov_datasets[0]['filename'])
    ref_size = np.ma.array(ref_cube.data).compressed().shape[0]
    if cov_cube.shape != (ref_size, ref_size):
        logger.warning(
            "Expected shape of covariance cube to be %s, got %s (after "
            "removal of all missing values)", (ref_size, ref_size),
            cov_cube.shape)
        return (None, other_datasets)
    return (cov_cube, other_datasets)


def _get_new_path(cfg, old_path):
    """Get new path."""
    basename = os.path.splitext(os.path.basename(old_path))[0]
    return get_diagnostic_filename(basename, cfg)


def _get_normalization_factor(coords, ref_cube, normalize):
    """Get normalization constant for calculation of means."""
    norm = 1.0
    if not normalize:
        return norm
    for coord in coords:
        coord_idx = ref_cube.coord_dims(coord)[0]
        norm *= ref_cube.shape[coord_idx]
    return norm


def _get_time_weights(cfg, cube, power=1, normalize=False):
    """Calculate time weights."""
    time_weights = None
    time_units = mlr.get_absolute_time_units(cube.coord('time').units)
    if cfg.get('time_weighted', True):
        time_weights = mlr.get_time_weights(cube, normalize=normalize)
        time_weights = time_weights**power
    return (time_weights, time_units**power)


def _square_root_metadata(cube):
    """Take the square root of the cube metadata."""
    cube.var_name = cube.var_name.replace('_squared_error', '_error')
    cube.long_name = cube.long_name.replace(' (squared error)', ' (error)')
    cube.units = cube.units.root(2)


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
    logger.info("Postprocessing errors using reference cube %s",
                ref_cube.summary(shorten=True))
    squared_error_cube = ref_cube.copy()
    squared_error_cube.data = np.ma.array(
        np.full(squared_error_cube.shape, 0.0),
        mask=np.ma.getmaskarray(squared_error_cube.data),
    )
    _add_squared_error_attributes(squared_error_cube)

    # Extract basename for error cubes
    basename = os.path.splitext(
        os.path.basename(ref_cube.attributes['filename']))[0] + '_error'
    basepath = get_diagnostic_filename(basename, cfg)

    # Extract covariance
    (cov_cube,
     error_datasets) = _get_covariance_dataset(error_datasets, ref_cube)

    # Extract squared errors
    for dataset in error_datasets:
        path = dataset['filename']
        cube = iris.load_cube(path)

        # Ignore cubes with wrong shape
        if cube.shape != ref_cube.shape:
            logger.warning(
                "Expected shape %s for error cubes, got %s, skipping",
                ref_cube.shape, cube.shape)
            continue

        # Add squared errors
        new_data = cube.data
        if 'squared_' not in cube.var_name:
            new_data **= 2
        squared_error_cube.data += new_data
        logger.debug("Added '%s' to squared error datasets", path)

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

    # Lower and upper error bounds
    if error_datasets:
        _calculate_lower_error_bound(cfg, squared_error_cube, basepath)
        _calculate_upper_error_bound(cfg, squared_error_cube, basepath)

        # Estimated real error using estimated covariance
        if cov_estim_datasets:
            _estimate_real_error(cfg, squared_error_cube,
                                 cov_estim_datasets[0], basepath)

    # Calculate real error if possible
    if cov_cube is not None:
        _calculate_real_error(cfg, ref_cube, cov_cube, basepath)


def postprocess_ref(cfg, ref_cube, data):
    """Postprocess reference cube."""
    logger.info("Postprocessing reference cube %s",
                ref_cube.summary(shorten=True))
    ref_cube = _collapse_regular_cube(cfg, ref_cube)
    _convert_units(cfg, ref_cube)
    ref_cube.attributes['source'] = data['filename']
    new_path = _get_new_path(cfg, data['filename'])
    io.iris_save(ref_cube, new_path)
    logger.info("Mean prediction: %s %s", ref_cube.data, ref_cube.units)


def split_datasets(datasets, tag, pred_name):
    """Split datasets into mean and error."""
    grouped_data = group_metadata(datasets, 'var_type')

    # Mean/reference dataset
    mean = grouped_data.get('prediction_output')
    if not mean:
        logger.warning(
            "No 'prediction_output' found for tag '%s' for prediction '%s'",
            tag, pred_name)
        return (None, None, None)
    if len(mean) > 1:
        logger.warning(
            "Got multiple 'prediction_output' datasets for tag '%s' for "
            "prediction '%s', using only first one (%s)", tag, pred_name,
            mean[0]['filename'])
    else:
        logger.debug(
            "Found reference dataset ('prediction_output') for tag '%s' for "
            "prediction '%s': %s", tag, pred_name, mean[0]['filename'])

    # Errors
    error = grouped_data.get('prediction_output_error', [])
    logger.debug("Found error datasets for tag '%s' for prediction '%s':", tag,
                 pred_name)
    logger.debug(pformat([d['filename'] for d in error]))

    # Estimation for covariance
    cov_estimation = grouped_data.get('prediction_input', [])
    cov_estimation = select_metadata(cov_estimation, tag=tag)
    if len(cov_estimation) > 1:
        logger.warning(
            "Got multiple 'prediction_input' datasets (used for covariance "
            "estimation) for tag '%s' for prediction '%s', using only first "
            "one (%s)", tag, pred_name, cov_estimation[0]['filename'])
        cov_estimation = [cov_estimation[0]]
    else:
        logger.debug(
            "Found reference dataset ('prediction_input') for covariance "
            "estimation of tag '%s' for prediction '%s': %s", tag, pred_name,
            [d['filename'] for d in cov_estimation])

    return (mean[0], error, cov_estimation)


def main(cfg):
    """Run the diagnostic."""
    input_data = mlr.get_input_data(cfg, pattern=cfg.get('pattern'))

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
            logger.debug("Loaded reference cube at '%s'", dataset['filename'])
            ref_cube = iris.load_cube(dataset['filename'])
            if ref_cube.ndim < 1:
                logger.warning(
                    "Postprocessing scalar dataset '%s' not possible",
                    dataset['filename'])
                continue

            # Process reference cube
            postprocess_ref(cfg, ref_cube, dataset)

            # Process errors
            postprocess_errors(cfg, ref_cube, error_datasets,
                               cov_estim_datastets)


# Run main function when this script is called
if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
