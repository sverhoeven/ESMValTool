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
        logger.debug("Converting units from '%s' to '%s'", cube.units.symbol,
                     units_to)
        try:
            cube.convert_units(units_to)
        except ValueError:
            logger.warning("Cannot convert units from '%s' to '%s'",
                           cube.units.symbol, units_to)


def _collapse_covariance_cube(cfg, cov_cube, ref_cube):
    """Collapse covariance cube with using desired operations."""
    cfg = deepcopy(cfg)
    cov_cube = cov_cube.copy()

    # Check shape of covariance_cube
    ref_size = np.ma.array(ref_cube.data).compressed().shape[0]
    if cov_cube.shape != (ref_size, ref_size):
        logger.warning(
            "Expected shape of covariance cube to be %s, got %s (after "
            "removal of all missing values)", (ref_size, ref_size),
            cov_cube.shape)
        return None

    # Calculate weights
    (weights, units, _) = _get_all_weights(cfg, ref_cube)
    weights = weights.ravel()
    weights = weights[~np.ma.getmaskarray(ref_cube.data).ravel()]
    weights = np.outer(weights, weights)

    # Calculate covariance
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
    pearson_dim0 = _corrcoef(ref)
    pearson_dim1 = _corrcoef(ref, rowvar=False)

    # Covariances
    cov_dim0 = (np.einsum('...i,...j->...ij', error, error) *
                np.einsum('...i,...j->...ij', weights, weights) * pearson_dim1)
    cov_dim1 = (np.einsum('i...,j...->...ij', error, error) *
                np.einsum('i...,j...->...ij', weights, weights) * pearson_dim0)

    # Errors over dimensions
    error_dim0 = np.ma.sqrt(np.ma.sum(cov_dim0, axis=(1, 2)))
    error_dim1 = np.ma.sqrt(np.ma.sum(cov_dim1, axis=(1, 2)))

    # Collaps further
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
    cfg = deepcopy(cfg)
    cube = cube.copy()
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


def _corrcoef(array, rowvar=True):
    """Fast version of :mod:`np.ma.corrcoef`."""
    if not rowvar:
        array = array.T
    demean = array - np.ma.mean(array, axis=1).reshape(-1, 1)
    res = np.ma.dot(demean, demean.T)
    row_norms = np.ma.sqrt(np.ma.sum(demean**2, axis=1))
    res /= np.ma.outer(row_norms, row_norms)
    return res


def _get_all_weights(cfg, cube, power=1):
    """Get all necessary weights (including mean calculation)."""
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
        for coord in cube.coords(dim_coords=True):
            if not coord.has_bounds():
                logger.debug("Guessing bounds of coordinate '%s' of cube",
                             coord.name())
                logger.debug(cube)
                coord.guess_bounds()
        if _has_valid_coords(cube, ['latitude', 'longitude']):
            logger.debug("Calculating area weights")
            area_weights = iris.analysis.cartography.area_weights(
                cube, normalize=normalize)
            area_weights = area_weights**power
    return (area_weights, Unit('m2')**power)


def _get_covariance_dataset(error_datasets):
    """Extract covariance dataset."""
    cov_datasets = []
    other_datasets = []
    for dataset in error_datasets:
        if '_cov' in dataset['short_name']:
            cov_datasets.append(dataset)
        else:
            other_datasets.append(dataset)
    if not cov_datasets:
        return (None, other_datasets)
    if len(cov_datasets) > 1:
        logger.warning(
            "Got multiple error datasets for covariance, using only first "
            "one (%s)", cov_datasets[0]['filename'])
    cov_cube = iris.load_cube(cov_datasets[0]['filename'])
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
    time_units = _get_time_units(cube.coord('time').units)
    if cfg.get('time_weighted', True):
        for coord in cube.coords(dim_coords=True):
            if not coord.has_bounds():
                logger.debug("Guessing bounds of coordinate '%s' of cube",
                             coord.name())
                logger.debug(cube)
                coord.guess_bounds()
        if _has_valid_coords(cube, ['time']):
            logger.debug("Calculating time weights")
            time = cube.coord('time')
            time_weights = time.bounds[:, 1] - time.bounds[:, 0]
            if normalize:
                time_weights /= np.ma.sum(time_weights)
            new_axis_pos = np.delete(np.arange(cube.ndim),
                                     cube.coord_dims('time'))
            for idx in new_axis_pos:
                time_weights = np.expand_dims(time_weights, idx)
            time_weights = np.broadcast_to(time_weights, cube.shape)
            time_weights = time_weights**power
    return (time_weights, time_units**power)


def _get_time_units(units):
    """Get non-relative time units."""
    if units.is_time_reference():
        units = Unit(units.symbol.split()[0])
        if not units.is_time():
            raise ValueError(
                f"Cannot convert time reference units {units.symbol} to "
                f"reasonable time units")
    return units


def _has_valid_coords(cube, coord_names):
    """Check if a cube has valid coordinates (length > 1)."""
    for coord_name in coord_names:
        try:
            coord = cube.coord(coord_name)
        except iris.exceptions.CoordinateNotFoundError:
            return False
        if coord.shape[0] <= 1:
            return False
    return True


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
    (cov_cube, error_datasets) = _get_covariance_dataset(error_datasets)

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

    # Lower and upper error bounds
    if error_datasets:
        _calculate_lower_error_bound(cfg, squared_error_cube, basepath)
        _calculate_upper_error_bound(cfg, squared_error_cube, basepath)

        # Estimated real error using estimated covariance
        if cov_estim_datasets:
            _estimate_real_error(cfg, squared_error_cube,
                                 cov_estim_datasets[0], basepath)

    # Calculate real error if possible
    if cov_cube:
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
    msg = f' for {pred_name}' if pred_name is not None else ''
    datasets = [d for d in datasets if not d.get('skip_for_postprocessing')]
    grouped_data = group_metadata(datasets, 'var_type')

    # Mean/reference dataset
    mean = grouped_data.get('prediction_output')
    if not mean:
        logger.warning("No 'prediction_output' for tag '%s'%s", tag, msg)
        return (None, None)
    if len(mean) > 1:
        logger.warning(
            "Got multiple 'prediction_output' datasets for tag '%s'%s, using "
            "only first one (%s)", tag, msg, mean[0]['filename'])
    else:
        logger.debug("Found reference dataset (mean) for tag '%s'%s: %s", tag,
                     msg, mean[0]['filename'])

    # Errors
    error = grouped_data.get('prediction_output_error', [])
    logger.debug("Found error datasets for tag '%s'%s:", tag, msg)
    logger.debug(pformat([d['filename'] for d in error]))

    # Estimation for covariance
    cov_estimation = grouped_data.get('prediction_input', [])
    cov_estimation = select_metadata(cov_estimation, tag=tag)
    if len(cov_estimation) > 1:
        logger.warning(
            "Got multiple 'prediction_input' datasets (used for covariance "
            "estimation) for tag '%s'%s, using only first one (%s)", tag, msg,
            cov_estimation[0]['filename'])
        cov_estimation = [cov_estimation[0]]
    else:
        logger.debug(
            "Found reference dataset for covariance estimation tag '%s'%s: %s",
            tag, msg, [d['filename'] for d in cov_estimation])

    return (mean[0], error, cov_estimation)


def main(cfg):
    """Run the diagnostic."""
    input_data = list(cfg['input_data'].values())
    input_data.extend(io.netcdf_to_metadata(cfg, pattern=cfg.get('pattern')))
    input_data = deepcopy(input_data)
    if not mlr.datasets_have_mlr_attributes(input_data, log_level='error'):
        return
    logger.debug("Found files")
    logger.debug(pformat([d['filename'] for d in input_data]))

    # Check cfg
    check_cfg(cfg)

    # Process data
    for (tag, tag_datasets) in group_metadata(input_data, 'tag').items():
        logger.info("Processing tag '%s'", tag)
        grouped_data = group_metadata(tag_datasets, 'prediction_name')
        for (pred_name, datasets) in grouped_data.items():
            if pred_name is not None:
                logger.info("Processing prediction '%s'", pred_name)
            (dataset, error_datasets,
             cov_estim_datastets) = split_datasets(datasets, tag, pred_name)

            # Extract cubes
            logger.debug("Loaded reference cube at %s", dataset['filename'])
            ref_cube = iris.load_cube(dataset['filename'])

            # Process reference cube
            postprocess_ref(cfg, ref_cube, dataset)

            # Process errors
            postprocess_errors(cfg, ref_cube, error_datasets,
                               cov_estim_datastets)


# Run main function when this script is called
if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
