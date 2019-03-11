#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple Pre-/Postprocessing for MLR models.

Description
-----------
This diagnostic performs simple pre-/postprocessing steps for variables in a
desired way.

Author
------
Manuel Schlund (DLR, Germany)

Project
-------
CRESCENDO

Configuration options in recipe
-------------------------------
aggregate_by : dict, optional
    Aggregate over given coordinates (dict keys) using a desired aggregator
    (dict values). Allowed aggregators are `mean`, `median`, `std` and `var`.
sum : list of str, optional
    Calculate the sum of over the specified coordinates.
mean : list of str, optional
    Calculate the mean over the specified coordinates.
trend : bool or str, optional (default: False)
    Calculate the temporal trend of the data, if `str` is given aditionally
    aggregate along that coordinate before trend calculation(e.g. `year`).
area_weighted : bool, optional (default: True)
    Calculate weighted averages/sums for area (using grid cell boundaries).
time_weighted : bool, optional (default: True)
    Calculate weighted averages/sums for time (using grid cell boundaries).
tag : str, optional
    Tag for the variable used in the MLR model.
convert_units_to : str, optional
    Convert units of the input data. Can also be given as dataset option.
pattern : str, optional
    Pattern matched against ancestor files.

"""

import copy
import logging
import os
from pprint import pformat

import iris
import numpy as np
from cf_units import Unit
from esmvaltool.diag_scripts.mlr import write_cube
from esmvaltool.diag_scripts.shared import (get_diagnostic_filename, io,
                                            run_diagnostic)
from scipy import stats

logger = logging.getLogger(os.path.basename(__file__))

AGGREGATORS = {
    'mean': iris.analysis.MEAN,
    'median': iris.analysis.MEDIAN,
    'std': iris.analysis.STD_DEV,
    'var:': iris.analysis.VARIANCE,
}


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


def _get_slope(x_arr, y_arr):
    """Get slope of linear regression of two (masked) arrays."""
    if np.ma.is_masked(y_arr):
        x_arr = x_arr[~y_arr.mask]
        y_arr = y_arr[~y_arr.mask]
    if len(y_arr) < 2:
        return np.nan
    reg = stats.linregress(x_arr, y_arr)
    return reg.slope


def _get_area_weights(cfg, cube):
    """Calculate area weights."""
    area_weights = None
    if cfg.get('area_weighted', True):
        for coord in cube.coords(dim_coords=True):
            if not coord.has_bounds():
                logger.debug("Guessing bounds of coordinate '%s' of cube",
                             coord.name)
                logger.debug(cube)
                coord.guess_bounds()
        if _has_valid_coords(cube, ['latitude', 'longitude']):
            logger.debug("Calculating area weights")
            area_weights = iris.analysis.cartography.area_weights(cube)
    return area_weights


def _get_time_weights(cfg, cube):
    """Calculate time weights."""
    time_weights = None
    if cfg.get('time_weighted', True):
        for coord in cube.coords(dim_coords=True):
            if not coord.has_bounds():
                logger.debug("Guessing bounds of coordinate '%s' of cube",
                             coord.name)
                logger.debug(cube)
                coord.guess_bounds()
        if _has_valid_coords(cube, ['time']):
            logger.debug("Calculating time weights")
            time = cube.coord('time')
            time_weights = time.bounds[:, 1] - time.bounds[:, 0]
            new_axis_pos = np.delete(
                np.arange(cube.ndim), cube.coord_dims('time'))
            for idx in new_axis_pos:
                time_weights = np.expand_dims(time_weights, idx)
            time_weights = np.broadcast_to(time_weights, cube.shape)
    return time_weights


def aggregate(cfg, cube):
    """Aggregate cube over specified coordinate."""
    cfg = copy.deepcopy(cfg)
    for (coord_name, aggregator) in cfg.get('aggregate_by', {}).items():
        iris_op = AGGREGATORS.get(aggregator)
        if iris_op is None:
            logger.warning("Unknown aggregation option '%s', skipping",
                           aggregator)
            continue
        logger.info("Aggregating coordinate %s by calculating %s", coord_name,
                    aggregator)
        cube = cube.aggregated_by(coord_name, iris_op)
    return cube


def calculate_sum_and_mean(cfg, cube):
    """Calculate sum and mean."""
    cfg = copy.deepcopy(cfg)
    ops = [('mean', iris.analysis.MEAN), ('sum', iris.analysis.SUM)]
    for (oper, iris_op) in ops:
        if cfg.get(oper):
            logger.info("Calculating %s over %s", oper, cfg[oper])
            if cfg[oper] == 'all':
                cfg[oper] = [
                    coord.name() for coord in cube.coords(dim_coords=True)
                ]

            # Latitude and longitude (weighted)
            area_weights = _get_area_weights(cfg, cube)
            if all([
                    area_weights is not None,
                    'latitude' in cfg[oper],
                    'longitude' in cfg[oper],
            ]):
                cube = cube.collapsed(['latitude', 'longitude'],
                                      iris_op,
                                      weights=area_weights)
                cfg[oper].remove('latitude')
                cfg[oper].remove('longitude')

            # Time (weighted)
            time_weights = _get_time_weights(cfg, cube)
            if all([time_weights is not None, 'time' in cfg[oper]]):
                cube = cube.collapsed(['time'], iris_op, weights=time_weights)
                cfg[oper].remove('time')

            # Remaining operations
            if cfg[oper]:
                cube = cube.collapsed(cfg[oper], iris_op)
    return cube


def calculate_trend(cfg, cube, data):
    """Calculate trend."""
    if cfg.get('trend'):
        if isinstance(cfg['trend'], str):
            logger.debug("Aggregating over %s for trend calculation",
                         cfg['trend'])
            cube = cube.aggregated_by(cfg['trend'], iris.analysis.MEAN)
            temp_units = cfg['trend']
        else:
            temp_units = (data['frequency']
                          if data['frequency'] != 'mon' else 'month')
        logger.info("Calculating %sly trend", temp_units)
        temp_units += '-1'

        # Use x-axis with incremental differences of 1
        x_data = np.arange(cube.coord('time').shape[0])
        y_data = np.moveaxis(cube.data, cube.coord_dims('time')[0], -1)

        # Calculate slope for (vectorized function)
        v_get_slope = np.vectorize(
            _get_slope, excluded=['x'], signature='(n),(n)->()')
        slopes = v_get_slope(x_data, y_data)
        cube = cube.collapsed('time', iris.analysis.MEAN)
        cube.data = np.ma.masked_invalid(slopes)
        cube.units *= Unit(temp_units)
        data['standard_name'] += '_trend'
        data['short_name'] += '_trend'
        data['long_name'] += ' (trend)'
        data['units'] += ' {}'.format(temp_units)
    return (cube, data)


def convert_units(cfg, cube, data):
    """Convert units if desired."""
    cfg_settings = cfg.get('convert_units_to')
    data_settings = data.get('convert_units_to')
    if cfg_settings or data_settings:
        units_to = cfg_settings
        if data_settings:
            units_to = data_settings
        logger.info("Converting units from '%s' to '%s'", cube.units.origin,
                    units_to)
        try:
            cube.convert_units(units_to)
        except ValueError:
            logger.warning("Cannot convert units from '%s' to '%s'",
                           cube.units.origin, units_to)
        else:
            data['units'] = units_to
    return (cube, data)


def main(cfg):
    """Run the diagnostic."""
    input_data = list(cfg['input_data'].values())
    input_data.extend(io.netcdf_to_metadata(cfg, pattern=cfg.get('pattern')))
    paths = [d['filename'] for d in input_data]
    logger.debug("Found files")
    logger.debug(pformat(paths))

    # Iterate over all data
    for data in input_data:
        path = data['filename']
        logger.info("Processing %s", path)
        data = dict(data)
        cube = iris.load_cube(path)

        # Aggregation
        cube = aggregate(cfg, cube)

        # Sum and mean
        cube = calculate_sum_and_mean(cfg, cube)

        # Trend
        (cube, data) = calculate_trend(cfg, cube, data)

        # Convert units
        (cube, data) = convert_units(cfg, cube, data)

        # Save new cube
        basename = os.path.splitext(os.path.basename(path))[0]
        new_path = get_diagnostic_filename(basename, cfg)
        data['filename'] = new_path
        if 'tag' in cfg and 'tag' not in data:
            data['tag'] = cfg['tag']
        write_cube(cube, data, new_path)


# Run main function when this script is called
if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
