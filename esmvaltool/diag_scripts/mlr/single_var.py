#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Preprocess single variables to use them as input for MLR models.

Description
-----------
This diagnostic preprocesses single variables in a desired way so that they can
be used as input data for MLR models.

Author
------
Manuel Schlund (DLR, Germany)

Project
-------
CRESCENDO

Configuration options in recipe
-------------------------------
sum : list of str, optional
    Calculate the sum of over the specified coordinates.
mean : list of str, optional
    Calculate the mean over the specified coordinates.
trend : bool or str, optional (default: False)
    Calculate the temporal trend of the data, if `str` is given aditionally
    aggregate along that coordinate before trend calculation(e.g. `year`).
weighted : bool, optional (default: True)
    Calculate weighted averages/sums (using grid cell boundaries).
tag : str, optional
    Tag for the variable used in the MLR model.

"""

import logging
import os

import iris
import numpy as np
from scipy import stats

from esmvaltool.diag_scripts.mlr import write_cube
from esmvaltool.diag_scripts.shared import (get_diagnostic_filename,
                                            run_diagnostic)

logger = logging.getLogger(os.path.basename(__file__))


def _get_slope(x_arr, y_arr):
    """Get slope of linear regression of two (masked) arrays."""
    if np.ma.is_masked(y_arr):
        x_arr = x_arr[~y_arr.mask]
        y_arr = y_arr[~y_arr.mask]
    if len(y_arr) < 2:
        return np.nan
    reg = stats.linregress(x_arr, y_arr)
    return reg.slope


def _get_weights(cfg, cube):
    """Calculate weights."""
    area_weights = None
    time_weights = None
    if cfg.get('weighted'):
        coords = [c.name() for c in cube.coords()]
        if 'latitude' in coords and 'longitude' in coords:
            logger.debug("Calculating area weights")
            area_weights = iris.analysis.cartography.area_weights(cube)
        if 'time' in coords:
            logger.debug("Calculating time weights")
            time = cube.coord('time')
            time_weights = time.bounds[:, 1] - time.bounds[:, 0]
            new_axis_pos = np.delete(cube.shape, cube.coord_dims('time')[0])
            for idx in new_axis_pos:
                time_weights = np.expand_dims(time_weights, idx)
            time_weights = np.broadcast_to(time_weights, cube.shape)
    return (area_weights, time_weights)


def calculate_sum_and_mean(cfg, cube):
    """Calculate sum and mean."""
    (area_weights, time_weights) = _get_weights(cfg, cube)
    ops = {'sum': iris.analysis.SUM, 'mean': iris.analysis.MEAN}
    for (oper, iris_op) in ops.items():
        if cfg.get(oper):
            logger.info("Calculating %s over %s", oper, cfg[oper])
            if area_weights is not None:
                cube = cube.collapsed(['latitude', 'longitude'],
                                      iris_op,
                                      weights=area_weights)
                cfg[oper].remove('latitude')
                cfg[oper].remove('longitude')
            if time_weights is not None:
                cube = cube.collapsed(['time'], iris_op, weights=time_weights)
                cfg[oper].remove('time')
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
        data['standard_name'] += '_trend'
        data['short_name'] += '_trend'
        data['long_name'] += ' (trend)'
        data['units'] += ' {}'.format(temp_units)
    return (cube, data)


def main(cfg):
    """Run the diagnostic."""
    if cfg['write_netcdf']:
        for (path, data) in cfg['input_data'].items():
            logger.info("Processing %s", path)
            data = dict(data)
            cube = iris.load_cube(path)

            # Sum and mean
            cube = calculate_sum_and_mean(cfg, cube)

            # Trend
            (cube, data) = calculate_trend(cfg, cube, data)

            # Save new cube
            basename = os.path.splitext(os.path.basename(path))[0]
            new_path = get_diagnostic_filename(basename, cfg)
            data['filename'] = new_path
            if 'tag' in cfg:
                data['tag'] = cfg['tag']
            write_cube(cube, data, new_path)
    else:
        logger.warning("Cannot save netcdf files because 'write_netcdf' is "
                       "set to 'False' in user configuration file.")


# Run main function when this script is called
if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
