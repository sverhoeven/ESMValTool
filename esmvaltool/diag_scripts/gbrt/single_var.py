#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Preprocess single variables to use them as input for GBRT models.

Description
-----------
This diagnostic preprocesses single variables in a desired way so that they can
be used as input data for GBRT models.

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
    Tag for the variable used in the GBRT model.

"""

import logging
import os

import cf_units
import iris
import numpy as np
from scipy import stats

from esmvaltool.diag_scripts.gbrt import write_cube
from esmvaltool.diag_scripts.shared import run_diagnostic

logger = logging.getLogger(os.path.basename(__file__))


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
        logger.debug("Calculating %s", oper)
        if cfg.get(oper):
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
            temp_units = cf_units.Unit(cfg['trend'])
        else:
            temp_units = (data['frequency']
                          if data['frequency'] != 'mon' else 'month')
        logger.debug("Calculating trend (units: %s)", temp_units)

        # Use x-axis with incremental differences of 1
        time = np.arange(cube.coord('time').shape[0])
        reg = stats.linregress(time, cube.data)
        cube = cube.collapsed('time', iris.analysis.MEAN)
        cube.data = reg.slope
        data['standard_name'] += '_trend'
        data['short_name'] += '_trend'
        data['long_name'] += ' (trend)'
        new_units = cf_units.Unit(data['units']) / temp_units
        data['units'] = new_units.name
    return (cube, data)


def main(cfg):
    """Run the diagnostic."""
    if cfg['write_netcdf']:
        for (path, data) in cfg['input_data'].items():
            logger.info("Processing %s", path)
            cube = iris.load_cube(path)

            # Sum and mean
            cube = calculate_sum_and_mean(cfg, cube)

            # Trend
            (cube, data) = calculate_trend(cfg, cube, data)

            # Save new cube
            new_path = os.path.join(cfg['work_dir'], os.path.basename(path))
            data['filename'] = new_path
            if 'tag' in cfg:
                data['tag'] = cfg['tag']
            write_cube(cube, data, new_path, cfg)
    else:
        logger.warning("Cannot save netcdf files because 'write_netcdf' is "
                       "set to 'False' in user configuration file.")


# Run main function when this script is called
if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
