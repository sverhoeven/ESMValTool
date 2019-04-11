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
anomaly : list, optional
    Calculate anomalies using reference datasets indicated by `ref: true`. Two
    datasets are matched using the list of metadata attributes given.
area_weighted : bool, optional (default: True)
    Calculate weighted averages/sums for area (using grid cell boundaries).
convert_units_to : str, optional
    Convert units of the input data. Can also be given as dataset option.
mean : list of str, optional
    Calculate the mean over the specified coordinates.
pattern : str, optional
    Pattern matched against ancestor files.
save_ref_data : bool, optional (default: False)
    Save data marked as `ref: true`.
sum : list of str, optional
    Calculate the sum of over the specified coordinates.
tag : str, optional
    Tag for the variable used in the MLR model.
time_weighted : bool, optional (default: True)
    Calculate weighted averages/sums for time (using grid cell boundaries).
trend : bool or str, optional (default: False)
    Calculate the temporal trend of the data, if `str` is given aditionally
    aggregate along that coordinate before trend calculation(e.g. `year`).

"""

import copy
import logging
import os
from functools import partial
from pprint import pformat

import iris
import numpy as np
from cf_units import Unit
from scipy import stats

from esmvaltool.diag_scripts.mlr import write_cube
from esmvaltool.diag_scripts.shared import (get_diagnostic_filename, io,
                                            run_diagnostic, select_metadata)

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


@partial(np.vectorize, excluded=['x_arr'], signature='(n),(n)->()')
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
            new_axis_pos = np.delete(np.arange(cube.ndim),
                                     cube.coord_dims('time'))
            for idx in new_axis_pos:
                time_weights = np.expand_dims(time_weights, idx)
            time_weights = np.broadcast_to(time_weights, cube.shape)
    return time_weights


def aggregate(cfg, cube):
    """Aggregate cube over specified coordinate."""
    for (coord_name, aggregator) in cfg.get('aggregate_by', {}).items():
        iris_op = AGGREGATORS.get(aggregator)
        if iris_op is None:
            logger.warning("Unknown aggregation option '%s', skipping",
                           aggregator)
            continue
        logger.debug("Aggregating coordinate %s by calculating %s", coord_name,
                     aggregator)
        try:
            cube = cube.aggregated_by(coord_name, iris_op)
        except iris.exceptions.CoordinateNotFoundError:
            if hasattr(iris.coord_categorisation, f'add_{coord_name}'):
                getattr(iris.coord_categorisation, f'add_{coord_name}')(cube,
                                                                        'time')
                logger.debug("Added coordinate '%s' to cube", coord_name)
                cube = cube.aggregated_by(coord_name, iris_op)
            else:
                logger.warning(
                    "'%s' is not a coordinate of cube %s and cannot be added "
                    "via iris.coord_categorisation", coord_name, cube)
    return cube


def calculate_anomalies(cfg, input_data):
    """Calculate anomalies using reference datasets."""
    metadata = cfg.get('anomaly')
    if not metadata:
        return input_data
    logger.info("Calculating anomalies")
    ref_data = select_metadata(input_data, ref=True)
    regular_data = select_metadata(input_data, ref=False)
    for data in regular_data:
        kwargs = {m: data[m] for m in metadata if m in data}
        ref = select_metadata(ref_data, **kwargs)
        if not ref:
            logger.warning(
                "No reference for dataset %s found, skipping "
                "anomaly calculation", data)
            continue
        if len(ref) > 1:
            logger.warning(
                "Reference data for dataset %s is not unique, found %s. "
                "Consider extending list of metadata for 'anomaly' option",
                data, ref)
            continue
        ref = ref[0]
        data['cube'].data -= ref['cube'].data
        data['standard_name'] += '_anomaly'
        data['short_name'] += '_anomaly'
        data['long_name'] += ' (anomaly)'
        data['anomaly'] = (
            f"Relative to {ref['short_name']} of {ref['dataset']} (project "
            f"{ref['project']}) of the {ref['exp']} run (years "
            f"{ref['start_year']} -- {ref['end_year']})")
    return input_data


def calculate_sum_and_mean(cfg, cube, data):
    """Calculate sum and mean."""
    cfg = copy.deepcopy(cfg)
    ops = [('mean', iris.analysis.MEAN), ('sum', iris.analysis.SUM)]
    for (oper, iris_op) in ops:
        if cfg.get(oper):
            logger.debug("Calculating %s over %s", oper, cfg[oper])
            if cfg[oper] == 'all':
                cfg[oper] = [
                    coord.name() for coord in cube.coords(dim_coords=True)
                ]

            # Latitude and longitude (weighted)
            area_weights = _get_area_weights(cfg, cube)
            if 'latitude' in cfg[oper] and 'longitude' in cfg[oper]:
                cube = cube.collapsed(['latitude', 'longitude'],
                                      iris_op,
                                      weights=area_weights)
                cfg[oper].remove('latitude')
                cfg[oper].remove('longitude')
                if oper == 'sum' and area_weights is not None:
                    cube.units *= Unit('m2')

            # Time (weighted)
            time_weights = _get_time_weights(cfg, cube)
            if 'time' in cfg[oper]:
                cube = cube.collapsed(['time'], iris_op, weights=time_weights)
                cfg[oper].remove('time')
                if oper == 'sum' and time_weights is not None:
                    time_units = (data['frequency']
                                  if data['frequency'] != 'mon' else 'month')
                    cube.units *= Unit(time_units)

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
            time_units = cfg['trend']
        else:
            time_units = (data['frequency']
                          if data['frequency'] != 'mon' else 'month')
        logger.debug("Calculating %sly trend", time_units)
        time_units += '-1'

        # Use x-axis with incremental differences of 1
        x_data = np.arange(cube.coord('time').shape[0])
        y_data = np.moveaxis(cube.data, cube.coord_dims('time')[0], -1)

        # Calculate slope for (vectorized function)
        slopes = _get_slope(x_data, y_data)
        cube = cube.collapsed('time', iris.analysis.MEAN)
        cube.data = np.ma.masked_invalid(slopes)
        cube.units *= Unit(time_units)
        data['standard_name'] += '_trend'
        data['short_name'] += '_trend'
        data['long_name'] += ' (trend)'
        data['units'] += ' {}'.format(time_units)
    return (cube, data)


def convert_units(cfg, cube, data):
    """Convert units if desired."""
    cfg_settings = cfg.get('convert_units_to')
    data_settings = data.get('convert_units_to')
    if cfg_settings or data_settings:
        units_to = cfg_settings
        if data_settings:
            units_to = data_settings
        logger.debug("Converting units from '%s' to '%s'", cube.units.symbol,
                     units_to)
        try:
            cube.convert_units(units_to)
        except ValueError:
            logger.warning("Cannot convert units from '%s' to '%s'",
                           cube.units.symbol, units_to)
        else:
            data['units'] = units_to
    return (cube, data)


def main(cfg):
    """Run the diagnostic."""
    input_data = list(cfg['input_data'].values())
    input_data.extend(io.netcdf_to_metadata(cfg, pattern=cfg.get('pattern')))
    input_data = copy.deepcopy(input_data)
    logger.debug("Found files")
    logger.debug(pformat([d['filename'] for d in input_data]))

    # Process data
    for data in input_data:
        path = data['filename']
        logger.info("Processing %s", path)
        cube = iris.load_cube(path)

        # Aggregation
        cube = aggregate(cfg, cube)

        # Sum and mean
        cube = calculate_sum_and_mean(cfg, cube, data)

        # Trend
        (cube, data) = calculate_trend(cfg, cube, data)

        # Convert units
        (cube, data) = convert_units(cfg, cube, data)

        # Cache cube
        basename = os.path.splitext(os.path.basename(path))[0]
        new_path = get_diagnostic_filename(basename, cfg)
        data['filename'] = new_path
        data['cube'] = cube
        if 'tag' in cfg and 'tag' not in data:
            data['tag'] = cfg['tag']
        if 'ref' not in data:
            data['ref'] = False

    # Calculate anomalies and save cubes
    input_data = calculate_anomalies(cfg, input_data)
    data_to_save = (input_data if cfg.get('save_ref_data', False) else
                    select_metadata(input_data, ref=False))
    for data in data_to_save:
        data.pop('ref')
        cube = data.pop('cube')
        write_cube(cube, data, data['filename'])


# Run main function when this script is called
if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
