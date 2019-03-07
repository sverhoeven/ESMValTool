#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple multi-model statistics to evaluate MLR models.

Description
-----------
This diagnostic applies simple multi-model statistics to evaluate MLR models.

Author
------
Manuel Schlund (DLR, Germany)

Project
-------
CRESCENDO

Configuration options in recipe
-------------------------------
mean : bool, optional (default: False)
    Calculate multi-model mean.
median : bool, optional (default: False)
    Calculate multi-model median.
std : bool, optional (default: False)
    Calculate multi-model standard deviation.
var : bool, optional (default: False)
    Calculate multi-model variance.
plot : bool, optional (default: True)
    Plot results.
convert_units_to : str, optional
    Convert units of the input data. Can also be given as dataset option.
pattern : str, optional
    Pattern matched against ancestor files.

"""

import logging
import os
from pprint import pformat

import iris
from esmvaltool.diag_scripts.shared import (get_diagnostic_filename, io,
                                            run_diagnostic)

logger = logging.getLogger(os.path.basename(__file__))

STATS = {
    'mean': iris.analysis.MEAN,
    'median': iris.analysis.MEDIAN,
    'std': iris.analysis.STD_DEV,
    'var:': iris.analysis.VARIANCE,
}


def add_mm_cube_attributes(cube, input_data, stat, path):
    """Add attribute to cube."""
    projects = {d['project'] for d in input_data}
    project = 'Multiple projects'
    if len(projects) == 1:
        project = projects.pop()

    # Modify attributes
    attrs = cube.attributes
    attrs['dataset'] = f'Multi-model {stat}'
    attrs['project'] = project
    attrs['filename'] = path


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


def preprocess_cube(cube, dataset):
    """Preprocess single cubes."""
    cube.attributes = {}
    cube.cell_methods = ()
    for coord in cube.coords(dim_coords=False):
        cube.remove_coord(coord)
    dataset_coord = iris.coords.AuxCoord(
        dataset, var_name='dataset', long_name='dataset')
    cube.add_aux_coord(dataset_coord, [])


def main(cfg):
    """Run the diagnostic."""
    input_data = list(cfg['input_data'].values())
    input_data.extend(io.netcdf_to_metadata(cfg, pattern=cfg.get('pattern')))
    paths = [d['filename'] for d in input_data]
    logger.debug("Found files")
    logger.debug(pformat(paths))
    datasets = []
    cubes = iris.cube.CubeList()

    # Iterate over all data
    for data in input_data:
        logger.info("Processing %s", data['filename'])
        data = dict(data)
        cube = iris.load_cube(data['filename'])

        # Convert units
        (cube, data) = convert_units(cfg, cube, data)

        # Add dataset coordinate and append to CubeList
        preprocess_cube(cube, data['dataset'])
        datasets.append(data['dataset'])
        cubes.append(cube)

    # Merge cubes
    mm_cube = cubes.merge_cube()

    # Calculate desired statistics
    stats = {}
    for (stat, iris_op) in STATS.items():
        if cfg.get(stat):
            stats[stat] = iris_op
    for (stat, iris_op) in stats.items():
        new_cube = mm_cube.collapsed('dataset', iris_op)
        new_path = get_diagnostic_filename(stat, cfg)
        add_mm_cube_attributes(new_cube, input_data, stat, new_path)
        io.save_iris_cube(new_cube, new_path)
        datasets.append(stat)
        cubes.append(new_cube)


# Run main function when this script is called
if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
