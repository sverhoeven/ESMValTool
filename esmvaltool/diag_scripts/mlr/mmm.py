#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Use simple multi-model mean for predictions.

Description
-----------
This diagnostic calculates the (unweighted) mean over all given datasets for a
given target variable.

Author
------
Manuel Schlund (DLR, Germany)

Project
-------
CRESCENDO

Configuration options in recipe
-------------------------------
collapse_over : str, optional (default: 'dataset')
    Dataset attribute to collapse over.
convert_units_to : str, optional
    Convert units of the input data. Can also be given as dataset option.
unweighted_mean : bool, optional (default: False)
    Calculate unweighted multi-model mean.
median : bool, optional (default: False)
    Calculate multi-model median.
std : bool, optional (default: False)
    Calculate multi-model standard deviation.
pattern : str, optional
    Pattern matched against ancestor files.
var : bool, optional (default: False)
    Calculate multi-model variance.

"""

import logging
import os
from pprint import pformat

import iris

from esmvaltool.diag_scripts.shared import (get_diagnostic_filename,
                                            group_metadata, io, run_diagnostic,
                                            select_metadata)

logger = logging.getLogger(os.path.basename(__file__))


def add_mmm_attributes(cube, datasets, tag):
    """Add attribute to cube."""
    projects = list({d['project'] for d in datasets})
    project = '|'.join(projects)
    cube.attributes['dataset'] = 'Multi-model mean'
    cube.attributes['project'] = project
    cube.attributes['tag'] = tag
    cube.attributes['var_type'] = 'prediction_output'


def convert_units(cfg, cube, data):
    """Convert units if desired."""
    cfg_settings = cfg.get('convert_units_to')
    data_settings = data.get('convert_units_to')
    if cfg_settings or data_settings:
        units_to = cfg_settings
        if data_settings:
            units_to = data_settings
        logger.info("Converting units from '%s' to '%s'", cube.units, units_to)
        try:
            cube.convert_units(units_to)
        except ValueError:
            logger.warning("Cannot convert units from '%s' to '%s'",
                           cube.units, units_to)


def get_cubes(cfg, datasets):
    """Extract data."""
    cubes = iris.cube.CubeList()
    dataset_labels = []
    for dataset in datasets:
        path = dataset['filename']
        dataset_label = dataset[cfg.get('collapse_over', 'dataset')]
        logger.info("Processing '%s'", path)
        cube = iris.load_cube(path)
        convert_units(cfg, cube, dataset)
        preprocess_cube(cube, dataset_label)
        cubes.append(cube)
        dataset_labels.append(dataset_label)
    return (cubes, dataset_labels)


def get_grouped_data(cfg, input_data=None):
    """Get input files."""
    if input_data is None:
        logger.debug("Loading input data from 'cfg' argument")
        input_data = list(cfg['input_data'].values())
        input_data.extend(
            io.netcdf_to_metadata(cfg, pattern=cfg.get('pattern')))
    else:
        logger.debug("Loading input data from 'input_data' argument")
    paths = [d['filename'] for d in input_data]
    logger.debug("Found files")
    logger.debug(pformat(paths))

    # Extract prediction input
    logger.info("Extracting files with var_type 'label'")
    input_data = select_metadata(input_data, var_type='label')
    paths = [d['filename'] for d in input_data]
    logger.debug("Found files")
    logger.debug(pformat(paths))

    # Return grouped data
    return group_metadata(input_data, 'tag')


def preprocess_cube(cube, dataset_label):
    """Preprocess single cubes."""
    cube.attributes = {}
    cube.cell_methods = ()
    for coord in cube.coords(dim_coords=False):
        cube.remove_coord(coord)
    dataset_label_coord = iris.coords.AuxCoord(dataset_label,
                                               var_name='dataset_label',
                                               long_name='dataset_label')
    cube.add_aux_coord(dataset_label_coord, [])


def main(cfg, input_data=None, description=None):
    """Run the diagnostic."""
    grouped_data = get_grouped_data(cfg, input_data=input_data)
    descr = '' if description is None else f'_for_{description}'
    if not grouped_data:
        logger.error("No input data found")
        return

    # Loop over all tags
    for (tag, datasets) in grouped_data.items():
        logger.info("Processing label '%s'", tag)

        # Extract data
        (cubes, dataset_labels) = get_cubes(cfg, datasets)

        # Merge cubes
        mm_cube = cubes.merge_cube()

        # Calculate (unweighted) multi-dataset mean
        if len(dataset_labels) > 1:
            mm_cube = mm_cube.collapsed(['dataset_label'], iris.analysis.MEAN)
        new_path = get_diagnostic_filename(f'mmm_{tag}_prediction{descr}', cfg)
        add_mmm_attributes(mm_cube, datasets, tag)
        io.iris_save(mm_cube, new_path)


# Run main function when this script is called
if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
