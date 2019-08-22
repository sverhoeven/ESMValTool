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
mlr_model_name : str, optional
    Human-readable name of the MLR model instance (e.g used for labels).
pattern : str, optional
    Pattern matched against ancestor files.

"""

import logging
import os
from pprint import pformat

import iris

import esmvaltool.diag_scripts.shared.iris_helpers as ih
from esmvaltool.diag_scripts import mlr
from esmvaltool.diag_scripts.shared import (get_diagnostic_filename,
                                            group_metadata, io, run_diagnostic,
                                            select_metadata)

logger = logging.getLogger(os.path.basename(__file__))


def _add_dataset_attributes(cube, datasets, cfg):
    """Add dataset-related attributes to cube."""
    dataset_names = list({d['dataset'] for d in datasets})
    projects = list({d['project'] for d in datasets})
    start_years = list({d['start_year'] for d in datasets})
    end_years = list({d['end_year'] for d in datasets})
    cube.attributes['dataset'] = '|'.join(dataset_names)
    cube.attributes['description'] = 'MMM prediction'
    cube.attributes['end_year'] = min(end_years)
    cube.attributes['mlr_model_name'] = cfg.get('mlr_model_name', 'MMM')
    cube.attributes['mlr_model_type'] = 'mmm'
    cube.attributes['project'] = '|'.join(projects)
    cube.attributes['start_year'] = min(start_years)
    cube.attributes['var_type'] = 'prediction_output'


def add_general_attributes(cube, **kwargs):
    """Add general attributes to cube."""
    for (key, val) in kwargs.items():
        if val is not None:
            cube.attributes[key] = val


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


def get_grouped_data(cfg, input_data=None):
    """Get input files."""
    if input_data is None:
        logger.debug("Loading input data from 'cfg' argument")
        input_data = mlr.get_input_data(cfg, pattern=cfg.get('pattern'))
    else:
        logger.debug("Loading input data from 'input_data' argument")
        valid_datasets = []
        for dataset in input_data:
            if mlr.datasets_have_mlr_attributes([dataset],
                                                log_level='warning'):
                valid_datasets.append(dataset)
            else:
                logger.warning("Skipping file %s", dataset['filename'])
        if not valid_datasets:
            logger.warning("No valid input data found")
        input_data = valid_datasets
    paths = [d['filename'] for d in input_data]
    logger.debug("Found files")
    logger.debug(pformat(paths))

    # Extract necessary data
    extracted_data = select_metadata(input_data, var_type='label')
    extracted_data.extend(
        select_metadata(input_data, var_type='prediction_reference'))
    logger.debug(
        "Extracted files with var_types 'label' and 'prediction_reference'")
    paths = [d['filename'] for d in extracted_data]
    logger.debug("Found files")
    logger.debug(pformat(paths))

    # Return grouped data
    return group_metadata(extracted_data, 'tag')


def get_mm_cube(cfg, datasets):
    """Extract data."""
    cubes = iris.cube.CubeList()
    cube_labels = []
    for dataset in select_metadata(datasets, var_type='label'):
        path = dataset['filename']
        cube_label = dataset[cfg.get('collapse_over', 'dataset')]
        logger.info("Processing '%s'", path)
        cube = iris.load_cube(path)
        convert_units(cfg, cube, dataset)
        ih.preprocess_cube_before_merging(cube, cube_label)
        cubes.append(cube)
        cube_labels.append(cube_label)
    mm_cube = cubes.merge_cube()
    if len(cube_labels) > 1:
        mm_cube = mm_cube.collapsed(['cube_label'], iris.analysis.MEAN)
    _add_dataset_attributes(mm_cube, datasets, cfg)
    return mm_cube


def get_reference_dataset(datasets, tag):
    """Get `prediction_reference` dataset."""
    ref_datasets = select_metadata(datasets, var_type='prediction_reference')
    if not ref_datasets:
        logger.debug(
            "Calculating residuals for '%s' not possible, no "
            "'prediction_reference' dataset given", tag)
        return (None, None)
    if len(ref_datasets) > 1:
        logger.warning(
            "Multiple 'prediction_reference' datasets for '%s' given, "
            "using only first one (%s)", tag, ref_datasets[0]['filename'])
    return (ref_datasets[0], ref_datasets[0].get('prediction_name'))


def get_residual_cube(mm_cube, ref_cube):
    """Calculate residuals."""
    logger.info("Calculating residuals")
    if mm_cube.shape != ref_cube.shape:
        logger.warning(
            "Shapes of 'label' and 'prediction_reference' data differs, "
            "got %s for 'label' and %s for 'prediction_reference'",
            mm_cube.shape, ref_cube.shape)
        return None
    res_cube = mm_cube.copy()
    res_cube.data -= ref_cube.data
    res_cube.attributes['residuals'] = 'predicted minus true values'
    res_cube.attributes['var_type'] = 'prediction_residual'
    res_cube.var_name += '_residual'
    res_cube.long_name += ' (residual)'
    return res_cube


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

        # Get reference dataset if possible
        (ref_dataset, pred_name) = get_reference_dataset(datasets, tag)

        # Calculate multi-model mean
        logger.info("Calculating multi-model mean")
        mm_cube = get_mm_cube(cfg, datasets)
        add_general_attributes(mm_cube, tag=tag, prediction_name=pred_name)
        mm_path = get_diagnostic_filename(f'mmm_{tag}_prediction{descr}', cfg)
        io.iris_save(mm_cube, mm_path)

        # Calculate residuals
        if ref_dataset is None:
            continue
        ref_cube = iris.load_cube(ref_dataset['filename'])
        res_cube = get_residual_cube(mm_cube, ref_cube)
        add_general_attributes(res_cube, tag=tag, prediction_name=pred_name)
        res_path = mm_path.replace('_prediction', '_prediction_residual')
        io.iris_save(res_cube, res_path)


# Run main function when this script is called
if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
