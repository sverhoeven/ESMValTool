#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple evaluation of residuals (coming from MLR model output).

Description
-----------
This diagnostic evaulates residuals created by an MLR model.

Author
------
Manuel Schlund (DLR, Germany)

Project
-------
CRESCENDO

Configuration options in recipe
-------------------------------
convert_units_to : str, optional
    Convert units of the input data.
pattern : str, optional
    Pattern matched against ancestor files.
seaborn_settings : dict, optional
    Options for seaborn's `set()` method (affects all plots), see
    <https://seaborn.pydata.org/generated/seaborn.set.html>.

"""

import logging
import os
from copy import deepcopy
from pprint import pformat

import iris
import numpy as np
import seaborn as sns
from cf_units import Unit

from esmvaltool.diag_scripts import mlr
from esmvaltool.diag_scripts.shared import (get_diagnostic_filename,
                                            group_metadata, io, run_diagnostic,
                                            select_metadata)

logger = logging.getLogger(os.path.basename(__file__))


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


def get_grouped_datasets(cfg):
    """Get grouped datasets (by MLR model name)."""
    input_data = list(cfg['input_data'].values())
    input_data.extend(io.netcdf_to_metadata(cfg, pattern=cfg.get('pattern')))
    logger.debug("Found files")
    logger.debug(pformat([d['filename'] for d in input_data]))
    return group_metadata(input_data, 'mlr_model_name')


def main(cfg):
    """Run the diagnostic."""
    sns.set(**cfg.get('seaborn_settings', {}))
    grouped_datasets = get_grouped_datasets(cfg)

    # Iterate over all MLR models
    for (model_name, datasets) in grouped_datasets.items():
        pass




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
