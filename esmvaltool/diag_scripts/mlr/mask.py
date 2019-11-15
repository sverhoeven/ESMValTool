#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple masking for MLR output.

Description
-----------
This diagnostic masks MLR model output based on a reference dataset and a
desired set of masking operations.

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
reference_dataset : dict
    Metadata describing the reference dataset. Must refer to exactly one
    dataset.
masking_operations : list of dict
    Masking operations which will be applied on the reference dataset to create
    the mask. Keys have to be :mod:`numpy.ma` conversion operations (see
    https://docs.scipy.org/doc/numpy/reference/routines.ma.html) and values
    all the keyword arguments of them.
mean : list of str, optional
    Preprocess reference dataset by calculate the mean over the specified
    coordinates.
pattern : str, optional
    Pattern matched against ancestor files.
sum : list of str, optional
    Preprocess reference dataset by calculating the sum of over the specified
    coordinates.
time_weighted : bool, optional (default: True)
    Calculate weighted averages/sums for time (using grid cell boundaries).

"""

import logging
import os

import iris
import numpy as np

from esmvaltool.diag_scripts import mlr
# TODO
# from esmvaltool.diag_scripts.mlr.preprocess import calculate_sum_and_mean
from esmvaltool.diag_scripts.shared import (get_diagnostic_filename, io,
                                            run_diagnostic, select_metadata)

logger = logging.getLogger(os.path.basename(__file__))


def check_cfg(cfg):
    """Check if all necessary configuration options are given."""
    for option in ('reference_dataset', 'masking_operations'):
        if option not in cfg:
            raise ValueError(
                f"Necessary option '{option}' for this script  not given in "
                f"recipe")


def get_ref_mask(cfg, input_data):
    """Create mask from reference dataset."""
    ref_metadata = cfg['reference_dataset']
    ref_datasets = select_metadata(input_data, **ref_metadata)
    if len(ref_datasets) != 1:
        raise ValueError(
            f"Cannot determine reference dataset, expected exactly one "
            f"dataset with metadata {ref_metadata}, got {len(ref_datasets):d}")
    logger.info("Using reference dataset '%s'", ref_datasets[0]['filename'])
    ref_cube = iris.load_cube(ref_datasets[0]['filename'])
    for (masking_op, kwargs) in cfg['masking_operations'].items():
        if not hasattr(np.ma, masking_op):
            raise AttributeError(
                f"Invalid masking operation, '{masking_op}' is not a function "
                f"of numpy.ma")
        ref_cube.data = getattr(np.ma, masking_op)(ref_cube.data, **kwargs)
    return np.ma.getmaskarray(ref_cube.data)


def main(cfg):
    """Run the diagnostic."""
    check_cfg(cfg)
    input_data = mlr.get_input_data(cfg, pattern=cfg.get('pattern'))
    ref_mask = get_ref_mask(cfg, input_data)

    # Mask input cubes
    for dataset in input_data:
        filename = dataset['filename']
        logger.info("Processing '%s'", filename)
        cube = iris.load_cube(filename)

        # Check shape
        if cube.shape != ref_mask.shape:
            raise ValueError(
                f"Shape of '{filename}' does not match shape of reference "
                f"dataset, got {cube.shape}, expected {ref_mask.shape}")

        # Apply mask and save file
        cube.data = np.ma.array(cube.data, mask=ref_mask)
        basename = os.path.splitext(os.path.basename(filename))[0]
        new_path = get_diagnostic_filename(basename, cfg)
        io.iris_save(cube, new_path)


# Run main function when this script is called
if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
