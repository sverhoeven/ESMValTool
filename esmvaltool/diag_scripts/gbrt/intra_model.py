#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Diagnostic script to create GBRT models for every climate model.

Description
-----------
This diagnostic creates a "Gradient Boosted Regression Trees" (GBRT) model to
predict future climate for each climate model given in the recipe.

Notes
-----
All datasets must have the attribute 'var_type' which specifies this dataset.
Possible values are 'feature' (independent variables used for
training/testing), 'label' (dependent variables, y-axis) or 'prediction_input'
(independent variables used for prediction of dependent variables, usually
observational data). All 'feature' and 'label' datasets must have the same
shape, except the attribute 'broadcast_from' is set to a list of suitable
coordinate indices (must be done for each feature/label). This also applies to
the 'prediction_input' data sets.

Author
------
Manuel Schlund (DLR, Germany)

Project
-------
CRESCENDO

Configuration options in recipe
-------------------------------
parameters : dict, optional
    Paramter used in the classifier, more information is available here:
    https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting.
use_coords_as_feature : dict, optional
    Use coordinates (e.g. 'air_pressure' or 'latitude') as features, coordinate
    names are given by the dictionary keys, the associated index by the
    dictionary values.
use_only_coords_as_features : bool, optional
    Use only the specified coordinates as features (default: False).

"""

import logging
import os

import iris
import numpy as np

from esmvaltool.diag_scripts.shared.gbrt import GBRTBase
from esmvaltool.diag_scripts.shared import group_metadata, run_diagnostic

logger = logging.getLogger(os.path.basename(__file__))


class IntraModelGBRT(GBRTBase):
    """Intra-model GBRT diagnostic."""

    def _collect_x_data(self, datasets, var_type):
        """Collect x data from `datasets`."""
        x_data = []
        names = []
        skipped_datasets = []
        coords = None
        cube = None

        # Iterate over data
        for dataset in datasets:
            if 'broadcast_from' in dataset:
                skipped_datasets.append(dataset)
                continue
            cube = iris.load_cube(dataset['filename'])
            name = dataset.get('label', dataset['short_name'])
            if coords is None:
                coords = cube.coords()
            else:
                if cube.coords() != coords:
                    raise ValueError("Expected fields with identical "
                                     "coordinates but '{}' for dataset '{}' "
                                     "('{}') is differing, consider "
                                     "regridding or the option "
                                     "'broadcast_from'".format(
                                         name, dataset['dataset'], var_type))
            if not self._cfg.get('use_only_coords_as_features'):
                x_data.append(cube.data.ravel())
                names.append(name)

        # Check if data was found
        if cube is None:
            if skipped_datasets:
                raise ValueError(
                    "Expected at least one '{}' dataset without "
                    "the option 'broadcast_from'".format(var_type))
            else:
                raise ValueError("No '{}' datasets found".format(var_type))

        # Add skipped data (which needs broadcasting)
        broadcasted_data = self._get_broadcasted_data(skipped_datasets,
                                                      cube.shape)
        x_data.extend(broadcasted_data[0])
        names.extend(broadcasted_data[1])

        # Add coordinate data if desired and possible
        coord_data = self._get_coordinate_data(cube)
        x_data.extend(coord_data[0])
        names.extend(coord_data[1])

        # Convert data to numpy array with correct shape
        x_data = np.array(x_data)
        if x_data.ndim > 1:
            x_data = np.swapaxes(x_data, 0, 1)
        return (x_data, names, cube)

    def _collect_y_data(self, datasets):
        """Collect y data from `datasets`."""
        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            raise ValueError("Expected exactly one dataset with var_type "
                             "'label', got {}".format(len(datasets)))
        cube = iris.load_cube(dataset['filename'])
        name = dataset.get('label', dataset['short_name'])
        return (cube.data.ravel(), name, cube)

    def _get_broadcasted_data(self, datasets, target_shape):
        """Get broadcasted data."""
        new_data = []
        names = []
        if not datasets:
            return (new_data, names)
        var_type = datasets[0]['var_type']
        for dataset in datasets:
            cube_to_broadcast = iris.load_cube(dataset['filename'])
            data_to_broadcast = cube_to_broadcast.data
            name = dataset.get('label', dataset['short_name'])
            try:
                new_axis_pos = np.delete(
                    np.arange(len(target_shape)), dataset['broadcast_from'])
            except IndexError:
                raise ValueError("Broadcasting failed for '{}', index out of "
                                 "bounds".format(name))
            logger.info("Broadcasting %s '%s' from %s to %s", var_type, name,
                        data_to_broadcast.shape, target_shape)
            for idx in new_axis_pos:
                data_to_broadcast = np.expand_dims(data_to_broadcast, idx)
            data_to_broadcast = np.broadcast_to(data_to_broadcast,
                                                target_shape)
            if not self._cfg.get('use_only_coords_as_features'):
                new_data.append(data_to_broadcast.ravel())
                names.append(name)
        return (new_data, names)

    def _get_coordinate_data(self, cube):
        """Get coordinate variables of a `cube` which can be used as x data."""
        new_data = []
        names = []

        # Iterate over desired coordinates
        for (coord, coord_idx) in self._cfg.get('use_coords_as_feature',
                                                {}).items():
            coord_array = cube.coord(coord).points
            try:
                new_axis_pos = np.delete(np.arange(len(cube.shape)), coord_idx)
            except IndexError:
                raise ValueError("'use_coords_as_feature' failed, index '{}'"
                                 "is out of bounds for coordinate "
                                 "'{}'".format(coord_idx, coord))
            for idx in new_axis_pos:
                coord_array = np.expand_dims(coord_array, idx)
            coord_array = np.broadcast_to(coord_array, cube.shape)
            new_data.append(coord_array.ravel())
            names.append(coord)

        # Check if data is empty if necessary
        if self._cfg.get('use_only_coords_as_features') and not new_data:
            raise ValueError("No data found, 'use_only_coords_as_features' "
                             "can only be used when 'use_coords_as_feature' "
                             "is specified")
        return (new_data, names)

    def _group_training_datasets(self, datasets):
        """Group input datasets (one GBRT model for every climate model)."""
        return group_metadata(datasets, 'dataset')


def main(cfg):
    """Run the diagnostic."""
    gbrt = IntraModelGBRT(cfg)
    logger.info("Initialized GBRT model with parameters %s", gbrt.parameters)

    # Fit and predict
    gbrt.fit()
    gbrt.predict()

    # Plots
    gbrt.plot_feature_importance()
    gbrt.plot_partial_dependence()
    gbrt.plot_prediction_error()


# Run main function when this script is called
if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
