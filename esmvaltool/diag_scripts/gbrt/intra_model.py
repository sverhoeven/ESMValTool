#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""Diagnostic script to create GBRT models for every climate model.

Description
-----------
This diagnostic performs the machine learning technique "Gradient Boosted
Regression Trees" (GBRT) for climate predictions for every individual climate
model given in the recipe.

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

import esmvaltool.diag_scripts.shared.plot.gbrt as plot
from esmvaltool.diag_scripts.shared.gbrt import GBRTBase
from esmvaltool.diag_scripts.shared import (group_metadata, run_diagnostic,
                                            select_metadata, save_iris_cube)


logger = logging.getLogger(os.path.basename(__file__))


class IntraModelGBRT(GBRTBase):
    """Intra-model GBRT diagnostic."""

    def _collect_x_data(self, datasets, var_type):
        """Collect x data from `datasets`."""
        x_data = []
        names = []
        coords = None
        cube = None
        skipped_datasets = []

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
                raise ValueError("Expected at least one '{}' dataset without "
                                 "the option 'broadcast_from'".format(
                                     var_type))
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
                new_axis_pos = np.delete(np.arange(len(target_shape)),
                                         dataset['broadcast_from'])
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
        for (coord, coord_idx) in self._cfg.get(
                'use_coords_as_feature', {}).items():
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


    def _group_input_datasets(self, datasets):
        """Group input datasets (one GBRT model for every climate model)."""
        return group_metadata(datasets, 'dataset')


def main(cfg):
    """Run the diagnostic."""
    gbrt = IntraModelGBRT(cfg)
    logger.info("Initialized GBRT model with parameters %s",
               gbrt.parameters)
    gbrt.fit()


# def _extract_data(input_data, var_type, cfg):
#     """Extract required var_type data from input_data."""
#     input_data = select_metadata(input_data, var_type=var_type)
#     new_data = []
#     names = []
#     coords = None
#     cube = None
#     skipped_data = []

#     # Iterate over data
#     for data in input_data:
#         if 'broadcast_from' in data:
#             skipped_data.append(data)
#             continue
#         cube = iris.load_cube(data['filename'])
#         name = data.get('label', data['short_name'])
#         if coords is None:
#             coords = cube.coords()
#         else:
#             if cube.coords() != coords:
#                 raise ValueError("Expected fields with identical coordinates "
#                                  "but '{}' for dataset '{}' ('{}') is "
#                                  "differing, consider regridding or the "
#                                  "option 'broadcast_from'".format(
#                                      name, data['dataset'], var_type))
#         if not cfg.get('use_only_coords_as_features'):
#             new_data.append(cube.data.ravel())
#             names.append(name)

#     # Process skipped data (which needs broadcasting)
#     broadcast_data = _add_broadcasted_data(skipped_data, cube, var_type, cfg)
#     new_data.extend(broadcast_data[0])
#     names.extend(broadcast_data[1])

#     return (new_data, names, cube)


# def _add_broadcasted_data(input_data, cube, var_type, cfg):
#     """Add data with the attribute 'broadcast_from'."""
#     new_data = []
#     names = []
#     if input_data and cube is None:
#         raise ValueError("Expected at least one '{}' dataset without the "
#                          "option 'broadcast_from'".format(var_type))
#     for data in input_data:
#         cube_to_broadcast = iris.load_cube(data['filename'])
#         data_to_broadcast = cube_to_broadcast.data
#         name = data.get('label', data['short_name'])
#         try:
#             new_axis_pos = np.delete(np.arange(len(cube.shape)),
#                                      data['broadcast_from'])
#         except IndexError:
#             raise ValueError("Broadcasting failed for '{}', index out of "
#                              "bounds".format(name))
#         logger.info("Broadcasting %s '%s' from %s to %s", var_type, name,
#                     data_to_broadcast.shape, cube.shape)
#         for idx in new_axis_pos:
#             data_to_broadcast = np.expand_dims(data_to_broadcast, idx)
#         data_to_broadcast = np.broadcast_to(data_to_broadcast, cube.shape)
#         if not cfg.get('use_only_coords_as_features'):
#             new_data.append(data_to_broadcast.ravel())
#             names.append(name)
#     return (new_data, names)


# def _add_coordinates(cube, cfg):
#     """Add coordinate variables to x data."""
#     x_data = []
#     names = []
#     if cube is None:
#         return (x_data, names)
#     for (coord, coord_idx) in cfg.get('use_coords_as_feature', {}).items():
#         coord_array = cube.coord(coord).points
#         try:
#             new_axis_pos = np.delete(np.arange(len(cube.shape)), coord_idx)
#         except IndexError:
#             raise ValueError("'use_coords_as_feature' failed, index '{}' is "
#                              "out of bounds for coordinate "
#                              "'{}'".format(coord_idx, coord))
#         for idx in new_axis_pos:
#             coord_array = np.expand_dims(coord_array, idx)
#         coord_array = np.broadcast_to(coord_array, cube.shape)
#         x_data.append(coord_array.ravel())
#         names.append(coord)
#     return (x_data, names)


# def extract_x_data(input_data, var_type, cfg, required_features=None):
#     """Extract x data from input files."""
#     allowed_types = ('feature', 'prediction_input')
#     if var_type not in allowed_types:
#         raise ValueError("Excepted one of '{}' for 'var_type', got "
#                          "'{}'".format(allowed_types, var_type))

#     # Extract regular data
#     (x_data, names, cube) = _extract_data(input_data, var_type, cfg)

#     # Add coordinate variables if desired
#     coord_data = _add_coordinates(cube, cfg)
#     x_data.extend(coord_data[0])
#     names.extend(coord_data[1])

#     # Check if data was found
#     if not x_data:
#         raise ValueError("No '{}' data found, maybe you used "
#                          "'use_only_coords_as_features' but did not specify "
#                          "any coordinates in "
#                          "'use_coords_as_feature'".format(var_type))

#     # Check if all required features are available (necessary for prediction)
#     if required_features is not None:
#         if len(names) > len(set(names)):
#             raise ValueError("Expected exactly one dataset for every feature "
#                              "for '{}', got duplicates".format(var_type))
#         if set(required_features) != set(names):
#             raise ValueError("Expected features '{}' for '{}' got "
#                              "'{}'".format(required_features, var_type, names))

#     # Return data, labels and last cube
#     x_data = np.array(x_data)
#     if x_data.ndim > 1:
#         x_data = np.swapaxes(x_data, 0, 1)
#     names = np.array(names)
#     return (x_data, names, cube)


# def extract_y_data(input_data, dataset_name):
#     """Extract y data from input files."""
#     input_data = select_metadata(input_data, var_type='label',
#                                  dataset=dataset_name)
#     if len(input_data) == 1:
#         input_data = input_data[0]
#     else:
#         raise ValueError("Expected exactly one dataset with var_type "
#                          "'label' for dataset '{}', got "
#                          "{}".format(dataset_name, len(input_data)))
#     label_cube = iris.load_cube(input_data['filename'])
#     return (label_cube.data.ravel(), label_cube)


# def main(cfg):
#     """Run the diagnostic."""
#     input_data = list(cfg['input_data'].values())
#     input_data.extend(gbrt.get_ancestor_data(cfg))

#     # Extract datasets and variables
#     all_features = select_metadata(input_data, var_type='feature')
#     all_labels = select_metadata(input_data, var_type='label')

#     # GBRT for every dataset (climate model)
#     for (model_name, features) in group_metadata(
#             all_features, 'dataset').items():
#         logger.info("Processing %s", model_name)

#         # Extract features
#         (x_data, feature_names, _) = extract_x_data(features, 'feature', cfg)

#         # Extract labels
#         (y_data, label_cube) = extract_y_data(input_data, model_name)

#         # Separate training and test data
#         (x_train, x_test, y_train, y_test) = train_test_split(x_data, y_data)

#         # Create regression model with desired parameters
#         params = {'n_estimators': 1000, 'max_depth': 4, 'min_samples_split': 2,
#                   'learning_rate': 0.01, 'loss': 'ls'}
#         params.update(cfg.get('parameters', {}))
#         logger.info("Use parameters %s for GBRT model", params)
#         clf = GradientBoostingRegressor(**params)
#         clf.fit(x_train, y_train)

#         # Compute test set deviance
#         test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
#         for (idx, y_pred) in enumerate(clf.staged_predict(x_test)):
#             test_score[idx] = clf.loss_(y_test, y_pred)

#         # Plots
#         plot.plot_prediction_error(clf, test_score, cfg,
#                                    filename='{}_prediction_error'.format(
#                                        model_name))
#         plot.plot_feature_importance(clf, feature_names, cfg,
#                                      filename='{}_feature_importance'.format(
#                                          model_name))
#         plot.plot_partial_dependence(clf, x_train, feature_names, cfg,
#                                      filename='{}_partial_dependence'.format(
#                                          model_name))

#         # Prediction
#         (x_pred, _, cube) = extract_x_data(input_data, 'prediction_input', cfg,
#                                            required_features=feature_names)
#         prediction = clf.predict(x_pred)
#         cube.data = prediction.reshape(cube.shape)
#         cube.attributes.update(params)
#         cube.var_name = label_cube.var_name
#         cube.standard_name = label_cube.standard_name
#         cube.long_name = label_cube.long_name
#         logger.info("Prediction successful:")
#         logger.info(cube)

#         # Save cubes if desired
#         if cfg['write_netcdf']:
#             description = 'Prediction from GBRT model of {}'.format(
#                 model_name)
#             cube.attributes['description'] = description
#             new_path = os.path.join(cfg['work_dir'],
#                                     '{}_gbrt_prediction.nc'.format(model_name))
#             save_iris_cube(cube, new_path, cfg)

#             # Corresponding 'label' cube
#             label_data = select_metadata(all_labels, dataset=model_name)[0]
#             label_cube = iris.load_cube(label_data['filename'])
#             exp = label_data.get('exp', '')
#             description = '{} projection of {}'.format(exp, model_name)
#             label_cube.attributes['description'] = description
#             new_path = os.path.join(cfg['work_dir'],
#                                     '{}_{}_projection.nc'.format(model_name,
#                                                                  exp))
#             save_iris_cube(cube, new_path, cfg)


# Run main function when this script is called
if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
