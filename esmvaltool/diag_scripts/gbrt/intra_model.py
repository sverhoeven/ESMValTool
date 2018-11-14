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
shape, also all 'prediction_input' datasets, but these two might be different.

Author
------
Manuel Schlund (DLR, Germany)

Project
-------
CRESCENDO

Configuration options in recipe
-------------------------------
allow_broadcasting : bool, optional
    Allow broadcasting for feature arrays (e.g. expance T2Ms field to T3M,
    default: False).
use_coord givefeature : dict, optional
    Use coordinates (e.g. 'air_pressure' or 'latitude') as features, coordinate
    names are given by the dictionary keys, the associated index by the
    dictionary values.

"""


import logging
import os
from pprint import pprint

import iris
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.model_selection import train_test_split
import xgboost as xgb

from esmvaltool.diag_scripts.shared import (group_metadata, run_diagnostic,
                                            select_metadata)
from esmvaltool.preprocessor._derive.uajet import DerivedVariable as Uajet

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger(os.path.basename(__file__))


def extract_x_data(input_data, var_type, cfg, required_features=None):
    """Extract x data from input files."""
    allowed_types = ('feature', 'prediction_input')
    if var_type not in allowed_types:
        raise ValueError("Excepted one of '{}' for 'var_type', got "
                         "'{}'".format(allowed_types, var_type))
    input_data = select_metadata(input_data, var_type=var_type)
    x_data = []
    names = []
    coords = None
    cube = None

    # Iterate over input data
    for data in input_data:
        cube = iris.load_cube(data['filename'])
        name = data.get('label', data['short_name'])
        if coords is None:
            coords = cube.coords()
        else:
            if cube.coords() != coords:
                raise ValueError("Expected x fields with identical "
                                 "coordinates, but '{}' for dataset '{}' is "
                                 "differing".format(name, data['dataset']))
        x_data.append(cube.data.ravel())
        names.append(name)

    # Add coordinate variables if desired
    for (coord, coord_idx) in cfg.get('use_coords_as_feature', []).items():
        coord_array = cube.coord(coord).points
        try:
            new_axis_pos = np.delete(np.arange(len(cube.shape)), coord_idx)
        except IndexError:
            raise ValueError("Coordinate index '{}' out of bounds for "
                             "coordinate '{}'".format(coord_idx, coord))
        for idx in new_axis_pos:
            coord_array = np.expand_dims(coord_array, idx)
        coord_array = np.broadcast_to(coord_array, cube.shape)
        x_data.append(coord_array.ravel())
        names.append(coord)

    # Check if all required features are available (necessary for prediction)
    if required_features is not None:
        if len(names) > len(set(names)):
            raise ValueError("Expected exactly one dataset for every feature "
                             "for '{}', got duplicates".format(var_type))
        if set(required_features) != set(names):
            raise ValueError("Expected features '{}' for '{}' got "
                             "'{}'".format(required_features, var_type, names))

    # Return data, labels and last cube
    x_data = np.array(x_data)
    if x_data.ndim > 1:
        x_data = np.swapaxes(x_data, 0, 1)
    names = np.array(names)
    return (x_data, names, cube)


def extract_y_data(input_data, dataset_name):
    """Extract y data from input files."""
    input_data = select_metadata(input_data, var_type='label',
                                 dataset=dataset_name)
    if len(input_data) == 1:
        input_data = input_data[0]
    else:
        raise ValueError("Expected exactly one dataset with var_type "
                         "'label' for dataset '{}', got "
                         "{}".format(dataset_name, len(input_data)))
    label_cube = iris.load_cube(input_data['filename'])
    return (label_cube.data.ravel(), label_cube)


def main(cfg):
    """Run the diagnostic."""
    input_data = cfg['input_data'].values()

    # Extract datasets and variables
    all_features = select_metadata(input_data, var_type='feature')
    all_labels = select_metadata(input_data, var_type='label')

    # GBRT for every dataset (climate model)
    for (model_name, features) in group_metadata(
            all_features, 'dataset').items():
        logger.info("Processing %s", model_name)

        # Extract features
        (x_data, feature_names, _) = extract_x_data(features, 'feature', cfg)

        # Extract labels
        (y_data, label_cube) = extract_y_data(input_data, model_name)

        # Separate training and test data
        (x_train, x_test, y_train, y_test) = train_test_split(x_data, y_data)

        # Create regression model
        params = {'n_estimators': 1000, 'max_depth': 4, 'min_samples_split': 2,
                  'learning_rate': 0.01, 'loss': 'ls'}
        clf = GradientBoostingRegressor(**params)
        clf.fit(x_train, y_train)

        # Plot training deviance
        if cfg['write_plots']:
            (_, axes) = plt.subplots()

            # Compute test set deviance
            test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
            for (idx, y_pred) in enumerate(clf.staged_predict(x_test)):
                test_score[idx] = clf.loss_(y_test, y_pred)

            # Plot
            axes.plot(np.arange(params['n_estimators']) + 1, clf.train_score_,
                      'b-', label='Training Set Deviance')
            axes.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
                      label='Test Set Deviance')
            axes.legend(loc='upper right')
            axes.set_title('Deviance')
            axes.set_xlabel('Boosting Iterations')
            axes.set_ylabel('Deviance')
            new_path = os.path.join(cfg['plot_dir'],
                                    '{}_prediction_error.{}'.format(
                                        model_name, cfg['output_file_type']))
            plt.savefig(new_path, orientation='landscape', bbox_inches='tight')
            logger.info("Wrote %s", new_path)
            plt.close()

        # Plot feature importance
        if cfg['write_plots']:
            (_, axes) = plt.subplots()
            feature_importance = clf.feature_importances_
            sorted_idx = np.argsort(feature_importance)
            pos = np.arange(sorted_idx.shape[0]) + 0.5
            axes.barh(pos, feature_importance[sorted_idx], align='center')
            axes.set_title('Variable Importance')
            axes.set_yticks(pos)
            axes.set_yticklabels(feature_names[sorted_idx])
            axes.set_xlabel('Relative Importance')
            new_path = os.path.join(cfg['plot_dir'],
                                    '{}_relative_importance.{}'.format(
                                        model_name, cfg['output_file_type']))
            plt.savefig(new_path, orientation='landscape', bbox_inches='tight')
            logger.info("Wrote %s", new_path)
            plt.close()

        # Plot partial dependence
        if cfg['write_plots']:
            for (idx, feature_name) in enumerate(feature_names):
                (_, [axes]) = plot_partial_dependence(clf, x_train, [idx])
                axes.set_title('Partial dependence')
                axes.set_xlabel(feature_name)
                axes.set_ylabel('Eastward wind')
                new_path = os.path.join(cfg['plot_dir'],
                                        '{}_partial_dependence_of_{}.'
                                        '{}'.format(model_name, feature_name,
                                                    cfg['output_file_type']))
                plt.savefig(new_path, orientation='landscape',
                            bbox_inches='tight')
                logger.info("Wrote %s", new_path)
                plt.close()

        # Prediction
        (x_pred, _, cube) = extract_x_data(input_data, 'prediction_input', cfg,
                                           required_features=feature_names)
        prediction = clf.predict(x_pred)
        cube.data = prediction.reshape(cube.shape)
        cube.attributes.update(params)
        cube.var_name = label_cube.var_name
        cube.standard_name = label_cube.standard_name
        cube.long_name = label_cube.long_name
        logger.info("Prediction successful:")
        logger.info(cube)

        # Save cube if desired
        if cfg['write_netcdf']:
            new_path = os.path.join(cfg['work_dir'],
                                    '{}_prediction.nc'.format(model_name))
            iris.save(cube, new_path)
            logger.info("Wrote %s", new_path)

        # Calculate jet positions
        uajet = Uajet('ua')
        cubes = iris.cube.CubeList([cube])
        predicted_uajet = uajet.calculate(cubes)
        label = select_metadata(all_labels, dataset=model_name)[0]
        cubes = iris.cube.CubeList([iris.load_cube(label['filename'])])
        label_uajet = uajet.calculate(cubes)
        logger.info("Predicted jet position: %f", predicted_uajet.data)
        logger.info("Model projected jet position: %f", label_uajet.data)


# Run main function when this script is called
if __name__ == '__main__':
    with run_diagnostic() as config:
        main(config)
